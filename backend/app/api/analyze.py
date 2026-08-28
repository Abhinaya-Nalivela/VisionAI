import json

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)

from sqlalchemy.orm import Session

import cv2
import numpy as np

from app.db.database import get_db
from app.db.models import AnalysisRecord
from app.services.feature_extraction import extract_features
from app.services.model_service import predict_quality
from app.services.local_anomaly import (
    extract_local_anomaly_features,
)
from app.schemas.analysis import AnalysisResponse


# ---------------------------------------------------------
# ROUTER
# ---------------------------------------------------------

router = APIRouter(
    prefix="/api",
    tags=["analysis"],
)


# ---------------------------------------------------------
# UPLOAD CONFIGURATION
# ---------------------------------------------------------

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

MAX_FILE_SIZE = 10 * 1024 * 1024


# ---------------------------------------------------------
# MODEL CLASSES
# ---------------------------------------------------------

DEGRADATION_CLASSES = {
    "blur",
    "underexposed",
    "overexposed",
    "noise",
    "degraded",
}


# ---------------------------------------------------------
# UTILITY
# ---------------------------------------------------------

def clamp(
    value,
    minimum=0.0,
    maximum=1.0,
):
    """
    Keep a numeric value between minimum and maximum.
    """

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


# ---------------------------------------------------------
# CV EVIDENCE
# ---------------------------------------------------------

def calculate_cv_evidence(
    features,
):
    """
    Build simple interpretable CV evidence.

    The Random Forest remains the primary quality
    classifier.

    These values provide supporting evidence and
    explainability for the final result.
    """

    brightness = float(
        features.get(
            "brightness",
            0.0,
        )
    )

    sharpness = float(
        features.get(
            "sharpness",
            0.0,
        )
    )

    noise_level = float(
        features.get(
            "noise_level",
            0.0,
        )
    )

    dark_pixel_ratio = float(
        features.get(
            "dark_pixel_ratio",
            0.0,
        )
    )

    bright_pixel_ratio = float(
        features.get(
            "bright_pixel_ratio",
            0.0,
        )
    )

    # -----------------------------------------------------
    # BLUR
    # -----------------------------------------------------

    if sharpness <= 25:
        blur = 1.0

    elif sharpness >= 150:
        blur = 0.0

    else:
        blur = (
            150.0 - sharpness
        ) / 125.0

    # -----------------------------------------------------
    # UNDEREXPOSURE
    # -----------------------------------------------------

    brightness_darkness = clamp(
        (
            85.0 - brightness
        ) / 50.0
    )

    dark_pixel_support = clamp(
        (
            dark_pixel_ratio - 0.20
        ) / 0.35
    )

    underexposed = clamp(
        brightness_darkness * 0.75
        + dark_pixel_support * 0.25
    )

    # -----------------------------------------------------
    # OVEREXPOSURE
    # -----------------------------------------------------

    brightness_excess = clamp(
        (
            brightness - 180.0
        ) / 45.0
    )

    bright_pixel_support = clamp(
        (
            bright_pixel_ratio - 0.20
        ) / 0.35
    )

    overexposed = clamp(
        brightness_excess * 0.75
        + bright_pixel_support * 0.25
    )

    # -----------------------------------------------------
    # NOISE
    # -----------------------------------------------------

    noise = clamp(
        (
            noise_level - 10.0
        ) / 15.0
    )

    return {
        "blur": round(
            blur,
            4,
        ),

        "underexposed": round(
            underexposed,
            4,
        ),

        "overexposed": round(
            overexposed,
            4,
        ),

        "noise": round(
            noise,
            4,
        ),
    }


# ---------------------------------------------------------
# FINAL DECISION
# ---------------------------------------------------------

def make_final_decision(
    ml_prediction,
    cv_evidence,
    local_anomaly,
):
    """
    Produce the final application decision.

    Design:

    1. The trained 6-class Random Forest is the primary
       AI decision system.

    2. CV evidence supports explainability and can confirm
       very strong measurable degradation.

    3. Local anomaly detection is deliberately conservative
       because evaluation showed that natural scene texture
       can overlap with synthetic localized defects.
    """

    ml_class = str(
        ml_prediction[
            "predicted_class"
        ]
    )

    ml_confidence = float(
        ml_prediction[
            "confidence"
        ]
    )

    probabilities = (
        ml_prediction[
            "probabilities"
        ]
    )

    clean_probability = float(
        probabilities.get(
            "clean",
            0.0,
        )
    )

    # -----------------------------------------------------
    # 1. RANDOM FOREST DEGRADATION
    # -----------------------------------------------------

    if (
        ml_class
        in DEGRADATION_CLASSES
    ):

        return {
            "final_class": (
                ml_class
            ),

            "confidence": round(
                ml_confidence,
                4,
            ),

            "quality_label": (
                "DEGRADED"
            ),

            "decision_source": (
                "random_forest"
            ),
        }

    # -----------------------------------------------------
    # 2. STRONG CV QUALITY PROBLEM
    # -----------------------------------------------------
    #
    # This is only a safety check for an obvious measurable
    # quality problem when the RF predicted clean.
    # -----------------------------------------------------

    strongest_cv_class = max(
        cv_evidence,
        key=cv_evidence.get,
    )

    strongest_cv_score = float(
        cv_evidence[
            strongest_cv_class
        ]
    )

    if strongest_cv_score >= 0.85:

        ml_support = float(
            probabilities.get(
                strongest_cv_class,
                0.0,
            )
        )

        confidence = clamp(
            strongest_cv_score * 0.70
            + ml_support * 0.30
        )

        return {
            "final_class": (
                strongest_cv_class
            ),

            "confidence": round(
                confidence,
                4,
            ),

            "quality_label": (
                "DEGRADED"
            ),

            "decision_source": (
                "strong_cv_override"
            ),
        }

    # -----------------------------------------------------
    # 3. CONSERVATIVE POTENTIAL DEFECT SIGNAL
    # -----------------------------------------------------
    #
    # IMPORTANT:
    #
    # The local anomaly detector is NOT treated as a
    # validated defect classifier.
    #
    # Our held-out evaluation showed substantial overlap
    # between clean and synthetic defect images.
    #
    # Therefore it can only flag an image when:
    #
    # - RF strongly believes the image is globally clean
    # - no strong global quality degradation exists
    # - localized anomaly evidence is extremely high
    #
    # This keeps false defect warnings conservative.
    # -----------------------------------------------------

    local_anomaly_score = float(
        local_anomaly.get(
            "local_anomaly_score",
            0.0,
        )
    )

    if (
        clean_probability >= 0.75
        and strongest_cv_score < 0.60
        and local_anomaly_score >= 0.90
    ):

        defect_confidence = clamp(
            local_anomaly_score * 0.60
            + clean_probability * 0.40
        )

        return {
            "final_class": (
                "potential_visual_defect"
            ),

            "confidence": round(
                defect_confidence,
                4,
            ),

            "quality_label": (
                "POTENTIALLY_DEFECTIVE"
            ),

            "decision_source": (
                "conservative_local_anomaly"
            ),
        }

    # -----------------------------------------------------
    # 4. ACCEPTABLE
    # -----------------------------------------------------

    confidence = max(
        clean_probability,
        ml_confidence,
    )

    confidence = max(
        confidence,
        0.50,
    )

    return {
        "final_class": (
            "clean"
        ),

        "confidence": round(
            clamp(
                confidence
            ),
            4,
        ),

        "quality_label": (
            "ACCEPTABLE"
        ),

        "decision_source": (
            "random_forest_clean"
        ),
    }


# ---------------------------------------------------------
# SEVERITY
# ---------------------------------------------------------

def determine_severity(
    quality_label,
    confidence,
):
    """
    Convert the final decision into a severity level.
    """

    confidence = clamp(
        confidence
    )

    if (
        quality_label
        == "ACCEPTABLE"
    ):
        return "LOW"

    if (
        quality_label
        == "POTENTIALLY_DEFECTIVE"
    ):

        if confidence >= 0.85:
            return "HIGH"

        return "MEDIUM"

    if confidence >= 0.80:
        return "HIGH"

    if confidence >= 0.55:
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------
# QUALITY SCORE
# ---------------------------------------------------------

def calculate_quality_score(
    quality_label,
    confidence,
):
    """
    Convert the decision into a user-friendly 0-100 score.

    Higher score = better image quality.
    """

    confidence = clamp(
        confidence
    )

    if (
        quality_label
        == "ACCEPTABLE"
    ):

        score = (
            70
            + confidence * 30
        )

    elif (
        quality_label
        == "DEGRADED"
    ):

        score = (
            70
            - confidence * 40
        )

    else:

        score = (
            45
            - confidence * 25
        )

    return int(
        round(
            max(
                0,
                min(
                    100,
                    score,
                ),
            )
        )
    )


# ---------------------------------------------------------
# ISSUES
# ---------------------------------------------------------

def build_issues(
    final_class,
    quality_label,
    confidence,
):
    """
    Build frontend-friendly issue information.
    """

    if (
        quality_label
        == "ACCEPTABLE"
    ):
        return []

    if (
        quality_label
        == "POTENTIALLY_DEFECTIVE"
    ):

        return [
            {
                "type": (
                    "potential_visual_defect"
                ),

                "confidence": round(
                    float(
                        confidence
                    ),
                    4,
                ),
            }
        ]

    return [
        {
            "type": (
                final_class
            ),

            "confidence": round(
                float(
                    confidence
                ),
                4,
            ),
        }
    ]


# ---------------------------------------------------------
# ANALYZE ENDPOINT
# ---------------------------------------------------------

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
)
async def analyze_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):

    # -----------------------------------------------------
    # 1. VALIDATE CONTENT TYPE
    # -----------------------------------------------------

    if (
        file.content_type
        not in ALLOWED_CONTENT_TYPES
    ):

        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported image format. "
                "Use JPG, JPEG, PNG, or WEBP."
            ),
        )

    # -----------------------------------------------------
    # 2. READ FILE
    # -----------------------------------------------------

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file is empty."
            ),
        )

    if (
        len(contents)
        > MAX_FILE_SIZE
    ):

        raise HTTPException(
            status_code=413,
            detail=(
                "Image exceeds the "
                "10 MB size limit."
            ),
        )

    # -----------------------------------------------------
    # 3. DECODE IMAGE
    # -----------------------------------------------------

    image_buffer = np.frombuffer(
        contents,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_buffer,
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "Image could not be decoded. "
                "The file may be corrupted "
                "or invalid."
            ),
        )

    # -----------------------------------------------------
    # 4. EXTRACT 18 CV/ML FEATURES
    # -----------------------------------------------------

    try:

        features = (
            extract_features(
                image
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Feature extraction failed: "
                f"{error}"
            ),
        )

    # -----------------------------------------------------
    # 5. SIX-CLASS RANDOM FOREST
    # -----------------------------------------------------

    try:

        ml_prediction = (
            predict_quality(
                features
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Model inference failed: "
                f"{error}"
            ),
        )

    probabilities = (
        ml_prediction[
            "probabilities"
        ]
    )

    # -----------------------------------------------------
    # 6. INTERPRETABLE CV EVIDENCE
    # -----------------------------------------------------

    cv_evidence = (
        calculate_cv_evidence(
            features
        )
    )

    # -----------------------------------------------------
    # 7. LOCAL ANOMALY SUPPORT
    # -----------------------------------------------------

    try:

        local_anomaly = (
            extract_local_anomaly_features(
                image
            )
        )

    except Exception:

        # Local anomaly is supporting evidence only.
        # Failure should not prevent the primary quality
        # model from returning an analysis.
        local_anomaly = {
            "local_anomaly_score": 0.0
        }

    # -----------------------------------------------------
    # 8. FINAL DECISION
    # -----------------------------------------------------

    decision = (
        make_final_decision(
            ml_prediction,
            cv_evidence,
            local_anomaly,
        )
    )

    final_class = (
        decision[
            "final_class"
        ]
    )

    confidence = float(
        decision[
            "confidence"
        ]
    )

    quality_label = (
        decision[
            "quality_label"
        ]
    )

    # -----------------------------------------------------
    # 9. SEVERITY
    # -----------------------------------------------------

    severity = (
        determine_severity(
            quality_label,
            confidence,
        )
    )

    # -----------------------------------------------------
    # 10. ISSUES
    # -----------------------------------------------------

    issues = (
        build_issues(
            final_class,
            quality_label,
            confidence,
        )
    )

    # -----------------------------------------------------
    # 11. QUALITY SCORE
    # -----------------------------------------------------

    quality_score = (
        calculate_quality_score(
            quality_label,
            confidence,
        )
    )

    # -----------------------------------------------------
    # 12. RESPONSE
    # -----------------------------------------------------

    result = {
        "filename": (
            file.filename
        ),

        "quality_score": (
            quality_score
        ),

        "quality_label": (
            quality_label
        ),

        "predicted_class": (
            final_class
        ),

        "confidence": round(
            confidence,
            4,
        ),

        "severity": (
            severity
        ),

        "issues": (
            issues
        ),

        # These are the original six-class
        # Random Forest probabilities.
        "probabilities": (
            probabilities
        ),

        "image_statistics": (
            features
        ),
    }

    # -----------------------------------------------------
    # TERMINAL INFORMATION
    # -----------------------------------------------------

    print(
        "\n"
        "----------------------------------------"
    )

    print(
        f"Image: {file.filename}"
    )

    print(
        "RF prediction:",
        ml_prediction[
            "predicted_class"
        ],
        ml_prediction[
            "confidence"
        ],
    )

    print(
        "Final prediction:",
        final_class,
        confidence,
    )

    print(
        "Quality label:",
        quality_label,
    )

    print(
        "Decision source:",
        decision[
            "decision_source"
        ],
    )

    print(
        "CV evidence:",
        cv_evidence,
    )

    print(
        "Local anomaly:",
        local_anomaly.get(
            "local_anomaly_score",
            0.0,
        ),
    )

    print(
        "RF probabilities:",
        probabilities,
    )

    print(
        "----------------------------------------"
        "\n"
    )

    # -----------------------------------------------------
    # 13. SAVE TO SQLITE
    # -----------------------------------------------------

    try:

        record = AnalysisRecord(
            filename=(
                file.filename
            ),

            quality_score=(
                quality_score
            ),

            quality_label=(
                quality_label
            ),

            predicted_class=(
                final_class
            ),

            confidence=round(
                confidence,
                4,
            ),

            severity=(
                severity
            ),

            issues_json=json.dumps(
                issues
            ),

            probabilities_json=json.dumps(
                probabilities
            ),

            statistics_json=json.dumps(
                features
            ),
        )

        db.add(
            record
        )

        db.commit()

        db.refresh(
            record
        )

    except Exception as error:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save analysis: "
                f"{error}"
            ),
        )

    # -----------------------------------------------------
    # 14. DATABASE ID
    # -----------------------------------------------------

    result[
        "analysis_id"
    ] = record.id

    return result