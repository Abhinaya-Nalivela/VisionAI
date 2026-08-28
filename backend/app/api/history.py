import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import AnalysisRecord
from app.schemas.analysis import HistoryResponse


router = APIRouter(
    prefix="/api",
    tags=["history"],
)


@router.get(
    "/history",
    response_model=HistoryResponse,
)
def get_analysis_history(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    db: Session = Depends(get_db),
):
    """
    Return previously stored image analysis results.
    """

    records = (
        db.query(AnalysisRecord)
        .order_by(AnalysisRecord.created_at.desc())
        .limit(limit)
        .all()
    )

    history = []

    for record in records:
        history.append(
            {
                "analysis_id": record.id,
                "filename": record.filename,
                "quality_score": record.quality_score,
                "quality_label": record.quality_label,
                "predicted_class": record.predicted_class,
                "confidence": record.confidence,
                "severity": record.severity,
                "issues": json.loads(record.issues_json),
                "probabilities": json.loads(
                    record.probabilities_json
                ),
                "image_statistics": json.loads(
                    record.statistics_json
                ),
                "created_at": record.created_at,
            }
        )

    return {
        "count": len(history),
        "items": history,
    }


@router.delete(
    "/history/{analysis_id}",
    status_code=status.HTTP_200_OK,
)
def delete_analysis_history_item(
    analysis_id: int,
    db: Session = Depends(get_db),
):
    """
    Delete one analysis history record.
    """

    record = (
        db.query(AnalysisRecord)
        .filter(AnalysisRecord.id == analysis_id)
        .first()
    )

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis history record not found",
        )

    db.delete(record)
    db.commit()

    return {
        "message": "History record deleted successfully",
        "analysis_id": analysis_id,
    }


@router.delete(
    "/history",
    status_code=status.HTTP_200_OK,
)
def clear_analysis_history(
    db: Session = Depends(get_db),
):
    """
    Delete all stored analysis history records.
    """

    deleted_count = db.query(AnalysisRecord).delete(
        synchronize_session=False
    )

    db.commit()

    return {
        "message": "Analysis history cleared successfully",
        "deleted_count": deleted_count,
    }