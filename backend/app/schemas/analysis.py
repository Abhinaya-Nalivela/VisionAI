from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel


# ---------------------------------------------------------
# ISSUE RESPONSE
# ---------------------------------------------------------

class IssueResponse(BaseModel):
    type: str
    confidence: float


# ---------------------------------------------------------
# IMAGE STATISTICS
# ---------------------------------------------------------

class ImageStatistics(BaseModel):

    # Basic image information
    width: int
    height: int
    channels: int

    # -----------------------------------------------------
    # GLOBAL IMAGE QUALITY FEATURES
    # -----------------------------------------------------

    brightness: float
    contrast: float
    sharpness: float

    dark_pixel_ratio: float
    bright_pixel_ratio: float

    saturation: float
    edge_density: float
    noise_level: float
    entropy: float

    # -----------------------------------------------------
    # ADDITIONAL TEXTURE / LOCAL FEATURES
    # -----------------------------------------------------

    gradient_mean: float
    gradient_std: float
    strong_edge_ratio: float

    local_contrast_mean: float
    local_contrast_std: float

    patch_intensity_std: float
    patch_intensity_range: float

    extreme_pixel_ratio: float
    laplacian_extreme_ratio: float


# ---------------------------------------------------------
# ANALYSIS RESPONSE
# ---------------------------------------------------------

class AnalysisResponse(BaseModel):

    filename: str

    quality_score: int
    quality_label: str

    predicted_class: str
    confidence: float

    severity: str

    issues: List[IssueResponse]

    probabilities: Dict[str, float]

    image_statistics: ImageStatistics

    analysis_id: int


# ---------------------------------------------------------
# HISTORY
# ---------------------------------------------------------

class HistoryItem(AnalysisResponse):
    created_at: datetime


class HistoryResponse(BaseModel):
    count: int
    items: List[HistoryItem]