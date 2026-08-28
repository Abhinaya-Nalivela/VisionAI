from fastapi import APIRouter


router = APIRouter(
    tags=["health"]
)


@router.get("/health")
def health_check():
    """
    Check whether the VisionAI backend is running.
    """

    return {
        "status": "ok",
        "service": "VisionAI API",
        "version": "1.0.0"
    }