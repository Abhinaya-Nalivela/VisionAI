from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.analyze import router as analyze_router
from app.api.history import router as history_router

from app.db.database import Base, engine
from app.db import models


# ---------------------------------------------------------
# FASTAPI APPLICATION
# ---------------------------------------------------------

app = FastAPI(
    title="VisionAI API",
    version="1.0.0",
    description=(
        "AI-powered image quality and defect detection API "
        "using computer vision and machine learning."
    ),
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
# Allows our React frontend to communicate with the backend.
# We can restrict this to the frontend URL before deployment.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# DATABASE INITIALIZATION
# ---------------------------------------------------------
# Creates the SQLite tables if they do not already exist.

Base.metadata.create_all(
    bind=engine
)


# ---------------------------------------------------------
# ROUTERS
# ---------------------------------------------------------

app.include_router(health_router)
app.include_router(analyze_router)
app.include_router(history_router)


# ---------------------------------------------------------
# ROOT ENDPOINT
# ---------------------------------------------------------

@app.get("/")
def root():
    """
    Basic API status endpoint.
    """

    return {
        "message": "VisionAI API is running",
        "status": "ok",
        "version": "1.0.0",
    }