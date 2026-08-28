from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.db.database import Base


class AnalysisRecord(Base):
    """
    Stores the result of every image analysis.
    """

    __tablename__ = "analysis_records"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    filename = Column(
        String(255),
        nullable=False,
    )

    quality_score = Column(
        Integer,
        nullable=False,
    )

    quality_label = Column(
        String(50),
        nullable=False,
    )

    predicted_class = Column(
        String(50),
        nullable=False,
    )

    confidence = Column(
        Float,
        nullable=False,
    )

    severity = Column(
        String(50),
        nullable=False,
    )

    issues_json = Column(
        Text,
        nullable=False,
        default="[]",
    )

    probabilities_json = Column(
        Text,
        nullable=False,
        default="{}",
    )

    statistics_json = Column(
        Text,
        nullable=False,
        default="{}",
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )