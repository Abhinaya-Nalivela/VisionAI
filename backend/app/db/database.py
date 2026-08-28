import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# ---------------------------------------------------------
# DATABASE LOCATION
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[2]

DATABASE_PATH = Path(
    os.getenv(
        "DATABASE_PATH",
        str(BACKEND_DIR / "visionai.db"),
    )
)

DATABASE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

DATABASE_URL = (
    f"sqlite:///{DATABASE_PATH.as_posix()}"
)


# ---------------------------------------------------------
# SQLALCHEMY ENGINE
# ---------------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    },
)


# ---------------------------------------------------------
# DATABASE SESSION
# ---------------------------------------------------------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ---------------------------------------------------------
# BASE MODEL
# ---------------------------------------------------------

Base = declarative_base()


# ---------------------------------------------------------
# FASTAPI DATABASE DEPENDENCY
# ---------------------------------------------------------

def get_db():
    """
    Create a database session for each request
    and close it after the request finishes.
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()