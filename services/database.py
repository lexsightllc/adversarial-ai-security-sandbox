# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from datetime import datetime
import os

from sqlalchemy import Column, String, Float, DateTime, Boolean, JSON, Integer, create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@postgres:5432/adversarial_sandbox_db"
)

# Configure connection pooling for better performance
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
    pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),  # Recycle connections after 1 hour
    pool_pre_ping=True,  # Test connections before using them
    echo=os.getenv("DB_ECHO", "False").lower() == "true",
    connect_args={
        "connect_timeout": 10,
        "options": "-c statement_timeout=30000",  # 30 second statement timeout
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Event listeners for connection pool management
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Set SQLite-specific settings (for testing)."""
    # This is useful for SQLite in-memory databases during tests
    if "sqlite" in str(dbapi_conn):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


@event.listens_for(engine, "pool_connect")
def receive_pool_connect(dbapi_conn, connection_record):
    """Log pool connection events."""
    connection_record.info["pool_id"] = id(dbapi_conn)

class Model(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    type = Column(String, nullable=False)
    version = Column(String, nullable=False)
    description = Column(String, nullable=True)
    status = Column(String, default="active")
    model_file_url = Column(String, nullable=True)
    metadata_data = Column("metadata", JSON, nullable=True)
    input_schema = Column(JSON, nullable=True)
    output_schema = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Model(id='{self.id}', name='{self.name}', type='{self.type}')>"


def _get_model_metadata(self):
    return self.metadata_data


def _set_model_metadata(self, value):
    self.metadata_data = value


Model.metadata = property(_get_model_metadata, _set_model_metadata)


class AttackResult(Base):
    __tablename__ = "attack_results"

    id = Column(String, primary_key=True, index=True)
    model_id = Column(String, index=True, nullable=False)
    attack_method_id = Column(String, nullable=False)
    original_input = Column(String, nullable=False)
    original_prediction = Column(String, nullable=False)
    original_confidence = Column(Float, nullable=False)
    adversarial_example = Column(String, nullable=False)
    adversarial_prediction = Column(String, nullable=False)
    adversarial_confidence = Column(Float, nullable=False)
    attack_success = Column(Boolean, nullable=False)
    perturbation_details = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    status = Column(String, default="completed")
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<AttackResult(id='{self.id}', model_id='{self.model_id}', status='{self.status}')>"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_db_and_tables():
    Base.metadata.create_all(bind=engine)
