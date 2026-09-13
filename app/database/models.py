from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from app.database.connection import Base


class Session(Base):
    __tablename__ = 'sessions'
    id = Column(String, primary_key=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    input_modality = Column(String)
    original_filename = Column(String)
    file_type = Column(String)
    file_size = Column(Integer)
    file_hash = Column(String)
    processing_status = Column(String)
    requested_operation = Column(String)
    user_query = Column(Text)
    processing_duration = Column(Float)


class ModelRun(Base):
    __tablename__ = 'model_runs'
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.id'))
    model_type = Column(String)
    model_name = Column(String)
    model_version = Column(String)
    framework = Column(String)
    device = Column(String)
    input_file_id = Column(String)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_ms = Column(Float)
    status = Column(String)
    error_message = Column(Text)


class RetrievalRun(Base):
    __tablename__ = 'retrieval_runs'
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.id'))
    query = Column(Text)
    created_at = Column(DateTime)


class WebSearchRun(Base):
    __tablename__ = 'web_search_runs'
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.id'))
    query = Column(Text)
    provider = Column(String)
    timestamp = Column(DateTime)
    result_count = Column(Integer)


class AuditEvent(Base):
    __tablename__ = 'audit_events'
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.id'))
    event_type = Column(String)
    agent = Column(String)
    crew = Column(String)
    graph_node = Column(String)
    model = Column(String)
    tool = Column(String)
    status = Column(String)
    duration = Column(Float)
    details = Column(Text)
