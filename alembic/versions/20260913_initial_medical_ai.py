"""initial medical ai metadata tables

Revision ID: 20260913_000001
Revises: 
Create Date: 2026-09-13 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = '20260913_000001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'sessions',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('created_at', sa.DateTime()),
        sa.Column('updated_at', sa.DateTime()),
        sa.Column('input_modality', sa.String()),
        sa.Column('original_filename', sa.String()),
        sa.Column('file_type', sa.String()),
        sa.Column('file_size', sa.Integer()),
        sa.Column('file_hash', sa.String()),
        sa.Column('processing_status', sa.String()),
        sa.Column('requested_operation', sa.String()),
        sa.Column('user_query', sa.Text()),
        sa.Column('processing_duration', sa.Float()),
    )
    op.create_table(
        'model_runs',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('session_id', sa.String(), sa.ForeignKey('sessions.id')),
        sa.Column('model_type', sa.String()),
        sa.Column('model_name', sa.String()),
        sa.Column('model_version', sa.String()),
        sa.Column('framework', sa.String()),
        sa.Column('device', sa.String()),
        sa.Column('input_file_id', sa.String()),
        sa.Column('started_at', sa.DateTime()),
        sa.Column('completed_at', sa.DateTime()),
        sa.Column('duration_ms', sa.Float()),
        sa.Column('status', sa.String()),
        sa.Column('error_message', sa.Text()),
    )
    op.create_table(
        'retrieval_runs',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('session_id', sa.String(), sa.ForeignKey('sessions.id')),
        sa.Column('query', sa.Text()),
        sa.Column('created_at', sa.DateTime()),
    )
    op.create_table(
        'web_search_runs',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('session_id', sa.String(), sa.ForeignKey('sessions.id')),
        sa.Column('query', sa.Text()),
        sa.Column('provider', sa.String()),
        sa.Column('timestamp', sa.DateTime()),
        sa.Column('result_count', sa.Integer()),
    )
    op.create_table(
        'audit_events',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('session_id', sa.String(), sa.ForeignKey('sessions.id')),
        sa.Column('event_type', sa.String()),
        sa.Column('agent', sa.String()),
        sa.Column('crew', sa.String()),
        sa.Column('graph_node', sa.String()),
        sa.Column('model', sa.String()),
        sa.Column('tool', sa.String()),
        sa.Column('status', sa.String()),
        sa.Column('duration', sa.Float()),
        sa.Column('details', sa.Text()),
    )


def downgrade():
    op.drop_table('audit_events')
    op.drop_table('web_search_runs')
    op.drop_table('retrieval_runs')
    op.drop_table('model_runs')
    op.drop_table('sessions')
