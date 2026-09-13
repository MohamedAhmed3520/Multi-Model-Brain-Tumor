from crewai import Task

from app.agents.agents import (
    AUDIO_AGENT,
    EVIDENCE_AGENT,
    RAG_AGENT,
    REPORT_AGENT,
    VISION_AGENT,
    WEB_AGENT,
)


VISION_TASK = Task(
    name='vision_analysis',
    description='Invoke classification, detection, and segmentation services and summarize structured evidence.',
    expected_output='Structured evidence summary with predictions and limitations.',
    agent=VISION_AGENT,
)

AUDIO_TASK = Task(
    name='speech_to_text',
    description='Transcribe audio with the Whisper adapter and recover language if available.',
    expected_output='Structured STT text and metadata.',
    agent=AUDIO_AGENT,
)

RAG_TASK = Task(
    name='retrieve_internal_evidence',
    description='Search internal medical knowledge using LangChain retrieval.',
    expected_output='Retrieved evidence chunks with metadata.',
    agent=RAG_AGENT,
)

WEB_TASK = Task(
    name='search_web',
    description='Use web search when external current guidance is needed.',
    expected_output='Web evidence with URL, title, domain, snippet, and ranking metadata.',
    agent=WEB_AGENT,
)

VALIDATION_TASK = Task(
    name='validate_evidence',
    description='Validate consistency, unsupported claims, and citation provenance.',
    expected_output='Evidence validation findings and uncertainty notes.',
    agent=EVIDENCE_AGENT,
)

REPORT_TASK = Task(
    name='final_report',
    description='Synthesize a final medical-friendly answer from model outputs and evidence.',
    expected_output='Structured final response with classifications, evidence, and limitations.',
    agent=REPORT_AGENT,
)

# Keep a compatibility alias for the repository’s older task naming style.
CrewTask = Task
