from crewai import Agent


VISION_AGENT = Agent(
    role='Vision Analysis Agent',
    goal='Inspect structured model evidence without inventing predictions',
    backstory='Reviews MRI-derived model outputs.',
    verbose=False,
)

AUDIO_AGENT = Agent(
    role='Audio/STT Agent',
    goal='Transcribe audio and surface structured language evidence',
    backstory='Uses the speech transcription model only.',
    verbose=False,
)

RAG_AGENT = Agent(
    role='Medical RAG Research Agent',
    goal='Retrieve and evaluate internal evidence with provenance',
    backstory='Searches the internal knowledge base using LangChain retrieval.',
    verbose=False,
)

WEB_AGENT = Agent(
    role='Web Research Agent',
    goal='Search web sources when information is current or external',
    backstory='Uses web search and preserves source metadata.',
    verbose=False,
)

EVIDENCE_AGENT = Agent(
    role='Evidence Validation Agent',
    goal='Validate consistency and identify unsupported claims',
    backstory='Checks project evidence without changing model outputs.',
    verbose=False,
)

REPORT_AGENT = Agent(
    role='Medical Report Agent',
    goal='Synthesize model outputs, evidence, limitations, and citations',
    backstory='Creates the final structured report.',
    verbose=False,
)
