import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

from app.agents.agents import (
    VISION_AGENT,
    AUDIO_AGENT,
    RAG_AGENT,
    WEB_AGENT,
    EVIDENCE_AGENT,
    REPORT_AGENT,
)
from app.agents.crew import CrewAIOrchestrator
from app.agents.tools import (
    ClassificationTool,
    DetectionTool,
    SegmentationTool,
    WhisperTool,
    KnowledgeBaseSearchTool,
    WebSearchTool,
)
from app.rag.bm25 import BM25Retriever
from app.rag.embeddings import SentenceTransformerEmbeddingService
from app.rag.reranker import BGEReranker
from app.rag.hybrid_retriever import HybridRetriever


def test_real_crewai_public_classes_and_registry_objects_exist():
    assert Agent
    assert Task
    assert Crew
    assert Process
    assert BaseTool

    assert isinstance(VISION_AGENT, Agent)
    assert isinstance(AUDIO_AGENT, Agent)
    assert isinstance(RAG_AGENT, Agent)
    assert isinstance(WEB_AGENT, Agent)
    assert isinstance(EVIDENCE_AGENT, Agent)
    assert isinstance(REPORT_AGENT, Agent)

    # The installed CrewAI Agent model does not permit a free-form `name` field;
    # its canonical user-facing identity is carried by the `role` string field.
    assert VISION_AGENT.role == 'Vision Analysis Agent'
    assert AUDIO_AGENT.role == 'Audio/STT Agent'
    assert RAG_AGENT.role == 'Medical RAG Research Agent'
    assert WEB_AGENT.role == 'Web Research Agent'
    assert EVIDENCE_AGENT.role == 'Evidence Validation Agent'
    assert REPORT_AGENT.role == 'Medical Report Agent'

    orchestrator = CrewAIOrchestrator()
    assert hasattr(orchestrator, 'crew')
    assert isinstance(orchestrator.crew, Crew)


def test_hybrid_retrieval_components_expose_requested_stack_objects():
    assert BM25Retriever
    assert SentenceTransformerEmbeddingService
    assert BGEReranker
    assert HybridRetriever

    # Sanity constructor-level instantiation without trying the network.
    bm25 = BM25Retriever()
    emb = SentenceTransformerEmbeddingService(model_name='sentence-transformers/all-MiniLM-L6-v2')
    reranker = BGEReranker(model_name='BAAI/bge-reranker-large')
    hybrid = HybridRetriever(dense_service=emb, bm25_service=bm25, reranker=reranker)

    assert bm25 is not None
    assert emb is not None
    assert reranker is not None
    assert hybrid is not None


def test_tool_classes_are_real_crewai_base_tool_subclasses():
    assert issubclass(ClassificationTool, BaseTool)
    assert issubclass(DetectionTool, BaseTool)
    assert issubclass(SegmentationTool, BaseTool)
    assert issubclass(WhisperTool, BaseTool)
    assert issubclass(KnowledgeBaseSearchTool, BaseTool)
    assert issubclass(WebSearchTool, BaseTool)
