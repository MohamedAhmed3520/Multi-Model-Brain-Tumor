from crewai.tools import BaseTool

from app.models.classification_service import ClassificationService
from app.models.detection_service import DetectionService
from app.models.segmentation_service import SegmentationService
from app.models.stt_service import STTService
from app.rag.retriever import Retriever
from app.search.web_search import WebSearch


class ClassificationTool(BaseTool):
    def __init__(self, service: ClassificationService):
        super().__init__(
            name='classification_tool',
            description='Run image classification over the provided MRI or medical imaging file.',
        )
        self.service = service

    def _run(self, image_path: str):
        return self.service.predict(image_path)


class DetectionTool(BaseTool):
    def __init__(self, service: DetectionService):
        super().__init__(
            name='detection_tool',
            description='Run object or lesion detection over the provided medical image.',
        )
        self.service = service

    def _run(self, image_path: str):
        return self.service.predict(image_path)


class SegmentationTool(BaseTool):
    def __init__(self, service: SegmentationService):
        super().__init__(
            name='segmentation_tool',
            description='Generate a segmentation mask and visual overlay from the provided medical image.',
        )
        self.service = service

    def _run(self, image_path: str):
        return self.service.predict(image_path)


class WhisperTool(BaseTool):
    def __init__(self, service: STTService):
        super().__init__(
            name='whisper_tool',
            description='Transcribe a clinical audio file into text evidence.',
        )
        self.service = service

    def _run(self, audio_path: str):
        return self.service.transcribe(audio_path)


class KnowledgeBaseSearchTool(BaseTool):
    def __init__(self, retriever: Retriever):
        super().__init__(
            name='knowledge_base_search_tool',
            description='Query the internal medical knowledge base via the structured retriever.',
        )
        self.retriever = retriever

    def _run(self, query: str):
        return self.retriever.search(query)


class WebSearchTool(BaseTool):
    def __init__(self, search: WebSearch):
        super().__init__(
            name='web_search_tool',
            description='Search the public web for current medical evidence.',
        )
        self.search = search

    def _run(self, query: str):
        return self.search.search(query)
