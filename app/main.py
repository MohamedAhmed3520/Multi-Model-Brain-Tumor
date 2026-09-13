from app.models.classification_service import ClassificationService
from app.models.detection_service import DetectionService
from app.models.segmentation_service import SegmentationService
from app.models.stt_service import STTService
from app.rag.ingestion import IngestionService
from app.rag.retriever import Retriever
from app.search.web_search import WebSearch
from app.agents.crew import CrewAIOrchestrator
from app.agent.graph import MedicalGraph


def build_default_services():
    classification = ClassificationService(model_path='./Tumor Models/best_model.h5')
    detection = DetectionService(model_path='./Tumor Models/best.pt')
    segmentation = SegmentationService(model_path='./Tumor Models/my_checkpoint.pth')
    stt = STTService(base_model='openai/whisper-large-v3', adapter_id='ma4389/Whisper-Fine')
    return {
        'classification': classification,
        'detection': detection,
        'segmentation': segmentation,
        'stt': stt,
        'retriever': Retriever(),
        'web': WebSearch(),
        'crew': CrewAIOrchestrator(),
        'graph': MedicalGraph(),
    }


if __name__ == '__main__':
    services = build_default_services()
    print('Loaded services:', ', '.join(services.keys()))
