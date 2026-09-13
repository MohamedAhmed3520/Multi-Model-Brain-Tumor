import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.classification import ClassificationResult
from app.schemas.detection import Detection, DetectionResult
from app.schemas.segmentation import SegmentationResult
from app.schemas.stt import STTResult
from app.schemas.rag import RetrievedEvidence, EvidenceBundle
from app.schemas.web import WebEvidence
from app.schemas.response import MedicalAIResponse


def test_pydantic_models_round_trip():
    classification = ClassificationResult(
        prediction='Brain Tumor',
        confidence=0.92,
        probabilities={'No Tumor': 0.08, 'Brain Tumor': 0.92},
        model_name='InceptionV3',
        model_version=None,
        inference_time_ms=12.0,
    )
    assert classification.prediction == 'Brain Tumor'

    detection = DetectionResult(
        detections=[Detection(label='tumor', confidence=0.7, bbox=[0.0, 1.0, 2.0, 3.0])],
        model_name='existing-detection',
        model_version=None,
        inference_time_ms=10.0,
    )
    assert detection.detections[0].label == 'tumor'

    segmentation = SegmentationResult(
        tumor_detected=True,
        threshold=0.5,
        coverage_ratio=0.2,
        dice_score=0.3,
        mask_path='mask.png',
        overlay_path='overlay.png',
        model_name='UNet',
        model_version=None,
        inference_time_ms=5.0,
    )
    assert segmentation.mask_path == 'mask.png'

    stt = STTResult(
        text='sample transcript',
        language='en',
        model_name='Whisper-Large-v3 + ma4389/Whisper-Fine',
        model_version='ma4389/Whisper-Fine',
        confidence=0.99,
        duration_seconds=1.0,
        inference_time_ms=10.0,
    )
    assert stt.text == 'sample transcript'

    evidence = RetrievedEvidence(
        content='grounded evidence',
        source='internal-report.pdf',
        page=1,
        section='Findings',
        score=0.81,
        modality='text',
        document_id='doc-1',
        chunk_id='chunk-1',
    )
    assert evidence.content == 'grounded evidence'

    web = WebEvidence(
        title='Current guideline',
        url='https://example.com/guideline',
        domain='example.com',
        snippet='guidance',
        rank=1,
        relevance_score=0.7,
    )
    assert web.domain == 'example.com'

    bundle = EvidenceBundle(user_query='When should I check?', retrieved_documents=[evidence], web_sources=[web])
    assert len(bundle.retrieved_documents) == 1

    response = MedicalAIResponse(
        summary='Summary',
        retrieved_evidence=[evidence],
        web_evidence=[web],
        limitations=['Model predictions are not a diagnosis.'],
        citations=['https://example.com/guideline'],
    )
    assert response.summary == 'Summary'
