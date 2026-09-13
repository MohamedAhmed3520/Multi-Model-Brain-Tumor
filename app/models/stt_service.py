import os
import time
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from app.schemas.stt import STTResult


class STTService:
    """Whisper STT service using fine-tuned LoRA adapter ma4389/Whisper-Fine over whisper-large-v3."""

    def __init__(self, base_model: str = 'openai/whisper-large-v3', adapter_id: str = 'ma4389/Whisper-Fine'):
        self.base_model = base_model
        self.adapter_id = adapter_id
        self.model_name = 'Whisper-Large-v3 + ma4389/Whisper-Fine'
        self.model_version = adapter_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AutoProcessor.from_pretrained(base_model)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(base_model)
        self.model = PeftModel.from_pretrained(self.model, adapter_id)
        self.model.eval()
        self.pipe = pipeline('automatic-speech-recognition', model=self.model, tokenizer=self.processor.tokenizer, feature_extractor=self.processor.feature_extractor, device=0 if self.device == 'cuda' else -1)

    def transcribe(self, audio_path: str) -> STTResult:
        start = time.time()
        text = self.pipe(audio_path)['text']
        language = None
        confidence = None
        return STTResult(
            text=text,
            language=language,
            model_name=self.model_name,
            model_version=self.model_version,
            confidence=confidence,
            duration_seconds=None,
            inference_time_ms=(time.time() - start) * 1000,
        )
