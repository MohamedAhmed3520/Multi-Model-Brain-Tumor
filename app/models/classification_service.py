import os
import time
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import load_model

from app.schemas.classification import ClassificationResult


class ClassificationService:
    """TensorFlow InceptionV3 wrapper preserving the artifact-informed workflow."""

    def __init__(self, model_path: str = './Tumor Models/best_model.h5', model_name: str = 'InceptionV3'):
        self.model_path = model_path
        self.model_name = model_name
        self.model_version = None
        self.model = load_model(self.model_path)

    def predict(self, image_path: str) -> ClassificationResult:
        start = time.time()
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        probs = self.model.predict(image_array, verbose=0)[0]

        classes = ['No Tumor', 'Brain Tumor']
        probability_map = {cls: float(probs[idx]) for idx, cls in enumerate(classes)}
        confidence = float(probs[1]) if len(probs) > 1 else 0.0
        inference_time_ms = (time.time() - start) * 1000
        return ClassificationResult(
            prediction='Brain Tumor' if confidence >= 0.5 else 'No Tumor',
            confidence=confidence,
            probabilities=probability_map,
            model_name=self.model_name,
            model_version=self.model_version,
            inference_time_ms=inference_time_ms,
        )
