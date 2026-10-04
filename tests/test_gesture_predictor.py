import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from gesture_predictor import GesturePredictor

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gesture_model.pkl")


@pytest.fixture
def predictor():
    return GesturePredictor(model_path=MODEL_PATH)


def test_predict_returns_valid_label_and_confidence(predictor):
    fake_landmarks = [0.0] * 63  # normalized-shape dummy input
    label, confidence = predictor.predict(fake_landmarks)

    assert label in predictor.model.classes_
    assert 0.0 <= confidence <= 1.0