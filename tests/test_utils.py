import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from utils import normalize_landmarks


def make_fake_landmarks():
    # 21 fake landmarks (x, y, z), wrist at index 0
    landmarks = []
    for i in range(21):
        landmarks.extend([0.5 + i * 0.01, 0.5 + i * 0.01, 0.0])
    return landmarks


def test_normalize_output_length():
    landmarks = make_fake_landmarks()
    result = normalize_landmarks(landmarks)
    assert len(result) == 63  # 21 landmarks * 3 coords


def test_normalize_is_translation_invariant():
    landmarks_a = make_fake_landmarks()
    landmarks_b = [v + 0.2 for v in landmarks_a]  # shift everything by 0.2

    norm_a = normalize_landmarks(landmarks_a)
    norm_b = normalize_landmarks(landmarks_b)

    assert np.allclose(norm_a, norm_b, atol=1e-4)


def test_normalize_wrist_is_origin():
    landmarks = make_fake_landmarks()
    result = np.array(normalize_landmarks(landmarks)).reshape(21, 3)
    assert np.allclose(result[0], [0, 0, 0], atol=1e-6)