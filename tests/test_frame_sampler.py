import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import time
from frame_sampler import FrameSampler


def test_low_confidence_predictions_are_ignored():
    sampler = FrameSampler(window_seconds=2, confidence_threshold=0.5)
    sampler.add_prediction("HELLO", 0.3)  # below threshold
    assert len(sampler.buffer) == 0


def test_high_confidence_predictions_are_buffered():
    sampler = FrameSampler(window_seconds=2, confidence_threshold=0.5)
    sampler.add_prediction("HELLO", 0.9)
    assert len(sampler.buffer) == 1


def test_majority_vote_picks_most_common_label():
    sampler = FrameSampler(window_seconds=2, confidence_threshold=0.5)
    sampler.add_prediction("HELLO", 0.9)
    sampler.add_prediction("HELLO", 0.8)
    sampler.add_prediction("YES", 0.6)

    result = sampler.get_best_prediction()
    assert result["label"] == "HELLO"


def test_buffer_clears_after_sampling():
    sampler = FrameSampler(window_seconds=2, confidence_threshold=0.5)
    sampler.add_prediction("HELLO", 0.9)
    sampler.get_best_prediction()
    assert len(sampler.buffer) == 0


def test_should_sample_respects_time_window():
    sampler = FrameSampler(window_seconds=1, confidence_threshold=0.5)
    assert sampler.should_sample() is False  # too soon
    time.sleep(1.1)
    assert sampler.should_sample() is True