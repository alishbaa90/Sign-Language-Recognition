import time
from collections import Counter

class FrameSampler:
    def __init__(self, window_seconds=2, confidence_threshold=0.40):
        self.window_seconds = window_seconds
        self.confidence_threshold = confidence_threshold
        self.buffer = []
        self.last_sample_time = time.time()

    def add_prediction(self, label, confidence):
        if confidence >= self.confidence_threshold:
            self.buffer.append((label, confidence))

    def should_sample(self):
        return (time.time() - self.last_sample_time) >= self.window_seconds

    def get_best_prediction(self):
        if not self.buffer:
            self.last_sample_time = time.time()
            return None

        # sabse zyada frequent + highest avg confidence wala label
        labels = [b[0] for b in self.buffer]
        most_common = Counter(labels).most_common(1)[0][0]
        confidences = [b[1] for b in self.buffer if b[0] == most_common]
        avg_conf = sum(confidences) / len(confidences)

        self.buffer = []
        self.last_sample_time = time.time()
        return {"label": most_common, "confidence": avg_conf}