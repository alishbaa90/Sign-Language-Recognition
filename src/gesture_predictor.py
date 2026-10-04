import pickle
import numpy as np
from utils import normalize_landmarks

class GesturePredictor:
    def __init__(self, model_path="../models/gesture_model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, landmarks):
        normalized = normalize_landmarks(landmarks)
        X = np.array(normalized).reshape(1, -1)
        probs = self.model.predict_proba(X)[0]
        best_idx = np.argmax(probs)
        label = self.model.classes_[best_idx]
        confidence = probs[best_idx]
        return label, confidence


if __name__ == "__main__":
    import cv2
    from hand_detector import HandDetector

    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    predictor = GesturePredictor()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, landmarks = detector.find_hands(frame)

        if landmarks:
            label, confidence = predictor.predict(landmarks[0])
            color = (0, 255, 0) if confidence >= 0.40 else (0, 0, 255)
            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Gesture Prediction Test", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()