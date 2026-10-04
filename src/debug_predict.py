import cv2
import numpy as np
from hand_detector import HandDetector
from gesture_predictor import GesturePredictor
from utils import normalize_landmarks

cap = cv2.VideoCapture(0)
detector = HandDetector()
predictor = GesturePredictor()

print("Press ESC to quit. Perform a sign and hold it.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame, landmarks = detector.find_hands(frame)

    if landmarks:
        normalized = normalize_landmarks(landmarks[0])
        probs = predictor.model.predict_proba([normalized])[0]

        # Sort and show only the top 5 predictions
        top5_idx = np.argsort(probs)[::-1][:5]

        y = 30
        for idx in top5_idx:
            cls = predictor.model.classes_[idx]
            p = probs[idx]
            color = (0, 255, 0) if idx == top5_idx[0] else (200, 200, 200)
            cv2.putText(frame, f"{cls}: {p*100:.1f}%", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 30

    cv2.imshow("Live Probabilities (Top 5)", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()