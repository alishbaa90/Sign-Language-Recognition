import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import pickle

from hand_detector import HandDetector
from utils import normalize_landmarks

SEQUENCE_LENGTH = 30

model = load_model("../models/lstm/dynamic_gesture_model.keras")
with open("../models/lstm/label_map.pkl", "rb") as f:
    labels = pickle.load(f)

cap = cv2.VideoCapture(0)
detector = HandDetector()
buffer = deque(maxlen=SEQUENCE_LENGTH)

print("Perform PLEASE, SORRY, BYE, or COME. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, landmarks = detector.find_hands(frame)

    if landmarks:
        normalized = normalize_landmarks(landmarks[0])
        buffer.append(normalized)

    if len(buffer) == SEQUENCE_LENGTH:
        sequence = np.expand_dims(np.array(buffer), axis=0)  # shape: (1, 30, 63)
        probs = model.predict(sequence, verbose=0)[0]
        best_idx = np.argmax(probs)
        label = labels[best_idx]
        confidence = probs[best_idx]

        color = (0, 255, 0) if confidence >= 0.6 else (0, 0, 255)
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("LSTM Dynamic Gesture Test", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()