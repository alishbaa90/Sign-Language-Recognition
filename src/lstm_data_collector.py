import cv2
import numpy as np
import os
import time
from hand_detector import HandDetector
from utils import normalize_landmarks

SEQUENCE_LENGTH = 30
DYNAMIC_WORDS = ["PLEASE", "SORRY", "BYE", "COME"]

def collect_sequences(label, num_sequences=60):
    os.makedirs("../data/lstm", exist_ok=True)
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    all_sequences = []

    for seq_num in range(num_sequences):
        print(f"\n[{label}] Sequence {seq_num + 1}/{num_sequences} - Get ready...")
        time.sleep(1)

        sequence = []
        frames_collected = 0

        while frames_collected < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                continue
            frame, landmarks = detector.find_hands(frame)

            if landmarks:
                normalized = normalize_landmarks(landmarks[0])
                sequence.append(normalized)
                frames_collected += 1
                cv2.putText(frame, f"{label} seq {seq_num+1}/{num_sequences} frame {frames_collected}/{SEQUENCE_LENGTH}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Collecting Sequence", frame)
            if cv2.waitKey(1) == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

        all_sequences.append(sequence)

    cap.release()
    cv2.destroyAllWindows()

    all_sequences = np.array(all_sequences)
    np.save(f"../data/lstm/{label}.npy", all_sequences)
    print(f"Saved: {label}.npy with shape {all_sequences.shape}")

def collect_all():
    for word in DYNAMIC_WORDS:
        input(f"\nPress ENTER when ready to record '{word}' (perform the FULL movement each time)...")
        collect_sequences(word)
    print("\nAll dynamic gestures collected!")

if __name__ == "__main__":
    mode = input("Collect (a)ll dynamic words or (s)ingle word? ").strip().lower()
    if mode == "a":
        collect_all()
    else:
        label = input("Enter dynamic word (PLEASE/SORRY/BYE/COME): ").strip().upper()
        collect_sequences(label)