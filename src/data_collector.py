import cv2
import csv
import time
from hand_detector import HandDetector
from utils import normalize_landmarks

WORDS = [
    "HELLO", "YES", "NO", "THANKYOU", "PLEASE", "SORRY", "GOOD", "BAD",
    "HELP", "STOP", "NAME", "WHAT", "WHERE", "HOW", "I", "YOU",
    "LOVE", "FRIEND", "EAT", "WATER"
]

def collect_data(label, num_samples=200):
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    count = 0

    print(f"\n>>> Get ready for: {label} <<<")
    print("Starting in 3 seconds...")
    time.sleep(3)

    with open(f'../data/{label}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            frame, landmarks = detector.find_hands(frame)
            if landmarks:
                normalized = normalize_landmarks(landmarks[0])
                writer.writerow(normalized)
                count += 1
                cv2.putText(frame, f"{label}: {count}/{num_samples}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Collecting Data", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done! {count} samples saved for '{label}'.")

def collect_all():
    for word in WORDS:
        input(f"\nPress ENTER when ready to record '{word}'...")
        collect_data(word)
    print("\nAll words collected!")

if __name__ == "__main__":
    mode = input("Collect (a)ll words or (s)ingle word? ").strip().lower()
    if mode == "a":
        collect_all()
    else:
        label = input("Enter sign label: ").strip().upper()
        collect_data(label)