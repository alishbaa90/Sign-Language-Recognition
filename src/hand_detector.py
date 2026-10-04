import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

class HandDetector:
    def __init__(self, max_hands=2, detection_conf=0.5):
        if not os.path.exists(MODEL_PATH):
            print("Downloading hand landmark model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        landmark_list = []
        if result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_lms in result.hand_landmarks:
                coords = []
                for lm in hand_lms:
                    coords.extend([lm.x, lm.y, lm.z])
                    if draw:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                landmark_list.append(coords)

        return frame, landmark_list


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, landmarks = detector.find_hands(frame)
        cv2.imshow("Test", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()