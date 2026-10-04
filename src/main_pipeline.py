import cv2
import csv
import os
from dotenv import load_dotenv

from hand_detector import HandDetector
from gesture_predictor import GesturePredictor
from frame_sampler import FrameSampler
from sentence_agent import run_sentence_agent
from tts_engine import generate_english_audio, generate_german_audio_sync
from video_recorder import VideoRecorder
from logger_config import get_logger

load_dotenv()
logger = get_logger("main_pipeline")


def run_pipeline(video_source=0):
    cap = cv2.VideoCapture(video_source)
    detector = HandDetector()
    predictor = GesturePredictor()
    sampler = FrameSampler(window_seconds=2, confidence_threshold=0.40)
    recorder = VideoRecorder()

    collected_signs = []
    predictions_log = []

    logger.info("Pipeline started. Waiting for gestures...")
    print("Perform your signs. Press ESC or wait until 4 signs are collected.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera.")
            break

        frame, landmarks = detector.find_hands(frame)

        if landmarks:
            label, confidence = predictor.predict(landmarks[0])
            sampler.add_prediction(label, confidence)

            color = (0, 255, 0) if confidence >= 0.40 else (0, 0, 255)
            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if sampler.should_sample():
            result = sampler.get_best_prediction()
            if result:
                collected_signs.append(result["label"])
                predictions_log.append(result)
                logger.info(f"Sampled gesture: {result}")

        recorder.write(frame)  # save every displayed frame to the output video

        cv2.imshow("Sign Language Recognition", frame)
        if cv2.waitKey(1) == 27 or len(collected_signs) >= 4:
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Session ended. Collected {len(collected_signs)} signs.")

    try:
        with open("../outputs/predictions_from_video.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "confidence"])
            writer.writeheader()
            writer.writerows(predictions_log)
        logger.info("Predictions CSV saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save predictions CSV: {e}")

    if not collected_signs:
        logger.warning("No signs were collected — skipping sentence generation.")
        print("No signs were collected — nothing to save.")
        recorder.release()
        return

    avg_conf = sum(p["confidence"] for p in predictions_log) / len(predictions_log)
    logger.info(f"Average confidence for this session: {avg_conf:.2f}")

    try:
        agent_result = run_sentence_agent(collected_signs, avg_confidence=avg_conf)
    except Exception as e:
        logger.error(f"Sentence agent failed: {e}")
        print("Something went wrong generating the sentence. Check logs/pipeline.log for details.")
        recorder.release()
        return

    print("Cleaned signs:", agent_result["cleaned_signs"])
    print("English (toned):", agent_result["toned_sentence"])
    print("German:", agent_result["german_sentence"])

    # Burn the final sentence in as a subtitle on a few extra closing frames
    ret, last_frame = cap.read() if cap.isOpened() else (False, None)
    subtitle_frame = frame.copy()
    subtitle_frame = recorder.add_subtitle(subtitle_frame, agent_result["toned_sentence"])
    for _ in range(40):  # hold the subtitle for ~2 seconds at 20fps
        recorder.write(subtitle_frame)

    recorder.release()
    logger.info("Output video with subtitles saved.")

    try:
        with open("../outputs/generated_sentences.txt", "w", encoding="utf-8") as f:
            f.write(f"English: {agent_result['toned_sentence']}\n")
            f.write(f"German: {agent_result['german_sentence']}\n")
        logger.info("Generated sentences saved to text file.")
    except Exception as e:
        logger.error(f"Failed to save generated sentences: {e}")

    try:
        generate_english_audio(agent_result["toned_sentence"])
        generate_german_audio_sync(agent_result["german_sentence"])
        logger.info("Audio files generated successfully.")
        print("Audio saved: sentence_en.mp3, sentence_de.mp3")
    except Exception as e:
        logger.error(f"Failed to generate audio: {e}")
        print("Audio generation failed. Check logs/pipeline.log for details.")


if __name__ == "__main__":
    run_pipeline()