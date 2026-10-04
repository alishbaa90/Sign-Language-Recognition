import cv2
from hand_detector import HandDetector
from gesture_predictor import GesturePredictor
from frame_sampler import FrameSampler
from sentence_agent import run_sentence_agent
from tts_engine import generate_english_audio, generate_german_audio_sync
from logger_config import get_logger

logger = get_logger("pipeline_core")


def process_video(video_path, max_signs=None, output_dir="../outputs"):
    """
    Processes a video file (or webcam index) end-to-end:
    hand detection -> gesture prediction -> 2s sampling -> sentence agent -> audio.
    Returns a dict with all results. Does NOT display any window (headless).
    """
    cap = cv2.VideoCapture(video_path)
    detector = HandDetector()
    predictor = GesturePredictor()
    sampler = FrameSampler(window_seconds=2, confidence_threshold=0.40)

    collected_signs = []
    predictions_log = []

    logger.info(f"Processing video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, landmarks = detector.find_hands(frame, draw=False)

        if landmarks:
            label, confidence = predictor.predict(landmarks[0])
            sampler.add_prediction(label, confidence)

        if sampler.should_sample():
            result = sampler.get_best_prediction()
            if result:
                collected_signs.append(result["label"])
                predictions_log.append(result)
                logger.info(f"Sampled gesture: {result}")

        if max_signs and len(collected_signs) >= max_signs:
            break

    cap.release()
    logger.info(f"Video processing complete. Collected {len(collected_signs)} signs.")

    response = {
        "collected_signs": collected_signs,
        "predictions_log": predictions_log,
        "english_sentence": None,
        "german_sentence": None,
        "audio_en_path": None,
        "audio_de_path": None,
        "low_confidence": False,
        "error": None,
    }

    if not collected_signs:
        response["error"] = "No hand gestures were detected in the video."
        return response

    avg_conf = sum(p["confidence"] for p in predictions_log) / len(predictions_log)

    try:
        agent_result = run_sentence_agent(collected_signs, avg_confidence=avg_conf)
    except Exception as e:
        logger.error(f"Sentence agent failed: {e}")
        response["error"] = f"Sentence generation failed: {e}"
        return response

    response["english_sentence"] = agent_result["toned_sentence"]
    response["german_sentence"] = agent_result["german_sentence"]
    response["low_confidence"] = agent_result.get("low_confidence_flag", False)

    audio_en_path = f"{output_dir}/sentence_en.mp3"
    audio_de_path = f"{output_dir}/sentence_de.mp3"

    try:
        generate_english_audio(agent_result["toned_sentence"], audio_en_path)
        generate_german_audio_sync(agent_result["german_sentence"], audio_de_path)
        response["audio_en_path"] = audio_en_path
        response["audio_de_path"] = audio_de_path
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        response["error"] = f"Audio generation failed: {e}"

    return response