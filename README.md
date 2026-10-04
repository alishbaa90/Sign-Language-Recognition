# 🤟 SignSense AI — Real-Time Sign Language Recognition

A real-time system that converts ASL (American Sign Language) hand gestures into natural spoken sentences, with bilingual (English + German) audio output. Built end-to-end: computer vision, a benchmarked ML classifier, an LSTM sequence model for dynamic gestures, a multi-step LLM agent for sentence generation, a REST API, and a live dashboard.

---

## Overview

Most sign-language demos stop at "detect a letter." SignSense AI goes further: it recognizes ASL signs, strings them into a natural, grammatically correct sentence using an LLM agent, and speaks that sentence aloud in two languages.

**Pipeline:**

\`\`\`
Webcam Input
   → Hand Landmark Detection (MediaPipe)
   → Gesture Classification (RandomForest, 99.97% accuracy)
   → 2-Second Smart Sampling (noise reduction + confidence gating)
   → LangGraph Agent (cleanup → grammar → tone → translation)
   → Bilingual Audio Output (gTTS + Edge TTS)
\`\`\`

A second, independent pipeline recognizes **dynamic (movement-based) gestures** using an LSTM trained on 30-frame landmark sequences — demonstrating both static and temporal gesture recognition approaches.

---

## Features

- **Static ASL vocabulary** — single-hand gestures, benchmarked across RandomForest, SVM, and a Neural Network, with RandomForest selected for production (best accuracy without requiring feature scaling at inference time)
- **Dynamic gesture recognition** — a separate LSTM model recognizes movement-based gestures (PLEASE, SORRY, BYE, COME) from 30-frame hand-landmark sequences
- **Smart frame sampling** — predictions are gathered over 2-second windows and resolved by majority vote, instead of reacting to every noisy frame
- **Agentic sentence generation** — a 4-node LangGraph agent turns raw, possibly repeated sign predictions into a natural sentence: deduplicate → build grammatical sentence → make it sound natural → translate to German
- **Confidence-aware fallback** — if average recognition confidence is low, the agent hedges the generated sentence instead of asserting it outright
- **Bilingual audio output** — English (Google TTS) and German (Edge TTS)
- **REST API** — a FastAPI service accepts an uploaded video and returns the recognized signs, generated sentences, and audio
- **Live dashboard** — a Streamlit app with a live camera feed, real-time gesture feedback, and the generated sentence + audio
- **Session recording** — the full recognition session is saved as a video with live landmark/confidence overlays and the final sentence burned in as a subtitle
- **Structured logging & error handling** — every pipeline stage logs to file and console, with failures in one stage (e.g. audio generation) not crashing the whole run
- **Automated tests** — a pytest suite covers the landmark normalization, frame-sampling, and prediction logic

---

## Tech Stack

| Layer | Tools |
|---|---|
| Hand tracking | MediaPipe (Tasks API — HandLandmarker) |
| Static gesture classification | scikit-learn (RandomForest, SVM, MLP — benchmarked) |
| Dynamic gesture classification | TensorFlow / Keras (LSTM) |
| Sentence generation | LangGraph + Gemini API |
| Audio | gTTS (English), Edge TTS (German) |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Testing | pytest |
| Core | Python, OpenCV, NumPy, Pandas |

---

## Project Structure

\`\`\`
sign-language-recognition/
├── data/                       # Collected landmark data (CSV) per word
│   └── lstm/                   # Sequence data (.npy) for dynamic gestures
├── models/
│   ├── gesture_model.pkl       # Production static-gesture classifier
│   └── lstm/                   # Trained LSTM model + label map
├── src/
│   ├── hand_detector.py        # MediaPipe hand-landmark detection
│   ├── utils.py                # Landmark normalization (translation/scale invariant)
│   ├── data_collector.py       # Static-gesture data collection
│   ├── lstm_data_collector.py  # Dynamic-gesture sequence collection
│   ├── train_compare.py        # Benchmarks RandomForest / SVM / MLP
│   ├── train_lstm.py           # LSTM training
│   ├── gesture_predictor.py    # Static gesture inference
│   ├── frame_sampler.py        # 2-second confidence-gated sampling
│   ├── sentence_agent.py       # LangGraph multi-step sentence agent
│   ├── tts_engine.py           # Bilingual audio generation
│   ├── video_recorder.py       # Session recording + subtitle burn-in
│   ├── logger_config.py        # Structured logging setup
│   ├── main_pipeline.py        # Live webcam pipeline (end-to-end)
│   ├── pipeline_core.py        # Headless pipeline used by the API
│   ├── api.py                  # FastAPI app
│   └── streamlit_app.py        # Live dashboard
├── tests/                      # pytest suite
├── outputs/                    # Generated video, audio, sentences, logs
└── requirements.txt
\`\`\`

---

## Setup

\`\`\`bash
# Clone and enter the project
git clone <repo-url>
cd sign-language-recognition

# Create and activate a virtual environment
python -m venv venv
venv\\Scripts\\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
\`\`\`

Create a `.env` file in the project root:

\`\`\`
GEMINI_API_KEY=your_key_here
\`\`\`

> **Note on the LSTM module:** TensorFlow does not yet support every Python version. If your main environment can't install TensorFlow, train/run the LSTM pieces (`lstm_data_collector.py`, `train_lstm.py`, `lstm_live_test.py`) from a separate virtual environment on a supported Python version (3.11–3.12). The static-gesture pipeline does not require TensorFlow.

---

## Usage

**Live webcam demo (terminal):**
\`\`\`bash
cd src
python main_pipeline.py
\`\`\`

**Live dashboard:**
\`\`\`bash
cd src
streamlit run streamlit_app.py
\`\`\`

**REST API:**
\`\`\`bash
cd src
python api.py
# Interactive docs: http://localhost:8000/docs
\`\`\`

**Run tests:**
\`\`\`bash
pytest tests/ -v
\`\`\`
