import streamlit as st
import cv2
from hand_detector import HandDetector
from gesture_predictor import GesturePredictor
from frame_sampler import FrameSampler
from sentence_agent import run_sentence_agent
from tts_engine import generate_english_audio, generate_german_audio_sync

NUM_SIGNS = 4
CONFIDENCE_THRESHOLD = 0.40

st.set_page_config(
    page_title="SignSense AI — Real-Time Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Cache heavy resources so they load only once ---
@st.cache_resource
def load_detector():
    return HandDetector()

@st.cache_resource
def load_predictor():
    return GesturePredictor()

# --- Professional styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .main { background-color: #0a0e14; }

        .hero {
            padding: 32px 0 8px 0;
            border-bottom: 1px solid #1f2733;
            margin-bottom: 28px;
        }
        .hero-badge {
            display: inline-block;
            background: linear-gradient(135deg, #4F46E5, #7C3AED);
            color: white;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 14px;
        }
        .hero h1 {
            font-size: 38px;
            font-weight: 800;
            margin: 0 0 8px 0;
            background: linear-gradient(135deg, #ffffff, #a5b4fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero p {
            color: #8b94a7;
            font-size: 15px;
            max-width: 680px;
            line-height: 1.6;
            margin-bottom: 20px;
        }

        .panel-title {
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #6b7280;
            margin-bottom: 10px;
        }

        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3em;
            font-weight: 600;
            font-size: 14px;
            border: none;
            transition: all 0.15s ease;
        }
        .primary-btn button {
            background: linear-gradient(135deg, #4F46E5, #6366F1);
            color: white;
        }
        .primary-btn button:hover { opacity: 0.9; }
        .stop-btn button {
            background-color: #161b26;
            color: #9ca3af;
            border: 1px solid #262e3d;
        }

        .result-card {
            background: #10151f;
            border: 1px solid #1f2733;
            border-radius: 12px;
            padding: 22px 24px;
            margin-bottom: 14px;
        }
        .lang-label {
            color: #6366F1;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
        }
        .sentence-text { font-size: 18px; font-weight: 600; color: #e5e7eb; margin: 6px 0 16px 0; line-height: 1.5; }

        .sign-chip {
            display: inline-block;
            background: #161b26;
            border: 1px solid #262e3d;
            color: #a5b4fc;
            padding: 6px 14px;
            border-radius: 20px;
            margin: 3px;
            font-weight: 600;
            font-size: 13px;
        }

        .empty-state {
            background: #10151f;
            border: 1px dashed #262e3d;
            border-radius: 12px;
            padding: 40px 20px;
            text-align: center;
            color: #6b7280;
            font-size: 14px;
        }

        .footer-note { color: #4b5563; font-size: 12px; margin-top: 30px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- Hero section ---
st.markdown("""
    <div class="hero">
        <span class="hero-badge">AI Engineering Project</span>
        <h1>🤟 SignSense AI</h1>
        <p>Turns real-time ASL hand gestures into natural spoken sentences,
        with bilingual English and German audio output.</p>
    </div>
""", unsafe_allow_html=True)

# --- Session state ---
if "collected_signs" not in st.session_state:
    st.session_state.collected_signs = []
if "predictions_log" not in st.session_state:
    st.session_state.predictions_log = []
if "running" not in st.session_state:
    st.session_state.running = False
if "result" not in st.session_state:
    st.session_state.result = None

col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown('<div class="panel-title">Live Camera Feed</div>', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    progress_placeholder = st.empty()

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        start_button = st.button("▶  Start Recognition")
        st.markdown('</div>', unsafe_allow_html=True)
    with btn_col2:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        stop_button = st.button("■  Stop")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel-title">Recognized Signs</div>', unsafe_allow_html=True)
    signs_placeholder = st.empty()
    st.markdown('<div class="panel-title" style="margin-top:20px;">Generated Sentence</div>', unsafe_allow_html=True)
    sentence_placeholder = st.empty()
    audio_en_placeholder = st.empty()
    audio_de_placeholder = st.empty()

    if not st.session_state.collected_signs and not st.session_state.result:
        signs_placeholder.markdown(
            "<div class='empty-state'>No active session.<br>Click <b>Start Recognition</b> and perform ASL signs in front of your camera.</div>",
            unsafe_allow_html=True
        )

if stop_button:
    st.session_state.running = False

if start_button:
    st.session_state.running = True
    st.session_state.collected_signs = []
    st.session_state.predictions_log = []
    st.session_state.result = None

    detector = load_detector()
    predictor = load_predictor()
    sampler = FrameSampler(window_seconds=2, confidence_threshold=CONFIDENCE_THRESHOLD)

    cap = cv2.VideoCapture(0)

    while st.session_state.running and len(st.session_state.collected_signs) < NUM_SIGNS:
        ret, frame = cap.read()
        if not ret:
            break

        frame, landmarks = detector.find_hands(frame)

        if landmarks:
            label, confidence = predictor.predict(landmarks[0])
            sampler.add_prediction(label, confidence)
            color = (99, 102, 241) if confidence >= CONFIDENCE_THRESHOLD else (107, 114, 128)
            cv2.putText(frame, f"{label} ({confidence*100:.0f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if sampler.should_sample():
            result = sampler.get_best_prediction()
            if result:
                st.session_state.collected_signs.append(result["label"])
                st.session_state.predictions_log.append(result)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        progress = len(st.session_state.collected_signs) / NUM_SIGNS
        progress_placeholder.progress(progress, text=f"Collecting signs: {len(st.session_state.collected_signs)}/{NUM_SIGNS}")

        chips = "".join(f"<span class='sign-chip'>{s}</span>" for s in st.session_state.collected_signs)
        signs_placeholder.markdown(
            f"<div class='result-card'>{chips or '<span style=color:#6b7280;>Listening...</span>'}</div>",
            unsafe_allow_html=True
        )

    cap.release()
    progress_placeholder.empty()

    if st.session_state.collected_signs:
        with st.spinner("Running LangGraph sentence agent..."):
            avg_conf = sum(p["confidence"] for p in st.session_state.predictions_log) / len(st.session_state.predictions_log)
            result = run_sentence_agent(st.session_state.collected_signs, avg_confidence=avg_conf)
            st.session_state.result = result

        with st.spinner("Generating bilingual audio..."):
            generate_english_audio(result["toned_sentence"])
            generate_german_audio_sync(result["german_sentence"])

    st.session_state.running = False

if st.session_state.result:
    result = st.session_state.result
    sentence_placeholder.markdown(f"""
        <div class="result-card">
            <div class="lang-label">English</div>
            <div class="sentence-text">{result['toned_sentence']}</div>
            <div class="lang-label">German</div>
            <div class="sentence-text">{result['german_sentence']}</div>
        </div>
    """, unsafe_allow_html=True)

    if result.get("low_confidence_flag"):
        st.warning("⚠️ Confidence was low this round — the sentence was hedged accordingly.")

    audio_en_placeholder.audio("../outputs/sentence_en.mp3")
    audio_de_placeholder.audio("../outputs/sentence_de.mp3")

st.markdown(
    '<div class="footer-note">MediaPipe · Scikit-learn · LangGraph · Gemini API · FastAPI · Streamlit</div>',
    unsafe_allow_html=True
)