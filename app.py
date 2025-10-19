"""
Emotion & Gesture-Aware Multilingual Transliteration/Translation (MVP)

Setup (CPU-only, typical laptop):
1) Create and activate a virtual environment (recommended)
   python -m venv .venv && source .venv/bin/activate   # Linux/macOS
   # On Windows (PowerShell):  py -m venv .venv;  .venv\Scripts\Activate.ps1

2) Install dependencies
   pip install -r requirements.txt

3) Run the app
   streamlit run app.py

Notes:
- First run may download a small sentiment model from Hugging Face.
- If translation API/model/camera are unavailable, the app gracefully falls back with helpful messages.
- No GPU required. Everything runs on CPU.
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import streamlit as st

# Optional heavy imports are wrapped in try/except so the app can still run
try:
    import cv2  # OpenCV for basic smile detection via Haar cascades
except Exception:  # pragma: no cover - handled at runtime in UI
    cv2 = None  # type: ignore

try:
    from transformers import pipeline  # Hugging Face sentiment pipeline
except Exception:  # pragma: no cover
    pipeline = None  # type: ignore

try:
    from deep_translator import GoogleTranslator  # Free translator (uses Google web)
except Exception:  # pragma: no cover
    GoogleTranslator = None  # type: ignore

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as indic_transliterate
except Exception:  # pragma: no cover
    sanscript = None  # type: ignore
    indic_transliterate = None  # type: ignore

from PIL import Image
import numpy as np
import requests


# ---------------------------
# Constants and configuration
# ---------------------------
LANG_OPTIONS = {"English": "en", "Hindi": "hi"}
SCRIPT_OPTIONS = {"Devanagari": "DEVANAGARI", "Latin (IAST)": "IAST"}

EMOJI_BY_EMOTION = {"happy": "😄", "sad": "😢", "neutral": "😐"}
EMOJI_BY_GESTURE = {"smile": "😊", "namaste": "🙏"}

# Public LibreTranslate endpoints to try (best-effort, may change/rate-limit)
LIBRETRANSLATE_ENDPOINTS = (
    "https://libretranslate.com/translate",
    "https://libretranslate.de/translate",
    "https://translate.argosopentech.com/translate",
)


# ---------------------------
# Helper data structures
# ---------------------------
@dataclass
class ProcessResult:
    text: str
    emotion: str
    emotion_score: Optional[float]
    gesture: Optional[str]
    translation_info: Optional[str] = None
    transliteration_info: Optional[str] = None
    warnings: Tuple[str, ...] = tuple()


# ---------------------------
# Caching of heavier resources
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline():
    """Load and cache a lightweight English sentiment model.

    Uses a 2-class model (POSITIVE/NEGATIVE). We derive NEUTRAL when the score is near 0.5.
    Returns None if transformers/torch are missing or model fails to load.
    """
    if pipeline is None:
        return None
    try:
        # Small, CPU-friendly model
        return pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_haar_cascades():
    """Load OpenCV Haar cascades for faces and smiles, if OpenCV is available."""
    if cv2 is None:
        return None, None
    try:
        face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        smile_path = cv2.data.haarcascades + "haarcascade_smile.xml"
        face_cascade = cv2.CascadeClassifier(face_path)
        smile_cascade = cv2.CascadeClassifier(smile_path)
        # Basic sanity check
        if face_cascade.empty() or smile_cascade.empty():
            return None, None
        return face_cascade, smile_cascade
    except Exception:
        return None, None


# ---------------------------
# Core features: translation
# ---------------------------
def translate_text(text: str, source_lang_code: str, target_lang_code: str) -> Tuple[str, Optional[str]]:
    """Translate text using deep_translator (GoogleTranslator). Fallback to LibreTranslate.

    Returns (translated_text, warning_message_if_any).
    """
    if not text or source_lang_code == target_lang_code:
        return text, None

    # Attempt 1: GoogleTranslator via deep_translator
    if GoogleTranslator is not None:
        try:
            translated = GoogleTranslator(source=source_lang_code, target=target_lang_code).translate(text)
            if isinstance(translated, str) and translated.strip():
                return translated, None
        except Exception as e:
            # Fall through to LibreTranslate
            warning = f"Primary translation unavailable ({type(e).__name__}). Trying fallback..."
        else:
            warning = None
    else:
        warning = "Primary translator not installed. Trying fallback..."

    # Attempt 2: LibreTranslate public endpoints (best-effort, may be rate-limited or HTML)
    for endpoint in LIBRETRANSLATE_ENDPOINTS:
        try:
            resp = requests.post(
                endpoint,
                data={
                    "q": text,
                    "source": source_lang_code,
                    "target": target_lang_code,
                    "format": "text",
                },
                timeout=10,
            )
            if not resp.ok:
                continue
            # Guard against non-JSON responses (e.g., HTML error pages)
            content_type = resp.headers.get("Content-Type", "")
            if "json" not in content_type.lower():
                continue
            data = resp.json()
            translated = data.get("translatedText")
            if isinstance(translated, str) and translated.strip():
                return translated, warning
        except Exception:
            # Try next endpoint
            continue

    # If we reach here, all fallbacks failed
    warning = warning or "Translation service unreachable right now; showing original text."

    # Give up gracefully
    return text, warning or "Translation unavailable; showing original text."


# -----------------------------------
# Core features: transliteration (HI)
# -----------------------------------
def transliterate_text(text: str, source_script_label: str, target_script_label: str) -> Tuple[str, Optional[str]]:
    """Transliterate text between Devanagari and Latin (IAST) using indic-transliteration.

    Returns (converted_text, warning_message_if_any).
    """
    if not text or source_script_label == target_script_label:
        return text, None

    if sanscript is None or indic_transliterate is None:
        return text, "Transliteration library not installed; skipping."

    try:
        src_scheme_name = SCRIPT_OPTIONS.get(source_script_label)
        tgt_scheme_name = SCRIPT_OPTIONS.get(target_script_label)
        if not src_scheme_name or not tgt_scheme_name:
            return text, "Unsupported script selection; skipping."

        src_scheme = getattr(sanscript, src_scheme_name)
        tgt_scheme = getattr(sanscript, tgt_scheme_name)
        converted = indic_transliterate(text, src_scheme, tgt_scheme)
        return converted, None
    except Exception as e:
        return text, f"Transliteration failed ({type(e).__name__})."


# -------------------------------------
# Core features: emotion classification
# -------------------------------------
_POSITIVE_HINTS = {
    # English
    "happy", "glad", "joy", "great", "good", "love", "awesome", "nice", "wonderful",
    # Hindi (simple forms)
    "khushi", "khush", "accha", "achha", "pyar", "pyaar", "badhiya",
}
_NEGATIVE_HINTS = {
    # English
    "sad", "bad", "angry", "terrible", "awful", "hate", "upset", "depressed",
    # Hindi (simple forms)
    "dukhi", "dukh", "bura", "buri", "udaas", "gussa",
}


def detect_emotion(text: str) -> Tuple[str, Optional[float]]:
    """Return (label, score) where label in {happy, sad, neutral}.

    Priority: use transformers pipeline if available; otherwise fall back to a keyword heuristic.
    """
    if not text or not text.strip():
        return "neutral", None

    pipe = get_sentiment_pipeline()
    if pipe is not None:
        try:
            # Run the model (cap very long inputs for latency)
            sample = text if len(text) < 5120 else text[:5120]
            result = pipe(sample)[0]
            label = str(result.get("label", "")).upper()
            score = float(result.get("score", 0.5))

            if label.startswith("POS"):
                # Make mid-confidence scores neutral for a calmer UX
                if 0.45 <= score <= 0.55:
                    return "neutral", score
                return "happy", score
            if label.startswith("NEG"):
                if 0.45 <= score <= 0.55:
                    return "neutral", score
                return "sad", score
            return "neutral", score
        except Exception:
            # Fall through to heuristic
            pass

    # Heuristic fallback (language-agnostic, very simple)
    lowered = text.lower()
    pos_hits = sum(1 for w in _POSITIVE_HINTS if w in lowered)
    neg_hits = sum(1 for w in _NEGATIVE_HINTS if w in lowered)
    if pos_hits > neg_hits:
        return "happy", None
    if neg_hits > pos_hits:
        return "sad", None
    return "neutral", None


# -----------------------------------
# Core features: gesture recognition
# -----------------------------------

def _image_file_to_cv2_bgr(uploaded_file) -> Optional[np.ndarray]:
    """Convert a Streamlit UploadedFile (image) to an OpenCV BGR numpy array."""
    if uploaded_file is None:
        return None
    try:
        bytes_data = uploaded_file.getvalue()
        pil = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        rgb = np.array(pil)
        bgr = rgb[:, :, ::-1]
        return bgr
    except Exception:
        return None


def detect_smile_from_bgr(bgr_image: np.ndarray) -> bool:
    """Detect smiles using Haar cascades. Returns True if a smile is likely present.

    This is a simple, best-effort demo. Lighting and camera angle matter.
    """
    face_cascade, smile_cascade = get_haar_cascades()
    if face_cascade is None or smile_cascade is None or cv2 is None:
        return False

    try:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y : y + h, x : x + w]
            # The smile cascade is noisy; these parameters are tuned for fewer false positives
            smiles = smile_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25)
            )
            if len(smiles) > 0:
                return True
        return False
    except Exception:
        return False


# ---------------------------
# Processing pipeline (orchestrator)
# ---------------------------

def process(
    text: str,
    *,
    translation_enabled: bool,
    source_lang_label: str,
    target_lang_label: str,
    transliteration_enabled: bool,
    source_script_label: str,
    target_script_label: str,
    uploaded_image,
    gesture_mode: str,  # "Auto detect smile", "Simulate Namaste", "None"
) -> ProcessResult:
    warnings: list[str] = []

    # Emotion analysis runs on original input for predictable behavior
    emo_label, emo_score = detect_emotion(text)

    working_text = text
    tr_info = None
    tl_info = None

    # Translation (optional)
    if translation_enabled:
        src = LANG_OPTIONS.get(source_lang_label, "en")
        tgt = LANG_OPTIONS.get(target_lang_label, "hi")
        translated, warn = translate_text(working_text, src, tgt)
        working_text = translated
        if warn:
            warnings.append(warn)
            tr_info = warn

    # Transliteration (optional)
    if transliteration_enabled:
        converted, warn = transliterate_text(
            working_text, source_script_label, target_script_label
        )
        working_text = converted
        if warn:
            warnings.append(warn)
            tl_info = warn

    # Gesture: detect or simulate
    gesture: Optional[str] = None
    if gesture_mode == "Auto detect smile":
        bgr = _image_file_to_cv2_bgr(uploaded_image)
        if bgr is not None and detect_smile_from_bgr(bgr):
            gesture = "smile"
        elif uploaded_image is not None:
            # Snapshot present but no smile detected
            gesture = None
    elif gesture_mode == "Simulate Namaste":
        gesture = "namaste"

    return ProcessResult(
        text=working_text,
        emotion=emo_label,
        emotion_score=emo_score,
        gesture=gesture,
        translation_info=tr_info,
        transliteration_info=tl_info,
        warnings=tuple(warnings),
    )


# ---------------------------
# Streamlit UI
# ---------------------------

def main() -> None:
    st.set_page_config(page_title="Emotion & Gesture Transliterator/Translator", page_icon="🌐")
    st.title("Emotion & Gesture-Aware Multilingual Transliteration/Translation")
    st.caption(
        "Single-page MVP: Translate, transliterate, detect emotion and basic gestures (smile/namaste)."
    )

    with st.expander("What this demo does", expanded=False):
        st.write(
            "- Translates text between English and Hindi using free public libraries."
        )
        st.write(
            "- Transliterates text between Devanagari and Latin (IAST) scripts using indic-transliteration."
        )
        st.write(
            "- Detects basic text emotion (happy/sad/neutral) via a small Hugging Face model or a simple heuristic."
        )
        st.write(
            "- Detects a smile from a webcam snapshot using OpenCV Haar cascades, or lets you simulate a Namaste gesture."
        )

    st.subheader("Input")
    text = st.text_area("Enter text", height=140, placeholder="Type something in English or Hindi…")

    col1, col2 = st.columns(2)
    with col1:
        translation_enabled = st.checkbox("Enable translation", value=True)
        src_lang = st.selectbox("Source language", list(LANG_OPTIONS.keys()), index=0)
        tgt_lang = st.selectbox("Target language", list(LANG_OPTIONS.keys()), index=1)
    with col2:
        transliteration_enabled = st.checkbox("Enable transliteration", value=False)
        src_script = st.selectbox("Source script", list(SCRIPT_OPTIONS.keys()), index=0)
        tgt_script = st.selectbox("Target script", list(SCRIPT_OPTIONS.keys()), index=1)

    st.subheader("Gesture input (optional)")
    gesture_mode = st.radio(
        "Choose gesture mode",
        ("None", "Auto detect smile", "Simulate Namaste"),
        index=0,
        help="Auto-detect smile from a webcam snapshot, or simulate a Namaste gesture.",
        horizontal=True,
    )
    uploaded_image = None
    if gesture_mode == "Auto detect smile":
        uploaded_image = st.camera_input("Webcam snapshot (click to capture)")
        if cv2 is None:
            st.info(
                "OpenCV not available. Smile detection will be disabled, but you can still upload a snapshot."
            )

    run = st.button("Run")

    if run:
        if not text.strip():
            st.warning("Please enter some text to process.")
            st.stop()

        with st.spinner("Processing…"):
            result = process(
                text=text,
                translation_enabled=translation_enabled,
                source_lang_label=src_lang,
                target_lang_label=tgt_lang,
                transliteration_enabled=transliteration_enabled,
                source_script_label=src_script,
                target_script_label=tgt_script,
                uploaded_image=uploaded_image,
                gesture_mode=gesture_mode,
            )

        # Compose decorative tags
        emo_tag = EMOJI_BY_EMOTION.get(result.emotion, "")
        gest_tag = EMOJI_BY_GESTURE.get(result.gesture, "") if result.gesture else ""

        st.subheader("Output")
        st.write(f"Detected emotion: {result.emotion} {emo_tag}")
        if result.emotion_score is not None:
            st.caption(f"(confidence ~ {result.emotion_score:.2f})")

        # Final text with tags
        st.markdown("**Final result:**")
        st.text(result.text + (" " + emo_tag if emo_tag else "") + (" " + gest_tag if gest_tag else ""))

        # Friendly warnings/help
        if result.warnings:
            for w in result.warnings:
                st.info(w)

        # Minimal debug info
        with st.expander("Details"):
            st.write({
                "translation": {
                    "enabled": translation_enabled,
                    "from": src_lang,
                    "to": tgt_lang,
                    "note": result.translation_info,
                },
                "transliteration": {
                    "enabled": transliteration_enabled,
                    "from": src_script,
                    "to": tgt_script,
                    "note": result.transliteration_info,
                },
                "gesture": {
                    "mode": gesture_mode,
                    "detected": result.gesture,
                },
            })

    # Footer
    st.caption(
        "This MVP uses free libraries: Streamlit, transformers, deep-translator, indic-transliteration, OpenCV, and more."
    )


if __name__ == "__main__":
    # Streamlit apps should be launched via: streamlit run app.py
    # For direct execution, show a helpful hint instead of running the UI.
    print("Run with: streamlit run app.py", file=sys.stderr)
