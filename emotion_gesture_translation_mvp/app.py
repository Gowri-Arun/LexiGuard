"""
Emotion & Gesture-Aware Multilingual Transliteration/Translation (MVP)

Setup & Run (no GPU required):
- Create and activate a Python 3.10+ virtual environment
    python3 -m venv .venv && source .venv/bin/activate
- Install dependencies
    pip install -r requirements.txt
- Launch the Streamlit app
    streamlit run app.py

Notes:
- Translation uses LibreTranslate (public demo URL by default) with graceful fallback.
- Emotion detection uses a small Hugging Face pipeline and runs on CPU.
- Gesture recognition attempts webcam via Streamlit's camera_input; if unavailable, use buttons.
- For a demo-friendly build, OpenCV/MediaPipe are optional; errors are handled with friendly messages.

"""

import io
import sys
import json
import time
import base64
import typing as t
from dataclasses import dataclass

import requests
import numpy as np
from PIL import Image

import streamlit as st

# Transformers are used only for the emotion detection pipeline
from transformers import pipeline

# Optional imports for gesture detection. We'll guard with try/except.
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - opencv may be unavailable
    cv2 = None

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover - mediapipe may be unavailable
    mp = None

# Transliteration (Devanagari ↔ Latin) using indic-transliteration
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except Exception:  # pragma: no cover
    sanscript = None
    transliterate = None


# -----------------------------
# Data classes and constants
# -----------------------------

SUPPORTED_LANGS = {
    "en": "English",
    "hi": "Hindi",
}

EMOTION_TO_EMOJI = {
    "joy": "😊",
    "happy": "😊",
    "sad": "😢",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😮",
    "love": "😍",
    "neutral": "😐",
}

GESTURE_TO_EMOJI = {
    "smile": "😄",
    "namaste": "🙏",
}

DEFAULT_LT_ENDPOINT = "https://libretranslate.de/translate"


@dataclass
class TranslationResult:
    text: str
    source_lang: str
    target_lang: str
    error: t.Optional[str] = None


@dataclass
class EmotionResult:
    label: str
    score: float
    error: t.Optional[str] = None


@dataclass
class GestureResult:
    detected: bool
    label: t.Optional[str] = None
    error: t.Optional[str] = None


# -----------------------------
# Translation (LibreTranslate)
# -----------------------------

def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    lt_endpoint: str = DEFAULT_LT_ENDPOINT,
    timeout: float = 8.0,
) -> TranslationResult:
    """Translate text via LibreTranslate public API.

    If the API fails, returns the original text with an error message.
    """
    if not text.strip():
        return TranslationResult(text="", source_lang=source_lang, target_lang=target_lang)

    if source_lang == target_lang:
        return TranslationResult(text=text, source_lang=source_lang, target_lang=target_lang)

    payload = {
        "q": text,
        "source": source_lang,
        "target": target_lang,
        "format": "text",
    }
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(lt_endpoint, headers=headers, data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        translated = data.get("translatedText")
        if not translated:
            raise RuntimeError("No translatedText in response")
        return TranslationResult(text=translated, source_lang=source_lang, target_lang=target_lang)
    except Exception as e:
        return TranslationResult(
            text=text,  # graceful: keep original text
            source_lang=source_lang,
            target_lang=target_lang,
            error=f"Translation unavailable: {e}"
        )


# -----------------------------
# Transliteration (Devanagari ↔ Latin)
# -----------------------------

def transliterate_text(text: str, from_script: str, to_script: str) -> str:
    """Transliterate between Devanagari and Latin using indic-transliteration.

    from_script/to_script should be one of: 'devanagari', 'iast' (or 'hk').
    We'll map UI choices to these.
    """
    if not text or transliterate is None or sanscript is None:
        return text

    if from_script == to_script:
        return text

    script_map = {
        "devanagari": sanscript.DEVANAGARI,
        "iast": sanscript.IAST,
        "hk": sanscript.HK,
    }

    src = script_map.get(from_script)
    dst = script_map.get(to_script)
    if not src or not dst:
        return text

    try:
        return transliterate(text, _from=src, _to=dst)
    except Exception:
        return text


# -----------------------------
# Emotion Detection (HF pipeline)
# -----------------------------

@st.cache_resource(show_spinner=False)
def get_emotion_pipeline():
    """Lazy-load a small zero-shot-like emotion classifier or a distilled model.

    We normalize labels to {happy, sad, neutral}.
    """
    try:
        # The "j-hartmann/emotion-english-distilroberta-base" model outputs emotion labels.
        # It's CPU-friendly. If offline, we'll handle errors. We request all scores
        # so we can select the best label reliably.
        clf = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
        )
        return clf
    except Exception:
        return None


def analyze_emotion(text: str) -> EmotionResult:
    if not text.strip():
        return EmotionResult(label="neutral", score=1.0)

    clf = get_emotion_pipeline()
    if clf is None:
        return EmotionResult(label="neutral", score=1.0, error="Emotion model unavailable")

    try:
        outputs = clf(text)
        # outputs can be list[dict] with 'label' and 'score' per class or list[list[...]] depending on top_k
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], list):
            # top_k=None returns list[list]; take the best label
            best = max(outputs[0], key=lambda x: x.get("score", 0.0))
        elif isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            best = max(outputs, key=lambda x: x.get("score", 0.0))
        else:
            return EmotionResult(label="neutral", score=1.0, error="Unexpected emotion output format")

        raw_label = str(best.get("label", "neutral")).lower()
        score = float(best.get("score", 0.0))

        # Normalize to {happy, sad, neutral}
        if any(k in raw_label for k in ["joy", "happiness", "love"]):
            norm = "happy"
        elif any(k in raw_label for k in ["sad", "sorrow"]):
            norm = "sad"
        elif any(k in raw_label for k in ["anger", "fear", "disgust"]):
            # Map negative emotions to sad for simplicity
            norm = "sad"
        elif "neutral" in raw_label:
            norm = "neutral"
        else:
            norm = "neutral"

        return EmotionResult(label=norm, score=score)
    except Exception as e:
        return EmotionResult(label="neutral", score=1.0, error=f"Emotion analysis failed: {e}")


# -----------------------------
# Gesture Detection (Smile via OpenCV / Namaste via MediaPipe Hands)
# -----------------------------

def detect_smile_from_image(pil_image: Image.Image) -> GestureResult:
    """Basic smile detection using OpenCV Haar cascades (very simple and demo-only)."""
    if cv2 is None:
        return GestureResult(detected=False, error="OpenCV not available")

    try:
        img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Use built-in Haar cascades shipped with OpenCV package
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        if face_cascade.empty() or smile_cascade.empty():
            return GestureResult(detected=False, error="Haar cascades not found")

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
            if len(smiles) > 0:
                return GestureResult(detected=True, label="smile")
        return GestureResult(detected=False)
    except Exception as e:
        return GestureResult(detected=False, error=f"Smile detection failed: {e}")


def detect_namaste_from_image(pil_image: Image.Image) -> GestureResult:
    """Detect a namaste gesture using MediaPipe Hands by checking hands proximity.

    Heuristic: If two hands detected with wrists close and palms facing, we flag 'namaste'.
    This is a simplistic heuristic intended only for MVP demo; may be unreliable.
    """
    if mp is None:
        return GestureResult(detected=False, error="MediaPipe not available")

    try:
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
            img = np.array(pil_image.convert("RGB"))
            results = hands.process(img)
            if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
                return GestureResult(detected=False)

            # Extract wrist landmarks for two hands
            wrist_points = []
            img_h, img_w, _ = img.shape
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_points.append((wrist.x * img_w, wrist.y * img_h))

            if len(wrist_points) >= 2:
                (x1, y1), (x2, y2) = wrist_points[:2]
                dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                # Heuristic threshold relative to image width
                if dist < max(40, img_w * 0.12):
                    return GestureResult(detected=True, label="namaste")
            return GestureResult(detected=False)
    except Exception as e:
        return GestureResult(detected=False, error=f"Namaste detection failed: {e}")


# -----------------------------
# UI Helpers
# -----------------------------

def tag_text_with_emotion_and_gesture(text: str, emotion: EmotionResult, gesture: GestureResult) -> str:
    """Append emoji/tags to the text based on emotion and gesture results."""
    parts = [text]

    # Emotion tag
    emo = emotion.label.lower() if emotion.label else "neutral"
    emo_emoji = EMOTION_TO_EMOJI.get(emo, EMOTION_TO_EMOJI["neutral"])
    parts.append(f"[{emo} {emo_emoji}]")

    # Gesture tag
    if gesture.detected and gesture.label:
        ges_emoji = GESTURE_TO_EMOJI.get(gesture.label, "")
        parts.append(f"[{gesture.label} {ges_emoji}]")

    return " ".join(parts)


# -----------------------------
# Streamlit App
# -----------------------------

def main():
    st.set_page_config(page_title="Emotion & Gesture-Aware Transliterator/Translator", page_icon="🌐")
    st.title("Emotion & Gesture-Aware Multilingual Transliteration/Translation")
    st.caption("MVP • English ↔ Hindi translation, Devanagari ↔ Latin transliteration, emotion and gesture tagging")

    with st.expander("About this demo"):
        st.write(
            "This MVP demonstrates text translation/transliteration with simple emotion detection and basic gesture recognition. "
            "If camera or models are unavailable, the app degrades gracefully."
        )

    # Sidebar controls
    st.sidebar.header("Settings")
    mode = st.sidebar.radio("Mode", ["Translate", "Transliterate"], index=0)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        src_lang = st.selectbox("Source Language", options=list(SUPPORTED_LANGS.keys()), index=0, format_func=lambda k: SUPPORTED_LANGS[k])
    with col2:
        tgt_lang = st.selectbox("Target Language", options=list(SUPPORTED_LANGS.keys()), index=1, format_func=lambda k: SUPPORTED_LANGS[k])

    # Scripts for transliteration
    script_choices = {"Devanagari": "devanagari", "Latin (IAST)": "iast"}
    col3, col4 = st.sidebar.columns(2)
    with col3:
        from_script = st.selectbox("From Script", options=list(script_choices.values()), index=0, format_func=lambda v: [k for k, val in script_choices.items() if val == v][0])
    with col4:
        to_script = st.selectbox("To Script", options=list(script_choices.values()), index=1, format_func=lambda v: [k for k, val in script_choices.items() if val == v][0])

    st.markdown("---")

    # Text input
    input_text = st.text_area("Input Text", height=150, placeholder="Type text here…")

    # Gesture input: webcam snapshot or buttons
    st.subheader("Gesture Input (optional)")
    cam_col, btn_col = st.columns(2)
    with cam_col:
        snapshot = st.camera_input("Take a snapshot (smile/namaste)", help="If your device has a webcam, try detecting a smile or a namaste gesture.")
    with btn_col:
        st.write("No camera? Use quick gesture buttons:")
        btn_smile = st.button("Trigger Smile 😄")
        btn_namaste = st.button("Trigger Namaste 🙏")

    # Process gestures
    gesture_result = GestureResult(detected=False)
    gesture_errors: list[str] = []

    if snapshot is not None:
        try:
            pil_img = Image.open(io.BytesIO(snapshot.getvalue()))
            # Try smile detection first
            smile_res = detect_smile_from_image(pil_img)
            if smile_res.detected:
                gesture_result = smile_res
            else:
                # Try namaste detection
                namaste_res = detect_namaste_from_image(pil_img)
                if namaste_res.detected:
                    gesture_result = namaste_res
                # track any errors for display
                if smile_res.error:
                    gesture_errors.append(smile_res.error)
                if namaste_res.error:
                    gesture_errors.append(namaste_res.error)
        except Exception as e:
            gesture_errors.append(f"Failed to read snapshot: {e}")

    # Button fallback gestures
    if btn_smile:
        gesture_result = GestureResult(detected=True, label="smile")
    if btn_namaste:
        gesture_result = GestureResult(detected=True, label="namaste")

    # Emotion detection on the input text (regardless of mode)
    emotion_result = analyze_emotion(input_text)

    # Main action button
    st.markdown("---")
    run = st.button("Run")

    output_text = ""
    translation_error_msg = None

    if run:
        if mode == "Translate":
            tr = translate_text(input_text, src_lang, tgt_lang)
            output_text = tr.text
            translation_error_msg = tr.error
        else:
            # Transliterate
            output_text = transliterate_text(input_text, from_script=from_script, to_script=to_script)

        # Tag with emotion and gesture
        output_text = tag_text_with_emotion_and_gesture(output_text, emotion_result, gesture_result)

    # Output area
    st.subheader("Output")
    st.text_area("Result", value=output_text, height=150)

    # Friendly messages and diagnostics
    if translation_error_msg:
        st.info(translation_error_msg)

    if emotion_result.error:
        st.info(emotion_result.error)

    if gesture_errors:
        with st.expander("Gesture diagnostics"):
            for err in gesture_errors:
                st.write(f"- {err}")

    st.markdown("---")
    st.caption("Tip: If translation fails due to API rate limits, try again later or switch to Transliterate mode.")


if __name__ == "__main__":
    main()
