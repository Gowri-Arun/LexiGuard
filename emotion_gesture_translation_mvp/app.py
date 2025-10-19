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
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ru": "Russian",
    "pt": "Portuguese",}

EMOTION_TO_EMOJI = {
    "joy": "😊",
    "happy": "😊",
    "sad": "😢",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😮",
    "love": "😍",
    "neutral": "😐",
    "confused": "😕",
    "embarrassed": "😳",
}

GESTURE_TO_EMOJI = {
    "smile": "😄",
    "namaste": "🙏",
    "wave": "👋",
    "thumbs_up": "👍",
    "clap": "👏",
    "fist_bump": "👊",
    "peace": "✌️",
    "raised-hand": "✋",
    "ok": "👌",
    "rock-on": "🤘",
    "shrug": "🤷",
}

DEFAULT_LT_ENDPOINT = "http://127.0.0.1:5000/translate"  # Change if using a different LibreTranslate server


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
    # Optional confidence score in [0.0, 1.0]
    confidence: t.Optional[float] = None


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

        # More granular normalization to the supported emotion set:
        # {joy, happy, sad, anger, fear, surprise, love, neutral, confused, embarrassed}
        if raw_label in ("joy", "joyful") or "joy" in raw_label:
            norm = "joy"
        elif raw_label in ("happy", "happiness") or "happy" in raw_label or "happiness" in raw_label:
            norm = "happy"
        elif any(k in raw_label for k in ("sad", "sorrow", "sadness")):
            norm = "sad"
        elif any(k in raw_label for k in ("anger", "angry")):
            norm = "anger"
        elif any(k in raw_label for k in ("fear", "afraid", "scared")):
            norm = "fear"
        elif any(k in raw_label for k in ("surprise", "surprised")):
            norm = "surprise"
        elif "love" in raw_label or "affection" in raw_label:
            norm = "love"
        elif any(k in raw_label for k in ("confus", "confused", "confusion")):
            norm = "confused"
        elif any(k in raw_label for k in ("embarrass", "embarrassed")):
            norm = "embarrassed"
        elif "neutral" in raw_label or raw_label.strip() == "":
            norm = "neutral"
        else:
            # As a best-effort fallback, if the raw label exactly matches one of our keys, keep it.
            candidate = raw_label.strip()
            if candidate in EMOTION_TO_EMOJI:
                norm = candidate
            else:
                # Unknown labels are treated as neutral to avoid surprising UI tags
                norm = "neutral"

        return EmotionResult(label=norm, score=score)
    except Exception as e:
        return EmotionResult(label="neutral", score=1.0, error=f"Emotion analysis failed: {e}")


# -----------------------------
# Gesture Detection (Smile via OpenCV / Namaste via MediaPipe Hands)
# -----------------------------

def detect_smile_from_image(pil_image: Image.Image) -> GestureResult:
    """Basic smile detection using OpenCV Haar cascades with light pre-processing and confidence.

    Steps:
    - Convert to grayscale
    - Apply CLAHE (adaptive histogram equalization) for contrast
    - Detect face → detect smile in face ROI
    - Confidence based on number/size of smile detections
    """
    if cv2 is None:
        return GestureResult(detected=False, error="OpenCV not available")

    try:
        img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # CLAHE for robustness under varying lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Use built-in Haar cascades shipped with OpenCV package
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        if face_cascade.empty() or smile_cascade.empty():
            return GestureResult(detected=False, error="Haar cascades not found")

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        best_conf = 0.0
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            # Slight Gaussian blur to reduce noise
            roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=15, minSize=(20, 20))
            if len(smiles) > 0:
                # Confidence heuristic: relative width and count of detections
                confs = []
                for (sx, sy, sw, sh) in smiles:
                    rel = min(1.0, (sw / max(1.0, w)) * 1.5)
                    confs.append(rel)
                best_conf = max(best_conf, float(min(1.0, sum(confs))))
        if best_conf > 0.0:
            return GestureResult(detected=True, label="smile", confidence=best_conf)
        return GestureResult(detected=False)
    except Exception as e:
        return GestureResult(detected=False, error=f"Smile detection failed: {e}")


def detect_namaste_from_image(pil_image: Image.Image) -> GestureResult:
    """Detect hand gestures using MediaPipe Hands with palm-size normalization and confidence.

    Improvements:
    - Normalize distances by estimated palm width for scale invariance
    - Provide a simple confidence score in [0,1]
    - Refined heuristics for two-hand (namaste/clap/fist_bump/shrug) and single-hand gestures
    """
    if mp is None:
        return GestureResult(detected=False, error="MediaPipe not available")

    try:
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
        ) as hands:
            img = np.array(pil_image.convert("RGB"))
            results = hands.process(img)
            if not results.multi_hand_landmarks:
                return GestureResult(detected=False)

            img_h, img_w, _ = img.shape

            # Helper: convert landmark to pixel coordinate
            def lm_pt(lm):
                return (lm.x * img_w, lm.y * img_h)

            def euclidean(a, b):
                return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)

            def palm_width(lms: list[tuple[float, float]]) -> float:
                idx = mp_hands.HandLandmark.INDEX_FINGER_MCP
                pky = mp_hands.HandLandmark.PINKY_MCP
                return euclidean(lms[idx], lms[pky])

            def tip_to_wrist_ratio(lms: list[tuple[float, float]], tip_idx: int, wrist_idx: int) -> float:
                pw = max(1.0, palm_width(lms))
                return euclidean(lms[tip_idx], lms[wrist_idx]) / pw

            def conf_less(value: float, threshold: float) -> float:
                # Higher confidence when value is much smaller than threshold
                return float(max(0.0, min(1.0, 1.0 - (value / max(1e-6, threshold)))))

            def conf_greater(value: float, threshold: float) -> float:
                # Higher confidence when value is much larger than threshold
                return float(max(0.0, min(1.0, (value / max(1e-6, threshold)) - 1.0)))

            # Collect landmark arrays per hand for easier heuristics
            hands_lms: list[list[tuple[float, float]]] = []
            for hand_landmarks in results.multi_hand_landmarks:
                pts = [lm_pt(lm) for lm in hand_landmarks.landmark]
                hands_lms.append(pts)

            # Two-hand heuristics
            if len(hands_lms) >= 2:
                l0, l1 = hands_lms[0], hands_lms[1]
                wrist0 = l0[mp_hands.HandLandmark.WRIST]
                wrist1 = l1[mp_hands.HandLandmark.WRIST]
                wrist_dist = euclidean(wrist0, wrist1)
                avg_palm = max(1.0, 0.5 * (palm_width(l0) + palm_width(l1)))

                # Finger-tip pairing distances
                tips_idx = [
                    mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_TIP,
                ]
                tip_dists = [euclidean(l0[idx], l1[idx]) for idx in tips_idx]
                mean_tip = float(np.mean(tip_dists)) if tip_dists else 1e6

                # Namaste: wrists close and tips close relative to palm size
                th_wrist_close = 1.6 * avg_palm
                th_tips_close = 1.5 * avg_palm
                if wrist_dist < th_wrist_close and mean_tip < th_tips_close:
                    c1 = conf_less(wrist_dist, th_wrist_close)
                    c2 = conf_less(mean_tip, th_tips_close)
                    return GestureResult(detected=True, label="namaste", confidence=min(c1, c2))

                # Clap: tips very close regardless of wrist distance
                th_tips_clap = 1.2 * avg_palm
                if mean_tip < th_tips_clap:
                    return GestureResult(detected=True, label="clap", confidence=conf_less(mean_tip, th_tips_clap))

                # Fist bump: wrists moderately close and fists (tips near wrists on both hands)
                folded_th = 1.15
                ratios0 = [
                    tip_to_wrist_ratio(l0, mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.WRIST),
                    tip_to_wrist_ratio(l0, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.WRIST),
                    tip_to_wrist_ratio(l0, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.WRIST),
                    tip_to_wrist_ratio(l0, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.WRIST),
                    tip_to_wrist_ratio(l0, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.WRIST),
                ]
                ratios1 = [
                    tip_to_wrist_ratio(l1, mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.WRIST),
                    tip_to_wrist_ratio(l1, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.WRIST),
                    tip_to_wrist_ratio(l1, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.WRIST),
                    tip_to_wrist_ratio(l1, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.WRIST),
                    tip_to_wrist_ratio(l1, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.WRIST),
                ]
                both_fists = all(r < folded_th for r in ratios0) and all(r < folded_th for r in ratios1)
                th_wrist_bump = 2.0 * avg_palm
                if both_fists and wrist_dist < th_wrist_bump:
                    c1 = conf_less(wrist_dist, th_wrist_bump)
                    c2 = min([conf_less(r, folded_th) for r in ratios0 + ratios1])
                    return GestureResult(detected=True, label="fist_bump", confidence=min(c1, c2))

                # Shrug-like: hands raised and apart
                avg_y = (wrist0[1] + wrist1[1]) / 2.0
                if avg_y < img_h * 0.45 and abs(wrist0[0] - wrist1[0]) > img_w * 0.2:
                    # Confidence increases the higher the hands are
                    conf = conf_less(avg_y, img_h * 0.45)
                    return GestureResult(detected=True, label="shrug", confidence=conf)

                return GestureResult(detected=False)

            # Single-hand heuristics (normalize by palm width)
            lms = hands_lms[0]
            wrist = lms[mp_hands.HandLandmark.WRIST]
            thumb_tip = lms[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = lms[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = lms[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = lms[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = lms[mp_hands.HandLandmark.PINKY_TIP]

            pw = max(1.0, palm_width(lms))

            # Ratios of tip-to-wrist distances to palm width
            r_thumb = euclidean(thumb_tip, wrist) / pw
            r_index = euclidean(index_tip, wrist) / pw
            r_middle = euclidean(middle_tip, wrist) / pw
            r_ring = euclidean(ring_tip, wrist) / pw
            r_pinky = euclidean(pinky_tip, wrist) / pw

            extended_th = 1.7
            folded_th = 1.2

            # OK sign: thumb tip close to index tip
            ok_th = 0.6 * pw
            d_ok = euclidean(thumb_tip, index_tip)
            if d_ok < ok_th:
                return GestureResult(detected=True, label="ok", confidence=conf_less(d_ok, ok_th))

            # Peace sign: index and middle extended, ring/pinky folded
            if r_index > extended_th and r_middle > extended_th and r_ring < extended_th and r_pinky < extended_th:
                c = min(conf_greater(r_index, extended_th), conf_greater(r_middle, extended_th), conf_less(r_ring, extended_th), conf_less(r_pinky, extended_th))
                return GestureResult(detected=True, label="peace", confidence=c)

            # Thumbs up: thumb extended and other fingers relatively folded
            others_avg = (r_index + r_middle + r_ring + r_pinky) / 4.0
            if r_thumb > extended_th and others_avg < extended_th:
                c = min(conf_greater(r_thumb, extended_th), conf_less(others_avg, extended_th))
                return GestureResult(detected=True, label="thumbs_up", confidence=c)

            # Rock-on: index and pinky extended, middle and ring folded
            if r_index > extended_th and r_pinky > extended_th and r_middle < extended_th and r_ring < extended_th:
                c = min(conf_greater(r_index, extended_th), conf_greater(r_pinky, extended_th), conf_less(r_middle, extended_th), conf_less(r_ring, extended_th))
                return GestureResult(detected=True, label="rock-on", confidence=c)

            # Fist: all fingertips close to wrist
            if all(r < folded_th for r in (r_thumb, r_index, r_middle, r_ring, r_pinky)):
                c = min(conf_less(r_thumb, folded_th), conf_less(r_index, folded_th), conf_less(r_middle, folded_th), conf_less(r_ring, folded_th), conf_less(r_pinky, folded_th))
                return GestureResult(detected=True, label="fist_bump", confidence=c)

            # Raised-hand / wave: open palm (all fingers extended)
            if all(r > extended_th for r in (r_index, r_middle, r_ring, r_pinky)):
                if wrist[1] < img_h * 0.35:
                    return GestureResult(detected=True, label="raised-hand", confidence=conf_less(wrist[1], img_h * 0.35))
                return GestureResult(detected=True, label="wave", confidence=0.5)

            return GestureResult(detected=False)
    except Exception as e:
        return GestureResult(detected=False, error=f"Namaste/hand gesture detection failed: {e}")


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
