## Emotion & Gesture-Aware Multilingual Transliteration/Translation (MVP)

A laptop-friendly Streamlit app that combines:
- Text translation (English ↔ Hindi via LibreTranslate demo API)
- Script transliteration (Devanagari ↔ Latin/IAST)
- Text emotion detection (happy/sad/neutral) using a Hugging Face pipeline
- Basic gesture recognition from webcam snapshot (smile via OpenCV Haar cascades, namaste via MediaPipe Hands heuristic), with button fallbacks

### Requirements
- Python 3.10+
- Internet access for first-time model download and LibreTranslate API
- Webcam optional (app still works without it)

### Setup
```bash
# From the project root
cd emotion_gesture_translation_mvp
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run app.py
```
This opens the app in your browser. If it doesn’t auto-open, visit the printed local URL.

### Using the app
- Mode: choose Translate or Transliterate in the sidebar.
- Translate: choose source and target languages (English/Hindi), enter text, click Run.
- Transliterate: choose From Script and To Script (Devanagari/Latin-IAST), enter text, click Run.
- Gesture input: take a webcam snapshot (smile/namaste) or click the Smile/Namaste buttons to simulate.
- Output: shows final text tagged with detected emotion and gesture (e.g., `[happy 😊] [smile 😄]`).

### Example inputs
- English → Hindi: "Hello, how are you?" → Expected like: "नमस्ते, आप कैसे हैं?" → Tags: `[happy 😊]` if text is positive; `[smile 😄]` if you smiled.
- Hindi → English: "आज मौसम सुहावना है" → Expected like: "The weather is pleasant today." → Tags depend on emotion.
- Transliteration (Devanagari → Latin): "नमस्ते भारत" → "namaste bhārata"

### Troubleshooting
- Camera not working:
  - Ensure your browser has camera permissions enabled.
  - Some environments (VMs/remote servers) do not expose a webcam; use the gesture buttons instead.
- Gesture detection errors:
  - OpenCV/MediaPipe may not be available on all systems. The app will show friendly messages and you can still use buttons.
- Emotion model download fails:
  - The first run downloads a small model from Hugging Face. If behind a firewall, connect to the internet or pre-download with a VPN and re-run.
  - If the model is unavailable, the app falls back to neutral.
- LibreTranslate failures or rate limits:
  - The app will preserve your input text and display an info message. Try again later or switch to Transliterate mode.

### Quick smoke test 
```bash
# From inside emotion_gesture_translation_mvp venv
python smoke_test.py
```
This script checks translation (with graceful fallback), transliteration, emotion analysis, and (optionally) gesture detection using images in `assets/` if present.

### Suggested module structure 
For maintainability beyond the MVP, split into modules:
- `core/translation.py`: LibreTranslate client utilities
- `core/transliteration.py`: Devanagari ↔ Latin helpers
- `core/emotion.py`: Hugging Face pipeline wrapper and label normalization
- `core/gesture.py`: OpenCV smile + MediaPipe namaste heuristics
- `ui/app.py`: Streamlit layout and event wiring
- `tests/`: smoke and unit tests

