"""
Quick smoke test for the MVP functions.
Run inside the project venv after installing requirements:

    python smoke_test.py

This checks translation (graceful on network failures), transliteration,
emotion analysis, and optional gesture detection from images in assets/.
"""
from __future__ import annotations

import os
from typing import Optional

from PIL import Image

import app as mvp


def try_translation() -> None:
    print("\n[Translation]")
    text = "Hello, how are you?"
    res = mvp.translate_text(text, source_lang="en", target_lang="hi", timeout=5.0)
    print("input:", text)
    print("output:", res.text)
    print("error:", res.error)


def try_transliteration() -> None:
    print("\n[Transliteration]")
    dev_text = "नमस्ते भारत"
    lat = mvp.transliterate_text(dev_text, from_script="devanagari", to_script="iast")
    print("dev→lat:", lat)
    back = mvp.transliterate_text(lat, from_script="iast", to_script="devanagari")
    print("lat→dev:", back)


def try_emotion() -> None:
    print("\n[Emotion]")
    text = "I am very happy today!"
    emo = mvp.analyze_emotion(text)
    print("input:", text)
    print("label:", emo.label, "score:", round(emo.score, 3), "error:", emo.error)


def load_image_if_exists(path: str) -> Optional[Image.Image]:
    if os.path.exists(path):
        try:
            return Image.open(path)
        except Exception:
            return None
    return None


def try_gestures() -> None:
    print("\n[Gestures]")
    # Skip if optional deps are not present
    if mvp.cv2 is None and mvp.mp is None:
        print("OpenCV/MediaPipe not available — skipping gesture tests")
        return

    smile_img = load_image_if_exists("assets/smile.jpg")
    namaste_img = load_image_if_exists("assets/namaste.jpg")

    if smile_img and mvp.cv2 is not None:
        s = mvp.detect_smile_from_image(smile_img)
        print("smile detected:", s.detected, "label:", s.label, "error:", s.error)
    else:
        print("smile test skipped (no image or OpenCV)")

    if namaste_img and mvp.mp is not None:
        n = mvp.detect_namaste_from_image(namaste_img)
        print("namaste detected:", n.detected, "label:", n.label, "error:", n.error)
    else:
        print("namaste test skipped (no image or MediaPipe)")


if __name__ == "__main__":
    try_translation()
    try_transliteration()
    try_emotion()
    try_gestures()
    print("\nSmoke test complete.")