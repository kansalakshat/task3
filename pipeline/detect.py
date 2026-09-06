"""Stage 1: detect + crop a face with MediaPipe, embed it with DeepFace (Facenet).

API note: this uses `mediapipe.tasks.python.vision.FaceDetector`, not the older
`mediapipe.solutions.face_detection`. Google dropped the legacy Solutions API
from the 0.10.30+ wheels, and 0.10.30 is the oldest release with a Python 3.13
wheel -- so on 3.13 there is no version that still has it. Both wrap the same
BlazeFace short-range model; the Tasks API returns a pixel bounding box where
Solutions returned a relative one.
"""

import json
import os
import urllib.request

import cv2
import numpy as np

OUT_DIR = "output"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

# Padding around MediaPipe's tight box. Facenet was trained on loosely-cropped
# faces, so feeding it a tight box measurably degrades the embedding.
PAD = 0.25


def _ensure_model():
    """Download the BlazeFace weights once (~230 KB), then reuse them.

    Downloads to a .part file and renames on success. Writing straight to
    MODEL_PATH means an interrupted download leaves a truncated file that
    os.path.exists() happily accepts forever after, and every later run dies
    with "The model is not a valid Flatbuffer buffer".
    """
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"  downloading face detector model -> {MODEL_PATH}")
        part = MODEL_PATH + ".part"
        try:
            urllib.request.urlretrieve(MODEL_URL, part)
            os.replace(part, MODEL_PATH)
        finally:
            if os.path.exists(part):
                os.remove(part)
    return MODEL_PATH


def _detect_box(image, min_confidence=0.5):
    """Return (x, y, w, h) of the highest-confidence face, or None."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_ensure_model()),
        min_detection_confidence=min_confidence,
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with vision.FaceDetector.create_from_options(options) as detector:
        result = detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )

    if not result.detections:
        return None

    best = max(result.detections, key=lambda d: d.categories[0].score)
    b = best.bounding_box
    px, py = b.width * PAD, b.height * PAD

    # Boxes can extend past the frame edge; clamp before slicing or numpy
    # silently returns an empty array.
    h, w = image.shape[:2]
    x0, y0 = max(0, int(b.origin_x - px)), max(0, int(b.origin_y - py))
    x1 = min(w, int(b.origin_x + b.width + px))
    y1 = min(h, int(b.origin_y + b.height + py))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def run(image_path, out_dir=OUT_DIR):
    """Crop the face and embed it. Returns dict with crop path + embedding."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"could not read image: {image_path}")

    box = _detect_box(image)
    if box is None:
        raise ValueError(f"no face detected in {image_path}")

    x, y, w, h = box
    crop = image[y : y + h, x : x + w]

    os.makedirs(out_dir, exist_ok=True)
    crop_path = os.path.join(out_dir, "face_crop.jpg")
    cv2.imwrite(crop_path, crop)

    from deepface import DeepFace

    # detector_backend="skip": MediaPipe already cropped, so don't re-detect.
    rep = DeepFace.represent(
        img_path=crop_path,
        model_name="Facenet",
        detector_backend="skip",
        enforce_detection=False,
    )
    embedding = rep[0]["embedding"]

    result = {
        "source_image": os.path.abspath(image_path),
        "face_crop": os.path.abspath(crop_path),
        "bounding_box": {"x": x, "y": y, "w": w, "h": h},
        "model": "Facenet",
        "dimensions": len(embedding),
        "embedding": embedding,
    }
    with open(os.path.join(out_dir, "embedding.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


def _demo():
    """Self-check: unreadable input and a face-less image must fail cleanly."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        for path, expect in [("nope.jpg", "could not read"), (None, "no face")]:
            if path is None:
                path = os.path.join(d, "blank.jpg")
                cv2.imwrite(path, np.zeros((200, 300, 3), np.uint8))
            try:
                run(path, out_dir=d)
                raise AssertionError(f"expected {expect!r} error")
            except ValueError as e:
                assert expect in str(e), e
    print("detect.py self-check OK")


if __name__ == "__main__":
    _demo()
