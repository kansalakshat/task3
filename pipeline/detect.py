"""Stage 1: detect + crop a face with MediaPipe, embed it with DeepFace (Facenet).

API note: this uses `mediapipe.tasks.python.vision.FaceDetector`, not the older
`mediapipe.solutions.face_detection`. Google dropped the legacy Solutions API
from the 0.10.30+ wheels, and 0.10.30 is the oldest release with a Python 3.13
wheel -- so on 3.13 there is no version that still has it. Both wrap the same
BlazeFace short-range model; the Tasks API returns a pixel bounding box where
Solutions returned a relative one.
"""

import json
import math
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

# Facenet512 over Facenet: same weights family, same call, 512-d instead of
# 128-d, and a materially lower error rate (LFW 99.6 vs 99.2). The extra
# dimensions are what make the stage-2 distance check discriminative enough to
# separate "same person" from "another face in the same pose".
MODEL_NAME = "Facenet512"

# Below this, the face is too few pixels to embed meaningfully -- the vector is
# mostly upscaling artefacts and will sit at a random distance from anything.
# Search thumbnails routinely contain faces this small, and an unguarded
# garbage vector is exactly how a false match gets through.
MIN_FACE_PX = 40


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


def _detect(image, min_confidence=0.5):
    """Return the highest-confidence face detection, or None."""
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
    return max(result.detections, key=lambda d: d.categories[0].score)


def align_and_crop(image):
    """Detect the strongest face, level the eyes, and return (crop, box).

    Returns (None, None) if there is no usable face. `box` is in *source*
    coordinates; the crop is taken after rotation, so the two differ by that
    rotation.

    BlazeFace gives six keypoints -- the first two are the eyes. Rotating so
    the eye line is horizontal is what Facenet's own training crops did, and
    a tilted face is the normal case in candid photos, so skipping this costs
    accuracy on exactly the inputs that matter.
    """
    det = _detect(image)
    if det is None:
        return None, None

    b = det.bounding_box
    if min(b.width, b.height) < MIN_FACE_PX:
        return None, None

    h, w = image.shape[:2]
    (rx, ry), (lx, ly) = [(k.x * w, k.y * h) for k in det.keypoints[:2]]
    matrix = cv2.getRotationMatrix2D(
        ((rx + lx) / 2, (ry + ly) / 2),
        math.degrees(math.atan2(ly - ry, lx - rx)),
        1.0,
    )
    rotated = cv2.warpAffine(
        image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE
    )

    # The box has to travel through the same rotation as the pixels, or the
    # crop lands off the face for any non-trivial tilt.
    corners = np.array([[
        (b.origin_x, b.origin_y),
        (b.origin_x + b.width, b.origin_y),
        (b.origin_x, b.origin_y + b.height),
        (b.origin_x + b.width, b.origin_y + b.height),
    ]], np.float32)
    pts = cv2.transform(corners, matrix)[0]
    px, py = b.width * PAD, b.height * PAD

    # Boxes can extend past the frame edge; clamp before slicing or numpy
    # silently returns an empty array.
    x0, y0 = max(0, int(pts[:, 0].min() - px)), max(0, int(pts[:, 1].min() - py))
    x1, y1 = min(w, int(pts[:, 0].max() + px)), min(h, int(pts[:, 1].max() + py))
    if x1 <= x0 or y1 <= y0:
        return None, None
    return rotated[y0:y1, x0:x1], (b.origin_x, b.origin_y, b.width, b.height)


def embed(image):
    """Embed the strongest face in a BGR image. None if there is no usable one.

    Used by stage 2 to embed candidate images off the web, so those go through
    the identical detect -> align -> crop -> Facenet512 path as the query face.
    A distance is only meaningful between vectors produced the same way.
    """
    crop, _ = align_and_crop(image)
    if crop is None or crop.size == 0:
        return None
    return _represent(crop)


def _represent(crop):
    from deepface import DeepFace

    # detector_backend="skip": we already detected, cropped and aligned.
    rep = DeepFace.represent(
        img_path=crop,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False,
    )
    return rep[0]["embedding"]


def run(image_path, out_dir=OUT_DIR):
    """Crop the face and embed it. Returns dict with crop path + embedding."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"could not read image: {image_path}")

    crop, box = align_and_crop(image)
    if crop is None:
        raise ValueError(f"no face detected in {image_path}")

    x, y, w, h = box

    os.makedirs(out_dir, exist_ok=True)
    crop_path = os.path.join(out_dir, "face_crop.jpg")
    cv2.imwrite(crop_path, crop)

    embedding = _represent(crop)

    result = {
        "source_image": os.path.abspath(image_path),
        "face_crop": os.path.abspath(crop_path),
        # In source coordinates. The crop is eye-aligned, so it is this box
        # rotated -- they only coincide for a perfectly level face.
        "bounding_box": {"x": x, "y": y, "w": w, "h": h},
        "aligned": True,
        "model": MODEL_NAME,
        "dimensions": len(embedding),
        "embedding": embedding,
    }
    with open(os.path.join(out_dir, "embedding.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


def _demo():
    """Self-check: bad input fails cleanly; alignment and embedding hold up."""
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

        # A face rotated on the page must embed to nearly the same vector as
        # the upright one -- that is the whole point of the alignment step, and
        # the check that fails if the box stops travelling with the pixels.
        img = cv2.imread("me.png")
        if img is None:
            print("detect.py self-check OK (no me.png; skipped alignment case)")
            return
        from deepface.modules import verification

        big = cv2.resize(img, None, fx=4, fy=4)
        h, w = big.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 20, 1.0)
        tilted = cv2.warpAffine(big, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

        upright, tilted_emb = embed(big), embed(tilted)
        assert upright is not None and tilted_emb is not None, "lost the face"
        assert len(upright) == 512, len(upright)
        dist = verification.find_cosine_distance(upright, tilted_emb)
        limit = verification.find_threshold(MODEL_NAME, "cosine")
        assert dist < limit, f"tilted face read as a different person: {dist:.3f}"

        # A face too small to embed honestly must be refused, not guessed at.
        assert embed(cv2.resize(img, (30, 28))) is None

    print("detect.py self-check OK")


if __name__ == "__main__":
    _demo()
