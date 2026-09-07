"""Stage 2: reverse-image-search the cropped face via SerpAPI.

Reverse image search APIs take a *public URL*, not a file upload, so the crop
is pushed to a public file host first and the resulting URL is handed to
SerpAPI. That upload is permanent and cannot be deleted (see upload_public) --
a real privacy cost, and the largest single reason this pipeline is
consent-only. See the README's scope section before pointing this at anyone.
"""

import datetime
import json
import os
import re

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from pipeline import detect

OUT_DIR = "output"

SERPAPI_URL = "https://serpapi.com/search"
UPLOAD_URL = "https://catbox.moe/user/api.php"
MAX_MATCHES = 5
TIMEOUT = 60

# Google Lens matches on whatever dominates the image. On a face crop that is
# routinely the sunglasses, the hairline or the background -- it will happily
# return sixty product pages that share nothing with the person. So every
# candidate is re-embedded and distance-checked against the query face before
# it counts as a match; CANDIDATES is how deep into the engine's ranking that
# check reaches, since the true match is often not in the engine's top 5.
CANDIDATES = 25
DISTANCE_METRIC = "cosine"

# Fetching a candidate image should not inherit the 60s API timeout; a slow
# host would otherwise stall the whole check.
FETCH_TIMEOUT = 15
MAX_FETCH_BYTES = 12 * 1024 * 1024

# The host rejects the default python-requests agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def upload_public(image_path):
    """Upload the crop and return a direct public image URL.

    NOT temporary. catbox.moe stores anonymous uploads permanently and offers
    no anonymous delete, so whatever goes through here is public for good.
    Hosts with real expiry were all dead when this was written (litterbox 403s,
    0x0.st disabled uploads, uguu returns a non-resolving CDN host, x0.at's TLS
    cert broke). This is the honest name for what actually happens -- read the
    README's scope section before running it on a real person.

    ponytail: single host, no fallback chain. If it is down the caller records
    an honest "search failed" rather than inventing a match.
    """
    with open(image_path, "rb") as f:
        resp = requests.post(
            UPLOAD_URL,
            files={"fileToUpload": f},
            data={"reqtype": "fileupload"},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
    resp.raise_for_status()
    url = resp.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"upload host returned no URL: {url[:200]}")
    return url


def _query_serpapi(image_url, api_key):
    resp = requests.get(
        SERPAPI_URL,
        params={"engine": "google_lens", "url": image_url, "api_key": api_key},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"SerpAPI: {data['error']}")
    return data.get("visual_matches", [])


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _clean(value):
    """Strip control characters from engine-supplied text.

    Titles and URLs originate on arbitrary web pages, so they are attacker
    controlled. Printed raw they can inject ANSI escape sequences into the
    terminal; stored raw they carry that into every downstream consumer.
    Cleaned here at capture rather than at each print site.
    """
    return _CONTROL_CHARS.sub("", value) if isinstance(value, str) else value


def _redact(text, secret):
    """Strip the API key out of error text.

    requests puts the full request URL in HTTPError messages, and that URL
    carries api_key=... . Without this the key lands in match_result.json and
    in terminal output -- i.e. in anything screenshotted or shared.
    """
    text = str(text)
    return text.replace(secret, "***REDACTED***") if secret else text


def _fetch_image(url):
    """Download an image URL into a BGR array. None if it is not usable."""
    resp = requests.get(url, headers=HEADERS, timeout=FETCH_TIMEOUT, stream=True)
    resp.raise_for_status()
    # Capped read: these URLs come off arbitrary web pages, so an unbounded
    # read is a remote party deciding how much memory this process uses.
    data = resp.raw.read(MAX_FETCH_BYTES + 1)
    if len(data) > MAX_FETCH_BYTES:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _face_distance(candidate, query_embedding):
    """Cosine distance from the query face to the face in `candidate`.

    None when the candidate holds no usable face at all -- which is the common
    case, and precisely what the old code counted as a match.
    """
    from deepface.modules import verification

    # Prefer the full image over the thumbnail: gstatic thumbnails are often
    # too small for the face to survive MIN_FACE_PX, and a match dropped for
    # being small is indistinguishable from one dropped for being wrong.
    for url in (candidate.get("image"), candidate.get("thumbnail")):
        if not url:
            continue
        try:
            image = _fetch_image(url)
        except (requests.RequestException, OSError):
            continue
        if image is None:
            continue
        embedding = detect.embed(image)
        if embedding is not None:
            return verification.find_cosine_distance(query_embedding, embedding)
    return None


def _verify(matches, query_embedding):
    """Keep only candidates whose face is the query face, best first.

    ponytail: serial fetch + embed, ~25 round trips. Thread the fetches if the
    wait becomes the complaint -- the embeddings share a TF graph and should
    stay on one thread.
    """
    from deepface.modules import verification

    limit = verification.find_threshold(detect.MODEL_NAME, DISTANCE_METRIC)
    scored = []
    for m in matches[:CANDIDATES]:
        distance = _face_distance(m, query_embedding)
        if distance is not None and distance <= limit:
            scored.append((distance, m))
    scored.sort(key=lambda pair: pair[0])
    return scored, limit


def _write(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "match_result.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def run(crop_path, out_dir=OUT_DIR, api_key=None, embedding=None):
    """Search the web for the cropped face. Never raises on 'no match found'.

    `embedding` is the stage 1 query vector. With it, every candidate the
    engine returns is re-embedded and distance-checked, and only same-face
    candidates are recorded. Without it the engine's ranking is taken on
    trust, which is worth roughly nothing -- see CANDIDATES.
    """
    # load_dotenv here, not just in main.py: the README documents running this
    # stage standalone, and without it .env is ignored and the search silently
    # reports "no key" even though one is configured.
    load_dotenv()
    api_key = api_key or os.getenv("SERPAPI_KEY")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = {
        "searched_at": now,
        "engine": "serpapi/google_lens",
        "matched": False,
        "matches": [],
        "note": "",
        "face_verified": embedding is not None,
        "model": detect.MODEL_NAME,
        "distance_metric": DISTANCE_METRIC,
    }

    if not api_key:
        result["note"] = "SERPAPI_KEY not set; search skipped."
        print("  ! SERPAPI_KEY not set -- skipping search (no match recorded).")
        _write(result, out_dir)
        return result

    try:
        image_url = upload_public(crop_path)
        result["query_image_url"] = image_url
        matches = _query_serpapi(image_url, api_key)
    except (requests.RequestException, RuntimeError, OSError) as e:
        # A dead API must not kill the pipeline, and must not invent a match.
        reason = _redact(e, api_key)
        result["note"] = f"search failed: {reason}"
        print(f"  ! search failed: {reason}")
        _write(result, out_dir)
        return result

    if embedding is None:
        # Unverified path: kept so this stage still runs standalone, but say
        # plainly that nothing confirmed these are the same face.
        kept = [(None, m) for m in matches[:MAX_MATCHES]]
        result["note"] = "engine ranking only; faces not verified."
    else:
        kept, limit = _verify(matches, embedding)
        result["threshold"] = limit
        result["candidates_checked"] = min(len(matches), CANDIDATES)

    for distance, m in kept[:MAX_MATCHES]:
        entry = {
            "title": _clean(m.get("title")),
            "url": _clean(m.get("link")),
            "source": _clean(m.get("source")),
            "thumbnail": _clean(m.get("thumbnail")),
        }
        if distance is not None:
            entry["distance"] = round(distance, 4)
        result["matches"].append(entry)
    result["matched"] = bool(result["matches"])
    result["total_returned"] = len(matches)
    if not result["matched"]:
        if embedding is not None and matches:
            result["note"] = (
                f"{result['candidates_checked']} candidate(s) checked; "
                "none contained this face."
            )
            print(f"  ! no face-verified match in {result['candidates_checked']} candidate(s).")
        else:
            result["note"] = "no visual matches returned by the engine."
            print("  ! no visual matches found.")

    _write(result, out_dir)
    return result


def _demo():
    """Self-check: failures must produce an honest no-match, never a crash."""
    import tempfile
    global load_dotenv

    with tempfile.TemporaryDirectory() as d:
        # Force the genuine no-key branch. Once a real .env exists, run() would
        # otherwise load a live key and this case would quietly stop testing
        # the thing it claims to test.
        real_loader, load_dotenv = load_dotenv, lambda *a, **k: None
        saved = os.environ.pop("SERPAPI_KEY", None)
        try:
            r = run("nonexistent.jpg", out_dir=d, api_key=None)
            assert r["matched"] is False and r["matches"] == [], r
            assert "not set" in r["note"], r
        finally:
            load_dotenv = real_loader
            if saved is not None:
                os.environ["SERPAPI_KEY"] = saved
        with open(os.path.join(d, "match_result.json")) as f:
            assert json.load(f)["matched"] is False

        # Key present but the image is unreadable -> graceful, no fabrication.
        r = run("nonexistent.jpg", out_dir=d, api_key="dummy")
        assert r["matched"] is False and "failed" in r["note"], r

    # Engine-supplied text must lose control chars but keep ordinary content.
    dirty = "evil" + chr(27) + "[31m" + chr(0) + "title"
    assert _clean(dirty) == "evil[31mtitle"
    assert _clean("https://ok.example/a-b_c") == "https://ok.example/a-b_c"
    assert _clean(None) is None

    # The whole point of stage 2's rewrite: a candidate with no face, and one
    # with a different face, must both be dropped -- the sunglasses-product
    # results that used to come back as confident "matches".
    global _face_distance
    real = _face_distance
    scores = {"same": 0.11, "other": 0.62, "faceless": None}
    _face_distance = lambda c, q: scores[c["link"]]
    try:
        kept, limit = _verify([{"link": k} for k in scores], [0.0] * 512)
        assert [m["link"] for _, m in kept] == ["same"], kept
        assert 0.11 < limit < 0.62, limit
    finally:
        _face_distance = real

    print("search.py self-check OK")


if __name__ == "__main__":
    _demo()
