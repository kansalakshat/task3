#!/usr/bin/env python3
"""Face-verification pipeline: detect -> reverse image search -> on-chain anchor.

===========================  SCOPE: CONSENT ONLY  ===========================
This tool is for self-verification and consenting-subject use ONLY -- checking
whether *your own* photos appear elsewhere online, or a demo run against a
teammate who has explicitly agreed to be searched.

It is NOT a stranger-identification tool. Do not point it at a face you do not
have permission to search. Stage 2 uploads the cropped face to a public
temporary image host in order to run the reverse image search, so running it on
someone else's face publishes their face without their consent -- on top of
being what this project exists not to do.
============================================================================

Usage:  python main.py path/to/photo.jpg
"""

import argparse
import os
import sys

from dotenv import load_dotenv

import verify
from pipeline import detect, search

OUT_DIR = "output"


def _banner(n, title):
    print(f"\n{'=' * 62}\n  STAGE {n}: {title}\n{'=' * 62}")


def main():
    p = argparse.ArgumentParser(
        description="Face verification pipeline (consent-only -- see README): "
                    "detect a face, reverse image search it, anchor the result on-chain."
    )
    p.add_argument("image", help="path to an input photo (yours, or a consenting subject's)")
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()

    load_dotenv()
    if not os.path.exists(args.image):
        sys.exit(f"error: no such file: {args.image}")

    print("\n  Consent-only tool. Run this on your own face, or on a face whose")
    print("  owner has explicitly agreed to be searched. See README.\n")

    _banner(1, "FACE DETECTION + ALIGNMENT + EMBEDDING (MediaPipe -> Facenet512)")
    try:
        face = detect.run(args.image, args.out_dir)
    except ValueError as e:
        sys.exit(f"  ! {e}")
    box = face["bounding_box"]
    print(f"  face at x={box['x']} y={box['y']} w={box['w']} h={box['h']}")
    print(f"  crop      -> {face['face_crop']}")
    print(f"  embedding -> {args.out_dir}/embedding.json "
          f"({face['dimensions']}-d {face['model']}, eye-aligned)")

    _banner(2, "REVERSE IMAGE SEARCH + FACE VERIFICATION (SerpAPI / Google Lens)")
    match = search.run(face["face_crop"], args.out_dir, embedding=face["embedding"])
    if match["matched"]:
        print(f"  {match['total_returned']} candidate(s) from the engine; "
              f"{len(match['matches'])} confirmed as this face:")
        for m in match["matches"]:
            print(f"    - [d={m['distance']:.3f}] {m['source'] or '?'}: {m['url']}")
    else:
        print(f"  no confident match recorded ({match['note']})")
    print(f"  results   -> {args.out_dir}/match_result.json")

    _banner(3, "BLOCKCHAIN VERIFICATION (Solidity / Hardhat / web3.py)")
    try:
        result = verify.run(args.out_dir)
    except (FileNotFoundError, ConnectionError, RuntimeError) as e:
        print(f"  ! stage 3 skipped: {e}")
        sys.exit(2)

    print(f"  record    -> {args.out_dir}/chain_record.json")
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
