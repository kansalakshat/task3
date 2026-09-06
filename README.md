# Face Verification Pipeline

**Hacker House Goa — Task 3**

An end-to-end demo of three stages wired together: detect a face in a photo,
search the web for where that face already appears, and anchor a tamper-evident
hash of the result on a blockchain so the finding can be re-verified later.

---

## ⚠️ Scope: consent-based use only

**This is a self-verification tool. It is not a stranger-identification tool.**

Legitimate uses:

- Checking whether **your own** photos have been scraped or reposted elsewhere.
- A demo run against a **teammate who has explicitly agreed** to be searched.

Do not run it on a face you do not have permission to search. Beyond the obvious
ethical problem, there is a concrete mechanical one: reverse image search APIs
accept a public image *URL*, not a file upload, so **Stage 2 uploads the cropped
face to a public file host** (`catbox.moe`) in order to run the query.

**That upload is permanent.** catbox stores anonymous uploads indefinitely and
offers no anonymous delete. Every host with a real expiry was dead when this was
built (litterbox 403s, 0x0.st has disabled uploads, uguu returns a
non-resolving CDN host, x0.at's TLS certificate is broken), so there is no
expiring option to fall back to. Treat every face you put through Stage 2 as
published to the open internet forever. That is the single strongest reason this
tool is consent-only.

This project makes **no production-grade identity-verification claim**. It
demonstrates a pipeline; it does not establish who anyone is.

---

## Architecture

```
  input photo
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1  Face detection + encoding                              │
│   mediapipe FaceDetector (BlazeFace)  → best-scoring face box   │
│   + 25% padding, clamped to frame     → output/face_crop.jpg    │
│   DeepFace.represent(Facenet)         → output/embedding.json   │
└─────────────────────────────────────────────────────────────────┘
      │  face_crop.jpg
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2  Reverse image search                                   │
│   upload crop → public URL (permanent, see scope)               │
│   SerpAPI  engine=google_lens  → visual_matches                 │
│   top 5 {url, title, source} + search timestamp                 │
│                                       → output/match_result.json│
│   No match / no key / API down → recorded honestly, never faked │
└─────────────────────────────────────────────────────────────────┘
      │  matched URL + timestamp
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3  Blockchain verification                                │
│   embedding_hash = keccak256(canonical embedding JSON)          │
│   record_id      = keccak256(embedding_hash | searched_at)      │
│   data_hash      = keccak256(embedding_hash | url | searched_at)│
│                                                                 │
│   Verification.sol  storeRecord(record_id, data_hash)  [write-once]
│   then re-read on chain + recompute from disk                   │
│                          → VERIFIED ✅ / MISMATCH ❌            │
│                          → output/chain_record.json             │
└─────────────────────────────────────────────────────────────────┘
```

Only the digest goes on chain. **No embedding, no image, no URL is written to
the blockchain** — chain writes are permanent and public, and biometric data
does not belong there.

---

## Repo layout

```
pipeline/detect.py       Stage 1 — MediaPipe crop + Facenet embedding
pipeline/search.py       Stage 2 — SerpAPI reverse image search
contracts/Verification.sol   Stage 3 — hash anchor contract
scripts/deploy.js        Hardhat deploy → deployment.json (address + ABI)
verify.py                Stage 3 — web3.py upload + re-verification
main.py                  CLI orchestrator for all three stages
output/                  Generated artifacts (gitignored)
```

---

## Setup

```bash
pip install -r requirements.txt     # mediapipe, deepface, web3, ...
npm install                         # hardhat, ethers
cp .env.example .env                # then fill in
```

First run downloads two model files: the BlazeFace detector (~230 KB, into
`models/`) and the Facenet weights (~90 MB, into `~/.deepface/weights/`).

### Troubleshooting

- **`ValueError: ... requires tf-keras package`** — DeepFace needs `tf-keras`
  alongside TensorFlow ≥ 2.16. It is in `requirements.txt`; if you installed
  piecemeal, run `pip install tf-keras`.
- **`An exception occurred while downloading facenet_weights.h5`** — DeepFace's
  downloader is flaky against GitHub releases. Fetch it manually:
  ```bash
  curl -L -o ~/.deepface/weights/facenet_weights.h5 \
    https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5
  ```
- **`HH502: Couldn't download compiler version list`** — transient; re-run
  `npx hardhat compile`.

### Keys

Everything is read from `.env` — nothing is hardcoded. See `.env.example`.

- `SERPAPI_KEY` — free tier is 100 searches/month.
- `SEPOLIA_RPC_URL` + `PRIVATE_KEY` — **testnet throwaway wallet only.** Never
  put a real or funded key in this file.

Both are optional. Without `SERPAPI_KEY`, Stage 2 records "no match" and the
pipeline continues. Without the Sepolia pair, Stage 3 runs against a local
Hardhat node.

---

## Running

### Option A — local Hardhat chain (no test ETH needed)

```bash
npx hardhat node                                          # terminal 1
npx hardhat run scripts/deploy.js --network localhost     # terminal 2
python main.py photo.jpg                                  # terminal 2
```

`verify.py` signs local transactions with Hardhat's deterministic dev account
#0 — a publicly-known, worthless key that is only ever used against
`127.0.0.1:8545`.

### Option B — Sepolia testnet

```bash
# .env has SEPOLIA_RPC_URL and PRIVATE_KEY, wallet funded from a faucet
npx hardhat run scripts/deploy.js --network sepolia
python main.py photo.jpg
```

### Stages independently

```bash
python -m pipeline.detect          # self-check (error paths)
python -m pipeline.search          # self-check (graceful-failure paths)
python verify.py --self-check      # self-check (hash chain, offline)
python verify.py                   # re-verify existing output/ against chain
```

Re-running `verify.py` on unmodified output prints `VERIFIED ✅`. Edit a single
character in `output/match_result.json` and it prints `MISMATCH ❌` — that is
the whole point of the stage.

---

## Choices

**MediaPipe: Tasks API, not `solutions`.** The brief specified
`mediapipe.solutions.face_detection`. Google dropped the legacy Solutions API
from the 0.10.30+ wheels, and 0.10.30 is the oldest release pip will install on
Python 3.13 — so on 3.13 no available version has it. Verified by unzipping the
0.10.30 wheel and finding zero `solutions` entries, not assumed. Stage 1 uses
`mediapipe.tasks.python.vision.FaceDetector`, which wraps the **same BlazeFace
short-range model**; the only practical difference is that it returns a pixel
bounding box instead of a relative one, and the ~230 KB `.tflite` weights are
downloaded to `models/` on first run. On Python ≤3.12 the old
`solutions.face_detection` call would be a drop-in swap inside `_detect_box`.

**Why SerpAPI over Bing Visual Search.** Microsoft retired the Bing Search APIs
(including Visual Search) in August 2025, so it is no longer an option for new
projects. SerpAPI is also simpler to wire up: one authenticated `GET`, JSON out,
no Azure resource provisioning.

Within SerpAPI, the pipeline uses `engine=google_lens` rather than the older
`google_reverse_image`. Google folded classic reverse image search into Lens, and
`google_lens` is the endpoint that actually returns a structured
`visual_matches` array.

**Why Sepolia.** It is the proof-of-stake testnet Ethereum client teams actively
maintain for application testing, faucets are easy to reach, and Etherscan
indexes it so a judge can click through to the transaction. Goerli is
deprecated; mainnet costs real money for a demo that stores a 32-byte hash. The
local Hardhat fallback exists so the demo never depends on a faucet working
during a hackathon.

**Why write-once records.** `storeRecord` reverts if a record ID already exists,
so a later run cannot silently overwrite an earlier proof. Because `record_id`
includes the search timestamp, each run gets its own record and re-running the
pipeline is not blocked.

---

## Known limitations

**Reverse image search**
- SerpAPI's free tier is 100 searches/month; the pipeline burns one per run.
- Google Lens returns visually similar images with **no confidence score**. The
  pipeline reports what the engine returned; it does not rank by identity
  likelihood, and a "match" may be a lookalike, a stock photo, or unrelated.
- Cropped faces search far worse than full images — the crop discards the
  background context reverse image search leans on. Expect false negatives.
- The crop is uploaded to `catbox.moe` because the API needs a fetchable URL,
  and that upload is **permanent and undeletable** (see the scope section).
  There is a single upload host and no fallback: free file hosts are unreliable
  in practice — x0.at worked in the morning and had a broken TLS certificate by
  the afternoon. If the host is down, Stage 2 records an honest failure and the
  pipeline continues rather than inventing a match.

**Face matching**
- Facenet embeddings are sensitive to pose, lighting, age, and occlusion. This
  demo produces an embedding; it does **not** implement a verified matching
  threshold, and no distance comparison gates the result.
- MediaPipe takes only the highest-confidence face. Group photos are not handled.
- Nothing here is liveness-checked. A photo of a photo passes Stage 1.

**Blockchain**
- The anchor proves *"this exact output existed and was submitted by this
  address at this block"*. It proves nothing about whether the match is
  **correct** — garbage in, notarised garbage out.
- Testnets are best-effort: Sepolia RPC endpoints rate-limit, faucets run dry,
  and testnet state carries no guarantee of permanence.
- The contract has no access control. Anyone can write any record ID, and
  because records are write-once, anyone who learns your `record_id` before you
  submit can permanently squat it and block your anchor. Not practically
  exploitable (the ID derives from your embedding hash, which you don't
  publish), but it is why this is a demo and not an identity system.
- On-chain `timestamp` is the miner's block time, not the search time.

**General**
- No production-grade identity verification is claimed or implied. Do not use
  this to make decisions about a person.

---

## License

MIT. Demo project — built for a hackathon, not for production.
