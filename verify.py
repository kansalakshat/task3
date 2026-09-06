"""Stage 3: anchor the pipeline output on-chain, then re-verify it.

Hash chain
----------
  embedding_hash = keccak256(canonical JSON of the embedding vector)
  record_id      = keccak256(embedding_hash | searched_at)
  data_hash      = keccak256(embedding_hash | matched_url | searched_at)

Only `data_hash` is written on chain -- no biometric data, no URLs. Chain
writes are permanent and public, so anything more would be irreversible.

Re-verification recomputes both hashes from output/*.json and compares against
what the contract returns. Any edit to either file changes the digest.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from web3 import Web3

# The VERIFIED/MISMATCH line uses emoji; the default Windows console codepage
# (cp1252) cannot encode them. main.py picks this up by importing this module.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = "output"
DEPLOYMENT_FILE = "deployment.json"
CHAIN_RECORD = "chain_record.json"

# Hardhat's deterministic dev account #0. Public, worthless, well known -- it
# only ever signs against a local node. Sepolia always uses PRIVATE_KEY.
LOCAL_DEV_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
LOCAL_RPC = "http://127.0.0.1:8545"

# Live networks where a transaction costs real money. Ethereum, Optimism, BSC,
# Polygon, Base, Arbitrum One -- refused outright.
MAINNET_CHAIN_IDS = {1, 10, 56, 137, 8453, 42161}


def compute_hashes(out_dir=OUT_DIR):
    """Recompute record_id and data_hash from the saved stage 1 + 2 outputs."""
    with open(os.path.join(out_dir, "embedding.json")) as f:
        embedding = json.load(f)["embedding"]
    with open(os.path.join(out_dir, "match_result.json")) as f:
        match = json.load(f)

    # `or ""` because a visual match can come back with a null link, and
    # f-string would otherwise hash the literal text "None".
    matched_url = (match["matches"][0]["url"] or "") if match.get("matches") else ""
    searched_at = match["searched_at"]

    embedding_hash = Web3.keccak(
        text=json.dumps(embedding, separators=(",", ":"))
    ).hex()
    record_id = Web3.keccak(text=f"{embedding_hash}|{searched_at}")
    data_hash = Web3.keccak(text=f"{embedding_hash}|{matched_url}|{searched_at}")

    return record_id, data_hash, {
        "embedding_hash": embedding_hash,
        "matched_url": matched_url,
        "searched_at": searched_at,
    }


def _connect():
    """Return (w3, contract, account, deployment) from deployment.json + .env."""
    load_dotenv()
    if not os.path.exists(DEPLOYMENT_FILE):
        raise FileNotFoundError(
            f"{DEPLOYMENT_FILE} not found -- deploy the contract first:\n"
            "  local:   npx hardhat node   (in another terminal)\n"
            "           npx hardhat run scripts/deploy.js --network localhost\n"
            "  sepolia: npx hardhat run scripts/deploy.js --network sepolia"
        )
    with open(DEPLOYMENT_FILE) as f:
        deployment = json.load(f)

    is_local = deployment["network"] in ("localhost", "hardhat")
    rpc = LOCAL_RPC if is_local else os.getenv("SEPOLIA_RPC_URL")
    key = LOCAL_DEV_KEY if is_local else os.getenv("PRIVATE_KEY")
    if not rpc or not key:
        raise RuntimeError("SEPOLIA_RPC_URL and PRIVATE_KEY must be set in .env")

    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 60}))
    if not w3.is_connected():
        raise ConnectionError(
            f"cannot reach {rpc}"
            + (" -- is `npx hardhat node` running?" if is_local else "")
        )

    # This is a testnet demo and it signs with a key from .env. Refuse to touch
    # a real-money network even if someone points SEPOLIA_RPC_URL at one, and
    # refuse a chain that isn't the one the contract was deployed to -- writing
    # to the wrong chain would silently produce an unverifiable record.
    chain_id = w3.eth.chain_id
    if chain_id in MAINNET_CHAIN_IDS:
        raise RuntimeError(
            f"refusing to run against chain id {chain_id} (a live mainnet). "
            "This project is testnet-only and signs with a key from .env."
        )
    if chain_id != deployment["chainId"]:
        raise RuntimeError(
            f"chain mismatch: connected to chain id {chain_id} but "
            f"deployment.json was made on {deployment['chainId']}. Redeploy, "
            "or point the RPC at the right network."
        )

    account = w3.eth.account.from_key(key if key.startswith("0x") else "0x" + key)
    address = Web3.to_checksum_address(deployment["address"])

    # A restarted local node wipes state while deployment.json still points at
    # the old address. Without this check web3 fails deep in a contract call
    # with an opaque BadFunctionCallOutput instead of saying "redeploy".
    if len(w3.eth.get_code(address)) == 0:
        raise ConnectionError(
            f"no contract at {address} on '{deployment['network']}' -- the chain "
            f"was probably restarted. Redeploy:\n"
            f"  npx hardhat run scripts/deploy.js --network {deployment['network']}"
        )

    contract = w3.eth.contract(address=address, abi=deployment["abi"])
    return w3, contract, account, deployment


def _store(w3, contract, account, record_id, data_hash):
    """Send storeRecord and wait for the receipt. Returns the tx hash."""
    tx = contract.functions.storeRecord(record_id, data_hash).build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "chainId": w3.eth.chain_id,
    })
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
    if receipt.status != 1:
        raise RuntimeError(f"transaction reverted: 0x{tx_hash.hex()}")
    return "0x" + tx_hash.hex(), receipt.blockNumber


def _previous_tx(out_dir, record_id):
    """(tx_hash, block) from an earlier run of the same record, else (None, None)."""
    path = os.path.join(out_dir, CHAIN_RECORD)
    try:
        with open(path) as f:
            prev = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None
    if prev.get("record_id") != record_id:
        return None, None
    return prev.get("tx_hash"), prev.get("block")


def run(out_dir=OUT_DIR):
    """Upload the digest, then re-verify it from disk. Returns a result dict."""
    record_id, data_hash, parts = compute_hashes(out_dir)
    w3, contract, account, deployment = _connect()

    print(f"  network:   {deployment['network']} (chainId {deployment['chainId']})")
    print(f"  contract:  {deployment['address']}")
    print(f"  record id: 0x{record_id.hex()}")

    existing, _, _ = contract.functions.getRecord(record_id).call()
    if existing == b"\x00" * 32:
        tx_hash, block = _store(w3, contract, account, record_id, data_hash)
        print(f"  stored in block {block}, tx {tx_hash}")
    else:
        # A re-verification run sends no transaction. Carry the original tx
        # forward from the last record so the proof keeps its explorer link
        # instead of being overwritten with nulls.
        tx_hash, block = _previous_tx(out_dir, "0x" + record_id.hex())
        print("  record already on chain -- skipping write, verifying instead")

    # --- re-verification: fetch from chain, compare with hashes recomputed
    # from the files above. contract.verifyRecord does the same check on chain.
    on_chain, timestamp, submitter = contract.functions.getRecord(record_id).call()
    verified = contract.functions.verifyRecord(record_id, data_hash).call()

    result = {
        "verified": bool(verified),
        "network": deployment["network"],
        "chain_id": deployment["chainId"],
        "contract": deployment["address"],
        "record_id": "0x" + record_id.hex(),
        "local_hash": "0x" + data_hash.hex(),
        "on_chain_hash": "0x" + on_chain.hex(),
        "on_chain_timestamp": timestamp,
        "submitter": submitter,
        "tx_hash": tx_hash,
        "block": block,
        "hash_inputs": parts,
    }
    with open(os.path.join(out_dir, CHAIN_RECORD), "w") as f:
        json.dump(result, f, indent=2)

    print(f"  local hash:    0x{data_hash.hex()}")
    print(f"  on-chain hash: 0x{on_chain.hex()}")
    print("\n  VERIFIED ✅" if verified else "\n  MISMATCH ❌")
    return result


def _demo():
    """Self-check: the hash chain is deterministic and edit-sensitive."""
    import tempfile

    emb = {"embedding": [0.1, -0.2, 0.3]}
    match = {"searched_at": "2026-01-01T00:00:00+00:00",
             "matches": [{"url": "https://example.com/post/1"}]}
    with tempfile.TemporaryDirectory() as d:
        def write(m):
            with open(os.path.join(d, "embedding.json"), "w") as f:
                json.dump(emb, f)
            with open(os.path.join(d, "match_result.json"), "w") as f:
                json.dump(m, f)

        write(match)
        a = compute_hashes(d)
        write(match)
        assert compute_hashes(d) == a, "hashes must be deterministic"

        # Tampering with the matched URL must change data_hash but not record_id.
        tampered = json.loads(json.dumps(match))
        tampered["matches"][0]["url"] = "https://evil.example/other"
        write(tampered)
        b = compute_hashes(d)
        assert b[0] == a[0], "record_id must not depend on the URL"
        assert b[1] != a[1], "data_hash must change when the URL changes"

        # No match at all is still a valid, distinct anchor -- not a crash.
        write({"searched_at": match["searched_at"], "matches": []})
        c = compute_hashes(d)
        assert c[1] != a[1] and c[0] == a[0]
    print("verify.py self-check OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Anchor and re-verify pipeline output on-chain.")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--self-check", action="store_true", help="run offline hash tests only")
    args = p.parse_args()
    if args.self_check:
        _demo()
    else:
        # Setup problems (no deployment, chain down, stale address) are user
        # errors, not crashes -- print the guidance, not a traceback.
        try:
            raise SystemExit(0 if run(args.out_dir)["verified"] else 1)
        except (FileNotFoundError, ConnectionError, RuntimeError) as exc:
            raise SystemExit(f"  ! {exc}")
