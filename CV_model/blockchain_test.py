import hashlib
import json
import requests


# This represents an ACTUAL DaanDristi ACCEPT event.
record = {
    "pocket_id": "webcam-0",
    "timestamp": "2026-10-06T22:00:00+00:00",
    "item_status": "NOTE_DETECTED",
    "predicted_denomination": "500",
    "classifier_confidence": 0.96,
    "confidence": 0.96,
    "source": "webcam",
    "decision": "ACCEPT",
    "target_bin": "BIN_500",
    "model_version": "mobilenet_v3_clean_v1",
}


# Create a stable JSON representation.
canonical_data = json.dumps(
    record,
    sort_keys=True,
    separators=(",", ":")
)


# Generate SHA-256 fingerprint.
proof_hash = "0x" + hashlib.sha256(
    canonical_data.encode("utf-8")
).hexdigest()


print("=== DAANDRISTI BLOCKCHAIN TEST ===")
print()
print("Donation record:")
print(json.dumps(record, indent=2))
print()
print("SHA-256 proof:")
print(proof_hash)
print()


# Send proof to the Node.js blockchain API.
response = requests.post(
    "http://localhost:3000/register-proof",
    json={
        "proofHash": proof_hash,
        "sourceId": record["pocket_id"]
    },
    timeout=30
)


print("API response:")
print(response.status_code)
print(response.json())