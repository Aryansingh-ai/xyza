# DivyaKosh Blockchain Architecture & Cryptographic Proof Specification

This document provides a complete technical trace of the **DivyaKosh blockchain proof registration and verification pipeline**, connecting the Python machine learning layer with the Node.js API, the AI Proof SDK, and the Monad smart contract.

---

## 1. Cryptographic Proof Architecture Overview

The purpose of the blockchain integration is to provide **tamper-evident provenance** for real-world donations. If an adversary accesses the local SQLite database or JSONL files and alters a ₹500 donation to ₹10, the cryptographic hash on the Monad blockchain will no longer match, immediately exposing fraud.

```
Python CV Layer (live_camera.py)
   │ (Only on ACCEPT)
   ▼
CV_model/src/blockchain.py
   │ Computes legacy SHA-256 hash & retrieves donationId
   │ HTTP POST { proofHash, sourceId, donationId }
   ▼
Blockchain/api/index.js (Express Port 3000)
   │ 1. Queries CV_model/sqlite/donation_box.db using donationId
   │ 2. Hashes model binary: SHA-256(mobilenet_v3_clean.pt) -> modelHash
   │ 3. Assembles Canonical AI Event (CORE_FIELDS)
   │ 4. Computes Canonical Proof Hash: SHA-256(canonicalize(event))
   │ 5. Invokes Smart Contract: contract.registerProof(registerHash, sourceId)
   │ 6. Awaits on-chain receipt (block number, gas, status)
   │ 7. Stores original snapshot in Blockchain/api/data/original_records.json
   ▼
Monad Testnet (Chain ID 10143)
   │ Contract: DonationProvenanceRegistry.sol (0xBCD966873f48563643Bde79760eEC98a567fd7e6)
   │ Emits: ProofRegistered(bytes32 indexed proofHash, string sourceId, uint256 timestamp, address indexed registeredBy)
```

---

## 2. Triggering Condition & Input Data

* **Triggering Event:** Proof generation is triggered **strictly on `ACCEPT` decisions**. If the outcome is `REVIEW` or `REJECT`, `register_blockchain_proof()` immediately skips processing (`CV_model/src/blockchain.py:65-67`).
* **Input Record Shape (from `live_camera.py`):**
  ```json
  {
    "pocket_id": "webcam-0",
    "timestamp": "2026-11-02T12:45:00.123456+00:00",
    "item_status": "NOTE_DETECTED",
    "predicted_denomination": "500",
    "classifier_confidence": 0.9994,
    "confidence": 0.9994,
    "source": "webcam",
    "decision": "ACCEPT",
    "target_bin": "BIN_500",
    "model_version": "mobilenet_v3_clean_v1"
  }
  ```

---

## 3. Two Hashing Schemes: Legacy vs. Canonical-v1

The codebase supports two distinct cryptographic proof schemes to ensure backward compatibility with historical records:

### Scheme 1: Legacy Raw-Record Hash (`PROOF_SCHEME_LEGACY`)
Implemented in `CV_model/src/blockchain.py:17-38` and `Blockchain/sdk/proof.js:263-286`:
1. Sorts all top-level keys alphabetically.
2. Compacts JSON without whitespace (`separators=(',', ':')`).
3. Computes SHA-256:
   $$\text{legacy\_hash} = \text{0x} + \text{SHA256}(\text{JSON\_compact}(\text{record}))$$

### Scheme 2: Canonical AI Event Proof (`PROOF_SCHEME_CANONICAL` — `canonical-v1`)
Implemented in `Blockchain/sdk/proof.js:457-694`. This is the **primary production scheme** used when the donation row exists in SQLite:

#### A. Model Binary Hashing (`getModelHash()`, `proof.js:143-218`)
* File: `CV_model/models/mobilenet_v3_clean.pt` (17,043,769 bytes).
* Reads the raw `.pt` weights in 1 MB chunks and computes a SHA-256 digest (`modelHash`).
* Caches the hash based on file size and `mtimeMs` so disk I/O is not repeated on every transaction.
* **Security guarantee:** If an attacker swaps out the model weights with an unauthorized model, the `modelHash` changes, invalidating all proofs.

#### B. Canonical AI Event Structure (`createCanonicalAIEvent()`, `proof.js:457-584`)
Assembles a frozen data structure containing mandatory `CORE_FIELDS`:
```json
{
  "schemaVersion": "1.0",
  "eventType": "AI_DONATION_VERIFICATION",
  "eventId": "47ab44db18afab48a12e2cc40a2dd260d55b57a3ad62933f5bab0108134bd014",
  "issuer": "DAANDRISHTI",
  "deviceId": "DAANDRISHTI-PI-01",
  "modelId": "mobilenet_v3_large",
  "modelVersion": "mobilenet_v3_clean_v1",
  "modelHash": "0x5a1b...e9c2",
  "prediction": "500",
  "confidence": 0.9994,
  "decision": "ACCEPT",
  "timestamp": "2026-11-02T12:45:00.123456+00:00",
  "source": "webcam",
  "sourceId": "webcam-0",
  "targetBin": "BIN_500",
  "itemStatus": "NOTE_DETECTED",
  "recordHash": "0x64ed0b6dbbb4d7d5a11c76e16d3c28cb2eb47a28dda8335a2689847a81d14741"
}
```

#### C. Deterministic Canonical Serialization (`canonicalize()`, `proof.js:597-674`)
* Recursively sorts all dictionary keys.
* Serializes null, booleans, strings, and finite numbers.
* Strictly rejects `undefined`, `NaN`, and `Infinity`.
* Generates an unspaced UTF-8 JSON string.

#### D. Event Proof Hash (`hashAIEvent()`, `proof.js:680-694`)
$$\text{proofHash} = \text{0x} + \text{SHA256}(\text{canonicalize}(\text{event}))$$
This 32-byte hex string (`bytes32`) is the **only data stored on the blockchain**. The image and model weights remain off-chain.

---

## 4. HTTP Protocol: Python Client to Node.js API

### Endpoint: `POST /register-proof`
* **Server:** `Blockchain/api/index.js` (Express.js on port `3000`).
* **Client:** `CV_model/src/blockchain.py:57-120` using Python `requests.post()`.
* **Timeout:** 30 seconds (`timeout=30`).

#### Request Payload:
```json
{
  "proofHash": "0x64ed0b6dbbb4d7d5a11c76e16d3c28cb2eb47a28dda8335a2689847a81d14741",
  "sourceId": "webcam-0",
  "donationId": "47ab44db18afab48a12e2cc40a2dd260d55b57a3ad62933f5bab0108134bd014"
}
```

#### Success Response (`200 OK`):
```json
{
  "success": true,
  "transactionHash": "0x8f2a41bc90123...4567",
  "blockNumber": 1284950,
  "explorerUrl": "https://testnet.monadexplorer.com/tx/0x8f2a41bc90123...4567",
  "proofHash": "0x3b89f1a23c4d5e...canonical_hash",
  "proofScheme": "canonical-v1",
  "legacyProofHash": "0x64ed0b6dbbb4d7d5a11c76e16d3c28cb2eb47a28dda8335a2689847a81d14741",
  "eventId": "47ab44db18afab48a12e2cc40a2dd260d55b57a3ad62933f5bab0108134bd014",
  "sourceId": "webcam-0"
}
```

#### Error Responses:
* `400 Bad Request`: `sourceId and either proofHash or donationId are required`.
* `404 Not Found`: `donationId not found in SQLite database`.
* `500 Internal Server Error`: `contract.registerProof` error (e.g., `"Error: Proof already exists onchain"` or RPC error).

---

## 5. Smart Contract & Monad On-Chain Layer

### Network & Contract Specifications
* **Blockchain Network:** Monad Testnet (EVM-compatible, High-Throughput PoS).
* **Chain ID:** `10143` (hex: `0x279f`).
* **RPC Endpoint:** `https://testnet-rpc.monad.xyz` (configured in `Blockchain/api/.env`).
* **Block Explorer:** `https://testnet.monadexplorer.com`.
* **Contract Name:** `DonationProvenanceRegistry`.
* **Contract Address:** `0xBCD966873f48563643Bde79760eEC98a567fd7e6` (verified in `Blockchain/contracts/ignition/deployments/chain-10143/deployed_addresses.json`).

### Contract Interface (`DonationProvenanceRegistry.sol`)
```solidity
struct Proof {
    bytes32 proofHash;
    string sourceId;
    uint256 timestamp;
    address registeredBy;
}

mapping(bytes32 => Proof) public proofs;
mapping(address => bool) public authorizedIssuers;
address public owner;

event ProofRegistered(
    bytes32 indexed proofHash,
    string sourceId,
    uint256 timestamp,
    address indexed registeredBy
);

function registerProof(bytes32 _proofHash, string calldata _sourceId) external onlyAuthorized;
function verifyProof(bytes32 _proofHash) external view returns (bool exists, string memory sourceId, uint256 timestamp, address registeredBy);
```

### Signer & Wallet Requirements
* The Node.js API requires `PRIVATE_KEY` in `Blockchain/api/.env`.
* The wallet address must:
  1. Hold testnet `MON` tokens to pay for transaction gas.
  2. Be registered as an `authorizedIssuer` or be the contract `owner`.

---

## 6. Verification & Tamper Detection Flow

### Verification Endpoint: `GET /verify-proof/:hash`
Implemented in `Blockchain/api/index.js:1328-1430`:
1. Queries the Monad smart contract: `const result = await contract.verifyProof(hash)`.
2. If `result[0] === true`:
   Returns `{ exists: true, sourceId: result[1], timestamp: result[2], registeredBy: result[3] }`.
3. If `result[0] === false`:
   Returns `{ exists: false, error: "Proof hash not found on Monad blockchain" }`.

### Audit Endpoint: `GET /audit`
Implemented in `Blockchain/api/index.js:1696-2020`:
Compares the state across three layers:
1. **SQLite Database:** `donations` table in `donation_box.db`.
2. **Original Registry:** `Blockchain/api/data/original_records.json` (frozen at the moment of registration).
3. **Monad Blockchain:** Reads on-chain `ProofRegistered` events via RPC or Envio indexer.

#### Tamper Flags:
* `TAMPERED`: SQLite record exists, but its computed hash does not match `original_records.json` or the on-chain hash (e.g. denomination was manually altered from ₹500 to ₹10).
* `UNANCHORED`: Donation exists in local SQLite, but has not yet been registered on-chain.
* `VERIFIED`: SQLite record, original snapshot, and on-chain hash match with 100% cryptographic parity.

---

## 7. Operational Status Matrix

To avoid ambiguity, teammates must distinguish these 7 lifecycle statuses:

| Status | Code Boundary | Verification Method | Meaning |
| :--- | :--- | :--- | :--- |
| **1. ML Decision Accepted** | `validator.py:656` | `classification.decision == "ACCEPT"` | MobileNet classified image crop with confidence >= 0.80. No persistence has occurred yet. |
| **2. Local Persisted** | `database.py:60` | Row present in `donations` table | Donation is safely stored in local SQLite database (`donation_box.db`) and JSONL log. |
| **3. Proof / Hash Generated** | `proof.js:680` | Valid 32-byte hex `proofHash` computed | Canonical AI event created with model hash; serialized and digested into SHA-256. |
| **4. API Request Accepted** | `api/index.js:1055` | HTTP `200 OK` response received | Express server validated inputs and located SQLite record. |
| **5. Transaction Submitted** | `api/index.js:1210` | Nonce assigned, tx broadcast | `contract.registerProof()` sent to Monad mempool (`tx.hash` generated). |
| **6. Transaction Confirmed** | `api/index.js:1227` | `receipt.blockNumber > 0` | Transaction mined in a Monad testnet block; proof permanently recorded in state. |
| **7. Proof Independently Verified** | `api/index.js:1328` | `contract.verifyProof(hash).exists == true` | Any third party can independently read the Monad contract and verify the proof matches the event. |

> [!CAUTION]
> **Persistence Linking Gap:**
> When `Blockchain/api` commits a transaction, it saves `tx.hash` and `blockNumber` to `Blockchain/api/data/original_records.json`, but does **not** update the original row in `CV_model/sqlite/donation_box.db` with the transaction hash. Linking is achieved via `donation_id` (`eventId`).

