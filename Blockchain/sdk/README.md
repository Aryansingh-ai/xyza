# AI Proof SDK (`dpp-monad/sdk/proof.js`)

Generic, chain-agnostic proof layer used by DAANDRISHTI. Any AI pipeline can use it to produce a deterministic, tamper-evident commitment to an AI inference event.

---

## What it does

Given **any AI inference result**, the SDK:

1. Builds a structured **canonical-v1 event** containing the inference metadata and model provenance.
2. Serializes it with **recursive key-sorting** (`canonicalize()`), so the same event always maps to the exact same byte string regardless of JS object key insertion order.
3. Hashes that string with **SHA-256** → `proofHash`.
4. `proofHash` is the single `bytes32` value registered on-chain via `registerProof()`.

The model binary and the event body stay **off-chain**. Only the hash goes on-chain.

---

## canonical-v1

The canonical-v1 wire format is **frozen**. Any existing on-chain hash computed by this scheme remains valid forever, even as the SDK gains new features.

**Key rules:**

- Object keys are sorted recursively before serialization.
- No whitespace, no trailing commas.
- `null` values are preserved; `undefined` and `NaN` throw.
- Strings are JSON-escaped; numbers use `JSON.stringify`.
- Arrays preserve element order.

**Example serialization:**

```js
// { b: 2, a: 1 }  →  {"a":1,"b":2}
```

---

## CORE_FIELDS

Every canonical-v1 event **must** contain these fields (non-empty strings unless noted):

| Field           | Type     | Notes                                  |
|----------------|----------|----------------------------------------|
| `schemaVersion` | string   | `"1.0"`                                |
| `eventType`     | string   | e.g. `"AI_FRAUD_DETECTION"`            |
| `eventId`       | string   | Stable unique ID for this inference    |
| `issuer`        | string   | Organization that owns the system      |
| `deviceId`      | string   | Physical or logical device identifier  |
| `modelId`       | string   | Model architecture name                |
| `modelHash`     | string   | `0x`-prefixed SHA-256 of model binary  |
| `prediction`    | string   | Raw model output label                 |
| `confidence`    | number   | Finite float in [0, 1]                 |
| `decision`      | string   | Business decision, e.g. `"ACCEPT"`     |
| `timestamp`     | string   | ISO-8601 UTC timestamp of the event    |

Extra domain-specific keys (`targetBin`, `transactionId`, `recordHash`, etc.) are **allowed** and included verbatim in the hash.

---

## API

### Generic primitives (application-agnostic)

```js
const {
  CORE_FIELDS,
  createAIEvent,
  verifyEventHash,
  hashAIEvent,
  canonicalizeEvent,
} = require("./proof");
```

#### `createAIEvent(fields)`

Validates all `CORE_FIELDS` and returns a plain event object ready for hashing. Throws if any required field is missing, blank, or has the wrong type.

```js
const event = createAIEvent({
  schemaVersion: "1.0",
  eventType:     "AI_FRAUD_DETECTION",
  eventId:       "fraud-txn-abc123",
  issuer:        "ACME-FINTECH",
  deviceId:      "FRAUD-ENGINE-01",
  modelId:       "xgboost_fraud_v2",
  modelHash:     "0xabcdef...",
  prediction:    "FRAUDULENT",
  confidence:    0.987,
  decision:      "BLOCK",
  timestamp:     "2026-10-08T10:00:00.000000+00:00",
  // extra domain fields:
  transactionId: "txn-abc123",
});
```

#### `hashAIEvent(event)` → `string`

Returns the `0x`-prefixed SHA-256 hash of the canonical serialization of `event`. This is the value to register on-chain.

#### `verifyEventHash(event, expectedHash)` → `{ ok: boolean, computedHash: string }`

Pure, non-throwing comparison. Returns `ok: true` if `hashAIEvent(event)` matches `expectedHash` (case-insensitive hex). Use this to check whether a local event matches the hash stored on-chain.

```js
const { ok, computedHash } = verifyEventHash(event, "0x7559aa...");
if (!ok) console.error("TAMPERED – hash mismatch");
```

---

### DAANDRISHTI-specific

```js
const {
  createCanonicalAIEvent,
  getCurrentProvenance,
  getModelHash,
  donationToLegacyRecord,
  hashLegacyRecord,
} = require("./proof");
```

`createCanonicalAIEvent(donationRow, provenance)` builds the canonical event from a `donation_box.db` SQLite row and the live provenance config. It delegates to `createAIEvent` internally so CORE_FIELDS are validated, then adds DAANDRISHTI-specific extras (`sourceId`, `targetBin`, `itemStatus`, `recordHash`).

---

## Hashing pipeline

```
SQLite row (or any inference result)
    ↓  createAIEvent() / createCanonicalAIEvent()
Canonical event object
    ↓  canonicalize()  (recursive key-sort, compact JSON)
Deterministic UTF-8 string
    ↓  SHA-256
proofHash  (0x-prefixed hex)
    ↓  registerProof(proofHash, sourceId)
Monad Testnet
```

---

## Tamper detection

If **any field** in the local event is modified after registration, `hashAIEvent(event)` will produce a different hash. `verifyEventHash(event, onChainHash)` will return `ok: false`. The on-chain record remains unchanged, providing proof of the original values.

---

## DAANDRISHTI reference implementation

DAANDRISHTI (`DaanDristi` donation box, Monad Hackathon) is the **live production user** of this SDK.

| Constant          | Value                          |
|------------------|-------------------------------|
| `issuer`          | `DAANDRISHTI`                  |
| `deviceId`        | `DAANDRISHTI-PI-01`            |
| `modelId`         | `mobilenet_v3_large`           |
| `eventType`       | `AI_DONATION_VERIFICATION`     |
| `proofScheme`     | `canonical-v1`                 |
| Model file        | `Monad_hackathon/models/mobilenet_v3_clean.pt` |
| Contract          | `0xBCD966873f48563643Bde79760eEC98a567fd7e6` (Monad Testnet) |

The real test event registered in Phase 2 of the E2E test:

- **donation_id / eventId:** `1ac8337803679a9ec5f754c26eac342462793fa147d2103b2737aafe0503d1cc`
- **transactionHash:** `0xa8d56d5f583d28125994654c73688c87dcf6fd90de99544be82c8c1d38998841`
- **blockNumber:** `69229077`
- **proofHash (on-chain):** `0x7559aa4c2d541af01708bb7a49b466d9ac94d5c53a71b7c8c3b42a910c5f3696`

---

## Limitations

- `createAIEvent` validates types but does **not** enforce semantic constraints (e.g. `confidence ∈ [0,1]`). Callers are responsible.
- `modelHash` is computed from a real file via `getModelHash(path)`. If the model file is absent, registration will throw — by design.
- The SDK has no network dependency. Blockchain registration is handled entirely by `api/index.js`.
- Legacy proofs (scheme `legacy-raw-record`) are verified separately via `hashLegacyRecord` and kept for backward compatibility only.
- The canonical-v1 format is frozen. Adding new fields to future event types must use a new schema version.
