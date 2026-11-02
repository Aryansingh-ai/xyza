/**
 * fraud-event.js
 *
 * Demonstrates the GENERIC SDK layer using a non-donation AI
 * event: a financial-fraud detection pipeline.
 *
 * Does NOT register anything on Monad. The purpose is to prove
 * that createAIEvent / verifyEventHash are reusable outside the
 * DAANDRISHTI donation context.
 *
 * Run:
 *   node dpp-monad/sdk/examples/fraud-event.js
 * (from the repo root, or adjust relative paths if needed)
 */

"use strict";

const path = require("path");
const {
  CORE_FIELDS,
  createAIEvent,
  verifyEventHash,
  hashAIEvent,
  getModelHash,
  PROOF_SCHEME_CANONICAL,
} = require("../proof");

// ── 1. Config ────────────────────────────────────────────────

// Use the real DAANDRISHTI model file as the stand-in model hash.
// A real fraud-detection deployment would point at its own model.
const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");
const MODEL_PATH =
  [
    path.join(REPO_ROOT, "CV_model", "models", "mobilenet_v3_clean.pt"),
    path.join(REPO_ROOT, "CV-model", "models", "mobilenet_v3_clean.pt"),
    path.join(REPO_ROOT, "Monad_hackathon", "models", "mobilenet_v3_clean.pt"),
  ].find((p) => fs.existsSync(p)) ||
  path.join(REPO_ROOT, "CV_model", "models", "mobilenet_v3_clean.pt");

const MODEL_HASH = getModelHash(MODEL_PATH);

// ── 2. Build a fraud-detection AI event ──────────────────────

const fraudEvent = createAIEvent({
  schemaVersion:  "1.0",
  eventType:      "AI_FRAUD_DETECTION",
  eventId:        "fraud-txn-9f7e4a23b1c2d5e6f7a8b9c0d1e2f3a4",
  issuer:         "ACME-FINTECH",
  deviceId:       "FRAUD-ENGINE-NODE-01",
  modelId:        "xgboost_fraud_v2",
  modelVersion:   "xgboost_fraud_v2.3.1",
  modelHash:      MODEL_HASH,
  prediction:     "FRAUDULENT",
  confidence:     0.9871234567890123,
  decision:       "BLOCK",
  timestamp:      "2026-10-08T10:00:00.000000+00:00",
  // Domain-specific extras (not in CORE_FIELDS — allowed):
  transactionId:  "txn-abc123",
  riskScore:      0.9871234567890123,
  triggeredRules: ["velocity_check", "geo_mismatch"],
});

// ── 3. Hash it ───────────────────────────────────────────────

const proofHash = hashAIEvent(fraudEvent);

// ── 4. Simulate on-chain storage: remember this hash ─────────

const onChainHash = proofHash; // would be sent to registerProof()

// ── 5. Run tests ─────────────────────────────────────────────

let passed = 0;
let failed = 0;

function assert(label, condition) {
  if (condition) {
    console.log(`  PASS  ${label}`);
    passed++;
  } else {
    console.error(`  FAIL  ${label}`);
    failed++;
  }
}

console.log("\n=== Fraud-Detection AI Proof SDK Demo ===\n");

// 5a. CORE_FIELDS completeness
console.log("--- CORE_FIELDS validation ---");
for (const f of CORE_FIELDS) {
  assert(`CORE_FIELD '${f}' present in event`, f in fraudEvent);
}

// 5b. Proof scheme label is exported correctly
assert(
  `proofScheme constant = '${PROOF_SCHEME_CANONICAL}'`,
  PROOF_SCHEME_CANONICAL === "canonical-v1"
);

// 5c. Deterministic hashing — same object, same hash every time
const hash2 = hashAIEvent(fraudEvent);
assert(
  "Deterministic: hash(event) === hash(event) called twice",
  proofHash === hash2
);

// 5d. Key-order independence — a shuffled copy hashes identically
const shuffled = {};
const keys = Object.keys(fraudEvent).sort(() => Math.random() - 0.5);
for (const k of keys) shuffled[k] = fraudEvent[k];
const hashShuffled = hashAIEvent(shuffled);
assert(
  "Key-order independence: shuffled object hashes the same",
  proofHash === hashShuffled
);

// 5e. verifyEventHash — correct hash passes
const verifyOk = verifyEventHash(fraudEvent, onChainHash);
assert(
  "verifyEventHash: correct hash → ok = true",
  verifyOk.ok === true
);

// 5f. verifyEventHash — wrong hash fails
const verifyBad = verifyEventHash(fraudEvent, "0x" + "0".repeat(64));
assert(
  "verifyEventHash: wrong hash → ok = false",
  verifyBad.ok === false
);

// 5g. Tamper detection — modify one field and confirm hash changes
const tampered = Object.assign({}, fraudEvent, { prediction: "LEGITIMATE" });
const tamperedHash = hashAIEvent(tampered);
assert(
  "Tamper detection: changing 'prediction' changes hash",
  tamperedHash !== proofHash
);
const verifyTampered = verifyEventHash(tampered, onChainHash);
assert(
  "Tamper detection: tampered event fails verifyEventHash",
  verifyTampered.ok === false
);

// 5h. Missing CORE_FIELD is rejected
let missingFieldThrew = false;
try {
  createAIEvent({
    ...fraudEvent,
    issuer: "", // blank = missing
  });
} catch {
  missingFieldThrew = true;
}
assert(
  "Validation: blank required field throws",
  missingFieldThrew
);

// 5i. Non-finite confidence is rejected
let badConfThrew = false;
try {
  createAIEvent({ ...fraudEvent, confidence: NaN });
} catch {
  badConfThrew = true;
}
assert(
  "Validation: NaN confidence throws",
  badConfThrew
);

// ── 6. Summary ───────────────────────────────────────────────

console.log(`\n=== Fraud event proof hash (canonical-v1) ===`);
console.log(`  ${proofHash}`);
console.log(`\n=== Results ===`);
console.log(`  Passed: ${passed}`);
console.log(`  Failed: ${failed}`);

if (failed > 0) {
  console.error("\nSome tests FAILED.");
  process.exit(1);
} else {
  console.log("\nAll tests PASSED.");
}
