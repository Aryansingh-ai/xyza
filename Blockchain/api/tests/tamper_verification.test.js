/**
 * Automated Test Suite for DivyaKosh Tampering Detection
 * 
 * Covers all required test cases:
 * - Test A: Untampered Record (verified against on-chain snapshot) -> VERIFIED
 * - Test B: Modified Record (modified in SQLite without reference update) -> TAMPERED
 * - Test C: Modified Amount/Denomination (detected as TAMPERED with field details)
 * - Test D: Missing Reference (unregistered record) -> UNVERIFIED (not falsely marked VERIFIED)
 * - Test E: Local Audit Log & Checksum Check (detects modification even without blockchain proof)
 * - Test F: Operational Error Handling
 */

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const Database = require("better-sqlite3");
const proof = require("../../sdk/proof");

// Helper to compute donation_id (matching importer.py make_donation_id)
function computeDonationId(donation) {
  const identity = {
    denomination: Number(donation.denomination),
    model_version: donation.model_version ?? null,
    pocket_id: donation.pocket_id ?? null,
    source: donation.source ?? null,
    timestamp: donation.timestamp ?? null
  };
  const sortedKeys = Object.keys(identity).sort();
  const canonicalObj = {};
  for (const k of sortedKeys) {
    canonicalObj[k] = identity[k];
  }
  const jsonStr = JSON.stringify(canonicalObj);
  return crypto.createHash("sha256").update(jsonStr, "utf8").digest("hex");
}

function verifyRecord(donation, originalSnapshot, inferenceIndex) {
  let status = "UNVERIFIED";
  let exists = false;
  let tamperDetails = null;
  let currentHash = null;
  let proofScheme = proof.PROOF_SCHEME_LEGACY;
  let originalDenomination = originalSnapshot?.originalDenomination || null;
  let originalBin = originalSnapshot?.originalBin || null;
  let currentDenomination = null;
  let currentBin = null;

  try {
    if (!donation || typeof donation !== "object") {
      throw new Error("Invalid donation record: expected object");
    }

    const record = proof.donationToLegacyRecord(donation);
    const legacyHash = proof.hashLegacyRecord(record);

    let canonicalHash = null;
    try {
      const provenance = originalSnapshot?.provenance || proof.getCurrentProvenance();
      const event = proof.createCanonicalAIEvent(donation, provenance);
      canonicalHash = proof.hashAIEvent(event);
    } catch {}

    currentHash = canonicalHash || legacyHash;
    proofScheme = canonicalHash ? proof.PROOF_SCHEME_CANONICAL : proof.PROOF_SCHEME_LEGACY;

    currentDenomination = record.predicted_denomination || null;
    currentBin = record.target_bin || null;

    if (originalSnapshot) {
      const matchCanonical =
        originalSnapshot.proofHash.toLowerCase() === currentHash.toLowerCase();
      const matchLegacy =
        legacyHash &&
        originalSnapshot.proofHash.toLowerCase() === legacyHash.toLowerCase();

      if (matchCanonical || matchLegacy) {
        status = "VERIFIED";
        exists = true;
      } else {
        status = "TAMPERED";
        const changes = [];
        if (originalDenomination && String(originalDenomination) !== String(currentDenomination)) {
          changes.push(`Denomination changed from ₹${originalDenomination} to ₹${currentDenomination}`);
        }
        if (originalBin && originalBin !== currentBin) {
          changes.push(`Bin changed from ${originalBin} to ${currentBin}`);
        }
        if (changes.length === 0) {
          changes.push("Record contents changed and the current hash no longer matches the blockchain proof.");
        }
        tamperDetails = changes.join(" | ");
      }
    } else {
      let checksumMismatch = false;
      if (donation.donation_id) {
        const expectedDonationId = computeDonationId(donation);
        if (donation.donation_id.toLowerCase() !== expectedDonationId.toLowerCase()) {
          checksumMismatch = true;
        }
      }

      const logEntry = inferenceIndex ? inferenceIndex.get(donation.timestamp) : null;
      let logMismatch = false;
      const changes = [];

      if (logEntry) {
        const logDenom = String(logEntry.predicted_denomination);
        if (logDenom && logDenom !== String(currentDenomination)) {
          logMismatch = true;
          originalDenomination = logDenom;
          changes.push(`Denomination changed from ₹${logDenom} (audit log) to ₹${currentDenomination} (SQLite)`);
        }
        if (logEntry.target_bin && logEntry.target_bin !== currentBin) {
          logMismatch = true;
          originalBin = logEntry.target_bin;
          changes.push(`Bin changed from ${logEntry.target_bin} (audit log) to ${currentBin} (SQLite)`);
        }
      }

      if (checksumMismatch || logMismatch) {
        status = "TAMPERED";
        if (changes.length > 0) {
          tamperDetails = changes.join(" | ");
        } else {
          tamperDetails = "Database row modified directly: record integrity checksum (donation_id) mismatch.";
        }
      } else {
        status = "UNVERIFIED";
      }
    }
  } catch (err) {
    status = "VERIFICATION_ERROR";
    tamperDetails = `Verification error: ${err.message}`;
  }

  return {
    status,
    exists,
    proofHash: currentHash,
    proofScheme,
    originalDenomination,
    currentDenomination,
    originalBin,
    currentBin,
    tamperDetails
  };
}

// -------------------------------------------------------------
// Test Execution
// -------------------------------------------------------------

let passed = 0;
let failed = 0;

function assert(condition, testName, message) {
  if (condition) {
    console.log(`  ✓ PASS: ${testName}`);
    passed++;
  } else {
    console.error(`  ✗ FAIL: ${testName} - ${message || ""}`);
    failed++;
  }
}

console.log("==================================================");
console.log("RUNNING TAMPER DETECTION VERIFICATION TESTS");
console.log("==================================================\n");

// Set up isolated mock environment
const mockTimestamp = "2026-11-02T12:00:00.000000+00:00";
const mockPocket = "webcam-0";
const mockModelVer = "mobilenet_v3_clean_v1";

const validDonation = {
  id: 1001,
  pocket_id: mockPocket,
  timestamp: mockTimestamp,
  item_status: "NOTE_DETECTED",
  denomination: 100,
  classifier_confidence: 0.95,
  source: "webcam",
  target_bin: "BIN_100",
  model_version: mockModelVer,
  created_at: mockTimestamp
};
validDonation.donation_id = computeDonationId(validDonation);

// Establish canonical proof for snapshot
const mockProvenance = proof.getCurrentProvenance();
const canonicalEvent = proof.createCanonicalAIEvent(validDonation, mockProvenance);
const canonicalHash = proof.hashAIEvent(canonicalEvent);

const validSnapshot = {
  recordId: `${mockPocket}|${mockTimestamp}`,
  proofHash: canonicalHash,
  proofScheme: "canonical-v1",
  originalDenomination: "100",
  originalBin: "BIN_100",
  originalRecord: proof.donationToLegacyRecord(validDonation),
  registeredAt: mockTimestamp,
  transactionHash: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
  blockNumber: 1000000,
  provenance: mockProvenance
};

const emptyInferenceIndex = new Map();
const populatedInferenceIndex = new Map();
populatedInferenceIndex.set(mockTimestamp, {
  pocket_id: mockPocket,
  timestamp: mockTimestamp,
  predicted_denomination: "100",
  target_bin: "BIN_100"
});

// Test A: Untampered Record
console.log("Test A: Untampered Record with Blockchain Reference");
{
  const result = verifyRecord(validDonation, validSnapshot, emptyInferenceIndex);
  assert(result.status === "VERIFIED", "Test A1", `Expected VERIFIED, got ${result.status}`);
  assert(result.exists === true, "Test A2", "Expected exists to be true");
  assert(result.tamperDetails === null, "Test A3", "Expected no tamper details");
}

// Test B: Modified Record (tampered in SQLite)
console.log("\nTest B: Modified Record with Blockchain Reference");
{
  const tamperedDonation = { ...validDonation, target_bin: "BIN_500" };
  const result = verifyRecord(tamperedDonation, validSnapshot, emptyInferenceIndex);
  assert(result.status === "TAMPERED", "Test B1", `Expected TAMPERED, got ${result.status}`);
  assert(result.exists === false, "Test B2", "Expected exists to be false");
  assert(result.tamperDetails && result.tamperDetails.includes("Bin changed"), "Test B3", `Expected bin change details, got ${result.tamperDetails}`);
}

// Test C: Modified Amount or Denomination
console.log("\nTest C: Modified Denomination (Amount)");
{
  const tamperedDonation = { ...validDonation, denomination: 500 };
  const result = verifyRecord(tamperedDonation, validSnapshot, emptyInferenceIndex);
  assert(result.status === "TAMPERED", "Test C1", `Expected TAMPERED, got ${result.status}`);
  assert(
    result.tamperDetails && result.tamperDetails.includes("Denomination changed from ₹100 to ₹500"),
    "Test C2",
    `Expected denomination change explanation, got ${result.tamperDetails}`
  );
  assert(result.originalDenomination === "100", "Test C3", `Expected orig 100, got ${result.originalDenomination}`);
  assert(result.currentDenomination === "500", "Test C4", `Expected curr 500, got ${result.currentDenomination}`);
}

// Test D: Missing Reference (Unregistered Record)
console.log("\nTest D: Missing Reference (Not Anchored on Blockchain)");
{
  // Untampered record without snapshot should be UNVERIFIED, NOT marked as VERIFIED!
  const result = verifyRecord(validDonation, null, populatedInferenceIndex);
  assert(result.status === "UNVERIFIED", "Test D1", `Expected UNVERIFIED, got ${result.status}`);
  assert(result.status !== "VERIFIED", "Test D2", "Must NEVER mark unverified record as VERIFIED");
  assert(result.tamperDetails === null, "Test D3", "Expected no tamper details for clean unverified");
}

// Test E: Tampered Record WITHOUT Blockchain Snapshot (Detected via Checksum / Audit Log)
console.log("\nTest E: Tampered Record Without Blockchain Snapshot");
{
  // Record has no on-chain snapshot, but attacker modified denomination from 100 to 50 directly in SQLite
  const modifiedUnregistered = { ...validDonation, denomination: 50 };
  const result = verifyRecord(modifiedUnregistered, null, populatedInferenceIndex);
  assert(result.status === "TAMPERED", "Test E1", `Expected TAMPERED via local check, got ${result.status}`);
  assert(
    result.tamperDetails && result.tamperDetails.includes("Denomination changed from ₹100 (audit log) to ₹50 (SQLite)"),
    "Test E2",
    `Expected audit log mismatch details, got ${result.tamperDetails}`
  );
}

// Test E2: Tampered Record with only checksum mismatch (log entry missing)
console.log("\nTest E2: Tampered Record With Only Checksum Mismatch (Audit Log Missing)");
{
  const modifiedUnregistered = { ...validDonation, denomination: 50 };
  const result = verifyRecord(modifiedUnregistered, null, emptyInferenceIndex);
  assert(result.status === "TAMPERED", "Test E2-1", `Expected TAMPERED via checksum, got ${result.status}`);
  assert(
    result.tamperDetails && result.tamperDetails.includes("donation_id"),
    "Test E2-2",
    `Expected checksum mismatch details, got ${result.tamperDetails}`
  );
}

// Test F: Operational Error Handling
console.log("\nTest F: Operational Error Handling");
{
  const badDonation = null;
  const result = verifyRecord(badDonation, validSnapshot, emptyInferenceIndex);
  assert(result.status === "VERIFICATION_ERROR", "Test F1", `Expected VERIFICATION_ERROR, got ${result.status}`);
  assert(result.tamperDetails && result.tamperDetails.includes("error"), "Test F2", "Expected error details");
}

console.log("\n==================================================");
console.log(`TOTAL TESTS: ${passed + failed} | PASSED: ${passed} | FAILED: ${failed}`);
console.log("==================================================");

if (failed > 0) {
  process.exit(1);
} else {
  console.log("\nAll tamper-detection unit tests PASSED successfully!");
}
