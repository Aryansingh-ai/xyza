"use strict";

// ======================================================
// AI PROOF SDK  (DAANDRISHTI is the reference impl)
// ======================================================
//
// Generic layer: createAIEvent / verifyEventHash / CORE_FIELDS
// DAANDRISHTI layer: createCanonicalAIEvent (delegates to generic)
//
// canonical-v1 hash = SHA-256( canonicalize(event) )
// The canonical-v1 wire format is frozen; no existing
// on-chain hash will ever change when this file is updated.
//
// SQLite donation row
//   -> Canonical AI Event (this file)
//   -> Canonical serialization
//   -> SHA-256 proof hash
//   -> existing Monad registerProof()
//
// Pure Node (fs / path / crypto only). No network, no
// SQLite dependency, no private keys.
//
// The proof hash is the ONLY thing that goes on-chain.
// The model itself and the event body stay off-chain.
//

const crypto = require("crypto");
const fs = require("fs");
const path = require("path");


// ======================================================
// CONSTANTS
// ======================================================

const SCHEMA_VERSION = "1.0";

const EVENT_TYPE = "AI_DONATION_VERIFICATION";

// Stored with each original snapshot so verification knows
// which hashing scheme a registered proof used.
const PROOF_SCHEME_CANONICAL = "canonical-v1";

const PROOF_SCHEME_LEGACY = "legacy-raw-record";

// Blockchain/sdk -> repo root (Divyakosh-final)
const REPO_ROOT = path.resolve(__dirname, "..", "..");

function resolveCvModelPath() {
  const candidates = [
    path.join(REPO_ROOT, "CV_model", "models", "mobilenet_v3_clean.pt"),
    path.join(REPO_ROOT, "CV-model", "models", "mobilenet_v3_clean.pt"),
    path.join(REPO_ROOT, "Monad_hackathon", "models", "mobilenet_v3_clean.pt"),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return candidates[0];
}

const DEFAULT_MODEL_PATH = resolveCvModelPath();

const DEFAULT_ISSUER = "DAANDRISHTI";

// No device id exists in the existing code (pocket_id
// "webcam-0" is the camera slot, not the device), so this
// is a stable configurable constant. Never random.
const DEFAULT_DEVICE_ID = "DAANDRISHTI-PI-01";

// Architecture used by load_model() in live_camera.py:
// models.mobilenet_v3_large(...)
const DEFAULT_MODEL_ID = "mobilenet_v3_large";


// ======================================================
// CORE_FIELDS
// ======================================================
//
// The required top-level keys that every canonical-v1 event
// MUST contain, regardless of the application domain.
// Optional domain keys (targetBin, itemStatus, recordHash,
// etc.) are allowed and hashed verbatim.
//
const CORE_FIELDS = Object.freeze([
  "schemaVersion",
  "eventType",
  "eventId",
  "issuer",
  "deviceId",
  "modelId",
  "modelHash",
  "prediction",
  "confidence",
  "decision",
  "timestamp",
]);


// ======================================================
// PROVENANCE CONFIG
// ======================================================
//
// Env overrides (all optional):
//   ISSUER, DEVICE_ID, MODEL_ID, MODEL_PATH
//

function getProvenanceConfig(env = process.env) {

  return {

    issuer:
      env.ISSUER || DEFAULT_ISSUER,

    deviceId:
      env.DEVICE_ID || DEFAULT_DEVICE_ID,

    modelId:
      env.MODEL_ID || DEFAULT_MODEL_ID,

    modelPath:
      env.MODEL_PATH
        ? path.resolve(REPO_ROOT, env.MODEL_PATH)
        : DEFAULT_MODEL_PATH

  };

}


// ======================================================
// MODEL HASH
// ======================================================
//
// SHA-256 of the real model file. Streamed in 1 MB chunks
// and cached until the file's size or mtime changes, so it
// is NOT recomputed per request.
//

let modelHashCache = null;

function getModelHash(modelPath = DEFAULT_MODEL_PATH) {

  let stat;

  try {

    stat = fs.statSync(modelPath);

  } catch (error) {

    throw new Error(
      `Model file not found: ${modelPath} ` +
      "(set MODEL_PATH to override)"
    );

  }

  const cacheKey =
    `${modelPath}|${stat.size}|${stat.mtimeMs}`;

  if (
    modelHashCache &&
    modelHashCache.key === cacheKey
  ) {

    return modelHashCache.hash;

  }

  const hasher =
    crypto.createHash("sha256");

  const buffer =
    Buffer.allocUnsafe(1024 * 1024);

  const fd =
    fs.openSync(modelPath, "r");

  try {

    let bytesRead;

    while (
      (bytesRead =
        fs.readSync(
          fd,
          buffer,
          0,
          buffer.length,
          null
        )) > 0
    ) {

      hasher.update(
        buffer.subarray(0, bytesRead)
      );

    }

  } finally {

    fs.closeSync(fd);

  }

  const hash =
    "0x" + hasher.digest("hex");

  modelHashCache = {
    key: cacheKey,
    hash
  };

  return hash;

}


// ======================================================
// CURRENT PROVENANCE
// ======================================================
//
// Used when REGISTERING a proof (and when checking a record
// that has no frozen provenance yet). Throws if the model
// file cannot be read: we never write a fake modelHash.
//

function getCurrentProvenance(env = process.env) {

  const config =
    getProvenanceConfig(env);

  return {

    issuer:
      config.issuer,

    deviceId:
      config.deviceId,

    modelId:
      config.modelId,

    modelHash:
      getModelHash(config.modelPath)

  };

}


// ======================================================
// LEGACY RAW-RECORD HASH (existing behavior, unchanged)
// ======================================================
//
// Same algorithm the API and blockchain.py have always used:
// top-level keys sorted, compact JSON, SHA-256.
// Kept so proofs already on Monad keep verifying.
//

function hashLegacyRecord(record) {

  const canonicalData =
    JSON.stringify(
      Object.keys(record)
        .sort()
        .reduce((obj, key) => {

          obj[key] = record[key];

          return obj;

        }, {})
    );

  return (
    "0x" +
    crypto
      .createHash("sha256")
      .update(canonicalData, "utf8")
      .digest("hex")
  );

}


// ======================================================
// SQLITE ROW -> LEGACY JSONL-SHAPED RECORD
// ======================================================
//
// Rebuilds the record shape live_camera.py writes to JSONL,
// so dashboard responses and legacy hashes keep working:
//
//   denomination (INTEGER)  -> predicted_denomination (string)
//   classifier_confidence   -> classifier_confidence + confidence
//   decision                -> "ACCEPT" (table stores ACCEPT only)
//

function donationToLegacyRecord(donation) {

  return {

    pocket_id:
      donation.pocket_id,

    timestamp:
      donation.timestamp,

    item_status:
      donation.item_status ?? null,

    predicted_denomination:
      String(donation.denomination),

    classifier_confidence:
      donation.classifier_confidence ?? null,

    confidence:
      donation.classifier_confidence ?? null,

    source:
      donation.source ?? null,

    decision:
      "ACCEPT",

    target_bin:
      donation.target_bin ?? null,

    model_version:
      donation.model_version ?? null

  };

}


// ======================================================
// GENERIC AI EVENT BUILDER
// ======================================================
//
// createAIEvent(fields)
//   fields  - plain object whose keys become the event body.
//             All CORE_FIELDS must be present and non-empty.
//             Extra domain-specific keys are passed through.
//
// Returns a new frozen-shape object (not frozen in JS sense,
// just a plain copy) ready for hashAIEvent().
//
// Deliberately free of donation assumptions: no denomination,
// no SQLite row, no Monad registration logic.
//

function requireText(value, name) {

  if (
    typeof value !== "string" ||
    value.trim() === ""
  ) {

    throw new Error(
      `Cannot build AI event: missing ${name}`
    );

  }

  return value;

}


function createAIEvent(fields) {

  if (!fields || typeof fields !== "object") {
    throw new Error("createAIEvent: fields must be a non-null object");
  }

  // Validate all CORE_FIELDS are present and non-empty strings
  // (confidence is the one numeric CORE_FIELD, handled below)
  const textCoreFields = CORE_FIELDS.filter(
    f => f !== "confidence"
  );

  for (const f of textCoreFields) {
    if (
      typeof fields[f] !== "string" ||
      fields[f].trim() === ""
    ) {
      throw new Error(
        `createAIEvent: required field '${f}' is missing or not a non-empty string`
      );
    }
  }

  // confidence must be a finite number
  if (
    typeof fields.confidence !== "number" ||
    !Number.isFinite(fields.confidence)
  ) {
    throw new Error(
      "createAIEvent: 'confidence' must be a finite number"
    );
  }

  // Spread all fields into a fresh object (caller controls
  // key ordering; canonicalize() sorts before hashing so
  // order in the JS object is irrelevant to the hash).
  return Object.assign({}, fields);

}


// ======================================================
// PURE HASH VERIFICATION
// ======================================================
//
// verifyEventHash(event, expectedHash)
//   event        - the canonical AI event object
//   expectedHash - the 0x-prefixed hex string stored on-chain
//
// Returns { ok: boolean, computedHash: string }.
// Never throws on hash mismatch (only throws on bad input).
//

function verifyEventHash(event, expectedHash) {

  if (!event || typeof event !== "object") {
    throw new Error("verifyEventHash: event must be a non-null object");
  }

  if (
    typeof expectedHash !== "string" ||
    !expectedHash.startsWith("0x")
  ) {
    throw new Error(
      "verifyEventHash: expectedHash must be a 0x-prefixed hex string"
    );
  }

  const computedHash = hashAIEvent(event);

  return {
    ok: computedHash.toLowerCase() === expectedHash.toLowerCase(),
    computedHash,
  };

}


// ======================================================
// CANONICAL AI EVENT  (DAANDRISHTI / donation-specific)
// ======================================================


function createCanonicalAIEvent(
  donation,
  provenance
) {

  if (!donation) {

    throw new Error(
      "Cannot build AI event: donation row is missing"
    );

  }

  if (!provenance) {

    throw new Error(
      "Cannot build AI event: provenance is missing"
    );

  }

  const denomination =
    donation.denomination;

  if (
    !Number.isInteger(denomination) ||
    denomination <= 0
  ) {

    throw new Error(
      "Cannot build AI event: denomination must be a positive integer"
    );

  }

  // Build via the generic createAIEvent so CORE_FIELDS are
  // validated, then augment with donation-specific extras.
  // The extra keys (modelVersion, source, sourceId, targetBin,
  // itemStatus, recordHash) are appended AFTER the generic
  // call so the final object shape is identical to before.
  // canonicalize() sorts keys before hashing, so the order of
  // the key declarations here does NOT affect the hash.
  return createAIEvent({

    schemaVersion:
      SCHEMA_VERSION,

    eventType:
      EVENT_TYPE,

    // Stable ID computed once at SQLite insert time.
    // Never changes when the API polls.
    eventId:
      requireText(
        donation.donation_id,
        "donation_id"
      ),

    issuer:
      requireText(
        provenance.issuer,
        "issuer"
      ),

    deviceId:
      requireText(
        provenance.deviceId,
        "deviceId"
      ),

    modelId:
      requireText(
        provenance.modelId,
        "modelId"
      ),

    modelVersion:
      donation.model_version ?? null,

    modelHash:
      requireText(
        provenance.modelHash,
        "modelHash"
      ),

    prediction:
      String(denomination),

    confidence:
      donation.classifier_confidence ?? null,

    // donations only ever holds ACCEPT records
    decision:
      "ACCEPT",

    // Event time from the AI record. Never created_at,
    // never API polling time.
    timestamp:
      requireText(
        donation.timestamp,
        "timestamp"
      ),

    source:
      donation.source ?? null,

    sourceId:
      requireText(
        donation.pocket_id,
        "pocket_id"
      ),

    targetBin:
      donation.target_bin ?? null,

    itemStatus:
      donation.item_status ?? null,

    // Existing raw-record hash, rebuilt from the row.
    // Ties the new event to the original record concept.
    recordHash:
      hashLegacyRecord(
        donationToLegacyRecord(donation)
      )

  });

}


// ======================================================
// CANONICAL SERIALIZATION
// ======================================================
//
// Recursive key sort, no whitespace, UTF-8.
// Same event -> same string -> same hash, always.
// Throws on undefined / NaN / Infinity instead of silently
// serializing them differently.
//

function canonicalize(value) {

  if (value === null) {
    return "null";
  }

  if (Array.isArray(value)) {

    return (
      "[" +
      value.map(canonicalize).join(",") +
      "]"
    );

  }

  switch (typeof value) {

    case "string":
    case "boolean":
      return JSON.stringify(value);

    case "number":

      if (!Number.isFinite(value)) {

        throw new Error(
          "Cannot canonicalize non-finite number"
        );

      }

      return JSON.stringify(value);

    case "object":

      return (
        "{" +
        Object.keys(value)
          .sort()
          .map(key => {

            if (value[key] === undefined) {

              throw new Error(
                `Cannot canonicalize undefined value at key '${key}'`
              );

            }

            return (
              JSON.stringify(key) +
              ":" +
              canonicalize(value[key])
            );

          })
          .join(",") +
        "}"
      );

    default:

      throw new Error(
        `Cannot canonicalize value of type ${typeof value}`
      );

  }

}


function canonicalizeEvent(event) {

  return canonicalize(event);

}


// ======================================================
// EVENT HASH (this is the on-chain proof hash)
// ======================================================

function hashAIEvent(event) {

  return (
    "0x" +
    crypto
      .createHash("sha256")
      .update(
        canonicalizeEvent(event),
        "utf8"
      )
      .digest("hex")
  );

}


// ======================================================
// EXPORTS
// ======================================================

module.exports = {

  // Constants
  SCHEMA_VERSION,
  EVENT_TYPE,
  PROOF_SCHEME_CANONICAL,
  PROOF_SCHEME_LEGACY,
  DEFAULT_MODEL_PATH,
  CORE_FIELDS,

  // Config / model
  getProvenanceConfig,
  getCurrentProvenance,
  getModelHash,

  // Hashing
  hashLegacyRecord,
  canonicalizeEvent,
  hashAIEvent,

  // Generic SDK primitives
  createAIEvent,
  verifyEventHash,

  // DAANDRISHTI-specific
  donationToLegacyRecord,
  createCanonicalAIEvent,

};
