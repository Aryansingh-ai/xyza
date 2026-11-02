const path = require("path");
require("dotenv").config({ path: path.resolve(__dirname, ".env") });
require("dotenv").config();

const express = require("express");
const { ethers } = require("ethers");
const fs = require("fs");
const crypto = require("crypto");
const Database = require("better-sqlite3");
const proof = require("../sdk/proof");

const app = express();

app.use((req, res, next) => {
  const origin = req.headers.origin;
  const allowedOriginEnv = process.env.ALLOWED_ORIGIN;

  if (origin) {
    if (
      !allowedOriginEnv ||
      origin === allowedOriginEnv ||
      /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)
    ) {
      res.setHeader("Access-Control-Allow-Origin", origin);
    }
  } else {
    res.setHeader("Access-Control-Allow-Origin", "*");
  }

  res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With");

  if (req.method === "OPTIONS") {
    return res.sendStatus(204);
  }

  next();
});

app.use(express.json());


// ======================================================
// CONFIGURATION
// ======================================================

const RPC_URL = process.env.RPC_URL;
const PRIVATE_KEY = process.env.PRIVATE_KEY;
const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS;

const REPO_ROOT = path.resolve(__dirname, "..", "..");

function resolveCvDir() {
  const candidates = ["CV_model", "CV-model", "Monad_hackathon"];
  for (const name of candidates) {
    const candidatePath = path.join(REPO_ROOT, name);
    if (fs.existsSync(candidatePath)) {
      return candidatePath;
    }
  }
  return path.join(REPO_ROOT, "CV_model");
}

const CV_DIR = resolveCvDir();

const ML_LOG_FILE =
  process.env.ML_LOG_FILE
    ? path.resolve(__dirname, process.env.ML_LOG_FILE)
    : path.join(CV_DIR, "logs", "inference.jsonl");

// Structured source of truth for accepted AI events.
// (inference.jsonl above is now audit-only for the API.)
const SQLITE_FILE =
  process.env.SQLITE_FILE
    ? path.resolve(__dirname, process.env.SQLITE_FILE)
    : path.join(CV_DIR, "sqlite", "donation_box.db");


// ======================================================
// ORIGINAL RECORD REGISTRY
// ======================================================
//
// This file is automatically maintained by the API.
//
// It stores the ORIGINAL accepted record at the moment
// it is successfully registered on Monad.
//
// We NEVER manually edit this file.
//

const DATA_DIR = path.join(__dirname, "data");

const ORIGINAL_RECORDS_FILE =
  process.env.ORIGINAL_RECORDS_FILE
    ? path.resolve(__dirname, process.env.ORIGINAL_RECORDS_FILE)
    : path.join(DATA_DIR, "original_records.json");


// Create parent directory if it does not exist
const originalRecordsDir = path.dirname(ORIGINAL_RECORDS_FILE);
if (!fs.existsSync(originalRecordsDir)) {
  fs.mkdirSync(originalRecordsDir, { recursive: true });
}


// Create registry file if it does not exist
if (!fs.existsSync(ORIGINAL_RECORDS_FILE)) {
  fs.writeFileSync(
    ORIGINAL_RECORDS_FILE,
    JSON.stringify([], null, 2),
    "utf8"
  );
}


// ======================================================
// SMART CONTRACT ABI
// ======================================================

const CONTRACT_ABI = [

  "function registerProof(bytes32 _proofHash, string calldata _sourceId) external",

  "function verifyProof(bytes32 _proofHash) external view returns (bool exists, string memory sourceId, uint256 timestamp, address registeredBy)"

];


// ======================================================
// BLOCKCHAIN CONNECTION
// ======================================================

const provider =
  new ethers.JsonRpcProvider(RPC_URL);

const wallet =
  new ethers.Wallet(
    PRIVATE_KEY,
    provider
  );

const contract =
  new ethers.Contract(
    CONTRACT_ADDRESS,
    CONTRACT_ABI,
    wallet
  );


// ======================================================
// ENVIO HYPERINDEX — READ-ONLY QUERY LAYER
// ======================================================
//
// Envio indexes ProofRegistered and IssuerStatusChanged
// events from the deployed contract on Monad Testnet.
//
// USAGE:
//   - findProofTransaction() queries Envio first (O(1)).
//     Falls back to the old getLogs scan only when Envio
//     returns null (proof too recent / indexer lag).
//   - /verify-proof/:hash continues to call contract.verifyProof()
//     for authoritative existence. Envio supplies the tx metadata.
//   - /all-records is NOT changed: still uses local hash compare,
//     still makes ZERO per-record RPC calls.
//   - Envio is never used for cryptographic tamper detection.
//
// CONFIGURATION (optional — set ENVIO_GRAPHQL_URL in .env):
//   Default: http://localhost:8080  (local Envio dev instance)
//   Cloud:   set ENVIO_GRAPHQL_URL to your deployed Envio endpoint
//
// When ENVIO_GRAPHQL_URL is not set or Envio is unreachable,
// the API falls back transparently to the getLogs scan.
//

const ENVIO_GRAPHQL_URL =
  process.env.ENVIO_GRAPHQL_URL ||
  "http://localhost:8080";


// -------------------------------------------------------
// queryEnvioProof(proofHash)
// -------------------------------------------------------
// Queries the Envio indexer for a RegisteredProof by its
// proofHash. Returns { txHash, blockNumber } or null.
//
// Never throws — Envio is optional infrastructure.
// Any error (network, indexer down, hash not yet indexed)
// returns null so the caller can fall back to getLogs.
//
async function queryEnvioProof(proofHash) {

  try {

    // Normalise: Envio stores the id as the proofHash string
    // exactly as it was emitted (0x-prefixed hex).
    const id = proofHash.toLowerCase();

    const query = `
      query GetProof($id: String!) {
        RegisteredProof_by_pk(id: $id) {
          id
          txHash
          blockNumber
          blockTimestamp
          sourceId
          registeredBy
        }
      }
    `;

    const response = await fetch(
      `${ENVIO_GRAPHQL_URL}/v1/graphql`,
      {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ query, variables: { id } }),
        signal:  AbortSignal.timeout(4000)
      }
    );

    if (!response.ok) {
      return null;
    }

    const json = await response.json();

    const record = json?.data?.RegisteredProof_by_pk;

    if (!record || !record.txHash) {
      return null;
    }

    return {
      txHash:      record.txHash,
      blockNumber: Number(record.blockNumber)
    };

  } catch {
    // Envio unavailable / timeout / parse error — fall through to getLogs
    return null;
  }

}


// ======================================================
// CORS
// ======================================================

app.use((req, res, next) => {

  res.header(
    "Access-Control-Allow-Origin",
    "*"
  );

  res.header(
    "Access-Control-Allow-Methods",
    "GET,POST,OPTIONS"
  );

  res.header(
    "Access-Control-Allow-Headers",
    "Content-Type"
  );

  next();

});


// ======================================================
// CREATE LEGACY RAW-RECORD HASH
// ======================================================
//
// Hash of the raw ML record (JSONL-shaped). Every proof
// registered BEFORE canonical AI events used this, and
// blockchain.py still sends it as a lookup key.
//
// The algorithm lives in the SDK so the canonical event's
// recordHash and this function can never drift apart.
//

function createProofHash(record) {

  return proof.hashLegacyRecord(record);

}


// ======================================================
// READ DONATIONS FROM SQLITE (PRIMARY SOURCE OF TRUTH)
// ======================================================
//
// Monad_hackathon/sqlite/donation_box.db is the structured
// source for accepted AI events. inference.jsonl stays as the
// audit stream written by live_camera.py; the API no longer
// reads it.
//
// Every row is returned as:
//
//   lineNumber : donations.id (name kept so the dashboard's
//                "recordNumber" keeps working)
//   donation   : the raw SQLite row
//   record     : legacy JSONL-shaped record rebuilt from the
//                row (dashboard compatibility + legacy hash)
//

function openDonationDatabase() {

  if (!fs.existsSync(SQLITE_FILE)) {

    throw new Error(
      `SQLite database not found: ${SQLITE_FILE}`
    );

  }

  return new Database(
    SQLITE_FILE,
    {
      readonly: true,
      fileMustExist: true,
      timeout: 5000
    }
  );

}


function toDonationItem(donation) {

  return {

    lineNumber:
      donation.id,

    donation,

    record:
      proof.donationToLegacyRecord(
        donation
      )

  };

}


function getAllDonationRecords() {

  const db =
    openDonationDatabase();

  try {

    return db
      .prepare(
        "SELECT * FROM donations ORDER BY id ASC"
      )
      .all()
      .map(toDonationItem);

  } finally {

    db.close();

  }

}


function getLatestDonationRecord() {

  const db =
    openDonationDatabase();

  try {

    const row =
      db
        .prepare(
          "SELECT * FROM donations ORDER BY id DESC LIMIT 1"
        )
        .get();

    if (!row) {

      throw new Error(
        "SQLite donations table is empty"
      );

    }

    return toDonationItem(row);

  } finally {

    db.close();

  }

}


function getDonationById(donationId) {

  const db =
    openDonationDatabase();

  try {

    const row =
      db
        .prepare(
          "SELECT * FROM donations WHERE donation_id = ?"
        )
        .get(donationId);

    return row
      ? toDonationItem(row)
      : null;

  } finally {

    db.close();

  }

}


// ======================================================
// ORIGINAL RECORD REGISTRY FUNCTIONS
// ======================================================

function readOriginalRecords() {

  try {

    if (!fs.existsSync(ORIGINAL_RECORDS_FILE)) {
      return [];
    }

    const content =
      fs.readFileSync(
        ORIGINAL_RECORDS_FILE,
        "utf8"
      );

    if (!content.trim()) {
      return [];
    }

    const records =
      JSON.parse(content);

    if (!Array.isArray(records)) {
      return [];
    }

    return records;

  } catch (error) {

    console.error(
      "Error reading original records:",
      error.message
    );

    return [];

  }

}


function writeOriginalRecords(records) {

  fs.writeFileSync(
    ORIGINAL_RECORDS_FILE,
    JSON.stringify(
      records,
      null,
      2
    ),
    "utf8"
  );

}


// ======================================================
// CREATE STABLE RECORD ID
// ======================================================
//
// The hash changes if the record is tampered with.
//
// Therefore we CANNOT use the hash as the identity
// when detecting tampering.
//
// Instead we use:
// sourceId + original timestamp
//
// Example:
// webcam-0|2026-10-07T04:05:25.210977+00:00
//

function createRecordId(record) {

  return `${record.pocket_id || "unknown"}|${record.timestamp || "unknown"}`;

}


// ======================================================
// LOCAL INTEGRITY HELPERS (FOR UN-ANCHORED CHECKS)
// ======================================================

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

function loadInferenceLogIndex() {
  const index = new Map();
  try {
    if (fs.existsSync(ML_LOG_FILE)) {
      const content = fs.readFileSync(ML_LOG_FILE, "utf8");
      const lines = content.split(/\r?\n/);
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const entry = JSON.parse(line);
          if (entry.timestamp) {
            index.set(entry.timestamp, entry);
          }
        } catch (_) {}
      }
    }
  } catch (err) {
    console.error("Warning reading ML_LOG_FILE:", err.message);
  }
  return index;
}


// ======================================================
// FIND DONATION FOR REGISTRATION
// ======================================================
//
// Resolves the SQLite row a /register-proof request refers to.
//
//   1. donationId  (exact, preferred: sent by blockchain.py)
//   2. proofHash   (legacy raw-record hash, old clients)
//

function findDonationForRegistration({
  donationId,
  proofHash
}) {

  if (donationId) {

    const item =
      getDonationById(
        donationId
      );

    if (item) {

      return item;

    }

  }

  if (proofHash) {

    const wanted =
      proofHash.toLowerCase();

    for (
      const item of getAllDonationRecords()
    ) {

      if (
        createProofHash(
          item.record
        ).toLowerCase() === wanted
      ) {

        return item;

      }

    }

  }

  return null;

}


// ======================================================
// SAVE ORIGINAL RECORD
// ======================================================

function saveOriginalRecord(record, proofHash, blockchainInfo, proofInfo) {

  const records =
    readOriginalRecords();

  const recordId =
    createRecordId(record);

  // Prevent duplicate snapshots
  const existingIndex =
    records.findIndex(
      item =>
        item.recordId === recordId
    );

  const snapshot = {

    recordId,

    proofHash,

    // Which hashing scheme proofHash used, and the frozen
    // provenance (issuer/device/model/modelHash) at registration
    // time. Verification re-uses it so a later model swap or config
    // change cannot make an untouched record look tampered.
    proofScheme:
      proofInfo?.proofScheme || proof.PROOF_SCHEME_LEGACY,

    legacyProofHash:
      proofInfo?.legacyProofHash || null,

    provenance:
      proofInfo?.provenance || null,

    canonicalEvent:
      proofInfo?.canonicalEvent || null,

    sourceId:
      record.pocket_id || "unknown",

    originalDenomination:
      record.predicted_denomination || null,

    originalBin:
      record.target_bin || null,

    originalRecord:
      record,

    registeredAt:
      new Date().toISOString(),

    transactionHash:
      blockchainInfo?.transactionHash || null,

    blockNumber:
      blockchainInfo?.blockNumber || null

  };


  if (existingIndex >= 0) {

    // Never replace an existing original
    console.log(
      "Original snapshot already exists:",
      recordId
    );

    return records[existingIndex];

  }


  records.push(snapshot);

  writeOriginalRecords(records);

  console.log();
  console.log(
    "=== ORIGINAL RECORD SAVED ==="
  );

  console.log(
    "Record ID:",
    recordId
  );

  console.log(
    "Original denomination:",
    snapshot.originalDenomination
  );

  console.log(
    "Original bin:",
    snapshot.originalBin
  );

  console.log(
    "Original proof hash:",
    proofHash
  );

  return snapshot;

}


// ======================================================
// FIND ORIGINAL SNAPSHOT
// ======================================================

function findOriginalSnapshot(record) {

  const records =
    readOriginalRecords();

  const recordId =
    createRecordId(record);

  return (
    records.find(
      item =>
        item.recordId === recordId
    ) || null
  );

}


// ======================================================
// FIND TRANSACTION FOR PROOF
// ======================================================
//
// Strategy:
//   1. Try Envio first — O(1) indexed GraphQL lookup, no RPC.
//   2. If Envio returns null (proof not yet indexed, indexer
//      behind by a few blocks, or Envio unavailable), fall back
//      to the original getLogs scan over the last 200k blocks.
//
// This replaces a raw 200k-block getLogs scan with an instant
// GraphQL query, while preserving identical behaviour for callers.
//
// /all-records does NOT call this function — it uses local hash
// comparison exclusively and makes zero RPC calls.
//

async function findProofTransaction(
  proofHash
) {

  // ----------------------------------------------------------
  // Step 1: Try Envio (O(1), no RPC)
  // ----------------------------------------------------------

  const envioResult =
    await queryEnvioProof(proofHash);

  if (envioResult) {

    return {

      transactionHash:
        envioResult.txHash,

      blockNumber:
        envioResult.blockNumber,

      explorerUrl:
        `https://testnet.monadexplorer.com/tx/${envioResult.txHash}`,

      source:
        "envio"

    };

  }


  // ----------------------------------------------------------
  // Step 2: RPC fallback — getLogs scan (original behaviour)
  // Used when Envio is unavailable or the proof is too recent
  // to have been indexed yet.
  // ----------------------------------------------------------

  try {

    const latestBlock =
      await provider.getBlockNumber();

    const fromBlock =
      Math.max(
        0,
        latestBlock - 200000
      );

    const eventTopic =
      ethers.id(
        "ProofRegistered(bytes32,string,uint256,address)"
      );

    const logs =
      await provider.getLogs({

        address:
          CONTRACT_ADDRESS,

        topics: [
          eventTopic,
          proofHash
        ],

        fromBlock,

        toBlock:
          latestBlock

      });

    if (logs.length === 0) {

      return null;

    }

    const latestMatchingLog =
      logs[logs.length - 1];

    return {

      transactionHash:
        latestMatchingLog.transactionHash,

      blockNumber:
        latestMatchingLog.blockNumber,

      explorerUrl:
        `https://testnet.monadexplorer.com/tx/${latestMatchingLog.transactionHash}`,

      source:
        "rpc-fallback"

    };

  } catch (error) {

    console.error(
      "Transaction lookup error:",
      error.message
    );

    return null;

  }

}


// ======================================================
// VERIFY A DONATION ON-CHAIN (CANONICAL + LEGACY)
// ======================================================
//
// Two hashes can represent one SQLite row:
//
//   canonical : SHA-256 of the canonical AI event (new)
//   legacy    : SHA-256 of the raw ML record     (old)
//
// New registrations use canonical. Proofs already on Monad
// from before this change used legacy, so we try canonical
// first and fall back to legacy. If neither exists on-chain
// the caller decides NOT_REGISTERED vs TAMPERED exactly as
// before.
//
// Provenance for the canonical event comes from the original
// snapshot (frozen at registration). Rows with no snapshot use
// the current config + current model file hash.
//

let canonicalUnavailableWarned = false;


function buildProofCandidates(item, originalSnapshot) {

  const legacyHash =
    createProofHash(
      item.record
    );

  let canonicalEvent = null;

  let canonicalHash = null;

  try {

    const provenance =
      originalSnapshot?.provenance ||
      proof.getCurrentProvenance();

    canonicalEvent =
      proof.createCanonicalAIEvent(
        item.donation,
        provenance
      );

    canonicalHash =
      proof.hashAIEvent(
        canonicalEvent
      );

  } catch (error) {

    if (!canonicalUnavailableWarned) {

      canonicalUnavailableWarned = true;

      console.warn(
        "Canonical proof unavailable, using legacy hash only:",
        error.message
      );

    }

  }

  return {

    legacyHash,

    canonicalEvent,

    canonicalHash

  };

}


async function verifyDonationOnChain(
  item,
  originalSnapshot
) {

  const {
    legacyHash,
    canonicalEvent,
    canonicalHash
  } = buildProofCandidates(
    item,
    originalSnapshot
  );


  let canonicalResult = null;

  if (canonicalHash) {

    canonicalResult =
      await contract.verifyProof(
        canonicalHash
      );

    if (canonicalResult[0]) {

      return {

        exists: true,

        proofHash: canonicalHash,

        proofScheme:
          proof.PROOF_SCHEME_CANONICAL,

        blockchainResult:
          canonicalResult,

        canonicalEvent,

        legacyHash

      };

    }

  }


  const legacyResult =
    await contract.verifyProof(
      legacyHash
    );

  if (legacyResult[0]) {

    return {

      exists: true,

      proofHash: legacyHash,

      proofScheme:
        proof.PROOF_SCHEME_LEGACY,

      blockchainResult:
        legacyResult,

      canonicalEvent,

      legacyHash

    };

  }


  // Not on-chain under either hash.
  // Report the preferred (canonical) CURRENT hash.
  return {

    exists: false,

    proofHash:
      canonicalHash || legacyHash,

    proofScheme:
      canonicalHash
        ? proof.PROOF_SCHEME_CANONICAL
        : proof.PROOF_SCHEME_LEGACY,

    blockchainResult:
      canonicalResult || legacyResult,

    canonicalEvent,

    legacyHash

  };

}


// ======================================================
// HOME
// ======================================================

app.get("/", (req, res) => {

  res.json({

    service:
      "DaanDristi Blockchain API",

    status:
      "running"

  });

});


// ======================================================
// REGISTER BLOCKCHAIN PROOF
// ======================================================
//
// Accepts:
//   { donationId, sourceId }            (blockchain.py, preferred)
//   { proofHash,  sourceId }            (legacy clients)
//   { donationId, proofHash, sourceId } (blockchain.py sends both)
//
// When the SQLite row is found, the hash registered on Monad is
// the CANONICAL AI event hash, not the raw-record hash the client
// sent. The response `proofHash` is the hash actually registered.
//
// When the row cannot be found, behavior falls back to the old
// flow: register the client-supplied hash as-is.
//

app.post(
  "/register-proof",
  async (req, res) => {

    try {

      const {
        proofHash,
        sourceId,
        donationId
      } = req.body;


      if (
        !sourceId ||
        (!proofHash && !donationId)
      ) {

        return res.status(400).json({

          success:
            false,

          error:
            "sourceId and either proofHash or donationId are required"

        });

      }


      console.log();

      console.log(
        "=========================================="
      );

      console.log(
        "REGISTERING BLOCKCHAIN PROOF"
      );

      console.log(
        "=========================================="
      );

      console.log(
        "Client proof hash:",
        proofHash || "(none)"
      );

      console.log(
        "Donation ID:",
        donationId || "(none)"
      );

      console.log(
        "Source ID:",
        sourceId
      );


      // ------------------------------------------------
      // Find the exact SQLite row for this request
      // ------------------------------------------------

      const item =
        findDonationForRegistration({
          donationId,
          proofHash
        });


      let registerHash;

      let proofScheme;

      let provenance = null;

      let canonicalEvent = null;

      let legacyProofHash =
        proofHash || null;


      if (item) {

        // Throws if the model file cannot be read:
        // we never register a fake modelHash.
        provenance =
          proof.getCurrentProvenance();

        canonicalEvent =
          proof.createCanonicalAIEvent(
            item.donation,
            provenance
          );

        registerHash =
          proof.hashAIEvent(
            canonicalEvent
          );

        proofScheme =
          proof.PROOF_SCHEME_CANONICAL;

        legacyProofHash =
          createProofHash(
            item.record
          );

        console.log(
          "Event ID:",
          canonicalEvent.eventId
        );

        console.log(
          "Canonical proof hash:",
          registerHash
        );

      } else {

        if (!proofHash) {

          return res.status(404).json({

            success:
              false,

            error:
              "donationId not found in SQLite database"

          });

        }

        console.warn(
          "WARNING: Could not find matching SQLite record."
        );

        console.warn(
          "Registering the client-supplied hash as-is."
        );

        registerHash =
          proofHash;

        proofScheme =
          proof.PROOF_SCHEME_LEGACY;

      }


      // ------------------------------------------------
      // Send transaction to Monad
      // ------------------------------------------------

      const tx =
        await contract.registerProof(
          registerHash,
          sourceId
        );

      console.log(
        "Transaction sent:",
        tx.hash
      );


      // ------------------------------------------------
      // Wait for confirmation
      // ------------------------------------------------

      const receipt =
        await tx.wait();

      console.log(
        "Transaction confirmed!"
      );

      console.log(
        "Block:",
        receipt.blockNumber
      );


      const explorerUrl =
        `https://testnet.monadexplorer.com/tx/${tx.hash}`;


      // ------------------------------------------------
      // AUTOMATICALLY SAVE ORIGINAL SNAPSHOT
      // ------------------------------------------------

      if (item) {

        saveOriginalRecord(
          item.record,
          registerHash,
          {
            transactionHash:
              tx.hash,

            blockNumber:
              receipt.blockNumber
          },
          {
            proofScheme,

            legacyProofHash,

            provenance,

            canonicalEvent
          }
        );

      }


      res.json({

        success:
          true,

        transactionHash:
          tx.hash,

        blockNumber:
          receipt.blockNumber,

        explorerUrl,

        proofHash:
          registerHash,

        proofScheme,

        legacyProofHash,

        eventId:
          canonicalEvent?.eventId || null,

        sourceId

      });


    } catch (error) {

      console.error(
        "Blockchain registration error:",
        error
      );

      res.status(500).json({

        success:
          false,

        error:
          error.message

      });

    }

  }
);


// ======================================================
// VERIFY BLOCKCHAIN PROOF
// ======================================================

app.get(
  "/verify-proof/:hash",
  async (req, res) => {

    try {

      const proofHash =
        req.params.hash;


      if (!proofHash) {

        return res.status(400).json({

          success:
            false,

          error:
            "Proof hash is required"

        });

      }


      console.log();

      console.log(
        "=== VERIFYING BLOCKCHAIN PROOF ==="
      );

      console.log(
        "Proof Hash:",
        proofHash
      );


      const result =
        await contract.verifyProof(
          proofHash
        );


      let transactionInfo =
        null;


      if (result[0]) {

        transactionInfo =
          await findProofTransaction(
            proofHash
          );

      }


      res.json({

        success:
          true,

        exists:
          result[0],

        sourceId:
          result[1],

        timestamp:
          result[2].toString(),

        registeredBy:
          result[3],

        transactionHash:
          transactionInfo?.transactionHash ||
          null,

        blockNumber:
          transactionInfo?.blockNumber ||
          null,

        explorerUrl:
          transactionInfo?.explorerUrl ||
          null

      });


    } catch (error) {

      console.error(
        "Blockchain verification error:",
        error
      );

      res.status(500).json({

        success:
          false,

        error:
          error.message

      });

    }

  }
);


// ======================================================
// GET LATEST ACCEPTED RECORD (SQLITE)
// ======================================================

app.get(
  "/latest-record",
  async (req, res) => {

    try {

      const item =
        getLatestDonationRecord();

      const record =
        item.record;


      if (
        record.decision !==
        "ACCEPT"
      ) {

        return res.json({

          success:
            true,

          record,

          blockchain: {

            exists:
              false,

            message:
              "Blockchain proof skipped because decision is not ACCEPT."

          }

        });

      }


      const originalSnapshot =
        findOriginalSnapshot(
          record
        );


      const verification =
        await verifyDonationOnChain(
          item,
          originalSnapshot
        );

      const proofHash =
        verification.proofHash;

      const blockchainResult =
        verification.blockchainResult;


      let transactionInfo =
        null;


      if (
        blockchainResult[0]
      ) {

        transactionInfo =
          await findProofTransaction(
            proofHash
          );

      }


      let status;
      let tamperDetails = null;

      if (blockchainResult[0]) {

        status =
          "VERIFIED";

      } else if (originalSnapshot) {

        status =
          "TAMPERED";
        tamperDetails = "Blockchain proof mismatch: current record contents do not match registered on-chain proof.";

      } else {

        // Check local cryptographic checksum & audit log
        const inferenceIndex = loadInferenceLogIndex();
        const donation = item.donation;
        const expectedId = computeDonationId(donation);
        const logEntry = inferenceIndex.get(donation.timestamp);

        const checksumMismatch =
          donation.donation_id &&
          donation.donation_id.toLowerCase() !== expectedId.toLowerCase();

        const logMismatch =
          logEntry &&
          String(logEntry.predicted_denomination) !== String(donation.denomination);

        if (checksumMismatch || logMismatch) {
          status = "TAMPERED";
          tamperDetails = logMismatch
            ? `Local database modified: denomination changed from ₹${logEntry.predicted_denomination} (audit log) to ₹${donation.denomination}`
            : "Local database modified: record checksum (donation_id) mismatch.";
        } else {
          status = "UNVERIFIED";
        }

      }


      res.json({

        success:
          true,

        record,

        // Canonical AI provenance event for this row
        event:
          verification.canonicalEvent,

        blockchain: {

          proofHash,

          proofScheme:
            verification.proofScheme,

          exists:
            blockchainResult[0],

          status,

          sourceId:
            blockchainResult[1],

          timestamp:
            blockchainResult[2].toString(),

          registeredBy:
            blockchainResult[3],

          transactionHash:
            transactionInfo?.transactionHash ||
            null,

          blockNumber:
            transactionInfo?.blockNumber ||
            null,

          explorerUrl:
            transactionInfo?.explorerUrl ||
            null

        },

        original: originalSnapshot
          ? {

              denomination:
                originalSnapshot.originalDenomination,

              bin:
                originalSnapshot.originalBin,

              proofHash:
                originalSnapshot.proofHash

            }
          : null

      });


    } catch (error) {

      console.error(
        "Latest record error:",
        error
      );

      res.status(500).json({

        success:
          false,

        error:
          error.message

      });

    }

  }
);


// ======================================================
// ⭐ VERIFY ALL ACCEPTED RECORDS
// ======================================================

// ------------------------------------------------------
// Console summary de-duplication
// ------------------------------------------------------
//
// /all-records is polled repeatedly by the dashboard.
// The verification itself still runs on every request,
// but the console summary is only printed the first time
// and whenever Total / Verified / Tampered / Not registered
// changes.
//

let lastPrintedSummaryKey = null;

function printVerificationSummaryIfChanged(
  total,
  verified,
  tampered,
  unverified,
  errors
) {

  const summaryKey =
    `${total}|${verified}|${tampered}|${unverified}|${errors || 0}`;

  if (summaryKey === lastPrintedSummaryKey) {
    return;
  }

  lastPrintedSummaryKey = summaryKey;

  console.log();

  console.log(
    "=========================================="
  );

  console.log(
    "VERIFYING ALL DONATION RECORDS"
  );

  console.log(
    "=========================================="
  );

  console.log(
    "Total:",
    total
  );

  console.log(
    "Verified (On-chain):",
    verified
  );

  console.log(
    "Tampered:",
    tampered
  );

  console.log(
    "Unverified (Awaiting proof):",
    unverified
  );

  if (errors) {
    console.log(
      "Verification errors:",
      errors
    );
  }

}


app.get(
  "/all-records",
  async (req, res) => {

    try {

      const allRecords =
        getAllDonationRecords();

      const inferenceIndex =
        loadInferenceLogIndex();

      const results = [];


      for (
        const item of allRecords
      ) {

        const record =
          item.record;

        const donation =
          item.donation;


        // Only ACCEPT records
        if (
          record.decision !==
          "ACCEPT"
        ) {

          continue;

        }


        // ----------------------------------------------
        // Find original snapshot (holds frozen provenance)
        // ----------------------------------------------

        const originalSnapshot =
          findOriginalSnapshot(
            record
          );


        // ----------------------------------------------
        // Calculate current hash LOCALLY (No RPC)
        // ----------------------------------------------

        const {
          legacyHash,
          canonicalEvent,
          canonicalHash
        } = buildProofCandidates(
          item,
          originalSnapshot
        );

        const currentHash =
          canonicalHash || legacyHash;

        const proofScheme =
          canonicalHash
            ? proof.PROOF_SCHEME_CANONICAL
            : proof.PROOF_SCHEME_LEGACY;


        // ----------------------------------------------
        // Determine status with multi-tier verification
        // ----------------------------------------------

        let status = "UNVERIFIED";
        let exists = false;
        let tamperDetails = null;

        const currentDenomination =
          record.predicted_denomination ||
          null;

        const currentBin =
          record.target_bin ||
          null;

        let originalDenomination =
          originalSnapshot
            ?.originalDenomination ||
          null;

        let originalBin =
          originalSnapshot
            ?.originalBin ||
          null;

        try {
          if (originalSnapshot) {
            // Tier 1: Authoritative Monad Blockchain Proof Snapshot comparison
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

              if (
                originalDenomination &&
                String(originalDenomination) !== String(currentDenomination)
              ) {
                changes.push(
                  `Denomination changed from ₹${originalDenomination} to ₹${currentDenomination}`
                );
              }

              if (
                originalBin &&
                originalBin !== currentBin
              ) {
                changes.push(
                  `Bin changed from ${originalBin} to ${currentBin}`
                );
              }

              if (changes.length === 0) {
                changes.push(
                  "Record contents changed and the current hash no longer matches the blockchain proof."
                );
              }

              tamperDetails = changes.join(" | ");
            }
          } else {
            // Tier 2: Local cryptographic checksum & append-only audit stream cross-reference
            // Check 2a: donation_id checksum
            let checksumMismatch = false;
            if (donation.donation_id) {
              const expectedDonationId = computeDonationId(donation);
              if (donation.donation_id.toLowerCase() !== expectedDonationId.toLowerCase()) {
                checksumMismatch = true;
              }
            }

            // Check 2b: inference.jsonl audit log entry
            const logEntry = inferenceIndex.get(donation.timestamp);
            let logMismatch = false;
            const changes = [];

            if (logEntry) {
              const logDenom = String(logEntry.predicted_denomination);
              if (logDenom && logDenom !== String(currentDenomination)) {
                logMismatch = true;
                originalDenomination = logDenom;
                changes.push(
                  `Denomination changed from ₹${logDenom} (audit log) to ₹${currentDenomination} (SQLite)`
                );
              }
              if (logEntry.target_bin && logEntry.target_bin !== currentBin) {
                logMismatch = true;
                originalBin = logEntry.target_bin;
                changes.push(
                  `Bin changed from ${logEntry.target_bin} (audit log) to ${currentBin} (SQLite)`
                );
              }
            }

            if (checksumMismatch || logMismatch) {
              status = "TAMPERED";
              if (changes.length > 0) {
                tamperDetails = changes.join(" | ");
              } else {
                tamperDetails =
                  "Database row modified directly: record integrity checksum (donation_id) mismatch.";
              }
            } else {
              // Passed local checks, but not yet anchored on Monad blockchain
              status = "UNVERIFIED";
            }
          }
        } catch (verifErr) {
          status = "VERIFICATION_ERROR";
          tamperDetails = `Verification error: ${verifErr.message}`;
        }


        // ----------------------------------------------
        // Build result
        // ----------------------------------------------

        results.push({

          recordNumber:
            item.lineNumber,

          record,

          status,

          exists,

          proofHash:
            currentHash,

          proofScheme,

          eventId:
            item.donation.donation_id,


          // Original information
          originalDenomination,

          originalBin,

          originalProofHash:
            originalSnapshot?.proofHash ||
            null,


          // Current information
          currentDenomination,

          currentBin,


          // Human-readable explanation
          tamperDetails,


          // Blockchain information (from snapshot instead of RPC)
          sourceId:
            originalSnapshot?.sourceId ||
            donation.pocket_id ||
            "webcam-0",

          timestamp:
            originalSnapshot?.registeredAt ||
            donation.created_at ||
            null,

          registeredBy:
            null, // Only available via direct RPC lookup

          transactionHash:
            originalSnapshot?.transactionHash ||
            null,

          blockNumber:
            originalSnapshot?.blockNumber ||
            null,

          explorerUrl:
            originalSnapshot?.transactionHash
              ? `https://testnet.monadexplorer.com/tx/${originalSnapshot.transactionHash}`
              : null

        });

      }


      // =================================================
      // COUNTS
      // =================================================

      const verifiedRecords =
        results.filter(
          item =>
            item.status ===
            "VERIFIED"
        ).length;


      const tamperedRecords =
        results.filter(
          item =>
            item.status ===
            "TAMPERED"
        ).length;


      const unverifiedRecords =
        results.filter(
          item =>
            item.status ===
            "UNVERIFIED" ||
            item.status ===
            "NOT_REGISTERED"
        ).length;


      const errorRecords =
        results.filter(
          item =>
            item.status ===
            "VERIFICATION_ERROR"
        ).length;


      printVerificationSummaryIfChanged(
        results.length,
        verifiedRecords,
        tamperedRecords,
        unverifiedRecords,
        errorRecords
      );


      res.json({

        success:
          true,

        totalRecords:
          results.length,

        verifiedRecords,

        tamperedRecords,

        unverifiedRecords,

        errorRecords,

        notRegisteredRecords:
          unverifiedRecords, // Backward compatibility

        records:
          results

      });


    } catch (error) {

      console.error(
        "All records verification error:",
        error
      );

      res.status(500).json({

        success:
          false,

        error:
          error.message

      });

    }

  }
);


// ======================================================
// ORIGINAL REGISTRY VIEW
// ======================================================
//
// Useful for debugging/demo.
// Shows what the API automatically preserved.
//

app.get(
  "/original-records",
  (req, res) => {

    try {

      const records =
        readOriginalRecords();

      res.json({

        success:
          true,

        count:
          records.length,

        records

      });

    } catch (error) {

      res.status(500).json({

        success:
          false,

        error:
          error.message

      });

    }

  }
);


// ======================================================
// PROOF HISTORY (ENVIO-POWERED)
// ======================================================
//
// Returns paginated proof history from the Envio indexer.
//
// Query params:
//   limit  (default 20, max 100)
//   offset (default 0)
//
// This endpoint is additive — it does not change any existing
// endpoint behaviour. If Envio is unavailable, it returns a
// clear error with instructions.
//
// Note: this is HISTORY data from the event log, not
// authoritative on-chain state. For proof existence, use
// /verify-proof/:hash which calls contract.verifyProof().
//

// ======================================================
// PROOF ANALYTICS (ENVIO-POWERED)
// ======================================================

app.get(
  "/proof-analytics",
  async (req, res) => {
    try {
      let offset = 0;
      const limit = 1000;
      let allRecords = [];
      let latestProcessedBlock = null;

      while (true) {
        const query = `
          query GetAnalytics($limit: Int!, $offset: Int!) {
            chain_metadata {
              latest_processed_block
            }
            RegisteredProof(limit: $limit, offset: $offset) {
              registeredBy
              blockTimestamp
            }
          }
        `;

        const response = await fetch(`${ENVIO_GRAPHQL_URL}/v1/graphql`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, variables: { limit, offset } }),
          signal: AbortSignal.timeout(6000)
        });

        if (!response.ok) {
          return res.status(503).json({
            success: false,
            error: "Envio indexer unavailable"
          });
        }

        const json = await response.json();
        if (json.errors) {
          return res.status(502).json({
            success: false,
            error: "Envio GraphQL error",
            details: json.errors
          });
        }

        if (offset === 0 && json?.data?.chain_metadata?.[0]) {
          latestProcessedBlock = json.data.chain_metadata[0].latest_processed_block;
        }

        const records = json?.data?.RegisteredProof || [];
        allRecords.push(...records);

        // Stop if we got less than limit (end of data) or hit hard cap of 10k
        if (records.length < limit || allRecords.length >= 10000) {
          break;
        }
        offset += limit;
      }

      const total = allRecords.length;
      const isPartial = allRecords.length >= 10000;
      
      const byIssuer = {};
      const byDate = {};

      for (const r of allRecords) {
        const issuer = r.registeredBy;
        byIssuer[issuer] = (byIssuer[issuer] || 0) + 1;
        
        const ts = Number(r.blockTimestamp) * 1000;
        const date = new Date(ts).toISOString().split('T')[0];
        byDate[date] = (byDate[date] || 0) + 1;
      }

      res.json({
        success: true,
        total,
        isPartial,
        byIssuer,
        byDate,
        latestProcessedBlock,
        envioUrl: ENVIO_GRAPHQL_URL
      });

    } catch (error) {
      const isTimeout = error.name === "TimeoutError" || error.message.includes("timeout");
      res.status(isTimeout ? 504 : 503).json({
        success: false,
        error: isTimeout ? "Envio indexer timed out" : "Envio indexer unreachable"
      });
    }
  }
);

app.get(
  "/proof-history",
  async (req, res) => {

    try {

      const limit =
        Math.min(
          parseInt(req.query.limit) || 20,
          100
        );

      const offset =
        parseInt(req.query.offset) || 0;


      const query = `
        query ProofHistory($limit: Int!, $offset: Int!) {
          RegisteredProof(
            order_by: { blockNumber: desc }
            limit: $limit
            offset: $offset
          ) {
            id
            proofHash
            sourceId
            registeredBy
            blockTimestamp
            blockNumber
            txHash
          }
        }
      `;


      const response =
        await fetch(
          `${ENVIO_GRAPHQL_URL}/v1/graphql`,
          {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({
              query,
              variables: { limit, offset }
            }),
            signal: AbortSignal.timeout(6000)
          }
        );


      if (!response.ok) {

        return res.status(503).json({

          success: false,

          error:
            "Envio indexer unavailable",

          hint:
            `Start the indexer: cd Blockchain/envio && pnpm dev. ` +
            `Or set ENVIO_GRAPHQL_URL to your deployed Envio endpoint.`

        });

      }


      const json =
        await response.json();


      if (json.errors) {

        return res.status(502).json({

          success: false,

          error:
            "Envio GraphQL error",

          details:
            json.errors

        });

      }


      const records =
        (json?.data?.RegisteredProof || []).map(
          (r) => ({
            proofHash:      r.proofHash,
            sourceId:       r.sourceId,
            registeredBy:   r.registeredBy,
            blockTimestamp: r.blockTimestamp.toString(),
            blockNumber:    r.blockNumber.toString(),
            txHash:         r.txHash,
            explorerUrl:
              `https://testnet.monadexplorer.com/tx/${r.txHash}`
          })
        );


      res.json({

        success: true,

        count:   records.length,

        limit,

        offset,

        envioUrl:
          ENVIO_GRAPHQL_URL,

        note:
          "History from Envio event index. " +
          "For authoritative proof existence use /verify-proof/:hash.",

        records

      });


    } catch (error) {

      const isTimeout =
        error.name === "TimeoutError" ||
        error.message.includes("timeout");

      res.status(isTimeout ? 504 : 503).json({

        success: false,

        error: isTimeout
          ? "Envio indexer timed out"
          : "Envio indexer unreachable",

        hint:
          `Start the indexer: cd Blockchain/envio && pnpm dev. ` +
          `Or set ENVIO_GRAPHQL_URL in .env to your deployed endpoint.`

      });

    }

  }
);


// ======================================================
// START SERVER
// ======================================================

const PORT = process.env.PORT || 3000;

const server = app.listen(PORT);

server.on(
  "listening",
  () => {

    console.log();

    console.log(
      "=========================================="
    );

    console.log(
      "DaanDristi Blockchain API"
    );

    console.log(
      "=========================================="
    );

    console.log(
      `API running on http://localhost:${PORT}`
    );

    console.log(
      "Wallet:",
      wallet.address
    );

    console.log(
      "Contract:",
      CONTRACT_ADDRESS
    );

    console.log(
      "SQLite (source of truth):",
      SQLITE_FILE
    );

    console.log(
      "ML Log (audit only):",
      ML_LOG_FILE
    );

    try {

      const provenance =
        proof.getCurrentProvenance();

      console.log(
        "Issuer:",
        provenance.issuer
      );

      console.log(
        "Device ID:",
        provenance.deviceId
      );

      console.log(
        "Model ID:",
        provenance.modelId
      );

      console.log(
        "Model hash:",
        provenance.modelHash
      );

    } catch (err) {

      console.warn(
        "Provenance unavailable:",
        err.message
      );

    }

    console.log(
      "Original Registry:",
      ORIGINAL_RECORDS_FILE
    );

    console.log(
      "=========================================="
    );

    console.log();

  }
);

server.on(
  "error",
  (err) => {

    if (err.code === "EADDRINUSE") {

      console.error(
        `\nFATAL: Port ${PORT} is already in use.`
      );

      console.error(
        `Stop the existing process and try again.\n`
      );

    } else {

      console.error(
        "Server error:",
        err.message
      );

    }

    process.exit(1);

  }
);

module.exports = { app, server };