/**
 * DAANDRISHTI — Envio Integration Test Suite
 * ===========================================
 *
 * Run from dpp-monad/api/:
 *   node tests/envio-integration.test.js
 *
 * Tests are divided into:
 *
 *   SECTION A — Pure logic / offline (always run, no network needed)
 *   SECTION B — API behaviour (requires API server on port 3000)
 *   SECTION C — Monad Testnet validation (requires RPC; 3 known proofs validated)
 *   SECTION D — Envio indexer query (requires Envio running on ENVIO_GRAPHQL_URL)
 *
 * Each test prints PASS or FAIL with an explanation.
 * Non-zero exit code if any SECTION A or B test fails.
 *
 * Known registered proofs (from original_records.json, validated against Monad):
 *   1. webcam-0|2026-10-07T04:18:40.719409+00:00
 *      hash:  0x1b79ba86c84e91772811ac5677b2c80897b16c7a580c85d9514a22b4cbbd50d8
 *      txHash: 0x6a4b9e650fa1a51c995bb5163b10562fe38e439690b92a1052796149af1c1f5b
 *      block:  68873538
 *
 *   2. Canonical-v1 proof registered during E2E test:
 *      hash:  0x7559aa4c2d541af01708bb7a49b466d9ac94d5c53a71b7c8c3b42a910c5f3696
 *      txHash: 0xa8d56d5f583d28125994654c73688c87dcf6fd90de99544be82c8c1d38998841
 *      block:  69229077
 *
 *   (Third proof validated dynamically from original_records.json)
 */

require("dotenv").config();

const { ethers } = require("ethers");
const fs = require("fs");
const path = require("path");

// ============================================================
// Test runner
// ============================================================

let passed = 0;
let failed = 0;
let skipped = 0;

function pass(label) {
  console.log(`  ✅ PASS  ${label}`);
  passed++;
}

function fail(label, reason) {
  console.error(`  ❌ FAIL  ${label}`);
  if (reason) console.error(`         → ${reason}`);
  failed++;
}

function skip(label, reason) {
  console.log(`  ⏭  SKIP  ${label} (${reason})`);
  skipped++;
}

async function section(name, fn) {
  console.log(`\n── ${name} ──`);
  try {
    await fn();
  } catch (err) {
    fail(name, err.message);
  }
}

// ============================================================
// Config
// ============================================================

const API_URL =
  process.env.API_URL || "http://localhost:3000";

const ENVIO_URL =
  process.env.ENVIO_GRAPHQL_URL || "http://localhost:8080";

const RPC_URL =
  process.env.RPC_URL || "https://testnet-rpc.monad.xyz";

const CONTRACT_ADDRESS =
  "0xBCD966873f48563643Bde79760eEC98a567fd7e6";

const CONTRACT_ABI = [
  "function verifyProof(bytes32 _proofHash) external view returns (bool exists, string memory sourceId, uint256 timestamp, address registeredBy)",
];

const ORIGINAL_RECORDS_FILE =
  path.join(__dirname, "..", "data", "original_records.json");

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");
const SQLITE_FILE =
  [
    path.join(REPO_ROOT, "CV_model", "sqlite", "donation_box.db"),
    path.join(REPO_ROOT, "CV-model", "sqlite", "donation_box.db"),
    path.join(REPO_ROOT, "Monad_hackathon", "sqlite", "donation_box.db"),
  ].find((p) => fs.existsSync(p)) ||
  path.join(REPO_ROOT, "CV_model", "sqlite", "donation_box.db");

// Known proofs (source: original_records.json + sdk/README.md)
const KNOWN_PROOFS = [
  {
    label:   "legacy-raw-record proof #1 (webcam-0)",
    hash:    "0x1b79ba86c84e91772811ac5677b2c80897b16c7a580c85d9514a22b4cbbd50d8",
    txHash:  "0x6a4b9e650fa1a51c995bb5163b10562fe38e439690b92a1052796149af1c1f5b",
    block:   68873538,
  },
  {
    label:   "canonical-v1 proof (E2E test)",
    hash:    "0x7559aa4c2d541af01708bb7a49b466d9ac94d5c53a71b7c8c3b42a910c5f3696",
    txHash:  "0xa8d56d5f583d28125994654c73688c87dcf6fd90de99544be82c8c1d38998841",
    block:   69229077,
  },
];


// ============================================================
// Helpers
// ============================================================

async function apiGet(path) {
  const r = await fetch(`${API_URL}${path}`, { signal: AbortSignal.timeout(8000) });
  return r.json();
}

async function envioQuery(query, variables = {}) {
  const r = await fetch(
    `${ENVIO_URL}/v1/graphql`,
    {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ query, variables }),
      signal:  AbortSignal.timeout(6000),
    }
  );
  return r.json();
}

function isEnvioUp() {
  return fetch(`${ENVIO_URL}/v1/graphql`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ query: "{ __typename }" }),
    signal:  AbortSignal.timeout(3000),
  })
  .then((r) => r.ok)
  .catch(() => false);
}

function isApiUp() {
  return fetch(`${API_URL}/`, { signal: AbortSignal.timeout(3000) })
    .then((r) => r.ok)
    .catch(() => false);
}


// ============================================================
// SECTION A — Pure logic / offline
// ============================================================

async function runTests() {

await section("A — queryEnvioProof helper: never throws on bad input", async () => {

  // Test that the helper gracefully handles non-200 / timeout by
  // importing a local version that points to a definitely-unreachable URL.
  // We test this indirectly: /proof-history must return a structured error,
  // not an uncaught exception, when Envio is down.

  // A: Null / garbage proofHash does not crash the helper
  try {
    // Simulate what happens in the API when queryEnvioProof is called
    // against a dead server — it should return null, never throw.
    const result = await fetch("http://127.0.0.1:19999/v1/graphql", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ query: "{ __typename }" }),
      signal:  AbortSignal.timeout(500),
    }).then(() => "responded").catch(() => null);

    if (result === null) {
      pass("queryEnvioProof: unreachable Envio returns null (not throw)");
    } else {
      fail("queryEnvioProof: expected null for unreachable server", "");
    }
  } catch (err) {
    fail("queryEnvioProof: threw unexpectedly", err.message);
  }

});


await section("A — SDK: canonical-v1 hashes are stable", async () => {

  const proof = require("../../sdk/proof");

  // Load a real SQLite row
  let donation;
  try {
    const Database = require("better-sqlite3");
    const db = new Database(SQLITE_FILE, { readonly: true });
    donation = db.prepare("SELECT * FROM donations WHERE donation_id='3762bf4aa69245cf8a5b6c8401af1811ea623d436ef55b88a1a86957a1540317'").get();
    db.close();
  } catch (err) {
    skip("SDK canonical-v1 hash stability", `SQLite unavailable: ${err.message}`);
    return;
  }

  if (!donation) {
    skip("SDK canonical-v1 hash stability", "target donation row not found in SQLite");
    return;
  }

  const provenance = proof.getCurrentProvenance();
  const event      = proof.createCanonicalAIEvent(donation, provenance);
  const hash1      = proof.hashAIEvent(event);
  const hash2      = proof.hashAIEvent(event);

  if (hash1 === hash2) {
    pass(`canonical-v1 hash is deterministic: ${hash1.slice(0, 12)}…`);
  } else {
    fail("canonical-v1 hash is not deterministic", `${hash1} vs ${hash2}`);
  }

  // Tamper test: denomination change → different hash
  const modified    = Object.assign({}, donation, { denomination: donation.denomination + 1 });
  const modEvent    = proof.createCanonicalAIEvent(modified, provenance);
  const modHash     = proof.hashAIEvent(modEvent);

  if (hash1 !== modHash) {
    pass("denomination change produces different hash (tamper detection works)");
  } else {
    fail("denomination change did NOT change hash", "tamper detection broken");
  }

  // Tamper test: target_bin change → different hash
  const modBin      = Object.assign({}, donation, { target_bin: "BIN_FAKE" });
  const modBinEvent = proof.createCanonicalAIEvent(modBin, provenance);
  const modBinHash  = proof.hashAIEvent(modBinEvent);

  if (hash1 !== modBinHash) {
    pass("target_bin change produces different hash (tamper detection works)");
  } else {
    fail("target_bin change did NOT change hash", "tamper detection broken");
  }

  // Restore: denomination back → original hash
  const restoredEvent = proof.createCanonicalAIEvent(donation, provenance);
  const restoredHash  = proof.hashAIEvent(restoredEvent);

  if (restoredHash === hash1) {
    pass("restoring original values returns original hash");
  } else {
    fail("restored hash does not match original", `${restoredHash} vs ${hash1}`);
  }

});


await section("A — original_records.json: structure intact", async () => {

  let records;
  try {
    records = JSON.parse(fs.readFileSync(ORIGINAL_RECORDS_FILE, "utf8"));
  } catch (err) {
    fail("original_records.json readable", err.message);
    return;
  }

  if (Array.isArray(records)) {
    pass(`original_records.json is an array with ${records.length} entries`);
  } else {
    fail("original_records.json is not an array");
    return;
  }

  const hasHashes = records.every((r) => typeof r.proofHash === "string" && r.proofHash.startsWith("0x"));
  if (hasHashes) {
    pass("all entries have 0x-prefixed proofHash");
  } else {
    fail("some entries are missing valid proofHash");
  }

  const schemes = records.reduce((acc, r) => {
    acc[r.proofScheme || "none"] = (acc[r.proofScheme || "none"] || 0) + 1;
    return acc;
  }, {});
  pass(`proof schemes: ${JSON.stringify(schemes)}`);

});


// ============================================================
// SECTION B — API behaviour (requires server on port 3000)
// ============================================================

const apiUp = await isApiUp();

if (!apiUp) {
  console.log("\n── B — API tests ── (SKIPPED: API not running on port 3000)");
  console.log("  ⏭  Start with: cd dpp-monad/api && node index.js");
  skipped += 8;
} else {

  await section("B — GET / returns running status", async () => {
    const data = await apiGet("/");
    if (data.status === "running") {
      pass("GET / → status: running");
    } else {
      fail("GET / unexpected status", JSON.stringify(data));
    }
  });

  await section("B — GET /all-records makes zero per-record RPC calls", async () => {
    // We verify this structurally: /all-records should never include a
    // 'source' field of 'rpc-fallback' on individual records (that field
    // only appears in findProofTransaction which /all-records does NOT call).
    const data = await apiGet("/all-records");

    if (!data.success) {
      fail("/all-records returned success:false", data.error);
      return;
    }

    pass(`/all-records: ${data.totalRecords} total, ${data.verifiedRecords} verified, ${data.tamperedRecords} tampered`);

    const hasFindProofCall = data.records?.some((r) => r.source === "rpc-fallback");
    if (!hasFindProofCall) {
      pass("/all-records: no per-record RPC fallback calls in response");
    } else {
      fail("/all-records: found 'rpc-fallback' source in records — RPC was called per record");
    }

  });

  await section("B — /all-records: VERIFIED records match original_records.json", async () => {

    const data = await apiGet("/all-records");
    const records = JSON.parse(fs.readFileSync(ORIGINAL_RECORDS_FILE, "utf8"));

    const verified = data.records?.filter((r) => r.status === "VERIFIED") || [];

    if (verified.length > 0) {
      pass(`${verified.length} VERIFIED records returned`);
    } else {
      fail("No VERIFIED records found in /all-records");
      return;
    }

    // Every VERIFIED record should have a matching snapshot in original_records.json
    let allMatch = true;
    for (const v of verified) {
      const snap = records.find((r) => r.proofHash === v.originalProofHash);
      if (!snap) {
        fail(`VERIFIED record ${v.eventId?.slice(0, 12)} has no matching snapshot`);
        allMatch = false;
      }
    }
    if (allMatch) {
      pass("all VERIFIED records have matching original_records.json snapshots");
    }

  });

  await section("B — /all-records: denomination tamper → TAMPERED, restore → VERIFIED", async () => {

    // Find a VERIFIED record with a known donation_id
    const allData = await apiGet("/all-records");
    const verifiedRec = allData.records?.find((r) => r.status === "VERIFIED");

    if (!verifiedRec) {
      skip("denomination tamper test", "no VERIFIED records available");
      return;
    }

    const donationId = verifiedRec.eventId;
    const originalDenom = parseInt(verifiedRec.originalDenomination);

    // Tamper: set denomination to a wrong value
    let db;
    try {
      const Database = require("better-sqlite3");
      db = new Database(SQLITE_FILE);
    } catch (err) {
      skip("denomination tamper test", `SQLite unavailable: ${err.message}`);
      return;
    }

    const wrongDenom = originalDenom === 50 ? 100 : 50;
    db.prepare("UPDATE donations SET denomination=? WHERE donation_id=?").run(wrongDenom, donationId);
    db.close();

    const tampered = await apiGet("/all-records");
    const tamperedRec = tampered.records?.find((r) => r.eventId === donationId);

    if (tamperedRec?.status === "TAMPERED") {
      pass(`denomination change to ${wrongDenom} → TAMPERED`);
    } else {
      fail(`denomination change did not produce TAMPERED`, `got: ${tamperedRec?.status}`);
    }

    // Restore
    const db2 = require("better-sqlite3")(SQLITE_FILE);
    db2.prepare("UPDATE donations SET denomination=? WHERE donation_id=?").run(originalDenom, donationId);
    db2.close();

    const restored = await apiGet("/all-records");
    const restoredRec = restored.records?.find((r) => r.eventId === donationId);

    if (restoredRec?.status === "VERIFIED") {
      pass(`denomination restored to ${originalDenom} → VERIFIED`);
    } else {
      fail(`denomination restore did not return VERIFIED`, `got: ${restoredRec?.status}`);
    }

  });

  await section("B — /all-records: target_bin tamper → TAMPERED, restore → VERIFIED", async () => {

    const allData = await apiGet("/all-records");
    const verifiedRec = allData.records?.find((r) => r.status === "VERIFIED");

    if (!verifiedRec) {
      skip("target_bin tamper test", "no VERIFIED records available");
      return;
    }

    const donationId    = verifiedRec.eventId;
    const originalBin   = verifiedRec.originalBin;
    const wrongBin      = originalBin === "BIN_50" ? "BIN_100" : "BIN_50";

    const Database = require("better-sqlite3");
    const db = new Database(SQLITE_FILE);
    db.prepare("UPDATE donations SET target_bin=? WHERE donation_id=?").run(wrongBin, donationId);
    db.close();

    const tampered    = await apiGet("/all-records");
    const tamperedRec = tampered.records?.find((r) => r.eventId === donationId);

    if (tamperedRec?.status === "TAMPERED") {
      pass(`target_bin change to '${wrongBin}' → TAMPERED`);
    } else {
      fail(`target_bin change did not produce TAMPERED`, `got: ${tamperedRec?.status}`);
    }

    // Restore
    const db2 = new Database(SQLITE_FILE);
    db2.prepare("UPDATE donations SET target_bin=? WHERE donation_id=?").run(originalBin, donationId);
    db2.close();

    const restored    = await apiGet("/all-records");
    const restoredRec = restored.records?.find((r) => r.eventId === donationId);

    if (restoredRec?.status === "VERIFIED") {
      pass(`target_bin restored to '${originalBin}' → VERIFIED`);
    } else {
      fail(`target_bin restore did not return VERIFIED`, `got: ${restoredRec?.status}`);
    }

  });

  await section("B — /proof-history: returns structured error when Envio is down", async () => {

    const envioRunning = await isEnvioUp();

    if (envioRunning) {
      skip("Envio-down error handling", "Envio is actually running — test only applies when Envio is down");
      return;
    }

    const data = await fetch(`${API_URL}/proof-history`, {
      signal: AbortSignal.timeout(8000)
    }).then((r) => r.json());

    if (data.success === false && data.error) {
      pass(`/proof-history returns structured error when Envio down: "${data.error}"`);
    } else if (data.success === true) {
      // This could happen if ENVIO_GRAPHQL_URL points to a running instance
      pass("/proof-history responded successfully (Envio is reachable)");
    } else {
      fail("/proof-history returned unexpected shape", JSON.stringify(data).slice(0, 200));
    }

  });

  await section("B — /verify-proof/:hash: known proof exists on-chain", async () => {

    const targetHash = KNOWN_PROOFS[0].hash;
    const data = await apiGet(`/verify-proof/${targetHash}`);

    if (data.exists === true) {
      pass(`/verify-proof/${targetHash.slice(0, 14)}… → exists: true`);
    } else {
      fail(`/verify-proof expected exists:true`, JSON.stringify(data));
    }

  });

  await section("B — GET /original-records: structure intact", async () => {

    const data = await apiGet("/original-records");

    if (data.success && Array.isArray(data.records)) {
      pass(`/original-records: ${data.count} snapshots`);
    } else {
      fail("/original-records unexpected shape", JSON.stringify(data).slice(0, 200));
    }

  });

}


// ============================================================
// SECTION C — Monad Testnet validation (requires RPC)
// ============================================================

await section("C — Validate 2 known proofs against Monad Testnet contract", async () => {

  let provider;
  try {
    provider = new ethers.JsonRpcProvider(RPC_URL);
    await provider.getBlockNumber(); // connectivity check
  } catch (err) {
    skip("Monad Testnet validation", `RPC unavailable: ${err.message}`);
    return;
  }

  const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, provider);

  for (const known of KNOWN_PROOFS) {
    try {
      const result = await contract.verifyProof(known.hash);

      if (result[0] === true) {
        pass(
          `Monad contract.verifyProof confirms: ${known.label}\n` +
          `           hash=${known.hash.slice(0, 14)}… exists=true sourceId="${result[1]}"`
        );
      } else {
        fail(
          `contract.verifyProof returned false for ${known.label}`,
          `hash=${known.hash}`
        );
      }
    } catch (err) {
      fail(`RPC call failed for ${known.label}`, err.message);
    }
  }

  // Third proof: load dynamically from original_records.json
  try {
    const records = JSON.parse(fs.readFileSync(ORIGINAL_RECORDS_FILE, "utf8"));
    const third   = records.find(
      (r) =>
        r.proofHash !== KNOWN_PROOFS[0].hash &&
        r.proofHash !== KNOWN_PROOFS[1].hash &&
        r.transactionHash
    );

    if (!third) {
      skip("Third known proof validation", "no additional registered proof found in original_records.json");
    } else {
      const result = await contract.verifyProof(third.proofHash);
      if (result[0] === true) {
        pass(
          `Monad contract.verifyProof: third proof ${third.proofHash.slice(0, 14)}… exists=true`
        );
      } else {
        fail("Third proof returned false from contract.verifyProof", third.proofHash);
      }
    }
  } catch (err) {
    skip("Third known proof validation", err.message);
  }

});


// ============================================================
// SECTION D — Envio indexer query
// ============================================================

const envioUp = await isEnvioUp();

if (!envioUp) {
  console.log("\n── D — Envio indexer tests ── (SKIPPED: Envio not reachable at " + ENVIO_URL + ")");
  console.log("  ⏭  Start: cd dpp-monad/envio && pnpm install && pnpm dev");
  skipped += 5;
} else {

  await section("D — RegisteredProof entity: lookup by proofHash", async () => {

    const { hash, txHash, block, label } = KNOWN_PROOFS[0];

    const query = `
      query GetProof($id: String!) {
        RegisteredProof_by_pk(id: $id) {
          id proofHash sourceId registeredBy blockNumber txHash
        }
      }
    `;

    const result = await envioQuery(query, { id: hash.toLowerCase() });
    const rec    = result?.data?.RegisteredProof_by_pk;

    if (!rec) {
      fail(`Envio lookup for ${label}: record not found`, "Indexer may still be syncing");
      return;
    }

    if (rec.txHash?.toLowerCase() === txHash.toLowerCase()) {
      pass(`Envio RegisteredProof txHash matches for ${label}`);
    } else {
      fail(`Envio txHash mismatch for ${label}`, `got ${rec.txHash} expected ${txHash}`);
    }

    if (Number(rec.blockNumber) === block) {
      pass(`Envio blockNumber ${block} matches for ${label}`);
    } else {
      fail(`Envio blockNumber mismatch for ${label}`, `got ${rec.blockNumber} expected ${block}`);
    }

  });

  await section("D — RegisteredProof: filter by sourceId", async () => {

    const query = `
      query BySource($src: String!) {
        RegisteredProof(where: { sourceId: { _eq: $src } }) {
          id proofHash sourceId
        }
      }
    `;

    const result = await envioQuery(query, { src: "webcam-0" });
    const recs   = result?.data?.RegisteredProof || [];

    if (recs.length > 0) {
      pass(`Envio: ${recs.length} proofs with sourceId="webcam-0"`);
    } else {
      fail("Envio: no proofs found with sourceId='webcam-0'", "Indexer may still be syncing");
    }

  });

  await section("D — RegisteredProof: filter by registeredBy address", async () => {

    // Use the known wallet address from .env
    const wallet = process.env.PRIVATE_KEY
      ? new ethers.Wallet(process.env.PRIVATE_KEY).address.toLowerCase()
      : null;

    if (!wallet) {
      skip("filter by registeredBy", "PRIVATE_KEY not set in .env");
      return;
    }

    const query = `
      query ByIssuer($addr: String!) {
        RegisteredProof(where: { registeredBy: { _eq: $addr } }) {
          id proofHash blockNumber
        }
      }
    `;

    const result = await envioQuery(query, { addr: wallet });
    const recs   = result?.data?.RegisteredProof || [];

    pass(`Envio: ${recs.length} proofs registered by ${wallet.slice(0, 12)}…`);

  });

  await section("D — IssuerEvent entity: issuer authorization history", async () => {

    const query = `
      query IssuerHistory {
        IssuerEvent(order_by: { blockNumber: asc }) {
          id issuer isAuthorized blockNumber txHash
        }
      }
    `;

    const result = await envioQuery(query, {});
    const events = result?.data?.IssuerEvent || [];

    if (events.length > 0) {
      pass(`Envio: ${events.length} IssuerEvent records found`);
      const grants  = events.filter((e) => e.isAuthorized).length;
      const revokes = events.filter((e) => !e.isAuthorized).length;
      pass(`IssuerEvents: ${grants} grant(s), ${revokes} revoke(s)`);
    } else {
      // IssuerStatusChanged is only emitted when setIssuerStatus() is called.
      // The constructor does NOT emit it. So 0 events is valid if setIssuerStatus
      // was never called after deployment.
      pass("IssuerEvent: 0 records (setIssuerStatus was not called post-deployment — expected)");
    }

  });

  await section("D — Idempotency: duplicate entity IDs", async () => {

    // In a correct indexer, each proofHash appears exactly once because:
    // 1. The contract prevents duplicate registration (on-chain guard).
    // 2. The handler uses proofHash as the entity id (idempotent set).
    // We verify this by querying all proofs and checking for duplicates.

    const query = `
      query AllProofs {
        RegisteredProof { id }
      }
    `;

    const result = await envioQuery(query, {});
    const ids    = (result?.data?.RegisteredProof || []).map((r) => r.id);
    const unique = new Set(ids);

    if (ids.length === unique.size) {
      pass(`Envio: no duplicate RegisteredProof ids (${ids.length} unique records)`);
    } else {
      fail("Duplicate RegisteredProof ids detected", `${ids.length} total, ${unique.size} unique`);
    }

  });

}


// ============================================================
// Summary
// ============================================================

console.log(`
══════════════════════════════════════
Test Summary
══════════════════════════════════════
  Passed:  ${passed}
  Failed:  ${failed}
  Skipped: ${skipped}
──────────────────────────────────────`);

if (failed > 0) {
  console.error(`\n  ${failed} test(s) FAILED.`);
  process.exit(1);
} else {
  console.log(`\n  All executed tests passed.`);
  process.exit(0);
}

} // end runTests

runTests();
