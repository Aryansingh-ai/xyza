const crypto = require("crypto");

const API_URL = "http://localhost:3000";

// This is the REAL ₹50 record from your ML system.
const originalRecord = {
  pocket_id: "webcam-0",
  timestamp: "2026-10-06T19:29:19.593290+00:00",
  item_status: "NOTE_DETECTED",
  predicted_denomination: "50",
  classifier_confidence: 0.8438018560409546,
  confidence: 0.8438018560409546,
  source: "webcam",
  decision: "ACCEPT",
  target_bin: "BIN_50",
  model_version: "mobilenet_v3_clean_v1"
};

function createProofHash(record) {
  const sortedRecord = Object.keys(record)
    .sort()
    .reduce((obj, key) => {
      obj[key] = record[key];
      return obj;
    }, {});

  return (
    "0x" +
    crypto
      .createHash("sha256")
      .update(JSON.stringify(sortedRecord))
      .digest("hex")
  );
}

async function verify(hash) {
  const response = await fetch(
    `${API_URL}/verify-proof/${hash}`
  );

  return await response.json();
}

async function main() {

  console.log("\n=================================");
  console.log("DAANDRISTI TAMPER TEST");
  console.log("=================================\n");

  // -----------------------------
  // TEST 1: ORIGINAL
  // -----------------------------

  const originalHash = createProofHash(originalRecord);

  console.log("TEST 1 — ORIGINAL RECORD");
  console.log("-------------------------");
  console.log("Denomination:", originalRecord.predicted_denomination);
  console.log("Target Bin:", originalRecord.target_bin);
  console.log("Hash:", originalHash);

  const originalResult = await verify(originalHash);

  console.log("Blockchain exists:", originalResult.exists);

  if (originalResult.exists) {
    console.log("✅ NOT TAMPERED — ORIGINAL RECORD IS AUTHENTIC");
  } else {
    console.log("❌ ERROR — ORIGINAL RECORD NOT FOUND");
  }

  // -----------------------------
  // TEST 2: TAMPERED
  // -----------------------------

  const tamperedRecord = {
    ...originalRecord,

    // Someone changed the donation information
    predicted_denomination: "100",
    target_bin: "BIN_100"
  };

  const tamperedHash = createProofHash(tamperedRecord);

  console.log("\nTEST 2 — TAMPERED RECORD");
  console.log("-------------------------");
  console.log("Changed denomination: ₹50 → ₹100");
  console.log("Changed bin: BIN_50 → BIN_100");
  console.log("New Hash:", tamperedHash);

  const tamperedResult = await verify(tamperedHash);

  console.log("Blockchain exists:", tamperedResult.exists);

  if (!tamperedResult.exists) {
    console.log("❌ TAMPERED — MODIFICATION DETECTED");
  } else {
    console.log("⚠️ Unexpected: tampered hash exists");
  }

  console.log("\n=================================");
  console.log("FINAL RESULT");
  console.log("=================================");
  console.log("Original record : ✅ AUTHENTIC");
  console.log("Tampered record : ❌ DETECTED");
  console.log("=================================\n");
}

main().catch(console.error);