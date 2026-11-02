const crypto = require("crypto");

const API_URL = "http://localhost:3000";


// ==========================================
// CREATE SHA-256 HASH
// ==========================================

function createProofHash(record) {
  const canonicalData = JSON.stringify(
    Object.keys(record)
      .sort()
      .reduce((obj, key) => {
        obj[key] = record[key];
        return obj;
      }, {})
  );

  return "0x" + crypto
    .createHash("sha256")
    .update(canonicalData, "utf8")
    .digest("hex");
}


// ==========================================
// VERIFY HASH ON BLOCKCHAIN
// ==========================================

async function verifyProof(proofHash) {

  const response = await fetch(
    `${API_URL}/verify-proof/${proofHash}`
  );

  return await response.json();
}


// ==========================================
// MAIN TAMPER TEST
// ==========================================

async function main() {

  console.log();
  console.log("==========================================");
  console.log("DAANDRISTI BLOCKCHAIN TAMPER TEST");
  console.log("==========================================");


  // ----------------------------------------
  // ORIGINAL RECORD
  // ----------------------------------------

  const originalRecord = {
    pocket_id: "webcam-0",
    timestamp: "2026-10-06T18:59:50.657784+00:00",
    item_status: "NOTE_DETECTED",
    predicted_denomination: "100",
    classifier_confidence: 0.8232235908508301,
    confidence: 0.8232235908508301,
    source: "webcam",
    decision: "ACCEPT",
    target_bin: "BIN_100",
    model_version: "mobilenet_v3_clean_v1"
  };


  console.log();
  console.log("========== ORIGINAL RECORD ==========");

  console.log(
    JSON.stringify(originalRecord, null, 2)
  );


  const originalHash = createProofHash(
    originalRecord
  );

  console.log();
  console.log("Original SHA-256:");
  console.log(originalHash);


  // ----------------------------------------
  // VERIFY ORIGINAL
  // ----------------------------------------

  console.log();
  console.log("Checking original hash on Monad...");

  const originalVerification =
    await verifyProof(originalHash);

  console.log();
  console.log("Original verification:");
  console.log(originalVerification);


  // ----------------------------------------
  // CREATE TAMPERED RECORD
  // ----------------------------------------

  const tamperedRecord = {
    ...originalRecord,

    // Someone changes ₹100 to ₹500
    predicted_denomination: "500",

    // Target bin also changed
    target_bin: "BIN_500"
  };


  console.log();
  console.log("========== TAMPERED RECORD ==========");

  console.log(
    JSON.stringify(tamperedRecord, null, 2)
  );


  const tamperedHash = createProofHash(
    tamperedRecord
  );

  console.log();
  console.log("Tampered SHA-256:");
  console.log(tamperedHash);


  // ----------------------------------------
  // VERIFY TAMPERED RECORD
  // ----------------------------------------

  console.log();
  console.log("Checking tampered hash on Monad...");

  const tamperedVerification =
    await verifyProof(tamperedHash);

  console.log();
  console.log("Tampered verification:");
  console.log(tamperedVerification);


  // ----------------------------------------
  // FINAL RESULT
  // ----------------------------------------

  console.log();
  console.log("==========================================");
  console.log("FINAL TAMPER DETECTION RESULT");
  console.log("==========================================");

  console.log();
  console.log(
    "Original Hash:",
    originalHash
  );

  console.log(
    "Tampered Hash:",
    tamperedHash
  );

  console.log();

  if (
    originalVerification.exists === true &&
    tamperedVerification.exists === false &&
    originalHash !== tamperedHash
  ) {

    console.log("✅ ORIGINAL RECORD: AUTHENTIC");
    console.log("❌ TAMPERED RECORD: DETECTED");

    console.log();
    console.log(
      "Blockchain successfully detected the modification."
    );

  } else {

    console.log(
      "⚠️ Tamper test did not produce the expected result."
    );
  }

  console.log();
  console.log("==========================================");
}


main().catch((error) => {

  console.error();
  console.error("❌ Test failed:");
  console.error(error);

});