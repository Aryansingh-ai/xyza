// DAANDRISHTI Provenance Indexer — Event Handlers
// ================================================
//
// Handles two events from DonationProvenanceRegistry on Monad Testnet:
//
//   ProofRegistered      → creates/updates a RegisteredProof entity
//   IssuerStatusChanged  → creates an IssuerEvent entity
//
// Key design decisions:
//
//  1. IDEMPOTENCY
//     ProofRegistered: id = proofHash. The on-chain contract already prevents
//     duplicate registrations (requires proofs[hash].timestamp == 0), so a
//     proofHash will appear in exactly one event. Setting by proofHash is safe.
//
//     IssuerStatusChanged: id = txHash + "-" + logIndex. Multiple status changes
//     can occur for the same issuer (grant, revoke, re-grant, etc.), so we store
//     each event as a separate record using the log's unique position as the ID.
//
//  2. NO PROOF REVOCATION
//     The deployed contract has no revokeProof function. IssuerStatusChanged
//     with isAuthorized=false revokes the issuer's ability to register new
//     proofs; it does NOT invalidate previously registered proofs.
//     RegisteredProof records are never deleted.
//
//  3. TAMPER DETECTION IS NOT DONE HERE
//     Envio is a read-only query layer. Cryptographic tamper detection uses the
//     local hash comparison in api/index.js against original_records.json.
//     These handlers only persist what the contract emitted.

const { indexer } = require("envio");


// ======================================================
// ProofRegistered
// ======================================================
//
// event ProofRegistered(
//   bytes32 indexed proofHash,
//   string          sourceId,
//   uint256         timestamp,
//   address indexed registeredBy
// )
//
// Entity id = proofHash (unique by contract invariant).
//

indexer.onEvent({
  contract: "DonationProvenanceRegistry",
  event: "ProofRegistered"
}, async ({ event, context }) => {

  // proofHash comes in as a hex string with 0x prefix from Envio
  const proofHash = event.params.proofHash;

  // In Envio v3 with field_selection.transaction_fields: ["hash"],
  // the transaction hash is available at event.transaction.hash.
  // Use optional chaining + logIndex fallback for robustness.
  const txHash = event.transaction?.hash ?? event.block.hash + "-" + event.logIndex;

  context.RegisteredProof.set({
    id:             proofHash,
    proofHash:      proofHash,
    sourceId:       event.params.sourceId,
    registeredBy:   event.params.registeredBy.toLowerCase(),
    // event.params.timestamp is the contract's block.timestamp (uint256, seconds)
    // stored as BigInt by Envio
    blockTimestamp: event.params.timestamp,
    blockNumber:    BigInt(event.block.number),
    txHash:         txHash,
  });

});


// ======================================================
// IssuerStatusChanged
// ======================================================
//
// event IssuerStatusChanged(address indexed issuer, bool isAuthorized)
//
// Entity id = txHash + "-" + logIndex.
// Multiple status changes per issuer are each stored as a separate event.
//

indexer.onEvent({
  contract: "DonationProvenanceRegistry",
  event: "IssuerStatusChanged"
}, async ({ event, context }) => {

  const txHash = event.transaction?.hash ?? event.block.hash + "-" + event.logIndex;
  const id = `${txHash}-${event.logIndex}`;

  context.IssuerEvent.set({
    id,
    issuer:         event.params.issuer.toLowerCase(),
    isAuthorized:   event.params.isAuthorized,
    blockTimestamp: BigInt(event.block.timestamp),
    blockNumber:    BigInt(event.block.number),
    txHash:         txHash,
  });

});
