// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title DonationProvenanceRegistry
 * @dev Privacy-preserving provenance registry for AI-verified real-world donation events.
 */
contract DonationProvenanceRegistry {
    
    struct Proof {
        bytes32 proofHash;
        string sourceId;
        uint256 timestamp;
        address registeredBy;
    }

    // Maps proofHash -> Proof details
    mapping(bytes32 => Proof) public proofs;

    // Authorized issuers/devices
    mapping(address => bool) public authorizedIssuers;
    address public owner;

    event ProofRegistered(
        bytes32 indexed proofHash,
        string sourceId,
        uint256 timestamp,
        address indexed registeredBy
    );
    event IssuerStatusChanged(address indexed issuer, bool isAuthorized);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can perform this action");
        _;
    }

    modifier onlyAuthorized() {
        require(authorizedIssuers[msg.sender] || msg.sender == owner, "Not an authorized issuer");
        _;
    }

    constructor() {
        owner = msg.sender;
        authorizedIssuers[msg.sender] = true;
    }

    function setIssuerStatus(address _issuer, bool _status) external onlyOwner {
        authorizedIssuers[_issuer] = _status;
        emit IssuerStatusChanged(_issuer, _status);
    }

    /**
     * @dev Registers a cryptographic proof of a donation event.
     * @param _proofHash SHA-256 or Keccak-256 hash of the canonical donation record.
     * @param _sourceId Unique identifier for the registering hardware/app (e.g., "DAAN_BOX_001").
     */
    function registerProof(bytes32 _proofHash, string calldata _sourceId) external onlyAuthorized {
        require(proofs[_proofHash].timestamp == 0, "Error: Proof already exists onchain");
        
        proofs[_proofHash] = Proof({
            proofHash: _proofHash,
            sourceId: _sourceId,
            timestamp: block.timestamp,
            registeredBy: msg.sender
        });

        emit ProofRegistered(_proofHash, _sourceId, block.timestamp, msg.sender);
    }

    /**
     * @dev Verifies whether a proof hash exists onchain.
     */
    function verifyProof(bytes32 _proofHash) external view returns (bool exists, string memory sourceId, uint256 timestamp, address registeredBy) {
        Proof memory p = proofs[_proofHash];
        if (p.timestamp > 0) {
            return (true, p.sourceId, p.timestamp, p.registeredBy);
        }
        return (false, "", 0, address(0));
    }
}