// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title Verification
/// @notice Anchors a keccak256 digest of (face embedding hash + matched URL +
///         timestamp) on-chain so the local output files can later be proven
///         unmodified. Stores only the digest -- no biometric data goes on
///         chain, which is deliberate: chain writes are permanent and public.
contract Verification {
    struct Record {
        bytes32 dataHash;
        uint256 timestamp;
        address submitter;
    }

    mapping(bytes32 => Record) private records;

    event RecordStored(bytes32 indexed recordId, bytes32 dataHash, uint256 timestamp);

    /// @notice Store a digest under `recordId`. Records are write-once so a
    ///         later run cannot silently overwrite an earlier proof.
    function storeRecord(bytes32 recordId, bytes32 dataHash) external {
        require(dataHash != bytes32(0), "empty hash");
        require(records[recordId].dataHash == bytes32(0), "record exists");
        records[recordId] = Record(dataHash, block.timestamp, msg.sender);
        emit RecordStored(recordId, dataHash, block.timestamp);
    }

    /// @notice Fetch a stored record. dataHash is zero if `recordId` is unknown.
    function getRecord(bytes32 recordId)
        external
        view
        returns (bytes32 dataHash, uint256 timestamp, address submitter)
    {
        Record memory r = records[recordId];
        return (r.dataHash, r.timestamp, r.submitter);
    }

    /// @notice Re-verification: true only if `recordId` exists and matches.
    function verifyRecord(bytes32 recordId, bytes32 dataHash) external view returns (bool) {
        return records[recordId].dataHash != bytes32(0)
            && records[recordId].dataHash == dataHash;
    }
}
