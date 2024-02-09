"""
Useful types for consuming EEST test runners or hive simulators.
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Union, cast

from ...common.json import JSONEncoder, field, to_json
from ...common.types import (
    Address,
    Bytes,
    Hash,
    HexNumber,
    blob_versioned_hashes_from_transactions,
)
from ...spec.blockchain.types import Bloom, FixtureBlock


class EngineParis:
    """
    Paris (Merge) Engine API structures:
    https://github.com/ethereum/execution-apis/blob/main/src/engine/paris.md
    """

    @dataclass(kw_only=True)
    class ExecutionPayloadV1:
        """
        Structure of a version 1 execution payload:
        https://github.com/ethereum/execution-apis/blob/main/src/engine/paris.md#executionpayloadv1
        """

        parent_hash: Hash = field(
            json_encoder=JSONEncoder.Field(
                name="parentHash",
            ),
        )
        """
        parentHash: DATA, 32 Bytes
        """
        coinbase: Address = field(
            json_encoder=JSONEncoder.Field(
                name="feeRecipient",
            ),
        )
        """
        feeRecipient: DATA, 20 Bytes
        """
        state_root: Hash = field(
            json_encoder=JSONEncoder.Field(
                name="stateRoot",
            ),
        )
        """
        stateRoot: DATA, 32 Bytes
        """
        receipts_root: Hash = field(
            json_encoder=JSONEncoder.Field(
                name="receiptsRoot",
            ),
        )
        """
        receiptsRoot: DATA, 32 Bytes
        """
        logs_bloom: Bloom = field(
            json_encoder=JSONEncoder.Field(
                name="logsBloom",
            ),
        )
        """
        logsBloom: DATA, 256 Bytes
        """
        prev_randao: Hash = field(
            json_encoder=JSONEncoder.Field(
                name="prevRandao",
            ),
        )
        """
        prevRandao: DATA, 32 Bytes
        """
        number: int = field(
            json_encoder=JSONEncoder.Field(
                name="blockNumber",
                cast_type=HexNumber,
            ),
        )
        """
        blockNumber: QUANTITY, 64 Bits
        """
        gas_limit: int = field(
            json_encoder=JSONEncoder.Field(
                name="gasLimit",
                cast_type=HexNumber,
            ),
        )
        """
        gasLimit: QUANTITY, 64 Bits
        """
        gas_used: int = field(
            json_encoder=JSONEncoder.Field(
                name="gasUsed",
                cast_type=HexNumber,
            ),
        )
        """
        gasUsed: QUANTITY, 64 Bits
        """
        timestamp: int = field(
            json_encoder=JSONEncoder.Field(
                name="timestamp",
                cast_type=HexNumber,
            ),
        )
        """
        timestamp: QUANTITY, 64 Bits
        """
        extra_data: Bytes = field(
            json_encoder=JSONEncoder.Field(
                name="extraData",
            ),
        )
        """
        extraData: DATA, 0 to 32 Bytes
        """
        base_fee_per_gas: int = field(
            json_encoder=JSONEncoder.Field(
                name="baseFeePerGas",
                cast_type=HexNumber,
            ),
        )
        """
        baseFeePerGas: QUANTITY, 64 Bits
        """
        block_hash: Hash = field(
            json_encoder=JSONEncoder.Field(
                name="blockHash",
            ),
        )
        """
        blockHash: DATA, 32 Bytes
        """
        transactions: List[str] = field(
            json_encoder=JSONEncoder.Field(
                name="transactions",
                to_json=True,
            ),
        )
        """
        transactions: Array of DATA
        """

        @classmethod
        def from_fixture_block(
            cls, fixture_block: FixtureBlock
        ) -> "EngineParis.ExecutionPayloadV1":
            """
            Converts a fixture block to a Paris execution payload.
            """
            header = fixture_block.block_header
            transactions = [
                "0x" + tx.serialized_bytes().hex() for tx in fixture_block.transactions
            ]

            kwargs = {
                field.name: getattr(header, field.name)
                for field in fields(header)
                if field.name in {f.name for f in fields(cls)}
            }

            return cls(**kwargs, transactions=transactions)

    @dataclass(kw_only=True)
    class NewPayloadV1:
        """
        Structure of a version 1 engine new payload:
        https://github.com/ethereum/execution-apis/blob/main/src/engine/paris.md#engine_newpayloadv1
        """

        execution_payload: "EngineParis.ExecutionPayloadV1" = field(
            json_encoder=JSONEncoder.Field(
                name="executionPayload",
                to_json=True,
            ),
        )

        @classmethod
        def from_fixture_block(cls, fixture_block: FixtureBlock) -> "EngineParis.NewPayloadV1":
            """
            Creates a Paris engine new payload from a fixture block.
            """
            return EngineParis.NewPayloadV1(
                execution_payload=EngineParis.ExecutionPayloadV1.from_fixture_block(fixture_block)
            )

        @classmethod
        def version(cls) -> int:
            """
            Returns the version of the engine new payload.
            """
            return 1

        def to_json_rpc(self) -> List[Dict[str, Any]]:
            """
            Serializes a Paris engine new payload dataclass to its JSON-RPC representation.
            """
            return [to_json(self.execution_payload)]


class EngineShanghai:
    """
    Shanghai Engine API structures:
    https://github.com/ethereum/execution-apis/blob/main/src/engine/shanghai.md
    """

    @dataclass(kw_only=True)
    class WithdrawalV1:
        """
        Structure of a version 1 withdrawal:
        https://github.com/ethereum/execution-apis/blob/main/src/engine/shanghai.md#withdrawalv1
        """

        index: int = field(
            json_encoder=JSONEncoder.Field(
                name="index",
                cast_type=HexNumber,
            ),
        )
        """
        index: QUANTITY, 64 Bits
        """
        validator_index: int = field(
            json_encoder=JSONEncoder.Field(
                name="validatorIndex",
                cast_type=HexNumber,
            ),
        )
        """
        validatorIndex: QUANTITY, 64 Bits
        """
        address: Address = field(
            json_encoder=JSONEncoder.Field(
                name="address",
            ),
        )
        """
        address: DATA, 20 Bytes
        """
        amount: int = field(
            json_encoder=JSONEncoder.Field(
                name="amount",
                cast_type=HexNumber,
            ),
        )
        """
        amount: QUANTITY, 64 Bits
        """

    @dataclass(kw_only=True)
    class ExecutionPayloadV2(EngineParis.ExecutionPayloadV1):
        """
        Structure of a version 2 execution payload:
        https://github.com/ethereum/execution-apis/blob/main/src/engine/shanghai.md#executionpayloadv2
        """

        withdrawals: List["EngineShanghai.WithdrawalV1"] = field(
            json_encoder=JSONEncoder.Field(
                name="withdrawals",
                to_json=True,
            ),
        )
        """
        withdrawals: Array of WithdrawalV1
        """

        @classmethod
        def from_fixture_block(
            cls, fixture_block: FixtureBlock
        ) -> "EngineShanghai.ExecutionPayloadV2":
            """
            Converts a fixture block to a Shanghai execution payload.
            """
            header = fixture_block.block_header
            transactions = [
                "0x" + tx.serialized_bytes().hex() for tx in fixture_block.transactions
            ]
            withdrawals = cast(List[EngineShanghai.WithdrawalV1], fixture_block.withdrawals)
            kwargs = {
                field.name: getattr(header, field.name)
                for field in fields(header)
                if field.name in {f.name for f in fields(cls)}
            }
            return cls(**kwargs, transactions=transactions, withdrawals=withdrawals)

    @dataclass(kw_only=True)
    class NewPayloadV2:
        """
        Structure of a version 2 engine new payload:
        https://github.com/ethereum/execution-apis/blob/main/src/engine/shanghai.md#engine_newpayloadv2
        """

        execution_payload: Union[
            "EngineShanghai.ExecutionPayloadV2", "EngineParis.ExecutionPayloadV1"
        ] = field(
            json_encoder=JSONEncoder.Field(
                name="executionPayload",
                to_json=True,
            ),
        )

        @classmethod
        def from_fixture_block(cls, fixture_block: FixtureBlock) -> "EngineShanghai.NewPayloadV2":
            """
            Creates a Shanghai engine new payload from a fixture block.
            """
            return EngineShanghai.NewPayloadV2(
                execution_payload=EngineShanghai.ExecutionPayloadV2.from_fixture_block(
                    fixture_block
                )
            )

        @classmethod
        def version(cls) -> int:
            """
            Returns the version of the engine new payload.
            """
            return 2

        def to_json_rpc(self) -> List[Dict[str, Any]]:
            """
            Serializes a Shanghai engine new payload dataclass to its JSON-RPC representation.
            """
            return [to_json(self.execution_payload)]


class EngineCancun:
    """
    Cancun Engine API structures:
    https://github.com/ethereum/execution-apis/blob/main/src/engine/cancun.md
    """

    @dataclass(kw_only=True)
    class ExecutionPayloadV3(EngineShanghai.ExecutionPayloadV2):
        """
        Structure of a version 3 execution payload:
        https://github.com/ethereum/execution-apis/blob/main/src/engine/cancun.md#executionpayloadv3
        """

        blob_gas_used: int = field(
            json_encoder=JSONEncoder.Field(
                name="blobGasUsed",
                cast_type=HexNumber,
            ),
        )
        """
        blobGasUsed: QUANTITY, 64 Bits
        """
        excess_blob_gas: int = field(
            json_encoder=JSONEncoder.Field(
                name="excessBlobGas",
                cast_type=HexNumber,
            ),
        )
        """
        excessBlobGas: QUANTITY, 64 Bits
        """

        @classmethod
        def from_fixture_block(
            cls, fixture_block: FixtureBlock
        ) -> "EngineCancun.ExecutionPayloadV3":
            """
            Converts a fixture block to a Cancun execution payload.
            """
            header = fixture_block.block_header
            transactions = [
                "0x" + tx.serialized_bytes().hex() for tx in fixture_block.transactions
            ]
            withdrawals = cast(List[EngineShanghai.WithdrawalV1], fixture_block.withdrawals)

            kwargs = {
                field.name: getattr(header, field.name)
                for field in fields(header)
                if field.name in {f.name for f in fields(cls)}
            }

            return cls(**kwargs, transactions=transactions, withdrawals=withdrawals)

    @dataclass(kw_only=True)
    class NewPayloadV3:
        """
        Structure of a version 3 engine new payload:
        https://github.com/ethereum/execution-apis/blob/main/src/engine/cancun.md#engine_newpayloadv3
        """

        execution_payload: "EngineCancun.ExecutionPayloadV3" = field(
            json_encoder=JSONEncoder.Field(
                to_json=True,
            ),
        )
        """
        executionPayload: ExecutionPayloadV3
        """
        expected_blob_versioned_hashes: List[Hash]
        """
        expectedBlobVersionedHashes: Array of DATA
        """
        parent_beacon_block_root: Hash
        """
        parentBeaconBlockRoot: DATA, 32 Bytes
        """

        @classmethod
        def from_fixture_block(cls, fixture_block: FixtureBlock) -> "EngineCancun.NewPayloadV3":
            """
            Creates a Cancun engine new payload from a fixture block.
            """
            execution_payload = EngineCancun.ExecutionPayloadV3.from_fixture_block(fixture_block)
            expected_blob_versioned_hashes = blob_versioned_hashes_from_transactions(
                fixture_block.transactions
            )
            parent_beacon_block_root = cast(Hash, fixture_block.block_header.beacon_root)
            return cls(
                execution_payload=execution_payload,
                expected_blob_versioned_hashes=[
                    Hash(blob_versioned_hash)
                    for blob_versioned_hash in expected_blob_versioned_hashes
                ],
                parent_beacon_block_root=parent_beacon_block_root,
            )

        @classmethod
        def version(cls) -> int:
            """
            Returns the version of the engine new payload.
            """
            return 3

        def to_json_rpc(self) -> List[Union[Dict[str, Any], List[Hash], Hash]]:
            """
            Serializes a Cancun engine new payload dataclass to its JSON-RPC representation.
            """
            return [
                to_json(self.execution_payload),
                self.expected_blob_versioned_hashes,
                self.parent_beacon_block_root,
            ]
