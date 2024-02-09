"""
Test suite for `ethereum_test_tools.consume.types` module, with a focus on the engine payloads.
"""

import json
import os
from typing import Any, Dict, Type, Union

import pytest

from ethereum_test_forks import Cancun, Fork, Paris, Shanghai
from ethereum_test_tools.common import Account, Environment, Hash, TestAddress, Transaction
from ethereum_test_tools.common.json import load_dataclass_from_json
from ethereum_test_tools.consume.engine.types import EngineCancun, EngineParis, EngineShanghai
from ethereum_test_tools.spec import BlockchainTest
from ethereum_test_tools.spec.blockchain.types import Block, Fixture, FixtureBlock
from evm_transition_tool import FixtureFormats, GethTransitionTool


def remove_info(fixture_json: Dict[str, Any]):  # noqa: D103
    for t in fixture_json:
        if "_info" in fixture_json[t]:
            del fixture_json[t]["_info"]


common_execution_payload_fields = {
    "coinbase": "0x2adc25665018aa1fe0e6bc666dac8fc2697ff9ba",
    "receipts_root": "0xc598f69a5674cae9337261b669970e24abc0b46e6d284372a239ec8ccbf20b0a",
    "logs_bloom": "0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",  # noqa: E501
    "prev_randao": "0x0000000000000000000000000000000000000000000000000000000000000000",
    "number": 1,
    "gas_limit": 100000000000000000,
    "gas_used": 43105,
    "timestamp": 12,
    "extra_data": "0x",
    "base_fee_per_gas": 7,
    "transactions": [
        "0xf861800a8405f5e10094100000000000000000000000000000000000000080801ba07e09e26678ed4fac08a249ebe8ed680bf9051a5e14ad223e4b2b9d26e0208f37a05f6e3f188e3e6eab7d7d3b6568f5eac7d687b08d307d3154ccd8c87b4630509b"  # noqa: E501
    ],
}


@pytest.mark.parametrize(
    "fork, expected_json_fixture, expected_engine_new_payload, expected_enp_json",
    [
        (
            Paris,
            "valid_simple_paris_blockchain.json",
            EngineParis.NewPayloadV1(
                execution_payload=EngineParis.ExecutionPayloadV1(
                    **common_execution_payload_fields,  # type: ignore
                    parent_hash=Hash(
                        0x86C6DC9CB7B8ADA9E27B1CF16FD81F366A0AD8127F42FF13D778EB2DDF7EAA90
                    ),
                    state_root=Hash(
                        0x19919608275963E6E20A1191996F5B19DB8208DD8DF54097CFD2B9CB14F682B6
                    ),
                    block_hash=Hash(
                        0xE9694E4B99986D312C6891CD7839B73D9E1B451537896818CEFEEAE97D7E3EA6
                    ),
                )
            ),
            "valid_simple_paris_enp.json",
        ),
        (
            Shanghai,
            "valid_simple_shanghai_blockchain.json",
            EngineShanghai.NewPayloadV2(
                execution_payload=EngineShanghai.ExecutionPayloadV2(
                    **common_execution_payload_fields,  # type: ignore
                    parent_hash=Hash(
                        0xCCB89B5B6043AA73114E6857F0783A02808EA6FF4CABD104A308EB4FE0114A9B
                    ),
                    state_root=Hash(
                        0x19919608275963E6E20A1191996F5B19DB8208DD8DF54097CFD2B9CB14F682B6
                    ),
                    block_hash=Hash(
                        0xC970B6BCF304CD5C71D24548A7D65DD907A24A3B66229378E2AC42677C1EEC2B
                    ),
                    withdrawals=[],
                )
            ),
            "valid_simple_shanghai_enp.json",
        ),
        (
            Cancun,
            "valid_simple_cancun_blockchain.json",
            EngineCancun.NewPayloadV3(
                execution_payload=EngineCancun.ExecutionPayloadV3(
                    **common_execution_payload_fields,  # type: ignore
                    parent_hash=Hash(
                        0xDA9249B7AFF004BCDFADFB5F668899746E36A5EEE8197D1589DEB4A3842251CE
                    ),
                    state_root=Hash(
                        0x3D760432D38FBF795FB9ADDD6B25A692D82498BD5F7B703A6DA6D8647C5B7820
                    ),
                    block_hash=Hash(
                        0x546546D8A2D99B3135A47DEBDFC708E6A2199B8D90E43325D2C0B3ADC3613709
                    ),
                    withdrawals=[],
                    blob_gas_used=0,
                    excess_blob_gas=0,
                ),
                expected_blob_versioned_hashes=[],
                parent_beacon_block_root=Hash(
                    0x0000000000000000000000000000000000000000000000000000000000000000
                ),
            ),
            "valid_simple_cancun_enp.json",
        ),
    ],
)
def test_valid_engine_new_payload_fields(
    fork: Fork,
    expected_json_fixture: str,
    expected_engine_new_payload: Any,
    expected_enp_json: str,
):
    """
    Test ...
    """
    # Create a blockchain test fixture
    t8n = GethTransitionTool()
    blockchain_fixture = BlockchainTest(  # type: ignore
        pre={
            0x1000000000000000000000000000000000000000: Account(code="0x4660015500"),
            TestAddress: Account(balance=1000000000000000000000),
        },
        post={
            "0x1000000000000000000000000000000000000000": Account(
                code="0x4660015500", storage={"0x01": "0x01"}
            ),
        },
        blocks=[
            Block(
                txs=[
                    Transaction(
                        ty=0x0,
                        chain_id=0x0,
                        nonce=0,
                        to="0x1000000000000000000000000000000000000000",
                        gas_limit=100000000,
                        gas_price=10,
                        protected=False,
                    )
                ]
            )
        ],
        genesis_environment=Environment(),
        tag="my_blockchain_test_valid_txs",
        fixture_format=FixtureFormats.BLOCKCHAIN_TEST,
    ).generate(
        t8n=t8n,
        fork=fork,
    )

    # Sanity check the fixture is equal to the expected
    with open(
        os.path.join(
            "src",
            "ethereum_test_tools",
            "tests",
            "test_consume",
            "fixtures",
            expected_json_fixture,
        )
    ) as f:
        expected = json.load(f)
    blockchain_fixture_json = {
        f"000/my_blockchain_test/{fork.name()}": blockchain_fixture.to_json(),
    }
    remove_info(blockchain_fixture_json)
    assert blockchain_fixture_json == expected

    # Load json fixture into Fixture dataclass
    fixture: Fixture
    for _, fixture_data in blockchain_fixture_json.items():
        fixture = load_dataclass_from_json(Fixture, fixture_data)

    # Extract the engine payloads from the fixture blocks
    # Ideally we don't know the fork at this point
    for fixture_block in fixture.blocks:
        fixture_block = load_dataclass_from_json(FixtureBlock, fixture_block)  # type: ignore
        if fixture.fork == "Merge":
            fork = Paris
        else:
            fork = globals()[fixture.fork]
        version = fork.engine_new_payload_version(
            fixture_block.block_header.number, fixture_block.block_header.timestamp  # type: ignore
        )
        PayloadClass: Type[
            Union[
                EngineParis.NewPayloadV1,
                EngineShanghai.NewPayloadV2,
                EngineCancun.NewPayloadV3,
            ]
        ]
        if version == 1:
            PayloadClass = EngineParis.NewPayloadV1
        elif version == 2:
            PayloadClass = EngineShanghai.NewPayloadV2
        elif version == 3:
            PayloadClass = EngineCancun.NewPayloadV3
        else:
            ValueError(f"Unexpected payload version: {version}")
            continue

        engine_new_payload = PayloadClass.from_fixture_block(fixture_block)  # type: ignore

    # Compare the engine payloads with the expected payloads
    assert engine_new_payload == expected_engine_new_payload

    # Check the json representation of the engine payloads that would be sent over json-rpc
    with open(
        os.path.join(
            "src",
            "ethereum_test_tools",
            "tests",
            "test_consume",
            "engine_json",
            expected_enp_json,
        )
    ) as f:
        expected = json.load(f)
    print(engine_new_payload.to_json_rpc())
    assert engine_new_payload.to_json_rpc() == expected
