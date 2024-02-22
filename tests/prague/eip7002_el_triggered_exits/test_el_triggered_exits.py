"""
abstract: Tests [EIP-7002: Execution layer triggerable exits](https://eips.ethereum.org/EIPS/eip-7002)
    Test execution layer triggered exits [EIP-7002: Execution layer triggerable exits](https://eips.ethereum.org/EIPS/eip-7002)

"""  # noqa: E501

from typing import Dict, List

import pytest
from ethereum.crypto.hash import keccak256

from ethereum_test_tools import Account, Block, BlockchainTestFiller, TestAddress, Transaction

from .common import Spec

REFERENCE_SPEC_GIT_PATH = "EIPS/eip-7002.md"
REFERENCE_SPEC_VERSION = "2ade0452efe8124378f35284676ddfd16dd56ecd"

pytestmark = pytest.mark.valid_from("Prague")


def test_multi_block_beacon_root_timestamp_calls(
    blockchain_test: BlockchainTestFiller,
):
    """
    Tests sending some exits to the exit precompile.
    """
    blocks: List[Block] = [
        Block(
            txs=[
                Transaction(
                    to=Spec.VALIDATOR_EXIT_PRECOMPILE_ADDRESS,
                    data=keccak256(b"exit"),
                ),
            ],
        ),
    ]
    pre = {
        TestAddress: Account(
            nonce=0,
            balance=0x10**10,
        ),
    }
    post: Dict = {}

    blockchain_test(
        pre=pre,
        blocks=blocks,
        post=post,
    )
