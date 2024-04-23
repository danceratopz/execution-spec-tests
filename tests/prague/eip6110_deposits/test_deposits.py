"""
abstract: Tests [EIP-6110: Supply validator deposits on chain](https://eips.ethereum.org/EIPS/eip-6110)
    Test [EIP-6110: Supply validator deposits on chain](https://eips.ethereum.org/EIPS/eip-6110).
"""  # noqa: E501

import pytest

from ethereum_test_tools import (
    Account,
    Block,
    BlockchainTestFiller,
    Code,
    CodeGasMeasure,
    Environment,
)
from ethereum_test_tools import Opcodes as Op
from ethereum_test_tools import TestAddress, Transaction

from .spec import Spec, ref_spec_6110

REFERENCE_SPEC_GIT_PATH = ref_spec_6110.git_path
REFERENCE_SPEC_VERSION = ref_spec_6110.version


@pytest.mark.valid_from("Prague")
def test_deposit_from_externally_owned_account(
    blockchain_test: BlockchainTestFiller,
):
    """
    Test making a deposit to the beacon chain deposit contract from an externally owned account.
    """

    data = bytes.fromhex(
        "22895118000000000000000000000000000000000000000000000000000000000000008000000000000000000000000000000000000000000000000000000000000000d00000000000000000000000000000000000000000000000000000000000000110a795a3ac3e900589d1cf09787f787b3c167f07ce1a3bf3caa33ec43490b849a9000000000000000000000000000000000000000000000000000000000000003095da32612ae7bf24dd1ca417dd7e47864e8348add0350a74affcd9b4326ed62ca03059c735a13dd7e6128beff8385abc00000000000000000000000000000000000000000000000000000000000000200100000000000000000000000d5311540bc5986b18e3a417117a8a1b84f81887000000000000000000000000000000000000000000000000000000000000006089c07596df32e14c35de654634c16adf1e1ddf23f1c80460feab01c6ad152a3c13fc31cbe3021403c2cdcb136aacd79d165c0a822a4d4460dddc393501f89a77acf96c1b1f4517b91e6a47a11d088240f3ca509274f5cd6a6e325875e8b23ecc"
    )

    pre = {
        TestAddress: Account(balance=33_000_000_000_000_000_000),
    }

    tx = Transaction(
        gas_limit=110_000,
        gas_price=0x07,
        to=Spec.DEPOSIT_CONTRACT_ADDRESS,
        value=32_000_000_000_000_000_000,
        data=data,
    )

    blockchain_test(
        genesis_environment=Environment(),
        pre=pre,
        post={},
        blocks=[
            Block(
                txs=[tx],
            )
        ],
    )
