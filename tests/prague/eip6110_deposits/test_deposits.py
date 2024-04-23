"""
abstract: Tests [EIP-6110: Supply validator deposits on chain](https://eips.ethereum.org/EIPS/eip-6110)
    Test [EIP-6110: Supply validator deposits on chain](https://eips.ethereum.org/EIPS/eip-6110).
"""  # noqa: E501

import pytest

from ethereum_test_tools import (
    Account,
    Address,
    Block,
    BlockchainTestFiller,
    Code,
    CodeGasMeasure,
    DepositRequest,
    Environment,
    Header,
)
from ethereum_test_tools import Opcodes as Op
from ethereum_test_tools import TestAddress, Transaction

from .spec import Spec, ref_spec_6110

REFERENCE_SPEC_GIT_PATH = ref_spec_6110.git_path
REFERENCE_SPEC_VERSION = ref_spec_6110.version

pytestmark = pytest.mark.valid_from("Prague")


def test_deposit_from_externally_owned_account(
    blockchain_test: BlockchainTestFiller,
):
    """
    Test making a deposit to the beacon chain deposit contract from an externally owned account.
    """
    deposit = DepositRequest(
        pubkey=0x01,
        withdrawal_credentials=0x02,
        amount=32_000_000_000,
        signature=0x03,
        index=0x0,
    )

    pre = {
        TestAddress: Account(balance=32_000_000_000_000_000_000 * 100),
    }

    tx = Transaction(
        gas_limit=1_000_000,
        gas_price=0x07,
        to=Spec.DEPOSIT_CONTRACT_ADDRESS,
        value=deposit.value,
        data=deposit.calldata,
    )

    blockchain_test(
        genesis_environment=Environment(),
        pre=pre,
        post={},
        blocks=[
            Block(
                txs=[tx],
                header_verify=Header(
                    requests_root=[deposit],
                ),
            )
        ],
    )


def test_deposit_from_contract(
    blockchain_test: BlockchainTestFiller,
):
    """
    Test making a deposit to the beacon chain deposit contract from a contract.
    """
    deposit = DepositRequest(
        pubkey=0x01,
        withdrawal_credentials=0x02,
        amount=32_000_000_000,
        signature=0x03,
        index=0x0,
    )

    relay_contract_address = Address(0x200)

    relay_contract_code = Op.CALLDATACOPY(0, 0, Op.CALLDATASIZE) + Op.CALL(
        Op.GAS, Spec.DEPOSIT_CONTRACT_ADDRESS, Op.CALLVALUE, 0, Op.CALLDATASIZE, 0, 0
    )

    pre = {
        TestAddress: Account(balance=32_000_000_000_000_000_000 * 100),
        relay_contract_address: Account(code=relay_contract_code, nonce=1),
    }

    tx = Transaction(
        gas_limit=1_000_000,
        gas_price=0x07,
        to=relay_contract_address,
        value=deposit.value,
        data=deposit.calldata,
    )

    blockchain_test(
        genesis_environment=Environment(),
        pre=pre,
        post={},
        blocks=[
            Block(
                txs=[tx],
                header_verify=Header(
                    requests_root=[deposit],
                ),
            )
        ],
    )
