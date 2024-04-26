"""
abstract: Tests [EIP-7685: General purpose execution layer requests](https://eips.ethereum.org/EIPS/eip-7685)
    Cross testing for withdrawal and deposit request for [EIP-7685: General purpose execution layer requests](https://eips.ethereum.org/EIPS/eip-7685)

"""  # noqa: E501

from dataclasses import dataclass
from typing import Dict, List

import pytest

from ethereum_test_tools import (
    Account,
    Address,
    Block,
    BlockchainTestFiller,
    BlockException,
    DepositRequest,
    Environment,
    Header,
    TestAddress,
    TestAddress2,
    TestPrivateKey,
    TestPrivateKey2,
    Transaction,
    WithdrawalRequest,
)

from ..eip6110_deposits.test_deposits import (
    DepositContract,
    DepositTransaction,
    DepositTransactionBase,
)
from ..eip7002_el_triggerable_withdrawals.test_withdrawal_requests import (
    WithdrawalRequestContract,
    WithdrawalRequestTransaction,
    WithdrawalRequestTransactionBase,
)
from .spec import ref_spec_7685

REFERENCE_SPEC_GIT_PATH = ref_spec_7685.git_path
REFERENCE_SPEC_VERSION = ref_spec_7685.version

pytestmark = pytest.mark.valid_from("Prague")


@dataclass
class SenderAccount:
    """Test sender account descriptor."""

    address: Address
    key: str


TestAccount1 = SenderAccount(TestAddress, TestPrivateKey)
TestAccount2 = SenderAccount(TestAddress2, TestPrivateKey2)

##############
#  Fixtures  #
##############


@pytest.fixture
def pre(
    requests: List[DepositTransactionBase | WithdrawalRequestTransactionBase],
) -> Dict[Address, Account]:
    """
    Initial state of the accounts. Every deposit transaction defines their own pre-state
    requirements, and this fixture aggregates them all.
    """
    pre = {}
    for d in requests:
        pre.update(d.pre())
    return pre


@pytest.fixture
def txs(
    requests: List[DepositTransactionBase | WithdrawalRequestTransactionBase],
) -> List[Transaction]:
    """List of transactions to include in the block."""
    return [d.transaction() for d in requests]


@pytest.fixture
def block_requests() -> List[DepositRequest] | None:
    """List of requests that overwrite the requests in the header. None by default."""
    return None


@pytest.fixture
def exception() -> BlockException | None:
    """Block exception expected by the tests. None by default."""
    return None


@pytest.fixture
def blocks(
    requests: List[DepositTransactionBase | WithdrawalRequestTransactionBase],
    block_requests: List[DepositRequest | WithdrawalRequest] | None,
    txs: List[Transaction],
    exception: BlockException | None,
) -> List[Block]:
    """List of blocks that comprise the test."""
    included_deposit_requests = []
    included_withdrawal_requests = []
    # Single block therefore base fee
    withdrawal_request_fee = 1
    for r in requests:
        if isinstance(r, DepositTransactionBase):
            included_deposit_requests += r.included_deposits()
        elif isinstance(r, WithdrawalRequestTransactionBase):
            included_withdrawal_requests += r.valid_withdrawal_requests(withdrawal_request_fee)

    return [
        Block(
            txs=txs,
            header_verify=Header(
                requests_root=included_deposit_requests + included_withdrawal_requests,
                requests=block_requests,
                exception=exception,
            ),
        )
    ]


################
#  Test cases  #
################


@pytest.mark.parametrize(
    "requests",
    [
        pytest.param(
            [
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=32_000_000_000,
                        signature=0x03,
                        index=0x0,
                    ),
                    nonce=0,
                ),
                WithdrawalRequestTransaction(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x01,
                        amount=0,
                    ),
                    fee=1,
                    nonce=1,
                ),
            ],
            id="single_deposit_from_eoa_single_withdrawal_from_eoa",
        ),
        pytest.param(
            [
                WithdrawalRequestTransaction(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x01,
                        amount=0,
                    ),
                    fee=1,
                    nonce=0,
                ),
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=32_000_000_000,
                        signature=0x03,
                        index=0x0,
                    ),
                    nonce=1,
                ),
            ],
            id="single_withdrawal_from_eoa_single_deposit_from_eoa",
        ),
        pytest.param(
            [
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=32_000_000_000,
                        signature=0x03,
                        index=0x0,
                    ),
                    nonce=0,
                ),
                WithdrawalRequestTransaction(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x01,
                        amount=0,
                    ),
                    fee=1,
                    nonce=1,
                ),
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=32_000_000_000,
                        signature=0x03,
                        index=0x1,
                    ),
                    nonce=2,
                ),
            ],
            id="two_deposits_from_eoa_single_withdrawal_from_eoa",
        ),
        pytest.param(
            [
                WithdrawalRequestTransaction(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x01,
                        amount=0,
                    ),
                    fee=1,
                    nonce=0,
                ),
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=32_000_000_000,
                        signature=0x03,
                        index=0x0,
                    ),
                    nonce=1,
                ),
                WithdrawalRequestTransaction(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x01,
                        amount=1,
                    ),
                    fee=1,
                    nonce=2,
                ),
            ],
            id="two_withdrawals_from_eoa_single_deposit_from_eoa",
        ),
        pytest.param(
            [
                DepositContract(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=32_000_000_000,
                        signature=0x03,
                        index=0x0,
                    ),
                    nonce=0,
                    contract_address=0x200,
                ),
                WithdrawalRequestContract(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x01,
                        amount=0,
                    ),
                    fee=1,
                    nonce=1,
                    contract_address=0x300,
                ),
            ],
            id="single_deposit_from_contract_single_withdrawal_from_contract",
        ),
        pytest.param(
            [
                WithdrawalRequestContract(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x01,
                        amount=0,
                    ),
                    fee=1,
                    nonce=0,
                    contract_address=0x300,
                ),
                DepositContract(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=32_000_000_000,
                        signature=0x03,
                        index=0x0,
                    ),
                    nonce=1,
                    contract_address=0x200,
                ),
            ],
            id="single_withdrawal_from_contract_single_deposit_from_contract",
        ),
        # TODO: Deposit and withdrawal in the same transaction
    ],
)
def test_valid_deposit_withdrawal_requests(
    blockchain_test: BlockchainTestFiller,
    pre: Dict[Address, Account],
    blocks: List[Block],
):
    """
    Test making a deposit to the beacon chain deposit contract and a withdrawal in the same block.
    """
    blockchain_test(
        genesis_environment=Environment(),
        pre=pre,
        post={},
        blocks=blocks,
    )


@pytest.mark.parametrize(
    "requests,block_requests,exception,engine_api_error",
    [
        pytest.param(
            [
                WithdrawalRequestTransaction(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x01,
                        amount=0,
                    ),
                    fee=1,
                    nonce=0,
                ),
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=32_000_000_000,
                        signature=0x03,
                        index=0x0,
                    ),
                    nonce=1,
                ),
            ],
            [
                WithdrawalRequest(
                    validator_public_key=0x01,
                    amount=0,
                    source_address=TestAddress,
                ),
                DepositRequest(
                    pubkey=0x01,
                    withdrawal_credentials=0x02,
                    amount=32_000_000_000,
                    signature=0x03,
                    index=0x0,
                ),
            ],
            # TODO: on the Engine API, the issue should be detected as an invalid block hash
            BlockException.INVALID_REQUESTS,
            id="single_deposit_from_eoa_single_withdrawal_from_eoa_incorrect_order",
        ),
    ],
)
def test_invalid_deposit_withdrawal_requests(
    blockchain_test: BlockchainTestFiller,
    pre: Dict[Address, Account],
    blocks: List[Block],
):
    """
    Negative testing for deposits and withdrawals in the same block.
    """
    blockchain_test(
        genesis_environment=Environment(),
        pre=pre,
        post={},
        blocks=blocks,
    )
