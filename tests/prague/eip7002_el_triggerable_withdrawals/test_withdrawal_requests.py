"""
abstract: Tests [EIP-7002: Execution layer triggerable withdrawals](https://eips.ethereum.org/EIPS/eip-7002)
    Test execution layer triggered exits [EIP-7002: Execution layer triggerable withdrawals](https://eips.ethereum.org/EIPS/eip-7002)

"""  # noqa: E501

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import pytest

from ethereum_test_tools import (
    Account,
    Address,
    Block,
    BlockchainTestFiller,
    Environment,
    Header,
    TestAddress,
    TestAddress2,
    TestPrivateKey,
    TestPrivateKey2,
    Transaction,
    WithdrawalRequest,
)

from .spec import Spec, ref_spec_7002

REFERENCE_SPEC_GIT_PATH = ref_spec_7002.git_path
REFERENCE_SPEC_VERSION = ref_spec_7002.version

pytestmark = pytest.mark.valid_from("Prague")


@dataclass
class SenderAccount:
    """Test sender account descriptor."""

    address: Address
    key: str


TestAccount1 = SenderAccount(TestAddress, TestPrivateKey)
TestAccount2 = SenderAccount(TestAddress2, TestPrivateKey2)


class WithdrawalRequestTransactionBase(ABC):
    """
    Base class for all types of withdrawal transactions we want to test.
    """

    def transaction(self) -> Transaction:
        """Return a transaction for the withdrawal request."""
        raise NotImplementedError

    def pre(self) -> Dict[Address, Account]:
        """Return the pre-state of the account."""
        raise NotImplementedError

    def valid_withdrawal_requests(self, current_block_fee: int) -> List[WithdrawalRequest]:
        """Return the list of withdrawal requests that should be valid in the block."""
        raise NotImplementedError


@dataclass(kw_only=True)
class WithdrawalRequestTransaction(WithdrawalRequestTransactionBase):
    """Class used to describe a withdrawal request originated from an externally owned account."""

    withdrawal_request: WithdrawalRequest
    valid: bool = True
    fee: int = 0
    gas_limit: int = 1_000_000
    sender_balance: int = 32_000_000_000_000_000_000 * 100
    sender_account: SenderAccount = TestAccount1
    nonce: int = 0
    calldata: bytes | None = None

    def transaction(self) -> Transaction:
        """Return a transaction for the withdrawal request."""
        return Transaction(
            nonce=self.nonce,
            gas_limit=self.gas_limit,
            gas_price=0x07,
            to=Spec.WITHDRAWAL_REQUEST_PREDEPLOY_ADDRESS,
            value=self.fee,
            data=self.calldata if self.calldata is not None else self.withdrawal_request.calldata,
            secret_key=self.sender_account.key,
        )

    def pre(self) -> Dict[Address, Account]:
        """Return the pre-state of the account."""
        return {
            self.sender_account.address: Account(balance=self.sender_balance),
        }

    def valid_withdrawal_requests(self, current_block_fee: int) -> List[WithdrawalRequest]:
        """Return the list of withdrawal requests that are valid."""
        if self.valid and self.fee >= current_block_fee:
            return [self.withdrawal_request.with_source_address(self.sender_account.address)]
        return []


##############
#  Fixtures  #
##############


@pytest.fixture
def included_withdrawal_requests(
    blocks_withdrawal_requests: List[List[WithdrawalRequestTransactionBase]],
) -> List[List[WithdrawalRequest]]:
    """
    Return the list of withdrawal requests that should be included in each block.
    """
    excess_withdrawal_requests = 0
    carry_over_withdrawal_requests: List[WithdrawalRequest] = []
    all_withdrawal_requests: List[List[WithdrawalRequest]] = []
    for block_withdrawal_requests in blocks_withdrawal_requests:
        current_block_fee = Spec.get_fee(excess_withdrawal_requests)

        current_block_valid_withdrawal_requests = carry_over_withdrawal_requests
        for w in block_withdrawal_requests:
            current_block_valid_withdrawal_requests += w.valid_withdrawal_requests(
                current_block_fee
            )

        current_block_included_withdrawal_requests = current_block_valid_withdrawal_requests[
            : Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK
        ]

        if (
            len(current_block_valid_withdrawal_requests)
            > Spec.TARGET_WITHDRAWAL_REQUESTS_PER_BLOCK
        ):
            excess_withdrawal_requests += (
                len(current_block_valid_withdrawal_requests)
                - Spec.TARGET_WITHDRAWAL_REQUESTS_PER_BLOCK
            )
        elif (
            len(current_block_valid_withdrawal_requests)
            < Spec.TARGET_WITHDRAWAL_REQUESTS_PER_BLOCK
        ):
            excess_withdrawal_requests = max(
                0,
                excess_withdrawal_requests
                - (
                    len(current_block_valid_withdrawal_requests)
                    - Spec.TARGET_WITHDRAWAL_REQUESTS_PER_BLOCK
                ),
            )

        if len(current_block_valid_withdrawal_requests) > Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK:
            carry_over_withdrawal_requests = current_block_valid_withdrawal_requests[
                Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK :
            ]
        else:
            carry_over_withdrawal_requests = []

        all_withdrawal_requests.append(current_block_included_withdrawal_requests)
    return all_withdrawal_requests


@pytest.fixture
def pre(
    blocks_withdrawal_requests: List[List[WithdrawalRequestTransactionBase]],
) -> Dict[Address, Account]:
    """
    Initial state of the accounts. Every withdrawal transaction defines their own pre-state
    requirements, and this fixture aggregates them all.
    """
    pre = {}
    for b in blocks_withdrawal_requests:
        for w in b:
            pre.update(w.pre())
    return pre


@pytest.fixture
def blocks(
    blocks_withdrawal_requests: List[List[WithdrawalRequestTransactionBase]],
    included_withdrawal_requests: List[List[WithdrawalRequest]],
) -> List[Block]:
    """
    Return the list of blocks that should be included in the test.
    """
    blocks: List[Block] = []
    for i in range(len(blocks_withdrawal_requests)):
        txs = [w.transaction() for w in blocks_withdrawal_requests[i]]
        blocks.append(
            Block(
                txs=txs,
                header_verify=Header(
                    requests_root=included_withdrawal_requests[i],
                ),
            )
        )
    return blocks


################
#  Test cases  #
################


@pytest.mark.parametrize(
    "blocks_withdrawal_requests",
    [
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestTransaction(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=0x01,
                            amount=0,
                        ),
                        fee=Spec.get_fee(0),
                    ),
                ],
            ],
            id="single_block_single_withdrawal_request_from_eoa",
        ),
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestTransaction(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=0x01,
                            amount=0,
                        ),
                        fee=0,
                    ),
                ],
            ],
            id="single_block_single_withdrawal_request_from_eoa_insufficient_fee",
        ),
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestTransaction(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=i + 1,
                            amount=0,
                        ),
                        fee=Spec.get_fee(0),
                        nonce=i,
                    )
                    for i in range(Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK)
                ],
            ],
            id="single_block_max_withdrawal_requests_from_eoa",
        ),
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestTransaction(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=i + 1,
                            amount=0,
                        ),
                        fee=Spec.get_fee(0),
                        nonce=i,
                    )
                    for i in range(Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK * 2)
                ],
                # Block 2, no new withdrawal requests, but queued requests from previous block
                [],
                # Block 3, no new nor queued withdrawal requests
                [],
            ],
            id="single_block_above_max_withdrawal_requests_from_eoa",
        ),
    ],
)
def test_withdrawal_requests(
    blockchain_test: BlockchainTestFiller,
    blocks: List[Block],
    pre: Dict[Address, Account],
):
    """
    Test making a withdrawal request to the beacon chain from an externally owned account.
    """
    blockchain_test(
        genesis_environment=Environment(),
        pre=pre,
        post={},
        blocks=blocks,
    )
