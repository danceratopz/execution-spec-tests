"""
abstract: Tests [EIP-6110: Supply validator deposits on chain](https://eips.ethereum.org/EIPS/eip-6110)
    Test [EIP-6110: Supply validator deposits on chain](https://eips.ethereum.org/EIPS/eip-6110).
"""  # noqa: E501

from abc import ABC
from dataclasses import dataclass
from typing import Dict, Iterable, List

import pytest

from ethereum_test_tools import (
    Account,
    Address,
    Block,
    BlockchainTestFiller,
    DepositRequest,
    Environment,
    Header,
)
from ethereum_test_tools import Opcodes as Op
from ethereum_test_tools import (
    TestAddress,
    TestAddress2,
    TestPrivateKey,
    TestPrivateKey2,
    Transaction,
)

from .spec import Spec, ref_spec_6110

REFERENCE_SPEC_GIT_PATH = ref_spec_6110.git_path
REFERENCE_SPEC_VERSION = ref_spec_6110.version

pytestmark = pytest.mark.valid_from("Prague")


@dataclass
class SenderAccount:
    """Test sender account descriptor."""

    address: Address
    key: str


TestAccount1 = SenderAccount(TestAddress, TestPrivateKey)
TestAccount2 = SenderAccount(TestAddress2, TestPrivateKey2)


class DepositTransactionBase(ABC):
    """
    Base class for all types of deposit transactions we want to test.
    """

    def transaction(self) -> Transaction:
        """Return a transaction for the deposit request."""
        raise NotImplementedError

    def pre(self) -> Dict[Address, Account]:
        """Return the pre-state of the account."""
        raise NotImplementedError

    def included_deposits(self) -> List[DepositRequest]:
        """Return the list of deposit requests that should be included in the block."""
        raise NotImplementedError


@dataclass(kw_only=True)
class DepositTransaction(DepositTransactionBase):
    """Class used to describe a deposit originated from an externally owned account."""

    deposit_request: DepositRequest
    included: bool
    gas_limit: int = 1_000_000
    sender_balance: int = 32_000_000_000_000_000_000 * 100
    sender_account: SenderAccount = TestAccount1
    nonce: int = 0

    def transaction(self) -> Transaction:
        """Return a transaction for the deposit request."""
        return Transaction(
            nonce=self.nonce,
            gas_limit=self.gas_limit,
            gas_price=0x07,
            to=Spec.DEPOSIT_CONTRACT_ADDRESS,
            value=self.deposit_request.value,
            data=self.deposit_request.calldata,
            secret_key=self.sender_account.key,
        )

    def pre(self) -> Dict[Address, Account]:
        """Return the pre-state of the account."""
        return {
            self.sender_account.address: Account(balance=self.sender_balance),
        }

    def included_deposits(self) -> List[DepositRequest]:
        """Return the list of deposit requests that should be included in the block."""
        return [self.deposit_request] if self.included else []


@dataclass(kw_only=True)
class DepositContract:
    """Class used to describe a deposit originated from a contract."""

    deposit_request: List[DepositRequest] | DepositRequest
    included: List[bool] | bool

    gas_limit: int = 1_000_000

    sender_account: SenderAccount = TestAccount1
    sender_balance: int = 32_000_000_000_000_000_000 * 100

    contract_balance: int = 32_000_000_000_000_000_000 * 100
    contract_address: Address = Address(0x200)

    extra_code: bytes = b""

    nonce: int = 0

    @property
    def deposit_requests(self) -> List[DepositRequest]:
        """Return the list of deposit requests."""
        if not isinstance(self.deposit_request, List):
            return [self.deposit_request]
        return self.deposit_request

    @property
    def contract_code(self) -> bytes:
        """Contract code used by the relay contract."""
        code = b""
        current_offset = 0
        for d in self.deposit_requests:
            code += Op.CALLDATACOPY(0, current_offset, len(d.calldata)) + Op.POP(
                Op.CALL(Op.GAS, Spec.DEPOSIT_CONTRACT_ADDRESS, d.value, 0, len(d.calldata), 0, 0)
            )
            current_offset += len(d.calldata)
        return code + self.extra_code

    def transaction(self) -> Transaction:
        """Return a transaction for the deposit request."""
        return Transaction(
            nonce=self.nonce,
            gas_limit=self.gas_limit,
            gas_price=0x07,
            to=self.contract_address,
            value=0,
            data=b"".join(d.calldata for d in self.deposit_requests),
            secret_key=self.sender_account.key,
        )

    def pre(self) -> Dict[Address, Account]:
        """Return the pre-state of the account."""
        return {
            self.sender_account.address: Account(balance=self.sender_balance),
            self.contract_address: Account(
                balance=self.contract_balance, code=self.contract_code, nonce=1
            ),
        }

    def included_deposits(self) -> List[DepositRequest]:
        """Return the list of deposit requests that should be included in the block."""
        if not isinstance(self.included, Iterable):
            return self.deposit_requests if self.included else []
        return [d for d, i in zip(self.deposit_requests, self.included) if i]


@pytest.fixture
def pre(deposit_requests: List[DepositTransactionBase]) -> Dict[Address, Account]:
    """Initial state of the accounts."""
    pre = {}
    for d in deposit_requests:
        pre.update(d.pre())
    return pre


@pytest.fixture
def txs(
    deposit_requests: List[DepositTransactionBase],
) -> List[Transaction]:
    """List of transactions to include in the block."""
    return [d.transaction() for d in deposit_requests]


@pytest.mark.parametrize(
    "deposit_requests",
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
                    included=True,
                ),
            ],
            id="single_deposit_from_eoa",
        ),
        pytest.param(
            [
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=120_000_000_000_000_000,
                        signature=0x03,
                        index=0x0,
                    ),
                    included=True,
                    sender_balance=120_000_001_000_000_000 * 10**9,
                ),
            ],
            id="single_deposit_from_eoa_huge_amount",
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
                    included=True,
                    nonce=0,
                ),
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=32_000_000_000,
                        signature=0x03,
                        index=0x1,
                    ),
                    included=True,
                    nonce=1,
                ),
            ],
            id="multiple_deposit_from_same_eoa",
        ),
        pytest.param(
            [
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=1_000_000_000,
                        signature=0x03,
                        index=i,
                    ),
                    included=True,
                    nonce=i,
                )
                for i in range(2000)
            ],
            id="multiple_deposit_from_same_eoa_high_count",
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
                    included=True,
                    sender_account=TestAccount1,
                ),
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=32_000_000_000,
                        signature=0x03,
                        index=0x1,
                    ),
                    included=True,
                    sender_account=TestAccount2,
                ),
            ],
            id="multiple_deposit_from_different_eoa",
        ),
        pytest.param(
            [
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=999_999,
                        signature=0x03,
                        index=0x0,
                    ),
                    included=False,
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
                    included=True,
                    nonce=1,
                ),
            ],
            id="multiple_deposit_from_same_eoa_first_reverts",
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
                    included=True,
                    nonce=0,
                ),
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=999_999,
                        signature=0x03,
                        index=0x0,
                    ),
                    included=False,
                    nonce=1,
                ),
            ],
            id="multiple_deposit_from_same_eoa_last_reverts",
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
                    # From traces, gas used by the first tx is 82,718 so reduce by one here
                    gas_limit=0x1431D,
                    included=False,
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
                    included=True,
                    nonce=1,
                ),
            ],
            id="multiple_deposit_from_same_eoa_first_oog",
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
                    included=True,
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
                    # From traces, gas used by the second tx is 68,594 so reduce by one here
                    gas_limit=0x10BF1,
                    included=False,
                    nonce=1,
                ),
            ],
            id="multiple_deposit_from_same_eoa_last_oog",
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
                    included=True,
                ),
            ],
            id="single_deposit_from_contract",
        ),
        pytest.param(
            [
                DepositContract(
                    deposit_request=[
                        DepositRequest(
                            pubkey=0x01,
                            withdrawal_credentials=0x02,
                            amount=32_000_000_000,
                            signature=0x03,
                            index=0x0,
                        ),
                        DepositRequest(
                            pubkey=0x01,
                            withdrawal_credentials=0x02,
                            amount=1_000_000_000,
                            signature=0x03,
                            index=0x1,
                        ),
                    ],
                    included=True,
                ),
            ],
            id="multiple_deposits_from_contract",
        ),
        pytest.param(
            [
                DepositContract(
                    deposit_request=[
                        DepositRequest(
                            pubkey=0x01,
                            withdrawal_credentials=0x02,
                            amount=999_999_999,
                            signature=0x03,
                            index=0x0,
                        ),
                        DepositRequest(
                            pubkey=0x01,
                            withdrawal_credentials=0x02,
                            amount=1_000_000_000,
                            signature=0x03,
                            index=0x0,
                        ),
                    ],
                    included=[False, True],
                ),
            ],
            id="multiple_deposits_from_contract_first_reverts",
        ),
        pytest.param(
            [
                DepositContract(
                    deposit_request=[
                        DepositRequest(
                            pubkey=0x01,
                            withdrawal_credentials=0x02,
                            amount=1_000_000_000,
                            signature=0x03,
                            index=0x0,
                        ),
                        DepositRequest(
                            pubkey=0x01,
                            withdrawal_credentials=0x02,
                            amount=999_999_999,
                            signature=0x03,
                            index=0x0,
                        ),
                    ],
                    included=[True, False],
                ),
            ],
            id="multiple_deposits_from_contract_last_reverts",
        ),
        pytest.param(
            [
                DepositContract(
                    deposit_request=[
                        DepositRequest(
                            pubkey=0x01,
                            withdrawal_credentials=0x02,
                            amount=32_000_000_000,
                            signature=0x03,
                            index=0x0,
                        ),
                        DepositRequest(
                            pubkey=0x01,
                            withdrawal_credentials=0x02,
                            amount=1_000_000_000,
                            signature=0x03,
                            index=0x1,
                        ),
                    ],
                    extra_code=Op.REVERT(0, 0),
                    included=False,
                ),
            ],
            id="multiple_deposits_from_contract_caller_reverts",
        ),
        pytest.param(
            [
                DepositContract(
                    deposit_request=[
                        DepositRequest(
                            pubkey=0x01,
                            withdrawal_credentials=0x02,
                            amount=32_000_000_000,
                            signature=0x03,
                            index=0x0,
                        ),
                    ],
                    nonce=0,
                    included=True,
                ),
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=32_000_000_000,
                        signature=0x03,
                        index=0x1,
                    ),
                    nonce=1,
                    included=True,
                ),
            ],
            id="single_deposit_from_contract_single_deposit_from_eoa",
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
                    included=True,
                    nonce=0,
                ),
                DepositContract(
                    deposit_request=[
                        DepositRequest(
                            pubkey=0x01,
                            withdrawal_credentials=0x02,
                            amount=32_000_000_000,
                            signature=0x03,
                            index=0x1,
                        ),
                    ],
                    included=True,
                    nonce=1,
                ),
            ],
            id="single_deposit_from_eoa_single_deposit_from_contract",
        ),
    ],
)
def test_deposit(
    blockchain_test: BlockchainTestFiller,
    deposit_requests: List[DepositTransactionBase],
    pre: Dict[Address, Account],
    txs: List[Transaction],
):
    """
    Test making a deposit to the beacon chain deposit contract from an externally owned account.
    """
    included_deposits = []

    for d in deposit_requests:
        included_deposits += d.included_deposits()

    blockchain_test(
        genesis_environment=Environment(),
        pre=pre,
        post={},
        blocks=[
            Block(
                txs=txs,
                header_verify=Header(
                    requests_root=included_deposits,
                ),
            )
        ],
    )
