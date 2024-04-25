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
    BlockException,
    DepositRequest,
    EngineAPIError,
    Environment,
    Header,
    Macros,
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


#############
#  Helpers  #
#############


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
    calldata: bytes | None = None

    def transaction(self) -> Transaction:
        """Return a transaction for the deposit request."""
        return Transaction(
            nonce=self.nonce,
            gas_limit=self.gas_limit,
            gas_price=0x07,
            to=Spec.DEPOSIT_CONTRACT_ADDRESS,
            value=self.deposit_request.value,
            data=self.calldata if self.calldata is not None else self.deposit_request.calldata,
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
class DepositContract(DepositTransactionBase):
    """Class used to describe a deposit originated from a contract."""

    deposit_request: List[DepositRequest] | DepositRequest
    included: List[bool] | bool

    tx_gas_limit: int = 1_000_000

    sender_account: SenderAccount = TestAccount1
    sender_balance: int = 32_000_000_000_000_000_000 * 100

    contract_balance: int = 32_000_000_000_000_000_000 * 100
    contract_address: int = 0x200

    call_type: Op = Op.CALL
    call_depth: int = 2
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
            value_arg = [d.value] if self.call_type in (Op.CALL, Op.CALLCODE) else []
            code += Op.CALLDATACOPY(0, current_offset, len(d.calldata)) + Op.POP(
                self.call_type(
                    Op.GAS,
                    Spec.DEPOSIT_CONTRACT_ADDRESS,
                    *value_arg,
                    0,
                    len(d.calldata),
                    0,
                    0,
                )
            )
            current_offset += len(d.calldata)
        return code + self.extra_code

    def transaction(self) -> Transaction:
        """Return a transaction for the deposit request."""
        return Transaction(
            nonce=self.nonce,
            gas_limit=self.tx_gas_limit,
            gas_price=0x07,
            to=self.entry_address,
            value=0,
            data=b"".join(d.calldata for d in self.deposit_requests),
            secret_key=self.sender_account.key,
        )

    @property
    def entry_address(self) -> Address:
        """Return the address of the contract entry point."""
        if self.call_depth == 2:
            return Address(self.contract_address)
        elif self.call_depth > 2:
            return Address(self.contract_address + self.call_depth - 2)
        raise ValueError("Invalid call depth")

    @property
    def extra_contracts(self) -> Dict[Address, Account]:
        """Extra contracts used to simulate call depth."""
        if self.call_depth <= 2:
            return {}
        return {
            Address(self.contract_address + i): Account(
                balance=self.contract_balance,
                code=Op.CALLDATACOPY(0, 0, Op.CALLDATASIZE)
                + Op.POP(
                    Op.CALL(
                        Op.GAS,
                        self.contract_address + i - 1,
                        0,
                        0,
                        Op.CALLDATASIZE,
                        0,
                        0,
                    )
                ),
                nonce=1,
            )
            for i in range(1, self.call_depth - 1)
        }

    def pre(self) -> Dict[Address, Account]:
        """Return the pre-state of the account."""
        return {
            self.sender_account.address: Account(balance=self.sender_balance),
            Address(self.contract_address): Account(
                balance=self.contract_balance, code=self.contract_code, nonce=1
            ),
        } | self.extra_contracts

    def included_deposits(self) -> List[DepositRequest]:
        """Return the list of deposit requests that should be included in the block."""
        if not isinstance(self.included, Iterable):
            return self.deposit_requests if self.included else []
        return [d for d, i in zip(self.deposit_requests, self.included) if i]


##############
#  Fixtures  #
##############


@pytest.fixture
def pre(deposit_requests: List[DepositTransactionBase]) -> Dict[Address, Account]:
    """
    Initial state of the accounts. Every deposit transaction defines their own pre-state
    requirements, and this fixture aggregates them all.
    """
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


################
#  Test cases  #
################


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
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=32_000_000_000,
                        signature=0x03,
                        index=0x0,
                    ),
                    included=False,
                    calldata=b"",
                ),
            ],
            id="send_eth_from_eoa",
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
                        DepositRequest(
                            pubkey=0x01,
                            withdrawal_credentials=0x02,
                            amount=1_000_000_000,
                            signature=0x03,
                            index=0x1,
                        ),
                    ],
                    extra_code=Macros.OOG(),
                    included=False,
                ),
            ],
            id="multiple_deposits_from_contract_caller_oog",
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
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=32_000_000_000,
                        signature=0x03,
                        index=0x2,
                    ),
                    included=True,
                    nonce=2,
                ),
            ],
            id="single_deposit_from_contract_between_eoa_deposits",
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
                DepositContract(
                    deposit_request=[
                        DepositRequest(
                            pubkey=0x01,
                            withdrawal_credentials=0x02,
                            amount=32_000_000_000,
                            signature=0x03,
                            index=0x2,
                        ),
                    ],
                    included=True,
                    nonce=2,
                    contract_address=0x300,
                ),
            ],
            id="single_deposit_from_eoa_between_contract_deposits",
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
                    call_type=Op.DELEGATECALL,
                    included=False,
                ),
            ],
            id="single_deposit_from_contract_delegatecall",
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
                    call_type=Op.STATICCALL,
                    included=False,
                ),
            ],
            id="single_deposit_from_contract_staticcall",
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
                    call_type=Op.CALLCODE,
                    included=False,
                ),
            ],
            id="single_deposit_from_contract_callcode",
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
                    call_depth=3,
                ),
            ],
            id="single_deposit_from_contract_call_depth_3",
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
                    call_depth=1024,
                    tx_gas_limit=2_500_000_000_000,
                ),
            ],
            id="single_deposit_from_contract_call_high_depth",
        ),
        # TODO: Send eth with the transaction to the contract
    ],
)
def test_deposit(
    blockchain_test: BlockchainTestFiller,
    deposit_requests: List[DepositTransactionBase],
    pre: Dict[Address, Account],
    txs: List[Transaction],
):
    """
    Test making a deposit to the beacon chain deposit contract.
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


@pytest.mark.parametrize(
    "deposit_requests,block_requests,exception,engine_api_error_code",
    [
        pytest.param(
            [],
            [
                DepositRequest(
                    pubkey=0x01,
                    withdrawal_credentials=0x02,
                    amount=1_000_000_000,
                    signature=0x03,
                    index=0x0,
                ),
            ],
            BlockException.INVALID_REQUESTS,
            None,
            id="no_deposits_non_empty_requests_list",
        ),
        pytest.param(
            [
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=1_000_000_000,
                        signature=0x03,
                        index=0x0,
                    ),
                    included=True,
                ),
            ],
            [],
            BlockException.INVALID_REQUESTS,
            None,
            id="single_deposit_empty_requests_list",
        ),
        pytest.param(
            [
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=1_000_000_000,
                        signature=0x03,
                        index=0x0,
                    ),
                    included=True,
                ),
            ],
            [
                DepositRequest(
                    pubkey=0x01,
                    withdrawal_credentials=0x02,
                    amount=1_000_000_000,
                    signature=0x03,
                    index=0x1,
                )
            ],
            BlockException.INVALID_REQUESTS,
            None,
            id="single_deposit_field_mismatch",
        ),
        pytest.param(
            [
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=1_000_000_000,
                        signature=0x03,
                        index=0x0,
                    ),
                    included=True,
                ),
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=1_000_000_000,
                        signature=0x03,
                        index=0x1,
                    ),
                    nonce=1,
                    included=True,
                ),
            ],
            [
                DepositRequest(
                    pubkey=0x01,
                    withdrawal_credentials=0x02,
                    amount=1_000_000_000,
                    signature=0x03,
                    index=0x1,
                ),
                DepositRequest(
                    pubkey=0x01,
                    withdrawal_credentials=0x02,
                    amount=1_000_000_000,
                    signature=0x03,
                    index=0x0,
                ),
            ],
            BlockException.INVALID_REQUESTS,
            None,
            id="two_deposits_out_of_order",
        ),
        pytest.param(
            [
                DepositTransaction(
                    deposit_request=DepositRequest(
                        pubkey=0x01,
                        withdrawal_credentials=0x02,
                        amount=1_000_000_000,
                        signature=0x03,
                        index=0x0,
                    ),
                    included=True,
                ),
            ],
            [
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
                    amount=1_000_000_000,
                    signature=0x03,
                    index=0x0,
                ),
            ],
            BlockException.INVALID_REQUESTS,
            None,
            id="single_deposit_duplicate_in_requests_list",
        ),
    ],
)
def test_deposit_negative(
    blockchain_test: BlockchainTestFiller,
    deposit_requests: List[DepositTransactionBase],
    block_requests: List[DepositRequest],
    exception: BlockException,
    engine_api_error_code: EngineAPIError | None,
    pre: Dict[Address, Account],
    txs: List[Transaction],
):
    """
    Test producing a block with the incorrect deposits in the body of the block,
    and/or Engine API payload.
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
                requests=block_requests,
                exception=exception,
                engine_api_error_code=engine_api_error_code,
            )
        ],
    )
