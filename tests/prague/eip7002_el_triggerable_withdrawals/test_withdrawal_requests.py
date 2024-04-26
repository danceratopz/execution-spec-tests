"""
abstract: Tests [EIP-7002: Execution layer triggerable withdrawals](https://eips.ethereum.org/EIPS/eip-7002)
    Test execution layer triggered exits [EIP-7002: Execution layer triggerable withdrawals](https://eips.ethereum.org/EIPS/eip-7002)

"""  # noqa: E501

from abc import ABC
from dataclasses import dataclass
from itertools import count
from typing import Dict, List

import pytest

from ethereum_test_tools import (
    Account,
    Address,
    Block,
    BlockchainTestFiller,
    BlockException,
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
    """
    Withdrawal request to be requested by the transaction.
    """
    valid: bool = True
    """
    Whether the withdrawal request is valid or not.
    """
    fee: int = 0
    """
    Fee to be paid for the withdrawal request.
    """
    gas_limit: int = 1_000_000
    """
    Gas limit for the transaction.
    """
    sender_balance: int = 32_000_000_000_000_000_000 * 100
    """
    Balance of the account that sends the transaction.
    """
    sender_account: SenderAccount = TestAccount1
    """
    Account that will send the transaction.
    """
    nonce: int = 0
    """
    Nonce of the sender account.
    """
    calldata: bytes | None = None
    """
    Calldata to be used in the transaction. By default it automatically generates the calldata
    according to the withdrawal request.
    """

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


@dataclass(kw_only=True)
class WithdrawalRequestContract(WithdrawalRequestTransactionBase):
    """Class used to describe a deposit originated from a contract."""

    withdrawal_request: List[WithdrawalRequest] | WithdrawalRequest
    """
    Withdrawal request or list of withdrawal requests to be requested by the contract.
    """
    valid: List[bool] | bool = True
    """
    Whether the withdrawal request is valid or not. If a list, it should have the same length as
    `withdrawal_request`.
    """
    fee: List[int] | int = 0
    """
    Fee to be paid for each withdrawal request. If a list, it should have the same length as
    `withdrawal_request`.
    """

    tx_gas_limit: int = 1_000_000
    """
    Gas limit for the transaction.
    """

    sender_account: SenderAccount = TestAccount1
    """
    Account that will send the transaction (not the actual caller to the pre-deploy contract)
    """
    sender_balance: int = 32_000_000_000_000_000_000 * 100
    """
    Balance of the account that sends the transaction.
    """
    nonce: int = 0
    """
    Nonce of the sender account.
    """

    contract_balance: int = 32_000_000_000_000_000_000 * 100
    """
    Balance of the contract that will make the call to the pre-deploy contract.
    """
    contract_address: int = 0x200
    """
    Address of the contract that will make the call to the pre-deploy contract.
    """

    call_gas: List[int] | int = -1
    """
    Gas to be used in the call. If -1, the gas is Op.GAS.
    """
    call_type: Op = Op.CALL
    """
    Type of call to be used to make the withdrawal request.
    """
    call_depth: int = 2
    """
    Frame depth of the pre-deploy contract when it executes the call.
    """
    extra_code: bytes = b""
    """
    Extra code to be added to the contract code.
    """

    @property
    def withdrawal_requests(self) -> List[WithdrawalRequest]:
        """Return the list of withdrawal requests."""
        if not isinstance(self.withdrawal_request, List):
            return [self.withdrawal_request]
        return self.withdrawal_request

    @property
    def fees(self) -> List[int]:
        """Return the list of fees for each withdrawal request."""
        if not isinstance(self.fee, List):
            return [self.fee] * len(self.withdrawal_requests)
        return self.fee

    @property
    def valid_list(self) -> List[bool]:
        """Return the list of fees for each withdrawal request."""
        if not isinstance(self.valid, List):
            return [self.valid] * len(self.withdrawal_requests)
        return self.valid

    @property
    def call_gas_list(self) -> List[int]:
        """Return the list of fees for each withdrawal request."""
        if not isinstance(self.call_gas, List):
            return [self.call_gas] * len(self.withdrawal_requests)
        return self.call_gas

    @property
    def contract_code(self) -> bytes:
        """Contract code used by the relay contract."""
        code = b""
        current_offset = 0
        for fee, gas, w in zip(self.fees, self.call_gas_list, self.withdrawal_requests):
            value_arg = [fee] if self.call_type in (Op.CALL, Op.CALLCODE) else []
            code += Op.CALLDATACOPY(0, current_offset, len(w.calldata)) + Op.POP(
                self.call_type(
                    Op.GAS if gas == -1 else gas,
                    Spec.WITHDRAWAL_REQUEST_PREDEPLOY_ADDRESS,
                    *value_arg,
                    0,
                    len(w.calldata),
                    0,
                    0,
                )
            )
            current_offset += len(w.calldata)
        return code + self.extra_code

    def transaction(self) -> Transaction:
        """Return a transaction for the deposit request."""
        return Transaction(
            nonce=self.nonce,
            gas_limit=self.tx_gas_limit,
            gas_price=0x07,
            to=self.entry_address,
            value=0,
            data=b"".join(w.calldata for w in self.withdrawal_requests),
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

    def valid_withdrawal_requests(self, current_block_fee: int) -> List[WithdrawalRequest]:
        """Return the list of withdrawal requests that are valid."""
        valid_requests: List[WithdrawalRequest] = []
        for w, fee, valid in zip(self.withdrawal_requests, self.fees, self.valid_list):
            if valid and fee >= current_block_fee:
                valid_requests.append(w.with_source_address(Address(self.contract_address)))
        return valid_requests


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
    carry_over_requests: List[WithdrawalRequest] = []
    per_block_included_requests: List[List[WithdrawalRequest]] = []
    for block_withdrawal_requests in blocks_withdrawal_requests:
        # Get fee for the current block
        current_block_fee = Spec.get_fee(excess_withdrawal_requests)

        # With the fee, get the valid withdrawal requests for the current block
        current_block_requests = []
        for w in block_withdrawal_requests:
            current_block_requests += w.valid_withdrawal_requests(current_block_fee)

        # Get the withdrawal requests that should be included in the block
        pending_requests = carry_over_requests + current_block_requests
        per_block_included_requests.append(
            pending_requests[: Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK]
        )
        carry_over_requests = pending_requests[Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK :]

        # Update the excess withdrawal requests
        excess_withdrawal_requests = Spec.get_excess_withdrawal_requests(
            excess_withdrawal_requests,
            len(current_block_requests),
        )
    return per_block_included_requests


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


#############
#  Helpers  #
#############


def get_n_fee_increments(n: int) -> List[int]:
    """
    Get the first N excess withdrawal requests that increase the fee.
    """
    excess_withdrawal_requests_counts = []
    last_fee = 1
    for i in count(0):
        if Spec.get_fee(i) > last_fee:
            excess_withdrawal_requests_counts.append(i)
            last_fee = Spec.get_fee(i)
        if len(excess_withdrawal_requests_counts) == n:
            break
    return excess_withdrawal_requests_counts


def get_n_fee_increment_blocks(n: int) -> List[List[WithdrawalRequestContract]]:
    """
    Return N blocks that should be included in the test such that each subsequent block has an
    increasing fee for the withdrawal requests.

    This is done by calculating the number of withdrawals required to reach the next fee increment
    and creating a block with that number of withdrawal requests plus the number of withdrawals
    required to reach the target.
    """
    blocks = []
    previous_excess = 0
    nonce = count(0)
    withdrawal_index = 0
    previous_fee = 0
    for required_excess_withdrawals in get_n_fee_increments(n):
        withdrawals_required = (
            required_excess_withdrawals
            + Spec.TARGET_WITHDRAWAL_REQUESTS_PER_BLOCK
            - previous_excess
        )
        tx_nonce = next(nonce)
        fee = Spec.get_fee(previous_excess)
        assert fee > previous_fee
        blocks.append(
            [
                WithdrawalRequestContract(
                    withdrawal_request=[
                        WithdrawalRequest(
                            validator_public_key=i,
                            amount=0,
                        )
                        for i in range(withdrawal_index, withdrawal_index + withdrawals_required)
                    ],
                    fee=fee,
                    # Increment the contract address to avoid overwriting the previous one
                    contract_address=0x200 + (tx_nonce * 0x100),
                    nonce=tx_nonce,
                )
            ],
        )
        previous_fee = fee
        withdrawal_index += withdrawals_required
        previous_excess = required_excess_withdrawals

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
                            validator_public_key=0x01,
                            amount=0,
                        ),
                        calldata=WithdrawalRequest(
                            validator_public_key=0x01,
                            amount=0,
                        ).calldata[:-1],
                        fee=Spec.get_fee(0),
                        valid=False,
                    ),
                ],
            ],
            id="single_block_single_withdrawal_request_from_eoa_input_too_short",
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
                        calldata=WithdrawalRequest(
                            validator_public_key=0x01,
                            amount=0,
                        ).calldata
                        + b"\x00",
                        fee=Spec.get_fee(0),
                        valid=False,
                    ),
                ],
            ],
            id="single_block_single_withdrawal_request_from_eoa_input_too_long",
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
                        fee=Spec.get_fee(0),
                    ),
                    WithdrawalRequestTransaction(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=0x02,
                            amount=Spec.MAX_AMOUNT - 1,
                        ),
                        fee=Spec.get_fee(0),
                        nonce=1,
                    ),
                ],
            ],
            id="single_block_multiple_withdrawal_request_from_same_eoa",
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
                        fee=Spec.get_fee(0),
                    ),
                    WithdrawalRequestTransaction(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=0x02,
                            amount=Spec.MAX_AMOUNT - 1,
                        ),
                        fee=Spec.get_fee(0),
                        sender_account=TestAccount2,
                    ),
                ],
            ],
            id="single_block_multiple_withdrawal_request_from_different_eoa",
        ),
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestTransaction(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=i + 1,
                            amount=0 if i % 2 == 0 else Spec.MAX_AMOUNT,
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
                            validator_public_key=0x01,
                            amount=0,
                        ),
                        fee=0,
                    ),
                    WithdrawalRequestTransaction(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=0x02,
                            amount=Spec.MAX_AMOUNT - 1,
                        ),
                        fee=Spec.get_fee(0),
                        nonce=1,
                    ),
                ],
            ],
            id="single_block_multiple_withdrawal_request_first_reverts",
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
                        fee=Spec.get_fee(0),
                    ),
                    WithdrawalRequestTransaction(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=0x02,
                            amount=Spec.MAX_AMOUNT - 1,
                        ),
                        fee=0,
                        nonce=1,
                    ),
                ],
            ],
            id="single_block_multiple_withdrawal_request_last_reverts",
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
                        fee=Spec.get_fee(0),
                        # Value obtained from trace minus one
                        gas_limit=114_247 - 1,
                        valid=False,
                    ),
                    WithdrawalRequestTransaction(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=0x02,
                            amount=0,
                        ),
                        fee=Spec.get_fee(0),
                        nonce=1,
                    ),
                ],
            ],
            id="single_block_multiple_withdrawal_request_first_oog",
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
                        fee=Spec.get_fee(0),
                    ),
                    WithdrawalRequestTransaction(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=0x02,
                            amount=0,
                        ),
                        fee=Spec.get_fee(0),
                        # Value obtained from trace minus one
                        gas_limit=80_047 - 1,
                        nonce=1,
                        valid=False,
                    ),
                ],
            ],
            id="single_block_multiple_withdrawal_request_last_oog",
        ),
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestTransaction(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=i + 1,
                            amount=0 if i % 2 == 0 else Spec.MAX_AMOUNT,
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
            id="multiple_block_above_max_withdrawal_requests_from_eoa",
        ),
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestContract(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=0x01,
                            amount=0,
                        ),
                        fee=Spec.get_fee(0),
                    ),
                ],
            ],
            id="single_block_single_withdrawal_request_from_contract",
        ),
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestContract(
                        withdrawal_request=[
                            WithdrawalRequest(
                                validator_public_key=i + 1,
                                amount=Spec.MAX_AMOUNT - 1 if i % 2 == 0 else 0,
                            )
                            for i in range(Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK)
                        ],
                        fee=Spec.get_fee(0),
                    ),
                ],
            ],
            id="single_block_multiple_withdrawal_requests_from_contract",
        ),
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestContract(
                        withdrawal_request=[
                            WithdrawalRequest(
                                validator_public_key=i + 1,
                                amount=Spec.MAX_AMOUNT - 1 if i % 2 == 0 else 0,
                            )
                            for i in range(Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK)
                        ],
                        fee=[0] + [Spec.get_fee(0)] * (Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK - 1),
                    ),
                ],
            ],
            id="single_block_multiple_withdrawal_requests_from_contract_first_reverts",
        ),
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestContract(
                        withdrawal_request=[
                            WithdrawalRequest(
                                validator_public_key=i + 1,
                                amount=Spec.MAX_AMOUNT - 1 if i % 2 == 0 else 0,
                            )
                            for i in range(Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK)
                        ],
                        fee=[Spec.get_fee(0)] * (Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK - 1) + [0],
                    ),
                ],
            ],
            id="single_block_multiple_withdrawal_requests_from_contract_last_reverts",
        ),
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestContract(
                        withdrawal_request=[
                            WithdrawalRequest(
                                validator_public_key=i + 1,
                                amount=Spec.MAX_AMOUNT - 1 if i % 2 == 0 else 0,
                            )
                            for i in range(Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK)
                        ],
                        fee=Spec.get_fee(0),
                        call_gas=[0x100] + [-1] * (Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK - 1),
                        valid=[False] + [True] * (Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK - 1),
                    ),
                ],
            ],
            id="single_block_multiple_withdrawal_requests_from_contract_first_oog",
        ),
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestContract(
                        withdrawal_request=[
                            WithdrawalRequest(
                                validator_public_key=i + 1,
                                amount=Spec.MAX_AMOUNT - 1 if i % 2 == 0 else 0,
                            )
                            for i in range(Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK)
                        ],
                        fee=Spec.get_fee(0),
                        call_gas=[-1] * (Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK - 1) + [0x100],
                        valid=[True] * (Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK - 1) + [False],
                    ),
                ],
            ],
            id="single_block_multiple_withdrawal_requests_from_contract_last_oog",
        ),
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestContract(
                        withdrawal_request=[
                            WithdrawalRequest(
                                validator_public_key=i + 1,
                                amount=Spec.MAX_AMOUNT - 1 if i % 2 == 0 else 0,
                            )
                            for i in range(Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK)
                        ],
                        fee=Spec.get_fee(0),
                        valid=False,
                        extra_code=Op.REVERT(0, 0),
                    ),
                ],
            ],
            id="single_block_multiple_withdrawal_requests_from_contract_caller_reverts",
        ),
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestContract(
                        withdrawal_request=[
                            WithdrawalRequest(
                                validator_public_key=i + 1,
                                amount=Spec.MAX_AMOUNT - 1 if i % 2 == 0 else 0,
                            )
                            for i in range(Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK)
                        ],
                        fee=Spec.get_fee(0),
                        valid=False,
                        extra_code=Macros.OOG(),
                    ),
                ],
            ],
            id="single_block_multiple_withdrawal_requests_from_contract_caller_oog",
        ),
        pytest.param(
            # Test the first 50 fee increments
            get_n_fee_increment_blocks(50),
            id="multiple_block_fee_increments",
        ),
        pytest.param(
            [
                # Block 1
                [
                    WithdrawalRequestContract(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=0x01,
                            amount=0,
                        ),
                        fee=Spec.get_fee(0),
                        call_type=Op.DELEGATECALL,
                        contract_address=0x200,
                        nonce=0,
                        valid=False,
                    ),
                    WithdrawalRequestContract(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=0x01,
                            amount=0,
                        ),
                        fee=Spec.get_fee(0),
                        call_type=Op.STATICCALL,
                        contract_address=0x300,
                        nonce=1,
                        valid=False,
                    ),
                    WithdrawalRequestContract(
                        withdrawal_request=WithdrawalRequest(
                            validator_public_key=0x01,
                            amount=0,
                        ),
                        fee=Spec.get_fee(0),
                        call_type=Op.CALLCODE,
                        contract_address=0x400,
                        nonce=2,
                        valid=False,
                    ),
                ],
            ],
            id="single_block_single_withdrawal_request_delegatecall_staticcall_callcode",
        ),
    ],
)
def test_withdrawal_requests(
    blockchain_test: BlockchainTestFiller,
    blocks: List[Block],
    pre: Dict[Address, Account],
):
    """
    Test making a withdrawal request to the beacon chain.
    """
    blockchain_test(
        genesis_environment=Environment(),
        pre=pre,
        post={},
        blocks=blocks,
    )


@pytest.mark.parametrize(
    "withdrawal_requests,block_requests,exception",
    [
        pytest.param(
            [],
            [
                WithdrawalRequest(
                    validator_public_key=0x01,
                    amount=0,
                    source_address=Address(0),
                ),
            ],
            BlockException.INVALID_REQUESTS,
            id="no_withdrawals_non_empty_requests_list",
        ),
        pytest.param(
            [
                WithdrawalRequestTransaction(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x01,
                        amount=0,
                    ),
                    fee=Spec.get_fee(0),
                ),
            ],
            [],
            BlockException.INVALID_REQUESTS,
            id="single_withdrawal_request_empty_requests_list",
        ),
        pytest.param(
            [
                WithdrawalRequestTransaction(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x01,
                        amount=0,
                    ),
                    fee=Spec.get_fee(0),
                ),
            ],
            [
                WithdrawalRequest(
                    validator_public_key=0x02,
                    amount=0,
                    source_address=TestAddress,
                )
            ],
            BlockException.INVALID_REQUESTS,
            id="single_withdrawal_request_public_key_mismatch",
        ),
        pytest.param(
            [
                WithdrawalRequestTransaction(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x01,
                        amount=0,
                    ),
                    fee=Spec.get_fee(0),
                ),
            ],
            [
                WithdrawalRequest(
                    validator_public_key=0x01,
                    amount=1,
                    source_address=TestAddress,
                )
            ],
            BlockException.INVALID_REQUESTS,
            id="single_withdrawal_request_amount_mismatch",
        ),
        pytest.param(
            [
                WithdrawalRequestTransaction(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x01,
                        amount=0,
                    ),
                    fee=Spec.get_fee(0),
                ),
            ],
            [
                WithdrawalRequest(
                    validator_public_key=0x01,
                    amount=0,
                    source_address=TestAddress2,
                )
            ],
            BlockException.INVALID_REQUESTS,
            id="single_withdrawal_request_source_address_mismatch",
        ),
        pytest.param(
            [
                WithdrawalRequestTransaction(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x01,
                        amount=0,
                    ),
                    fee=Spec.get_fee(0),
                    nonce=0,
                ),
                WithdrawalRequestTransaction(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x02,
                        amount=0,
                    ),
                    fee=Spec.get_fee(0),
                    nonce=1,
                ),
            ],
            [
                WithdrawalRequest(
                    validator_public_key=0x02,
                    amount=0,
                    source_address=TestAddress,
                ),
                WithdrawalRequest(
                    validator_public_key=0x01,
                    amount=0,
                    source_address=TestAddress,
                ),
            ],
            BlockException.INVALID_REQUESTS,
            id="two_withdrawal_requests_out_of_order",
        ),
        pytest.param(
            [
                WithdrawalRequestTransaction(
                    withdrawal_request=WithdrawalRequest(
                        validator_public_key=0x01,
                        amount=0,
                    ),
                    fee=Spec.get_fee(0),
                ),
            ],
            [
                WithdrawalRequest(
                    validator_public_key=0x01,
                    amount=0,
                    source_address=TestAddress,
                ),
                WithdrawalRequest(
                    validator_public_key=0x01,
                    amount=0,
                    source_address=TestAddress,
                ),
            ],
            BlockException.INVALID_REQUESTS,
            id="single_withdrawal_requests_duplicate_in_requests_list",
        ),
    ],
)
def test_withdrawal_requests_negative(
    blockchain_test: BlockchainTestFiller,
    withdrawal_requests: List[WithdrawalRequestTransactionBase],
    block_requests: List[WithdrawalRequest],
    exception: BlockException,
):
    """
    Test blocks where the requests list and the actual withdrawal requests that happened in the
    block's transactions do not match.
    """
    # No previous block so fee is the base
    fee = 1
    current_block_requests = []
    for w in withdrawal_requests:
        current_block_requests += w.valid_withdrawal_requests(fee)
    included_withdrawal_requests = current_block_requests[: Spec.MAX_WITHDRAWAL_REQUESTS_PER_BLOCK]

    pre = {}
    for w in withdrawal_requests:
        pre.update(w.pre())

    blockchain_test(
        genesis_environment=Environment(),
        pre=pre,
        post={},
        blocks=[
            Block(
                txs=[w.transaction() for w in withdrawal_requests],
                header_verify=Header(
                    requests_root=included_withdrawal_requests,
                ),
                requests=block_requests,
                exception=exception,
            )
        ],
    )
