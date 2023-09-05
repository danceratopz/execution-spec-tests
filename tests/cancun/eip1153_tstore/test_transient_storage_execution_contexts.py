"""
abstract: Tests for [EIP-1153: Transient Storage](https://eips.ethereum.org/EIPS/eip-1153)

    Test cases for `TSTORE` and `TLOAD` opcode calls in different execution contexts.
"""  # noqa: E501

from enum import Enum, unique

import pytest

from ethereum_test_tools import Account, Environment
from ethereum_test_tools import Opcodes as Op
from ethereum_test_tools import StateTestFiller, TestAddress, Transaction

from .spec import ref_spec_1153

REFERENCE_SPEC_GIT_PATH = ref_spec_1153.git_path
REFERENCE_SPEC_VERSION = ref_spec_1153.version

pytestmark = [pytest.mark.valid_from("Shanghai")]

# Address used to call the test bytecode on every test case.
caller_address = 0x100

# Address of the callee contract
callee_address = 0x200


@unique
class TStorageCallContextTestCases(Enum):
    """
    Transient storage test cases for different contract subcall contexts.
    """

    CALL = {
        "pytest_param": pytest.param(id="call"),
        "description": (
            "TSTORE0001: Caller and callee contracts use their own transient storage when callee "
            "is called via CALL."
        ),
        "caller_bytecode": (
            Op.TSTORE(0, 420)
            + Op.SSTORE(2, Op.CALL(Op.GAS(), callee_address, 0, 0, 0, 0, 0))
            + Op.SSTORE(0, Op.TLOAD(0))
            + Op.SSTORE(1, Op.TLOAD(1))
        ),
        "callee_bytecode": (
            Op.TSTORE(1, 69) + Op.SSTORE(0, Op.TLOAD(0)) + Op.SSTORE(1, Op.TLOAD(1))
        ),
        "expected_caller_storage": {0: 420, 1: 0, 2: 1},
        "expected_callee_storage": {0: 0, 1: 69},
    }
    STATICCALL = {
        "pytest_param": pytest.param(id="staticcall"),
        "description": ("TSTORE0002: A STATICCALL caller can not use transient storage."),
        "caller_bytecode": (
            Op.TSTORE(0, 420)
            + Op.STATICCALL(Op.GAS(), callee_address, 0, 0, 0, 0)
            + Op.SSTORE(0, Op.TLOAD(0))
        ),
        "callee_bytecode": Op.SSTORE(0, Op.TLOAD(0)),
        "expected_caller_storage": {0: 0},  # TODO: Should this be 420?
        "expected_callee_storage": {0: 0},
    }
    CALLCODE = {
        "pytest_param": pytest.param(id="callcode"),
        "description": (
            "TSTORE0003: Caller and callee contracts share transient storage "
            "when callee is called via CALLCODE."
        ),
        "caller_bytecode": (
            Op.TSTORE(0, 420)
            + Op.CALLCODE(Op.GAS(), callee_address, 0, 0, 0, 0, 0)
            + Op.SSTORE(0, Op.TLOAD(0))
        ),
        "callee_bytecode": Op.SSTORE(1, Op.TLOAD(0)),
        "expected_caller_storage": {0: 420, 1: 420},
        "expected_callee_storage": {0: 0, 0: 0},
    }
    DELEGATECALL = {
        "pytest_param": pytest.param(id="delegatecall"),
        "description": (
            "TSTORE0004: Caller and callee contracts share transient storage "
            "when callee is called via DELEGATECALL."
        ),
        "caller_bytecode": (
            Op.TSTORE(0, 420)
            + Op.DELEGATECALL(Op.GAS(), callee_address, 0, 0, 0, 0)
            + Op.SSTORE(0, Op.TLOAD(0))
        ),
        "callee_bytecode": Op.SSTORE(1, Op.TLOAD(0)),
        "expected_caller_storage": {0: 420, 1: 420},
        "expected_callee_storage": {0: 0, 0: 0},
    }

    def __init__(self, test_case):
        self._value_ = test_case["pytest_param"]
        self.description = test_case["description"]
        self.env = Environment()
        self.pre = {
            TestAddress: Account(balance=10**40),
            caller_address: Account(code=test_case["caller_bytecode"]),
            callee_address: Account(code=test_case["callee_bytecode"]),
        }
        self.post = {
            caller_address: Account(storage=test_case["expected_caller_storage"]),
            callee_address: Account(storage=test_case["expected_callee_storage"]),
        }
        self.txs = [
            Transaction(
                to=caller_address,
                gas_limit=1_000_000,
            )
        ]


@pytest.mark.parametrize(
    "test_case",
    [case for case in TStorageCallContextTestCases],
    ids=[case._value_.id for case in TStorageCallContextTestCases],
)
def test_tstore_tload(
    test_case: TStorageCallContextTestCases,
    state_test: StateTestFiller,
):
    """
    Test transient storage with a subcall using the following opcodes:

    - `CALL`
    - `CALLCODE`
    - `DELEGATECALL`
    - `STATICCALL`
    """
    state_test(
        env=test_case.env,
        pre=test_case.pre,
        post=test_case.post,
        txs=test_case.txs,
    )