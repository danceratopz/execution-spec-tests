"""
Common procedures to test
[EIP-7002: Execution layer triggerable exits](https://eips.ethereum.org/EIPS/eip-7002)
"""  # noqa: E501

from dataclasses import dataclass


# Constants
@dataclass(frozen=True)
class Spec:
    """
    Parameters from the EIP-7002 specifications as defined at
    https://eips.ethereum.org/EIPS/eip-7002#configuration

    If the parameter is not currently used within the tests, it is commented
    out.
    """

    VALIDATOR_EXIT_PRECOMPILE_ADDRESS = 0x229F8EDAF3C9C852E29034AF9EA3CE16B50AE017
    EXCESS_EXITS_STORAGE_SLOT = 0
    EXIT_COUNT_STORAGE_SLOT = 1
    EXIT_MESSAGE_QUEUE_HEAD_STORAGE_SLOT = 2  # Pointer to head of the exit message queue
    EXIT_MESSAGE_QUEUE_TAIL_STORAGE_SLOT = 3  # Pointer to the tail of the exit message queue
    EXIT_MESSAGE_QUEUE_STORAGE_OFFSET = (
        4  # The start memory slot of the in-state exit message queue
    )
    MAX_EXITS_PER_BLOCK = 16  # Maximum number of exits that can be de-queued into a block
    TARGET_EXITS_PER_BLOCK = 2
    MIN_EXIT_FEE = 1
    EXIT_FEE_UPDATE_FRACTION = 17
    EXCESS_RETURN_GAS_STIPEND = 2300
