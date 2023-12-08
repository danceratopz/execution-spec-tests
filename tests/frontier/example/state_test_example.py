"""
Converted test: src/GeneralStateTestsFiller/stExample/labelsExampleFiller.yml
Retesteth version: retesteth-0.3.2-cancun+commit.00e9689f.Linux.g++
An example how to use labels in expect section
"""

from collections import namedtuple
from typing import Any, Mapping, Optional

import pytest

from ethereum_test_forks import Berlin, Cancun, Fork, Frontier, Homestead, London, Merge, Shanghai
from ethereum_test_tools import (
    Account,
    Code,
    Environment,
    StateTestFiller,
    TestAddress,
    Transaction,
)

ExpectSectionIndex = namedtuple("ExpectSectionIndex", ["d", "g", "v", "f"])


class ExpectSection:
    """Manage expected post states for state tests transactions"""

    def __init__(self):
        self.sections: list[tuple[ExpectSectionIndex, dict[str, Account]]] = []

    def add_expect(self, ind: ExpectSectionIndex, expect: dict[str, Account]) -> None:
        """Adds a section with a given indexes and expect dictionary."""
        self.sections.append((ind, expect))

    def get_expect(self, tx_ind: ExpectSectionIndex) -> Optional[dict[str, Account]]:
        """Returns the element associated with the given id, if it exists."""
        for section_ind, section in self.sections:
            if (
                (tx_ind.d in section_ind.d or -1 in section_ind.d)
                and (tx_ind.g in section_ind.g or -1 in section_ind.g)
                and (tx_ind.v in section_ind.v or -1 in section_ind.v)
                and (tx_ind.f in section_ind.f)
            ):
                return section
        return None


@pytest.fixture
def env():  # noqa: D103
    return Environment(
        coinbase="0x2adc25665018aa1fe0e6bc666dac8fc2697ff9ba",
        difficulty=0x020000,
        gas_limit=71794957647893862,
        number=1,
        timestamp=1000,
    )


@pytest.fixture
def pre():  # noqa: D103
    return {
        "0x095e7baea6a6c7c4c2dfeb977efac326af552d87": Account(
            balance=1000000000000000000,
            nonce=0,
            code="0x60003560005500",
            storage={},
        ),
        "0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b": Account(
            balance=1000000000000000000,
            nonce=0,
            code="0x",
            storage={},
        ),
    }


@pytest.fixture
def expect():  # noqa: D103
    expect_section = ExpectSection()
    expect_section.add_expect(
        ExpectSectionIndex(d=[0], g=[0], v=[-1], f=[Berlin, London, Merge, Shanghai, Cancun]),
        {
            "0x095e7baea6a6c7c4c2dfeb977efac326af552d87": Account(
                storage={
                    "0x00": "0x0100000000000000000000000000000000000000000000000000000000000000"
                },
            ),
        },
    )
    expect_section.add_expect(
        ExpectSectionIndex(d=[1], g=[0], v=[-1], f=[Berlin, London, Merge, Shanghai, Cancun]),
        {
            "0x095e7baea6a6c7c4c2dfeb977efac326af552d87": Account(
                storage={
                    "0x00": "0x0200000000000000000000000000000000000000000000000000000000000000"
                },
            ),
        },
    )
    expect_section.add_expect(
        ExpectSectionIndex(d=[2, 3], g=[0], v=[-1], f=[Berlin, London, Merge, Shanghai, Cancun]),
        {
            "0x095e7baea6a6c7c4c2dfeb977efac326af552d87": Account(
                storage={
                    "0x00": "0x0300000000000000000000000000000000000000000000000000000000000000"
                },
            ),
        },
    )
    return expect_section


@pytest.mark.valid_from("Berlin")
@pytest.mark.parametrize(
    "tr_data",
    [
        (0, "0x01", ":label transaction1"),
        (1, "0x02", ":label transaction2"),
        (2, "0x03", ":label transaction3"),
        (3, "0x03", ":label transaction3"),
    ],
)
@pytest.mark.parametrize("tr_gasLimit", [(0, 400000)])
@pytest.mark.parametrize("tr_value", [(0, 100000)])
def test_labelsExample_py(
    env: Environment,
    pre: dict,
    expect: ExpectSection,
    fork: Fork,
    tr_data,
    tr_gasLimit,
    tr_value,
    state_test: StateTestFiller,
):
    """
    An example how to use labels in expect section
    """

    tx = Transaction(
        ty=0x0,
        chain_id=0x0,
        nonce=0,
        to="0x095e7baea6a6c7c4c2dfeb977efac326af552d87",
        gas_price=10,
        protected=False,
    )

    dataInd, dataValue, dataLabel = tr_data
    gasInd, gasValue = tr_gasLimit
    valueInd, valueValue = tr_value

    tx_index = ExpectSectionIndex(d=dataInd, g=gasInd, v=valueInd, f=fork)
    post = expect.get_expect(tx_index)

    tx.data = dataValue
    tx.gas_limit = gasValue
    tx.value = valueValue
    state_test(env=env, pre=pre, post=post, txs=[tx])
