"""
Converted test: src/GeneralStateTestsFiller/stExample/labelsExampleFiller.yml
Retesteth version: retesteth-0.3.2-cancun+commit.00e9689f.Linux.g++
An example how to use labels in expect section
"""

from enum import unique
from typing import Mapping

import pytest

from ethereum_test_forks import Fork
from ethereum_test_tools import Account, Environment, StateTestFiller, Transaction

from .parametrization_helpers import PytestParameterEnum


@unique
class ExampleIdTestCases(PytestParameterEnum):
    """
    An example of how to parametrize tests with a custom id.
    """

    transaction_one = {
        "description": "Set the transaction data to 0x00 and check the storage",
        "pytest_id": "tr_value_(0, 100000)-tr_gasLimit_(0, 400000)-tr_data_(0, '0x01', ':label transaction1')",  # noqa: E501
        "tx_data": "0x01",
        "expected_storage": {
            0: "0x0100000000000000000000000000000000000000000000000000000000000000"
        },
    }
    transaction_two = {
        "description": "Set the transaction data to 0x01 and check the storage",
        "pytest_id": "tr_value_(0, 100000)-tr_gasLimit_(0, 400000)-tr_data_(1, '0x02', ':label transaction2')",  # noqa: E501
        "tx_data": "0x02",
        "expected_storage": {
            0: "0x0200000000000000000000000000000000000000000000000000000000000000"
        },
    }
    transaction_three = {
        "description": "Set the transaction data to 0x03 and check the storage",
        "pytest_id": "tr_value_(0, 100000)-tr_gasLimit_(0, 400000)-tr_data_(2, '0x03', ':label transaction3')",  # noqa: E501
        "tx_data": "0x03",
        "expected_storage": {
            0: "0x0300000000000000000000000000000000000000000000000000000000000000"
        },
    }
    transaction_four = {
        "description": "Set the transaction data to 0x03 and check the storage",
        "pytest_id": "tr_value_(0, 100000)-tr_gasLimit_(0, 400000)-tr_data_(3, '0x03', ':label transaction3')",  # noqa: E501
        "tx_data": "0x03",
        "expected_storage": {
            0: "0x0300000000000000000000000000000000000000000000000000000000000000"
        },
    }

    def __init__(self, value):
        value = {
            "env": Environment(
                coinbase="0x2adc25665018aa1fe0e6bc666dac8fc2697ff9ba",
                difficulty=0x020000,
                gas_limit=71794957647893862,
                number=1,
                timestamp=1000,
            ),
            "pre": {
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
            },
            "tx": Transaction(
                ty=0x0,
                chain_id=0x0,
                nonce=0,
                to="0x095e7baea6a6c7c4c2dfeb977efac326af552d87",
                gas_price=10,
                protected=False,
                data=value["tx_data"],
                gas_limit=400000,
                value=100000,
            ),
            "post": {
                "0x095e7baea6a6c7c4c2dfeb977efac326af552d87": Account(
                    storage=value["expected_storage"],
                )
            },
        } | {k: value[k] for k in value.keys() if k in self.special_keywords()}
        super().__init__(value)


@ExampleIdTestCases.parametrize()
@pytest.mark.valid_from("Berlin")
def test_example_id(
    state_test: StateTestFiller,
    fork: Fork,
    env: Environment,
    pre: Mapping,
    tx: Transaction,
    post: Mapping,
):
    """
    An example how to set test ids.
    """
    state_test(env=env, pre=pre, post=post, txs=[tx])
