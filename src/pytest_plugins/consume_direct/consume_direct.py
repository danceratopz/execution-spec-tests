"""
A pytest plugin to execute the blocktest on the specified fixture directory.
"""
from pathlib import Path
from typing import Generator, Optional

import pytest

from evm_transition_tool import TransitionTool
from pytest_plugins.consume.consume import TestCase


def pytest_addoption(parser):  # noqa: D103
    consume_group = parser.getgroup(
        "consume", "Arguments related to consuming fixtures via a client"
    )

    consume_group.addoption(
        "--evm-bin",
        action="store",
        dest="evm_bin",
        type=Path,
        default=None,
        help=(
            "Path to an evm executable that provides `blocktest`. Default: First 'evm' entry in "
            "PATH."
        ),
    )
    consume_group.addoption(
        "--traces",
        action="store_true",
        dest="evm_collect_traces",
        default=False,
        help="Collect traces of the execution information from the transition tool.",
    )
    debug_group = parser.getgroup("debug", "Arguments defining debug behavior")
    debug_group.addoption(
        "--evm-dump-dir",
        action="store",
        dest="base_dump_dir",
        type=Path,
        default=None,
        help="Path to dump the transition tool debug output.",
    )


def pytest_configure(config):  # noqa: D103
    evm = TransitionTool.from_binary_path(
        binary_path=config.getoption("evm_bin"),
        # TODO: The verify_fixture() method doesn't currently use this option.
        trace=config.getoption("evm_collect_traces"),
    )
    try:
        blocktest_help_string = evm.get_blocktest_help()
    except NotImplementedError as e:
        pytest.exit(str(e))
    config.evm = evm
    config.evm_use_single_test = "--run" in blocktest_help_string


@pytest.fixture(autouse=True, scope="session")
def evm(request) -> Generator[TransitionTool, None, None]:
    """
    Returns the interface to the evm binary that will consume tests.
    """
    yield request.config.evm
    request.config.evm.shutdown()


@pytest.fixture(scope="session")
def evm_use_single_test(request) -> bool:
    """
    Helper specifying whether to execute one test per fixture in each json file.
    """
    return request.config.evm_use_single_test


@pytest.fixture(scope="function")
def test_dump_dir(
    request, json_fixture_path: Path, fixture_name: str, evm_use_single_test: bool
) -> Optional[Path]:
    """
    The directory to write evm debug output to.
    """
    base_dump_dir = request.config.getoption("base_dump_dir")
    if not base_dump_dir:
        return None
    if evm_use_single_test:
        if len(fixture_name) > 142:
            # ensure file name is not too long for eCryptFS
            fixture_name = fixture_name[:70] + "..." + fixture_name[-70:]
        return base_dump_dir / json_fixture_path.stem / fixture_name
    return base_dump_dir / json_fixture_path.stem


@pytest.fixture(scope="function")
def json_fixture_path(test_case: TestCase):
    """
    Provide the path to the current JSON fixture file.
    """
    return test_case.json_file


# @pytest.fixture(scope="function")
# def fixture_format(fixture_data: TestCase):
#     """
#     The format of the current fixture.
#     """
#     return fixture_data.fixture_format


@pytest.fixture(scope="function")
def fixture_name(test_case: TestCase):
    """
    The name of the current fixture.
    """
    return test_case.fixture_name