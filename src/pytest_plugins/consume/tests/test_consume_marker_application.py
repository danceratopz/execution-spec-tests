import json
from pathlib import Path
import pytest # This import is for the test file itself

# Helper function to create a dummy index.json for consume tests
def create_dummy_index_json_for_consume(
    pytester,
    test_id: str,
    fixture_json_name: str, 
    markers_data: list | None 
):
    index_content = {
        "root_hash": "0xabc",
        "created_at": "2023-01-01T00:00:00",
        "test_count": 1,
        "forks": ["Cancun"],
        "fixture_formats": ["state_test"],
        "test_cases": [
            {
                "id": test_id,
                "json_path": fixture_json_name, 
                "fixture_hash": "0x123",
                "fork": "Cancun",
                "format": "state_test", 
                "markers": markers_data if markers_data is not None else [] 
            }
        ]
    }
    # pytester.path is the root of the temporary test directory (e.g., /tmp/pytest-of-user/pytest-0/test_X0)
    # We want to create .meta/index.json inside this root.
    meta_dir = pytester.path / ".meta"
    meta_dir.mkdir(exist_ok=True)
    index_file = meta_dir / "index.json"
    with open(index_file, "w") as f:
        json.dump(index_content, f)
    return index_file

# Helper function to create a minimal dummy fixture file that index.json can point to
def create_minimal_dummy_fixture(pytester, fixture_json_name: str, test_id: str):
    fixture_content = {
        test_id: {
            "_info": {
                "fixture-format": "state_test",
                "fork": "Cancun",
            },
            "transaction": {}, "pre": {}, "post": {} 
        }
    }
    # This fixture file should also be at the root of the pytester.path,
    # as json_path in index.json is relative to the input_dir.
    fixture_file = pytester.path / fixture_json_name
    with open(fixture_file, "w") as f:
        json.dump(fixture_content, f)
    return fixture_file

def test_consume_applies_markers(pytester):
    test_id_in_index = "example_test_from_index"
    dummy_fixture_filename = "dummy_fixture_for_consume.json"
    
    markers_to_apply = [
        {"name": "slow", "args": [], "kwargs": {}},
        {"name": "custom_consume_marker", "args": [100], "kwargs": {"y": 200}},
    ]
    
    create_dummy_index_json_for_consume(
        pytester,
        test_id=test_id_in_index,
        fixture_json_name=dummy_fixture_filename,
        markers_data=markers_to_apply
    )
    create_minimal_dummy_fixture(pytester, dummy_fixture_filename, test_id_in_index)

    # This test file will be run by pytest, and consume plugin will parametrize it
    pytester.makepyfile(f"""
import pytest # Import pytest here for the test file being created by makepyfile

# This is the function that the consume plugin will find and parametrize
def test_state_test_runner(test_case): 
    # test_case will be an instance of TestCaseIndexFile or TestCaseStream
    # populated by the consume plugin based on index.json
    assert test_case.id == "{test_id_in_index}"
""")

    # Run pytest; the consume plugin should pick up index.json from pytester.path
    # because --input defaults to the current directory if not specified,
    # and pytester runs from its temporary directory root.
    result = pytester.runpytest(f"--input={str(pytester.path)}")
    result.assert_outcomes(passed=1) 

    items = pytester.getitems()
    assert len(items) == 1
    item = items[0]

    # item.own_markers contains the markers applied to this specific test item
    applied_markers = {marker.name: marker for marker in item.own_markers}

    assert "slow" in applied_markers
    assert not applied_markers["slow"].args  # Access args from the MarkDecorator object
    assert not applied_markers["slow"].kwargs # Access kwargs from the MarkDecorator object
    
    assert "custom_consume_marker" in applied_markers
    custom_marker = applied_markers["custom_consume_marker"]
    assert custom_marker.args == (100,) # args is a tuple
    assert custom_marker.kwargs == {"y": 200}
    
    # Check for default markers (fork and format)
    assert "Cancun" in applied_markers # Fork marker
    assert "state_test" in applied_markers # Format marker

def test_consume_no_markers_in_index(pytester):
    test_id_in_index = "test_no_custom_markers"
    dummy_fixture_filename = "dummy_fixture_no_markers.json"

    create_dummy_index_json_for_consume(
        pytester,
        test_id=test_id_in_index,
        fixture_json_name=dummy_fixture_filename,
        markers_data=[] # Empty list for markers in index.json
    )
    create_minimal_dummy_fixture(pytester, dummy_fixture_filename, test_id_in_index)

    pytester.makepyfile(f"""
import pytest # Import pytest here for the test file being created by makepyfile

def test_state_test_runner(test_case): 
    assert test_case.id == "{test_id_in_index}"
""")
    
    result = pytester.runpytest(f"--input={str(pytester.path)}")
    result.assert_outcomes(passed=1)

    items = pytester.getitems()
    assert len(items) == 1
    item = items[0]
    
    custom_marker_count = 0
    # Default markers that are always expected (fork and format)
    default_markers = {"Cancun", "state_test"} 

    for marker in item.own_markers:
        if marker.name not in default_markers:
             custom_marker_count +=1 # Count any markers that are not the default ones
    
    assert custom_marker_count == 0, "No custom markers should have been applied"
    assert "Cancun" in {m.name for m in item.own_markers}
    assert "state_test" in {m.name for m in item.own_markers}
