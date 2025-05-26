import json
import shutil
from pathlib import Path

import pytest

from cli.gen_index import generate_fixtures_index # Main function to test
from ethereum_test_fixtures.consume import IndexFile, TestCaseIndexFile # For validation
from ethereum_test_fixtures.base import PytestMarkerInfo # For creating expected data

# Helper function to create dummy fixture files (simplified)
def create_dummy_fixture_file(
    output_dir: Path,
    fixture_name: str,
    test_id: str,
    markers_info: list | None,
    content_override: dict | None = None
):
    fixture_file_path = output_dir / f"{fixture_name}.json"
    file_content = {}
    if content_override:
        file_content = content_override
    else:
        # Construct a minimal valid fixture structure
        info_dict = {
            "fixture-format": "state_test", # Example format
            "hash": "0x123", # Dummy hash
            "fork": "Cancun", # Example fork
        }
        if markers_info is not None:
            # gen_index expects markers to be dicts here, as they come from JSON
            info_dict["markers"] = [marker.model_dump() for marker in markers_info]

        file_content[test_id] = {
            "_info": info_dict,
            # Other minimal fixture data if necessary for Fixtures.model_validate_json
            "transaction": {}, 
            "pre": {},
            "post": {}
        }
    
    fixture_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fixture_file_path, "w") as f:
        json.dump(file_content, f)
    return fixture_file_path

@pytest.fixture
def temp_fixture_dir(tmp_path: Path) -> Path:
    # Create a temporary directory for fixture files
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    # Create .meta subdir as gen_index expects it for output
    meta_dir = fixture_dir / ".meta"
    meta_dir.mkdir()
    return fixture_dir

def test_gen_index_populates_markers(temp_fixture_dir: Path):
    # 1. Prepare: Create fixture files with marker data
    marker1 = PytestMarkerInfo(name="slow", args=[], kwargs={})
    marker2 = PytestMarkerInfo(name="eip", args=[7702], kwargs={})
    
    create_dummy_fixture_file(
        temp_fixture_dir,
        "fixture_with_markers",
        "test_one",
        markers_info=[marker1, marker2]
    )
    create_dummy_fixture_file(
        temp_fixture_dir,
        "fixture_no_markers",
        "test_two",
        markers_info=[] # Empty list of markers
    )
    create_dummy_fixture_file(
        temp_fixture_dir,
        "fixture_none_markers", # Markers field absent in _info
        "test_three",
        markers_info=None
    )

    # 2. Act: Run generate_fixtures_index
    generate_fixtures_index(input_path=temp_fixture_dir, quiet_mode=True, force_flag=True)

    # 3. Assert: Check the generated index.json
    index_json_path = temp_fixture_dir / ".meta" / "index.json"
    assert index_json_path.exists()

    with open(index_json_path, "r") as f:
        index_data_raw = json.load(f)
    
    # Validate with Pydantic model to make access easier and ensure structure
    index_file_obj = IndexFile.model_validate(index_data_raw)

    assert len(index_file_obj.test_cases) == 3
    
    found_test_one = False
    found_test_two = False
    found_test_three = False

    # Sort test_cases by id for consistent order in assertions
    sorted_test_cases = sorted(index_file_obj.test_cases, key=lambda tc: tc.id)

    for tc_index_file in sorted_test_cases:
        if tc_index_file.id == "test_one":
            found_test_one = True
            assert tc_index_file.markers is not None
            assert len(tc_index_file.markers) == 2
            # Pydantic should have converted dicts to PytestMarkerInfo objects
            # Sort markers by name for consistent order
            sorted_markers = sorted(tc_index_file.markers, key=lambda m: m.name)
            assert sorted_markers[0].name == "eip" # eip comes before slow alphabetically
            assert sorted_markers[0].args == [7702]
            assert sorted_markers[1].name == "slow"
        elif tc_index_file.id == "test_two":
            found_test_two = True
            assert tc_index_file.markers is not None # Should be an empty list
            assert len(tc_index_file.markers) == 0
        elif tc_index_file.id == "test_three":
            found_test_three = True
            # If markers_info was None, serialized_markers in gen_index.py was None.
            # The TestCaseIndexFile field is Optional[List[PytestMarkerInfo]] = None
            # So it should be None.
            assert tc_index_file.markers is None 

    assert found_test_one and found_test_two and found_test_three

# Add more tests if necessary for other scenarios or edge cases.
# For example, a fixture file that is malformed, or other variations.
