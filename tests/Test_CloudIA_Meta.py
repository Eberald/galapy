# Author: Enrico Veraldi
# check requirements distributed same along the package metadata. Check correct test discovery of pytest

from importlib.metadata import metadata
import pathlib
import pytest

import galapy.spectroscopy

_CLOUDIA_DIR = pathlib.Path(galapy.spectroscopy.__file__).resolve().parent

@pytest.mark.unit
def test_requirements_txt_matches_distributed_metadata():
    """
    Test function to verify that the package dependencies listed in the `requirements.txt`
    file align with the metadata dependencies marked for the "cloudia" extra in the
    package's distribution metadata. This ensures consistency between the defined
    dependencies in the file and the actual package metadata.

    Attributes:
        req_file: The path to the `requirements.txt` file used for the validation process.

    Raises:
        AssertionError: Raised when the `requirements.txt` file does not exist or when
        the dependencies listed in the file do not match those specified in the "cloudia"
        extra metadata of the package.
    """
    req_file = _CLOUDIA_DIR / "requirements.txt"
    assert req_file.is_file(), f"Error 404: file not found: {req_file}"

    #requirements packages
    file_requirements = {
        line.strip().lower()
        for line in req_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    #package requirements (metadata)
    md = metadata("galapy-fit")
    requires_dist = md.get_all("Requires-Dist") or []

    #filter extra packages marked as cloudia in the package
    metadata_requirements = set()
    for req in requires_dist:
        if 'extra == "cloudia"' in req or "extra == 'cloudia'" in req:
            pkg_spec = req.split(";")[0].strip().lower()
            metadata_requirements.add(pkg_spec)

    assert file_requirements == metadata_requirements, (
        f"Dependencies are different along the package\n"
        f"In requirements.txt: {sorted(file_requirements)}\n"
        f"In metadata package [cloudia]: {sorted(metadata_requirements)}"
    )


@pytest.mark.unit
def test_pytest_naming_convention_discovery():
    """
    Asserts that the current test file follows the naming convention for pytest discovery.

    This test checks if the current file's name starts with "Test_" to ensure compatibility
    with pytest's naming conventions for test discovery.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the file name does not start with "Test_".
    """
    current_file = pathlib.Path(__file__)
    assert current_file.name.startswith("Test_"), (
        f"File {current_file.name} not named as Test_*.py "
    )


@pytest.mark.unit
def test_install_cloudy_entrypoint_is_declared_in_metadata():
    """
    Test if the entry point for installing Cloudy is declared in the metadata.

    This unit test ensures that the 'galapy-install-cloudy' entry point exists in the
    metadata of the 'galapy-fit' distribution under the 'console_scripts' group.
    It verifies that the entry point name and value are correctly defined.

    Raises:
        AssertionError: If the 'galapy-install-cloudy' entry point is missing or its
        associated value is incorrect.

    """
    from importlib.metadata import distribution
    cs = {ep.name: ep.value for ep in distribution('galapy-fit').entry_points
          if ep.group == 'console_scripts'}
    assert 'galapy-install-cloudy' in cs, (
        f"entry point absent in metadata (console_scripts={sorted(cs)}):")
    assert cs['galapy-install-cloudy'] == 'galapy.spectroscopy.utils.install_cloudy:main'