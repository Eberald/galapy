# Author: Enrico Veraldi
# CloudIA packaging tests

from importlib.metadata import metadata
import importlib
import pathlib
import subprocess
import sys

import pytest
import galapy.spectroscopy

# installation path
_CLOUDIA_DIR = pathlib.Path(galapy.CloudIA.__file__).resolve().parent


#================== GENERAL MODULES IMPORT TEST =====================
@pytest.mark.unit
def test_cloudia_package_is_importable():
    """
    Tests whether the 'cloudia' package is properly importable by verifying the existence
    of its '__init__.py' file in the specified directory.

    Raises
    ------
    AssertionError
        If the '__init__.py' file is not found in the '_CLOUDIA_DIR' directory.
    """
    assert (_CLOUDIA_DIR / "__init__.py").is_file()


@pytest.mark.unit
def test_cloudia_cli_is_importable():
    """
    Test if the CloudIA CLI module is successfully importable.

    This test ensures that the CloudIA CLI component is correctly
    installed and can be imported without any issues.

    @pytest.mark.unit
    """
    import galapy.spectroscopy.cloudia


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_name",
    [
        "galapy.CloudIA.scripts",
        "galapy.CloudIA.scripts.cloudy_common",
        #"galapy.CloudIA.scripts.cloudy_common.abundances",
        "galapy.CloudIA.scripts.cloudy_hii",
        "galapy.CloudIA.scripts.cloudy_pdr",
    ],
)
def test_pipeline_modules_are_importable(module_name):
    """
    Test to ensure that specified pipeline modules can be imported.

    This test verifies that the given pipeline modules are properly defined and
    can be imported without errors. Each module's importability is checked to
    ensure their presence and correct implementation in the project structure.

    Parameters:
        module_name: str
            Fully qualified name of the module to test for importability.

    Raises:
        ImportError: If the specified module cannot be imported.

    Tags:
        Unit Test, Parametrized

    Related:
        - importlib.import_module
    """
    importlib.import_module(module_name)


"""
WIP, placeholder for check presence chemistry (N/O, grain, C/O, He/H)
@pytest.mark.unit
def test_shared_chemistry_is_reachable():
    from galapy.CloudIA.scripts.cloudy_common.abundances import (
        nitrogen_offset,
        carbon_offset,
        helium_scale_factor,
        grain_scale_from_xi_d,
    ) """


#========================= TEST NOT IMPORT HEAVY DEPENDENCES (PYTORCH ETC) =====================

@pytest.mark.unit
def test_import_cloudia_does_not_pull_heavy_deps():
    """
    Test to ensure that importing galapy.CloudIA does not pull in heavy dependencies.

    Summary:
    This test verifies that the importation of the `galapy.CloudIA` module does not cause heavy dependencies
    such as `torch`, `gpytorch`, or `sklearn` to be loaded into the system. The test executes the Python
    interpreter as a subprocess and checks for the presence of these modules in the `sys.modules` dictionary.

    Raises:
    subprocess.CalledProcessError: If the subprocess invocation fails or if heavy dependencies
    are loaded, the subprocess will exit with a non-zero status, causing this exception to be raised.
    """
    code = (
        "import sys; import galapy.CloudIA; "
        "heavy = [m for m in ('torch', 'gpytorch', 'sklearn') if m in sys.modules]; "
        "sys.exit('Heavy imports present: %s' % heavy if heavy else 0)"
    )
    subprocess.check_call([sys.executable, "-c", code], cwd="/tmp")


# ======================== CHECK FILES AND CONFIGURATIONS  ARE PRESENT =======================
"""
WIP, placeholder for check the templates cloudy and lines folders
@pytest.mark.unit
@pytest.mark.parametrize("rel_path", ["configs/lines", "configs/templates"])
def test_config_directories_exist(rel_path):
    assert (_CLOUDIA_DIR / rel_path).is_dir()"""

"""
WIP, placeholder check jinja templates cloudy
@pytest.mark.unit
@pytest.mark.parametrize(
    "rel_file",
    [
        "scripts/cloudy_hii/templates/hii.in.j2",
        "scripts/cloudy_pdr/templates/pdr.in.j2",
    ],
)
def test_pipeline_resources_exist(rel_file):
    assert (_CLOUDIA_DIR / rel_file).is_file()"""


#========================== METADATA VALIDATION =========================

@pytest.mark.unit
def test_extra_cloudia_is_declared_in_metadata():
    """
    Test if "cloudia" is included in the project's extra packages.

    Summary:
    This test checks that the metadata of the "galapy-fit" package includes
    "cloudia" in the extra packages declared under "Provides-Extra". Ensures
    that the specific extra package is properly declared for the project.

    Parameters:
    None

    Assertions:
    Asserts that "cloudia" is present in the list of extra packages provided
    by the metadata.
    """
    md = metadata("galapy-fit")
    extras = md.get_all("Provides-Extra") or []
    assert "cloudia" in extras, (
        "CLOUDIA is not present in the extra packages of the project"
    )


@pytest.mark.unit
def test_base_install_has_no_heavy_deps():
    """
    Unit test to verify that the `galapy-fit` package does not include heavy
    dependencies (torch, gpytorch, scikit-learn, ollama) in its base installation.

    The test retrieves metadata for the `galapy-fit` package, and filters
    requirements to exclude extras. It then checks that none of the base
    requirements overlap with the specified heavy packages.

    Parameters
    ----------
    None

    Raises
    ------
    AssertionError
        If any of the heavy dependencies are present in the base package
        requirements for `galapy-fit`.
    """
    md = metadata("galapy-fit")
    base_requirements = [
        r for r in (md.get_all("Requires-Dist") or []) if "extra ==" not in r
    ]
    heavy_packages = {"torch", "gpytorch", "scikit-learn", "ollama"}

    for req in base_requirements:
        pkg_name = (
            req.split(";")[0].split("[")[0].split()[0].strip().lower()
        )
        assert pkg_name not in heavy_packages, (
            f"Heavy dependence '{pkg_name}' is present in galapy base installation"
        )