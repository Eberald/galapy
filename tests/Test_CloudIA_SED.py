# Author: Enrico Veraldi
# check the sed extraction in CLOUDY format
# galapy/spectroscopy/utils/__init__.py and galapy/spectroscopy/utils/install_cloudy.py

import os
import numpy as np
import pytest

import galapy.spectroscopy.utils.spectra as ssp
import galapy.internal.constants as CONST

#TEST GRID
WAVE = np.array([100., 300., 500., 700., 900., 911.0, 911.5,
                 911.76, 912.0, 1000., 2000., 6000., 1.0e5])
TGRID = np.array([5.0e5, 1.0e6, 1.5e6, 2.0e6, 5.0e6, 1.0e7,
                  2.0e7, 5.0e7, 7.0e7, 1.0e8, 5.0e8, 1.0e9])
ZGRID = np.array([0.0001, 0.0005, 0.0010, 0.0040, 0.0080, 0.0200])


@pytest.fixture
def cube():
    """
    Fixture to generate a 3D cube for testing purposes.

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        A tuple containing:
        - l: numpy.ndarray
            The processed wave array as a float array.
        - TGRID: numpy.ndarray
            A copy of the temperature grid.
        - ZGRID: numpy.ndarray
            A copy of the redshift grid.
        - L: numpy.ndarray
            A 3D array containing the computed cube data, where each element
            represents a combination of wave, temperature, and redshift values.
    """
    l = np.asarray(WAVE, dtype=float)
    L = np.empty((l.size, TGRID.size, ZGRID.size), dtype=float)
    base = (l / 1000.0) ** -2.0
    for it in range(TGRID.size):
        for iz in range(ZGRID.size):
            L[:, it, iz] = base * (1.0 + it) * (100.0 ** iz)
    return l, TGRID.copy(), ZGRID.copy(), L


@pytest.fixture
def patched_cube(monkeypatch, cube):
    """
    Fixture to provide a patched version of the `load_ssp_cube` function in the `ess` module.

    Parameters:
        monkeypatch: pytest.MonkeyPatch
            A pytest fixture used to safely modify objects during testing.
        cube: Any
            The cube object to be returned by the patched `load_ssp_cube` function.

    Returns:
        Any: The cube object passed as an input argument.
    """
    monkeypatch.setattr(ssp, 'load_ssp_cube', lambda ssp_lib='parsec22.NT': cube)
    return cube


def _read_sed(path):
    """
    Reads spectral energy distribution (SED) data from a file.

    Parameters:
        path (str): The file path to the spectral energy distribution (SED) file.

    Returns:
        tuple: A tuple containing the following elements:
            - numpy.ndarray: Array of wavelength values extracted from the file.
            - numpy.ndarray: Array of numerical values from the second column of the file.
            - list: List of raw strings representing each relevant line from the file.
    """
    lam, col2, raw = [], [], []
    with open(path) as fh:
        for line in fh:
            s = line.split()
            if not s or s[0].startswith('#'):
                continue
            lam.append(float(s[0]))
            col2.append(float(s[1]))
            raw.append(line.rstrip('\n'))
    return np.asarray(lam), np.asarray(col2), raw


def _QH_reference_from_cube(cube, it, iz, lyman_A=None):
    """
    reference QH from Ronconi+24
    """
    ref = getattr(ssp, 'cloudy_sed_QH_reference', None)
    if ref is not None:  # implementazione canonica: nel modulo
        return float(ref(cube, it, iz))
    lyman_A = CONST.LyLimit if lyman_A is None else lyman_A
    l, _t, _Z, L = cube
    lam = np.asarray(l, dtype=float)
    L_lam = np.asarray(L[:, it, iz], dtype=float)
    euv = lam < lyman_A
    if not np.any(euv):
        return 0.0
    x = lam[euv] * 1e-8  # A -> cm
    nuFnu = L_lam[euv] * lam[euv] * CONST.Lsun  # erg/s per Msun
    o = np.argsort(x)
    return float(np.trapezoid(nuFnu[o], x[o]) /
                 (CONST.hP['erg*s'] * CONST.clight['cm/s']))

@pytest.mark.unit
def test_nuFnu_linear_and_strictly_positive(tmp_path):
    """
    Tests that the second column of the Cloudy-formatted SED file is strictly positive
    and has linear scaling with respect to nuFnu. Additionally, this verifies boundary
    conditions with small or zero values.

    Parameters
    ----------
    tmp_path : pytest.TempPathFactory
        Temporary directory provided by pytest for creating and managing
        test-specific files.

    Assertions
    ----------
    Asserts that the first value in the second column matches its expected positive
    value calculated using the input nuFnu and scaled by the luminosity conversion
    constant.
    Asserts that all elements in column 2 are strictly greater than zero.
    Asserts correct behavior for edge cases, specifically ensuring small or zero
    values are handled properly.
    """
    p = tmp_path / 'a.sed'
    ssp.write_cloudy_sed(np.array([1000., 2000., 3000.]),
                     np.array([1., 0., -5.]), str(p))
    _, col2, _ = _read_sed(p)
    assert col2[0] == pytest.approx(1.0 * 1000.0 * CONST.Lsun, rel=1e-6)  # (a)
    assert np.all(col2 > 0.0)  # (b)
    assert col2[1] == pytest.approx(1e-300) and col2[2] == pytest.approx(1e-300)


@pytest.mark.unit
def test_units_keyword_is_parseable_by_cloudy(tmp_path):
    """
    Unit test to validate the parseability of keyword 'units' by CLOUDY within
    specific SED files generated by the `write_cloudy_sed`
    function. This test ensures correct formatting and content of the SED files in
    accordance with CLOUDY's requirements.

    Parameters:
        tmp_path (pathlib.Path): Temporary file path provided by pytest's fixture.

    Raises:
        AssertionError: If any of the assert conditions fail, indicating that the
        specific keywords or data expected within the generated SED file were not
        parsed or formatted correctly.
    """
    p = tmp_path / 'b.sed'
    ssp.write_cloudy_sed(np.array([1000., 2000., 3000.]), np.ones(3), str(p))
    _, _, raw = _read_sed(p)
    tok = raw[0].split()
    assert float(tok[0]) == pytest.approx(1000.0) and float(tok[1]) > 0.0
    kw = [t.upper() for t in tok[2:]]
    assert any(k[:4] == 'NUFN' for k in kw), f"prefix NUFN absent: {kw}"
    i = next((j for j, k in enumerate(kw) if k[:4] == 'UNIT'), None)
    assert i is not None, f"prefix UNIT absent: {kw}"
    assert i + 1 < len(kw), "'units' with no unit after specified"
    assert kw[i + 1][:4] == 'ANGS', (
        f"after  'units' is {kw[i + 1]!r}: CLOUDY not read it")
    assert not any(k[:4] == 'EXTR' for k in kw)
    q = tmp_path / 'c.sed'
    ssp.write_cloudy_sed(np.array([1000., 2000.]), np.ones(2), str(q), extrapolate=True)
    kw_q = [t.upper() for t in _read_sed(q)[2][0].split()[2:]]
    assert any(k[:4] == 'EXTR' for k in kw_q)
    assert any(k[:4] == 'NUFN' for k in kw_q)


@pytest.mark.unit
def test_wavelengths_are_strictly_monotonic(tmp_path):
    """
    Tests that the wavelengths in the generated SED file are strictly monotonic.

    This test verifies that the wavelengths in the output SED file,
    created using `write_cloudy_sed`, are sorted in ascending order,
    and that the wavelength array does not have any non-monotonic
    values. This ensures that the wavelengths are properly ordered
    for downstream applications.

    Parameters
    ----------
    tmp_path : Path
        A temporary file path provided by pytest fixture where the test SED
        file will be stored.

    Raises
    ------
    AssertionError
        Raises an assertion error if the generated SED file does not have
        strictly monotonic wavelength values or if the sorting does not
        match the expected result.
    """
    p = tmp_path / 'd.sed'
    ssp.write_cloudy_sed(np.array([3000., 1000., 2000., 1000.]),
                     np.array([1., 2., 3., 9.]), str(p))
    lam, _, _ = _read_sed(p)
    assert lam.tolist() == pytest.approx([1000., 2000., 3000.])
    assert np.all(np.diff(lam) > 0.0)


@pytest.mark.unit
def test_extract_writes_one_file_per_requested_node(tmp_path, patched_cube):
    """
    Test the behavior of the extract_ssp_seds function to ensure correct file creation and metadata handling.

    Summary:
    This test verifies that the extract_ssp_seds function creates one file per requested node.
    It checks the number of generated files, metadata entries, and ensures that all nodes are processed correctly
    without silent overwrites.

    Parameters:
    tmp_path : pathlib.Path
        A temporary filesystem path provided by pytest's tmp_path fixture for creating test files.
    patched_cube : MockedCubeClass
        A patched cube object representing the data to be extracted and processed.

    Raises:
    AssertionError
        If the number of generated files does not match the expected count, or if metadata entries
        or node processing results in mismatches.
    """
    n, meta = ssp.extract_ssp_seds(str(tmp_path / 'SED'))
    files = sorted(p.name for p in (tmp_path / 'SED').glob('*.sed'))
    assert n == len(ssp.taus) * len(ssp.Z) == 48
    assert len(files) == n, (
        f"n={n} but {len(files)} files found on disc: {n - len(files)} nodes "
        "overwritten")
    assert len(meta) == n
    assert len({e['sed_file'] for e in meta}) == n


@pytest.mark.unit
def test_metadata(tmp_path, patched_cube):
    """
    Unit test to verify the integrity and correctness of metadata extracted during SSP SED processing.

    This test ensures that the metadata produced by `ssp.extract_ssp_seds` contains the required keys, 
    validates file paths, and checks specific field uniqueness within the returned metadata.

    Args:
        tmp_path (pathlib.Path): Temporary directory path provided by pytest to store test output.
        patched_cube: Mocked or patched cube data required for test execution.

    Raises:
        AssertionError: If any required key is missing in the metadata, if file paths in the metadata
        are invalid, or if the field `Qh_unit` does not have a unique value across the metadata entries.
    """
    outdir = tmp_path / 'SED'
    _n, meta = ssp.extract_ssp_seds(str(outdir))
    required = {'tau_SSP', 'Z_star', 'sed_file', 'it', 'iz', 'Qh_unit'}
    for e in meta:
        assert required <= set(e), f"missink keys: {required - set(e)}"
        assert os.path.basename(e['sed_file']) == e['sed_file']
        assert (outdir / e['sed_file']).is_file()
    assert len({e['Qh_unit'] for e in meta}) == len(meta)


@pytest.mark.physics
@pytest.mark.parametrize('it,iz', [(1, 0), (9, 5)])
def test_QH_round_trip(tmp_path, cube, it, iz):
    """
    Test QH (ionizing photon rate) round-trip consistency through SED file I/O.

    QH is the rate of hydrogen-ionizing photons (photons/s per M_sun) computed by
    integrating the SED below the Lyman limit (912 Å). This test validates that
    writing a CLOUDY SED file and reading it back preserves QH to within 5% error.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for test output.
    cube : tuple
        Test spectral cube (wavelength, temp_grid, metal_grid, luminosity).
    it : int
        Temperature grid index.
    iz : int
        Metallicity grid index.
    """
    p = tmp_path / f'node_{it}_{iz}.sed'
    ssp.to_cloudy_sed(str(p), cube=cube, it=it, iz=iz)
    q_ref = _QH_reference_from_cube(cube, it, iz)
    assert q_ref > 0.0, "degenerate node"
    rel = abs(ssp.cloudy_sed_QH(str(p)) - q_ref) / q_ref
    assert rel < 0.05, f"FAIL, difference higher 5% {rel:.3%} (node it={it}, iz={iz})"
    assert rel < 1e-4, f"degenerate difference: {rel:.2e}"


@pytest.mark.physics
def test_QH_absolute_scale_is_anchored(tmp_path):
    """
    Tests that the absolute scaling of QH is correctly anchored for the given
    SED with a flat distribution. This ensures
    that the calculated ionizing photon emission rate is consistent with the
    expected theoretical result.

    Parameters:
    tmp_path (pathlib.Path): Temporary directory path provided by pytest to store
                             generated files during the test.

    Assertions:
    Verifies that the ionizing photon number rate calculated using
    `ssp.cloudy_sed_QH` matches the expected theoretical value, allowing for a
    small relative tolerance of 1e-6.
    """
    A = 7.0e40  # erg/s
    lam = np.array([100., 300., 500., 700., 900.])  #  < LyLimit
    L_lam = A / (lam * CONST.Lsun)  # nuFnu = L_lambda*lambda*Lsun = A
    p = tmp_path / 'flat.sed'
    ssp.write_cloudy_sed(lam, L_lam, str(p))
    expected = A * (900.0 - 100.0) * 1e-8 / (CONST.hP['erg*s'] *
                                             CONST.clight['cm/s'])
    assert ssp.cloudy_sed_QH(str(p)) == pytest.approx(expected, rel=1e-6)


@pytest.mark.regression
def test_truncation_is_actually_applied(tmp_path, patched_cube):
    """
    Test that truncation is correctly applied during SSP SED extraction.
    The test validates that no SED files created during the extraction contain
    values below the Lyman limit, ensuring proper truncation.

    Attributes:
        tmp_path (PosixPath): Temporary directory path created for the test.
        patched_cube: A mocked cube object used for the SSP extraction operation.

    Parameters:
        tmp_path: A temporary directory fixture for file operations during the test.
        patched_cube: A fixture or mock object representing the cube used for SSP
                      extraction.

    Raises:
        AssertionError: If metadata generated during extraction is not empty.
        AssertionError: If any offenders (SED nodes with data below the Lyman limit)
                        are detected.
    """
    outdir = tmp_path / 'SED'
    n, meta = ssp.extract_ssp_seds(str(outdir), truncate_lyman=CONST.LyLimit)
    assert meta == []
    offenders = [(p.name, float(_read_sed(p)[0].min()))
                 for p in sorted(outdir.glob('*.sed'))
                 if _read_sed(p)[0].min() < CONST.LyLimit]
    assert not offenders, f"nodes PDR not cut: {offenders[:5]} ({len(offenders)})"
