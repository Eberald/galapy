# Author: Enrico Veraldi
# check the sed extraction in CLOUDY format
# galapy/spectroscopy/utils/spectra.py

import json
import numpy as np
import pytest

import galapy.internal.constants as CONST
import galapy.spectroscopy.utils.spectra as spc

#===================CONST FOR ANALYZE THE SSPs
RJ_BREAK_A = 1.5e6
H_MINUS_A = 1.64e4
RJ_SLOPE = -4.0
RJ_TOL = 0.30
RJ_RUN = 3
PROBE_MIN_A = 1.0e5
SAFETY_DEX = 0.2

#===================FUNCTION DIAGNOSTIC USED FOR TEST
def log_slope(lam, y):
    """
    Calculates the logarithmic slope of a data set.

    This function computes the slope of the logarithm of `y` with respect to the
    logarithm of `lam` for valid and finite input values.

    Parameters:
    lam : numpy.ndarray or list
        The input array or list containing `lam` values. Must be finite and
        greater than zero to be considered valid.
    y : numpy.ndarray or list
        The input array or list containing `y` values. Must be finite and
        greater than zero to be considered valid.

    Returns:
    tuple
        A tuple where the first element is a 1D numpy array of sorted valid
        `lam` values, and the second element is a 1D numpy array of computed
        logarithmic slopes corresponding to the sorted `lam` values. If there
        are fewer than three valid data points, the slope array will contain
        NaN values.
    """
    lam = np.asarray(lam, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y) & (y > 0) & np.isfinite(lam) & (lam > 0)
    o = np.argsort(lam[m])
    lv, yv = lam[m][o], y[m][o]
    if lv.size < 3:
        return lv, np.full(lv.shape, np.nan)
    return lv, np.gradient(np.log10(yv), np.log10(lv))


def rj_break(lam, L_lambda, probe_min_A=PROBE_MIN_A, tol=RJ_TOL, run=RJ_RUN):
    """
    Determines the break point in a series of log-scaled data based on specified criteria.

    This function analyzes the input data using log values and slopes to identify a break
    point where the conditions on log value thresholds, slope tolerances, and run criteria
    are met.

    Parameters:
    lam : numpy.ndarray
        Wavelength or independent variable values.
    L_lambda : numpy.ndarray
        Corresponding values associated with lam, typically representing intensity or
        another dependent variable.
    probe_min_A : float, optional
        Minimum allowed value for the log-scaled independent variable, lam. Defaults
        to PROBE_MIN_A.
    tol : float, optional
        Tolerance level to compare the slopes against the reference slope
        (RJ_SLOPE). Defaults to RJ_TOL.
    run : int, optional
        Number of consecutive matches required to confirm the break point. Defaults
        to RJ_RUN.

    Returns:
    float or None
        The break point value from the log-scaled independent variable that satisfies the
        specified conditions, or None if no such break point exists.
    """
    lv, sl = log_slope(lam, L_lambda)
    idx = np.where(lv >= probe_min_A)[0]
    if idx.size < run:
        return None
    bad = np.abs(sl[idx] - RJ_SLOPE) > tol
    for k in range(bad.size - run + 1):
        if bad[k:k + run].all():
            return float(lv[idx][k])
    return None

def value_pathologies(L_lambda, lam, lam_min_A=PROBE_MIN_A):
    """
    Analyzes and counts specific conditions in input arrays related to their finite and positivity states,
    including subsets restricted by a wavelength threshold.

    Parameters:
    - L_lambda: list | ndarray
      Input array specifying the intensity values to analyze.
    - lam: list | ndarray
      Input array specifying the corresponding wavelengths.
    - lam_min_A: float
      The wavelength threshold in Ångströms. Used to determine which array elements are
      considered for counting specific pathologies. Defaults to PROBE_MIN_A.

    Returns:
    - dict
      A dictionary containing the count of pathologies:
        - 'n_nonfinite': Count of non-finite (e.g., NaN or infinite) values in L_lambda.
        - 'n_nonpositive': Count of values in L_lambda that are non-positive (zero or negative).
        - 'n_nonfinite_red': Count of non-finite values in L_lambda for elements that satisfy lam >= lam_min_A.
        - 'n_nonpositive_red': Count of non-positive values in L_lambda for elements that satisfy lam >= lam_min_A.
    """
    y = np.asarray(L_lambda, dtype=float)
    lam = np.asarray(lam, dtype=float)
    nf = ~np.isfinite(y)
    np_ = np.isfinite(y) & (y <= 0)
    red = lam >= lam_min_A
    return {
        'n_nonfinite': int(nf.sum()),
        'n_nonpositive': int(np_.sum()),
        'n_nonfinite_red': int((nf & red).sum()),
        'n_nonpositive_red': int((np_ & red).sum()),
    }

def power_beyond(lam, L_lambda, lam_cut_A):
    """
    Calculates the proportion of the total integrated power captured in a
    specified wavelength range to the total power across all wavelengths.

    Parameters:
    lam: array-like
        Wavelength values. Must be a numeric array-like object.
    L_lambda: array-like
        Power or intensity values corresponding to the `lam` wavelengths.
        Must be a numeric array-like object.
    lam_cut_A: float
        Wavelength cutoff in Angstroms. All wavelengths greater than or equal
        to this value are considered part of the specified range.

    Returns:
    float
        Proportion of the integrated power within the specified wavelength
        range (`lam >= lam_cut_A`) compared to the total integrated power across
        all wavelengths. If insufficient data is available for computation or
        total power is non-positive, returns 0.0. If the input data does not allow
        integration, returns NaN.

    Raises:
        Does not explicitly raise errors; assumes input validation and will
        generate NaN, 0.0, or other calculated results based on input validity.
    """
    lam = np.asarray(lam, dtype=float)
    y = np.asarray(L_lambda, dtype=float)
    m = np.isfinite(y) & (y > 0)
    lv, yv = lam[m], y[m]
    o = np.argsort(lv)
    lv, yv = lv[o], yv[o]
    if lv.size < 2:
        return float('nan')
    tot = np.trapezoid(yv, lv)
    tail = lv >= float(lam_cut_A)
    if tail.sum() < 2 or tot <= 0:
        return 0.0
    return float(np.trapezoid(yv[tail], lv[tail]) / tot)

def tail_index(lam, L_lambda, lam_lo_A=1.0e7):
    """
    Calculate the tail index, scatter, and the number of points for the specified data.

    Parameters:
    lam: array-like
        The wavelength values.
    L_lambda: array-like
        The corresponding luminosity values for the wavelengths.
    lam_lo_A: float, optional
        The lower bound for the wavelength in angstroms. Defaults to 1.0e7.

    Returns:
    dict
        A dictionary containing:
            - 'index': The median value of the log slope for wavelengths greater
              than or equal to lam_lo_A, or None if there are insufficient points.
            - 'scatter': The standard deviation of the log slope for wavelengths
              greater than or equal to lam_lo_A, or None if there are insufficient
              points.
            - 'n_points': The number of valid points used in the calculation.

    Notes:
    The function first calculates the log slope using the provided wavelength
    and luminosity values. It then filters the data to include only wavelengths
    meeting the specified lower bound. If fewer than three points meet this
    criterion, no further processing is performed, and None is returned for the
    index and scatter. Otherwise, the median and standard deviation of the log
    slopes for the filtered data are computed and returned.
    """
    lv, sl = log_slope(lam, L_lambda)
    m = lv >= float(lam_lo_A)
    if m.sum() < 3:
        return {'index': None, 'scatter': None, 'n_points': int(m.sum())}
    return {'index': float(np.median(sl[m])), 'scatter': float(np.std(sl[m])),
    'n_points': int(m.sum())}

def diagnose_node(lam, L_lambda, lam_cut_A):
    """
    Diagnoses the properties of a given system based on the provided inputs. The function computes various
    metrics, such as the maximum value of the input lambda array, identifies a critical break point in
    lambda using the Rao-Jia model, computes the fraction of power beyond a cutoff threshold, analyzes the
    tail behavior of the distribution, and evaluates additional anomalies.

    Parameters:
    lam : numpy.ndarray
        An array representing the lambda values for the diagnosis.
    L_lambda : numpy.ndarray
        Corresponding power or intensity values associated with the lambda array.
    lam_cut_A : float
        The cutoff value of lambda for computing the power fraction.

    Returns:
    dict
        A dictionary containing the following keys and their respective computed values:
        - 'lambda_max_table_A': Maximum value of the input lambda array as a float.
        - 'lambda_rj_break_A': The Rao-Jia critical break point value of lambda.
        - 'power_fraction_beyond_cut': Fraction of the power beyond the cutoff value.
        - 'tail_powerlaw': The tail index describing the distribution's tail characteristics.
        Additional diagnostic metrics from value_pathologies are also included.
    """
    lam_rj = rj_break(lam, L_lambda)
    out = {
        'lambda_max_table_A': float(np.max(lam)),
        'lambda_rj_break_A': lam_rj,
        'power_fraction_beyond_cut': power_beyond(lam, L_lambda, lam_cut_A),
        'tail_powerlaw': tail_index(lam, L_lambda),
    }
    out.update(value_pathologies(L_lambda, lam))
    return out

def recommend_lambda_cut(nodes, safety_dex=SAFETY_DEX):
    """
    Recommend a cutoff value for lambda based on safety margin and node parameters.

    This function calculates a recommended cutoff value for lambda using the
    lowest value from the 'lambda_rj_break_A' attribute found in a list of
    nodes. The cutoff is calculated by applying a safety margin to the
    logarithm of the minimum lambda value.

    Parameters:
        nodes (list[dict]): A list of nodes, where each node is represented as a dictionary
            containing a key 'lambda_rj_break_A' with a numeric value or None.
        safety_dex (float): A safety margin to adjust the cutoff, applied as a
            subtraction in the logarithmic scale. Default is a constant value `SAFETY_DEX`.

    Returns:
        float or None: The recommended lambda cutoff value as a float if any valid
            'lambda_rj_break_A' value exists in the list of nodes, otherwise None.
    """
    br = [n['lambda_rj_break_A'] for n in nodes if n['lambda_rj_break_A'] is not None]
    if not br:
        return None
    return float(10.0 ** (np.log10(min(br)) - float(safety_dex)))

#============================= TEST

@pytest.fixture
def ssp():
    """
    Provides a pytest fixture for generating a synthetic spectral profile.

    This fixture creates a synthetic spectrum with a power-law profile
    representing common astrophysical models. A transition occurs at
    a predefined wavelength (`RJ_BREAK_A`), where the slope changes.
    The spectrum is useful for testing and validating spectral analysis
    functions or models.

    Returns:
        tuple: A tuple `(lam, y)` where:
            - lam (numpy.ndarray): Array of wavelengths sampled logarithmically.
            - y (numpy.ndarray): Corresponding flux values following a
              power-law distribution with a modified tail.

    """
    lam = np.logspace(1, 10, 3000)
    y = 1e-5 * (lam / 1e4) ** -4.0
    tail = lam > RJ_BREAK_A
    y[tail] = y[tail][0] * (lam[tail] / lam[tail][0]) ** -1.2
    return lam, y

def _read(path):
    """
    Reads data from a file and parses it into two separate numpy arrays: one for wavelengths and one
    for corresponding values.

    Parameters:
    path (str): The file path to read data from. The file should contain numerical data where each
        line represents a pair of values: a wavelength and its corresponding value. Lines starting
        with '#' or empty lines are ignored.

    Returns:
    tuple[numpy.ndarray, numpy.ndarray]: A tuple containing two numpy arrays. The first array contains
        the parsed wavelengths, and the second array contains their corresponding values. Both arrays
        are of type float.

    Raises:
    ValueError: If the file contains non-numeric data in the relevant lines.
    """
    lam, val = [], []
    for raw in open(path):
        raw = raw.strip()
        if not raw or raw.startswith('#'):
            continue
        tok = raw.split()
        lam.append(float(tok[0])); val.append(float(tok[1]))
    return np.array(lam), np.array(val)

@pytest.mark.unit
def test_sed_format_six_facts(tmp_path, ssp):
    """
    Test the correct formatting of a spectral energy distribution (SED) file.

    This test ensures that the generated SED file complies with the required
    format specifications and verifies the correctness of its content. It
    checks various conditions regarding units, extrapolation, positivity of
    flux values, and monotonicity of the wavelength values.

    Parameters:
        tmp_path (Path): Temporary directory path where the test files
            will be written.
        ssp (tuple): A tuple containing the wavelength array and the
            corresponding flux array generated for a test stellar population.

    Raises:
        AssertionError: If any of the conditions for proper formatting,
            positivity, or monotonicity is not met.
    """
    lam, y = ssp
    p = spc.write_cloudy_sed(lam, y, tmp_path / 'a.sed')
    lines = [l for l in open(p) if not l.startswith('#')]
    assert 'nuFnu units Angstroms' in lines[0]
    assert 'extrapolate' not in lines[0]
    assert all('units' not in l for l in lines[1:])
    w, v = _read(p)
    assert np.all(v > 0)
    assert np.all(np.diff(w) > 0)
    assert w.size >= 2

@pytest.mark.unit
def test_duplicate_wavelengths_are_deduplicated(tmp_path):
    """
    Test that duplicate wavelengths are properly deduplicated.

    This test verifies the functionality for removing duplicate wavelengths while
    ensuring that the resulting array remains sorted and contains unique values. It
    assesses the `write_cloudy_sed` method to confirm that duplicates in the input
    data do not propagate to the output.

    Parameters:
    tmp_path : pathlib.Path
        Temporary directory provided by pytest for storing test artifacts.
    """
    lam = np.array([1e3, 1e3, 2e3, 3e3])
    p = spc.write_cloudy_sed(lam, np.ones(4), tmp_path / 'd.sed')
    w, _ = _read(p)
    assert w.size == 3 and np.all(np.diff(w) > 0)


@pytest.mark.unit
def test_nonpositive_flux_is_floored_not_dropped(tmp_path):
    """
    Test that non-positive flux values in the input are floored instead of dropped.

    This unit test ensures that flux values that are non-positive are set to a floor value
    and retained in the output rather than being dropped entirely. It verifies that the
    output size is consistent with the input size and that all flux values in the resulting
    output are strictly positive.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory path provided by the pytest fixture to simulate a working
        environment for writing files.

    Raises
    ------
    AssertionError
        If the size of the output wavelengths array does not match the input size or
        if any flux values in the output are not strictly positive.
    """
    lam = np.array([1e3, 2e3, 3e3])
    p = spc.write_cloudy_sed(lam, np.array([1.0, 0.0, 1.0]), tmp_path / 'f.sed')
    w, v = _read(p)
    assert w.size == 3 and np.all(v > 0)

@pytest.mark.unit
def test_blue_truncation_applied(tmp_path, ssp):
    """
    Test that blue truncation is correctly applied when writing a Cloudy SED file.

    This test ensures that the minimum wavelength in the output SED file is greater than
    or equal to the specified Lyman limit. It validates the behavior of the `_read()`
    function and the Cloudy SED writing process.

    Parameters:
        tmp_path (Path): Temporary directory path for test file creation.
        ssp (Tuple): A tuple containing wavelength array and flux array.

    Raises:
        AssertionError: If the minimum wavelength in the output file is less than
        the Lyman limit.
    """
    lam, y = ssp
    w, _ = _read(spc.write_cloudy_sed(lam, y, tmp_path / 'b.sed',
                                      lambda_min_A=CONST.LyLimit))
    assert w.min() >= CONST.LyLimit


@pytest.mark.unit
def test_red_truncation_applied(tmp_path, ssp):
    """
    Unit test for verifying if the red truncation is applied correctly to the SED output.

    The function tests whether the maximum wavelength in the output spectral energy
    distribution (SED) file does not exceed the threshold defined in CONST.SED_cut.

    Parameters:
    tmp_path : pathlib.Path
        Temporary directory path provided by pytest for file operations.
    ssp : Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the wavelength array (lam) and flux array (y).

    Assertions:
    - Confirms the maximum wavelength of the generated SED file is less than or
      equal to CONST.SED_cut.
    """
    lam, y = ssp
    w, _ = _read(spc.write_cloudy_sed(lam, y, tmp_path / 'r.sed',
                                      lambda_max_A=CONST.SED_cut))
    assert w.max() <= CONST.SED_cut


@pytest.mark.unit
def test_both_truncations_compose(tmp_path, ssp):
    """
    Unit test to verify that truncations in the spectral energy distribution (SED)
    are correctly applied and can compose together into a valid output.

    Parameters:
    tmp_path : pytest.TempPathFactory
        Temporary path fixture to write test files.
    ssp : tuple
        Tuple containing wavelength array (lam) and corresponding spectrum (y).

    Raises:
    AssertionError
        If the minimum wavelength of the result is below the lower limit or the
        maximum wavelength exceeds the upper limit.
    """
    lam, y = ssp
    w, _ = _read(spc.write_cloudy_sed(lam, y, tmp_path / 'c.sed',
                                      lambda_min_A=CONST.LyLimit,
                                      lambda_max_A=CONST.SED_cut))
    assert w.min() >= CONST.LyLimit and w.max() <= CONST.SED_cut


@pytest.mark.unit
def test_sectors_differ_only_at_the_blue_end(tmp_path, ssp):
    """
    Test function to verify that sectors differ only at the blue end of the spectrum.

    This test checks the maximum values of the spectra generated by two different
    SED files (`h.sed` and `p.sed`) written using the `write_cloudy_sed` function.
    The spectra are compared to ensure they agree at their maximum points while
    differing due to their applied wavelength limits.

    Parameters:
    tmp_path : pytest.TempPathFactory
        A temporary path provided by pytest for creating necessary files during the
        test execution.

    ssp : tuple
        A tuple containing wavelength and flux data, where:
        lam : ndarray
            Array of wavelengths.
        y : ndarray
            Array of corresponding flux values.
    """
    lam, y = ssp
    wh, _ = _read(spc.write_cloudy_sed(lam, y, tmp_path / 'h.sed',
                                       lambda_max_A=CONST.SED_cut))
    wp, _ = _read(spc.write_cloudy_sed(lam, y, tmp_path / 'p.sed',
                                       lambda_min_A=CONST.LyLimit,
                                       lambda_max_A=CONST.SED_cut))
    assert wh.max() == wp.max()


@pytest.mark.unit
def test_QH_invariant_under_red_cut(tmp_path, ssp):
    """
    Performs a unit test to verify that the total hydrogen ionizing photon
    emission rate (QH) of a spectral energy distribution (SED) remains
    invariant under the application of a wavelength cutoff.

    This test generates two SED files: one without any wavelength
    restriction and one with a wavelength maximum specified. The total QH
    values for both SEDs are then calculated and compared to ensure they are
    equal, confirming the invariance of QH under the imposed cutoff.

    Arguments:
        tmp_path (pathlib.Path): A temporary filesystem path created during
            the test for writing intermediate SED files.
        ssp (tuple): A tuple containing spectral energy distribution
            information (lam and y), where lam represents the wavelength array and
            y represents the flux array.

    Raises:
        AssertionError: If the calculated QH values for the full SED and the
            wavelength-cropped SED are not equal.
    """
    lam, y = ssp
    q_full = spc.cloudy_sed_QH(spc.write_cloudy_sed(lam, y, tmp_path / 'q1.sed'))
    q_cut = spc.cloudy_sed_QH(spc.write_cloudy_sed(lam, y, tmp_path / 'q2.sed',lambda_max_A=CONST.SED_cut))
    assert q_full == q_cut

@pytest.mark.unit
def test_QH_is_zero_on_lyman_truncated_file(tmp_path, ssp):
    """
    Tests whether the ionizing photon production rate (Q_H) is correctly calculated as zero
    for a spectrum truncated at the Lyman limit.

    Parameters:
    tmp_path : pathlib.Path
        Temporary directory path for storing the spectrum file during the test.
    ssp : tuple
        A tuple containing the wavelength array and corresponding spectrum array.

    Raises:
    AssertionError
        If the calculated Q_H value is not equal to 0.0.
    """
    lam, y = ssp
    p = spc.write_cloudy_sed(lam, y, tmp_path / 'z.sed', lambda_min_A=CONST.LyLimit)
    assert spc.cloudy_sed_QH(p) == 0.0

@pytest.mark.unit
def test_rj_slope_is_minus_four_where_physics_says_so(ssp):
    """
    Tests that the slope in the Rayleigh-Jeans regime is equal to minus four, as predicted
    by physics. This ensures that the computed logarithmic slope adheres to the theoretical
    expectations in the range where Rayleigh-Jeans assumptions apply.

    Parameters:
    ssp : Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing two arrays:
        - lam : The wavelengths corresponding to the spectral energy distribution.
        - y : The flux or intensity values of the spectral energy distribution.

    Raises:
    AssertionError
        If the computed logarithmic slope deviates from -4.0 with a tolerance of 0.05 for
        values of log flux density (lv) between 1e5 and half the Rayleigh-Jeans break
        constant (RJ_BREAK_A).
    """
    lam, y = ssp
    lv, sl = log_slope(lam, y)
    m = (lv > 1e5) & (lv < RJ_BREAK_A / 2)
    assert (np.allclose(sl[m], -4.0, atol=0.05))

@pytest.mark.unit
def test_rj_break_recovers_injected_value(ssp):
    """
    Tests the correctness of the `rj_break` function when recovering an
    injected value.

    This test ensures that the `rj_break` function produces a value close
    to the constant `RJ_BREAK_A` based on the provided inputs. The test
    checks if the returned value is not `None` and if its relative
    difference compared to `RJ_BREAK_A` is within a tolerance of 5%.

    Parameters:
    ssp (tuple): A tuple containing `lam` and `y` values to inject into the
                 `rj_break` function.
    """
    lam, y = ssp
    got = rj_break(lam, y)
    assert got is not None
    assert abs(got - RJ_BREAK_A) / RJ_BREAK_A < 0.05

@pytest.mark.unit
def test_pure_rj_library_has_no_break():
    """
    Tests if the Rayleigh-Jeans (RJ) library does not encounter a break in the given data range.

    This unit test is designed to verify that the `rj_break` function behaves as
    expected when invoked with a logarithmically spaced array of values and a
    functionally defined RJ spectrum. The primary goal of the test is to ensure
    that the function identifies the absence of a break condition in these inputs.

    Parameters:
    lam : ndarray
        A logarithmically spaced array which defines the wavelengths, represented in a range
        between 10^1 and 10^10.

    Raises:
    AssertionError
        If the `rj_break` function fails to return `None`, indicating an unexpected
        break condition.
    """
    lam = np.logspace(1, 10, 2000)
    assert rj_break(lam, (lam / 1e4) ** -4.0) is None

@pytest.mark.unit
def test_pathologies_are_counted(ssp):
    """
    Tests the functionality of counting pathologies in a spectrum.

    This test verifies the behavior of the `value_pathologies` function when
    applied to a spectrum with specific pathological modifications. The test
    modifies the spectrum by zeroing out values corresponding to a specific
    range of wavelengths and ensures the reported counts of non-positive
    values are as expected.

    Parameters:
    ssp : tuple
        A tuple containing:
        - lam (array-like): Wavelength values of the spectrum.
        - y (array-like): Intensity values of the spectrum.
    """
    lam, y = ssp
    y = y.copy()
    y[(lam > 3e6) & (lam < 4e6)] = 0.0
    rep = value_pathologies(y, lam)
    assert rep['n_nonpositive'] > 0 and rep['n_nonpositive_red'] > 0

@pytest.mark.unit
def test_power_beyond_cut_is_negligible_for_rj_tail(ssp):
    """
    Unit test to validate that power beyond the cut for the Rayleigh-Jeans tail is negligible.

    This test ensures that the function `power_beyond` produces a result with a value less
    than the specified threshold (1e-4) when applied to the Rayleigh-Jeans (RJ) tail of
    a spectrum.

    Arguments:
        ssp (tuple): A tuple containing the wavelength array (`lam`) and `y` array,
                     representing the spectrum.

    Raises:
        AssertionError: If the result of `power_beyond` exceeds the threshold (1e-4).
    """
    lam, y = ssp
    assert power_beyond(lam, y, CONST.SED_cut) < 1e-4

@pytest.mark.unit
def test_tail_index_discriminates_powerlaw(ssp):
    """
    Unit test for the `tail_index` function to verify its ability to distinguish
    a power-law distribution within the provided data. The test ensures that the
    calculated tail index value aligns closely with the expected result and that
    the associated uncertainty scatter remains minimal.

    Parameters:
    ssp (tuple): A tuple containing `lam`, a parameter array, and `y`, the
        observed data.

    Raises:
    AssertionError: If the computed `tail_index` does not meet expected accuracy
        or scatter constraints.
    """
    lam, y = ssp
    t = tail_index(lam, y)
    assert abs(t['index'] + 1.2) < 0.05 and t['scatter'] < 0.05

@pytest.mark.unit
def test_QH_roundtrip_within_5pct(tmp_path, ssp):
    """
    Tests the round-trip functionality of the SED (Spectral Energy Distribution) processing for
    photoionization computations, specifically ensuring that the calculated hydrogen-ionizing photon
    flux (QH) deviates by no more than 5% from the reference value.

    Parameters:
    tmp_path : pathlib.Path
        Temporary directory path for writing the intermediate SED file.
    ssp : tuple
        A tuple containing wavelength array (lam) and the corresponding flux array (y).

    Raises:
    AssertionError
        If the round-trip QH calculation deviates from the reference value by more than 5%.
    """
    lam, y = ssp
    HC = CONST.hP["erg*s"]*CONST.clight["cm/s"]
    L_SUN_ERG = CONST.Lsun
    m = lam < 911.6
    x = np.sort(lam[m] * 1e-8)
    ref = float(np.trapezoid((y[m] * lam[m] * L_SUN_ERG)[np.argsort(lam[m])], x) / HC)
    got = spc.cloudy_sed_QH(spc.write_cloudy_sed(lam, y, tmp_path / 'rt.sed'))
    assert (abs(got - ref) / ref < 0.05)

@pytest.mark.unit
def test_pdr_sed_is_truncated_both_sides(tmp_path, ssp):
    """
    This test function validates that the spectral energy distribution (SED) output by the `_read` function
    is truncated within the specified wavelength range and maintains strict ordering. Specifically, the
    test ensures that:

    1. The minimum wavelength value of the output SED is greater than or equal to the defined constant
       `LyLimit`.
    2. The maximum wavelength value of the output SED is less than or equal to the defined constant
       `SED_cut`.
    3. The wavelength values in the output SED are strictly increasing.

    Parameters:
    - tmp_path (Path): Temporary directory path for storing intermediate files during the test execution.
    - ssp (Tuple[np.ndarray, np.ndarray]): A tuple containing a sequence of wavelength values (`lam`) and
      their corresponding intensity values (`y`).

    Assertions:
    - Ensures the minimum wavelength is greater than or equal to `LyLimit`.
    - Ensures the maximum wavelength is less than or equal to `SED_cut`.
    - Ensures the wavelength values are strictly monotonically increasing.
    """
    lam, y = ssp
    w, _ = _read(spc.write_cloudy_sed(lam, y, tmp_path / 'pdr.sed',
                                     lambda_min_A=CONST.LyLimit,
                                     lambda_max_A=CONST.SED_cut))
    assert w.min() >= CONST.LyLimit
    assert w.max() <= CONST.SED_cut
    assert np.all(np.diff(w) > 0)

@pytest.mark.unit
def test_recommend_cut_is_conservative(ssp):
    """
    A unit test for the `recommend_lambda_cut` function, ensuring that the function behaves
    conservatively in its recommendations based on input data regarding lambda values and
    threshold criteria.

    Args:
        ssp (tuple): A tuple where the first element is `lam` (a diagnostic lambda value),
            and the second element is `y` (associated data, context not provided).

    Raises:
        AssertionError: If the following conditions are not met:
            - The recommended lambda cut value is not `None` and is strictly less than `RJ_BREAK_A`.
            - The function produces correct lambda recommendations when provided with a list of
              nodes containing `lambda_rj_break_A` values, including edge cases such as `None`.

    Notes:
        - The function `recommend_lambda_cut` determines a recommended lambda cut value based
          on inputs.
        - Node data points with different `lambda_rj_break_A` thresholds are tested, and the
          function ensures correct behavior when these values are present or absent.
    """
    lam, y = ssp
    rec = recommend_lambda_cut([diagnose_node(lam, y, CONST.SED_cut)])
    assert rec is not None and rec < RJ_BREAK_A
    nodes = [{'lambda_rj_break_A': 2.0e6}, {'lambda_rj_break_A': 8.0e5},
    {'lambda_rj_break_A': None}]
    assert recommend_lambda_cut(nodes) < 8.0e5
    assert recommend_lambda_cut([{'lambda_rj_break_A': None}]) is None

@pytest.mark.unit
def test_report_is_json_serializable(ssp):
    """
    Unit test to verify if the function `test_report_is_json_serializable` ensures proper serialization of given input data
    into JSON format and the validity of extracted lambda values.

    The test validates the following:
    - The function generates a report dictionary (`rep`) containing diagnostic information and recommended lambda cut values.
    - The report is serialized to JSON format without errors.
    - The first node in the report has a `lambda_max_table_A` value greater than zero.

    Parameters:
    ssP : tuple
        A tuple containing input parameters `lam` and `y` used for function processing within the test.

    Raises:
    AssertionError
        If the JSON serialization fails or if the `lambda_max_table_A` value in the report does not satisfy the test condition.

    """
    lam, y = ssp
    rep = {'nodes': [diagnose_node(lam, y, CONST.SED_cut)],
    'recommended_lambda_cut_A': recommend_lambda_cut(
    [diagnose_node(lam, y, CONST.SED_cut)])}
    assert json.loads(json.dumps(rep))['nodes'][0]['lambda_max_table_A'] > 0
