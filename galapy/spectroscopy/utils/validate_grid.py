# Author: Enrico Veraldi
"""
Validation of CloudIA grids
Inputs (all optional except the sector, with default paths derived from data/):
    --sector        hii or pdr (required)
    --data        merged h5 (default: data/products/<sector>_grid.h5)
    --config        grid charter (default: data/grids/<sector>/grid_<sector>.yaml or database)
    --lhs           continuous sample before node expansion (default: data/grids/<sector>/lhs_<sector>.npy)
    --ssp-meta      SSP node metadata (default: data/cloudy_seds/ssp_metadata.json)
    --raw-workdirs  directory of kept CLOUDY working directories (default: data/<sector>/work)
    --report        JSON report (default: data/validation/report_hii.json)
    --markdown      Markdown report (default: data/validation/report_hii.md)

Statuses: PASS, FAIL, SKIP (an input is missing, the detail says which),
PARTIAL (every part that ran passed, some were skipped), ERROR (the check
raised). With --strict a blocking check that ends in SKIP fails the run.

Sectors:  --sector (hii or pdr)

Usage:
    galapy-validate-grid --sector hii
"""

import argparse
import functools
import json
import pathlib
import re
import sys
import traceback
from dataclasses import dataclass

import h5py
import numpy as np
from tqdm import tqdm

import galapy.internal.constants as CONST
import galapy.internal.cloudy as CLOUDY
from galapy.spectroscopy.utils import CLOUDY_REQUIRED
from galapy.spectroscopy.utils.run_core import converged_from_out
from galapy.spectroscopy.utils.hii.parse_one_hii import (parse_cloudy_con, parse_cloudy_linelist, air_to_vacuum_A,
                                                         integrate_grain_abundance, remove_pumping)


PASS, FAIL, SKIP, PARTIAL, ERROR = 'PASS', 'FAIL', 'SKIP', 'PARTIAL', 'ERROR'

# numpy >= 2 names it trapezoid; Python 3.8 only has numpy < 2 (same fallback as galapy.internal.interp)
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz

# dictionaries of .h5
GRID_POINT_PREFIX = 'grid_point_'
CHARTER_TO_ATTRIBUTE = {'f_esc_target': 'f_esc'}
SCALAR_DATASETS = {'f_esc_meas': 'f_esc_meas', 'T_fuv': 'T_fuv_hii', 'N_fuv': 'N_fuv_hii',
                   'dust_mass': 'dust_mass_per_Msun'}
NON_PARAMETER_ATTRIBUTES = {'not_converged', 'energy_balance_rel', 'sed_support_A', 'dust_mass_units',
                            'reprocessing_excess', 'absorbed_fraction', 'lyc_emit_fraction',
                            'f_esc_meas', 'cmb_incident_ratio'}

# readings from the models CLOUDY
CLOUDY_EXIT_OK = 'Cloudy exited OK'
EXEC_TIME_PATTERN = re.compile(r'ExecTime\(s\)\s*([0-9]+(?:\.[0-9]*)?)')
DECK_TABLE_SED = re.compile(r'^[ \t]*table[ \t]+SED[ \t]+"([^"]+)"', re.IGNORECASE | re.MULTILINE)
DECK_LINE_LIST = re.compile(r'^[ \t]*save[ \t]+(?:last[ \t]+)?line[ \t]+list[ \t]+"[^"]+"[ \t]+"([^"]+)"',
                            re.IGNORECASE | re.MULTILINE)

#=======================================================================================================================
# Some thresholds for check the consistency of the grid
THRESHOLDS = {
# Minimum fraction of planned runs that must converge;
    'convergence.min_fraction': 0.99,
# Maximum median relative imbalance |E_inc - E_trans - E_emit| / E_star; verifies global energy conservation
    'energy_balance.max_median': 0.01,
# Minimum absorbed energy fraction; filters out thin models dominated by output noise
    'energy_balance.min_absorbed_fraction': 2e-3,
# Maximum relative energy excess at lambda >= 911.76 A; ensures diffuse emission does not exceed absorbed energy
    'energy_balance.max_observable_excess': 0.02,
# Minimum fraction of graded models satisfying observable energy balance; tolerates edge numerical noise
    'energy_balance.min_pass_fraction': 0.95,
# Maximum relative difference between SED and SSP table Q_H; checks ionizing photon budget consistency
    'ionizing_budget.max_rel_diff': 0.05,
# Theoretical Case B H-alpha/H-beta flux ratio at T_e ~ 1e4 K;
    'case_b.reference': 2.86,
# Maximum relative deviation of median H-alpha/H-beta from 2.86; validates realistic T_e and line RT
    'case_b.max_rel_dev': 0.05,
# Central log U of the Case B evaluation window; selects conditions typical of standard galactic HII regions
    'case_b.logU_centre': -3.0,
# Half-width around log U = -3; samples continuous LHS models near the reference ionization state
    'case_b.logU_half_width': 0.25,
# Maximum redshift for Case B check; isolates low-z models where CMB heating is negligible
    'case_b.max_z_cmb': 1.0,
# (a, x0, b) of Kewley+01; upper theoretical limit for pure stellar photoionization
    'bpt.kewley01': (0.61, 0.47, 1.19),
# Empirical Kauffmann+03 BPT line params;
    'bpt.kauffmann03': (0.61, 0.05, 1.30),
# Upper log U limit for BPT grading; restricts validation to the valid range of the Kewley line
    'bpt.max_logU': -2.0,
# Minimum fraction of log U <= -2 models below Kewley line; rules out unphysical AGN-like excitation (not present in this version)
    'bpt.min_fraction_below': 0.95,
# Minimum Kolmogorov-Smirnov p-value; tests uniform marginal coverage across all LHS
    'lhs.min_p_value': 0.01,
# Allowed range for nu*F_nu dust emission peak; confirms physical equilibrium grain temps (~10-500 K)
    'grain.peak_window_um': (5.0, 500.0),
# Wavelength boundary (3000 A); defines the Wien tail regime where thermal dust emission must vanish
    'grain.uv_limit_um': 0.3,
# Maximum ratio of UV dust emission to IR peak; verifies stochastic heating is off and UV dust is negligible
    'grain.max_uv_to_peak': 1e-3,
# Minimum wavelength of far-IR domain; selects region where dust continuum dominates over nebular gas
    'grain.fir_min_um': 20.0,
# Allowed median ratio of grain to nebular continuum in far-IR; checks physical order of magnitude
    'grain.fir_ratio_median_window': (0.1, 10.0),
# Minimum fraction of models passing grain sanity tests; ensures reliable dust continuum predictions
    'grain.min_pass_fraction': 0.99,
# Fractional excess (10%) above baselines to flag line spikes; detects incomplete line subtraction
    'line_free.spike_excess': 0.10,
#  Maximum fraction of (model, line) pairs with residuals; ensures clean continua
    'line_free.max_flagged_fraction': 0.01,
# Half-width (bins) around line wavelength; integration window to verify rasterised line flux closure
    'closure.half_window_bins': 3,
# Maximum relative deficit of the rasterised energy below the tabulated flux (pumping removed); ensures that col9
# holds the lines that col4 - col9 removes
    'closure.max_rel_dev': 0.02,
# Flux threshold (10% of H-beta); the upper-side excess of strong lines is reported, not graded: col9 is less
# attenuated by the dust of the cloud than the emergent line list
    'closure.strong_line_min_hbeta': 0.1,
# Longest graded wavelength (10 um): beyond it col9 carries the lines net of the CMB (isotropic background), the
# line list does not
    'closure.max_graded_A': 1.0e5,
# Pumped fraction (10%) above which a window is reported, not graded: the removal of the pumping is exact only to
# ~1-20% of the pumped part (pumped and collisional photons are not transferred alike; Si IV 1397, Si III] 1892 on
# real runs), so only below 10% is that error within the 2% tolerance
    'closure.max_pumped_fraction': 0.1,
# Theoretical [O III] 4959/5007 ratio from C25 Einstein A-values;
    'oiii.reference': 0.346,
# Maximum relative deviation from [O III] 0.346; detects corrupted atomic data. 2%: the EMERGENT ratio is
# lowered by up to ~1% by the internal dust (4959 more absorbed than 5007), the intrinsic one is 0.3459
    'oiii.max_rel_dev': 0.02,
# Minimum [O III] 5007 / H-beta graded: below, O++ is absent (old SSP) and the pumping-free remainder is print noise
    'oiii.min_hbeta': 1e-3,
# Maximum residual (dex) from Stromgren dust mass scaling; verifies physical dust mass integration
    'dust.max_abs_residual': 0.2,
# Minimum fraction of models within Stromgren dust scaling; checks geometry and dust density consistency
    'dust.min_pass_fraction': 0.9,
# Target post-starburst stellar ages (70 and 100 Myr); selects nodes where massive stars have died off
    'old_nodes.ages': (7e7, 1e8),
# Maximum ratio of Q_H(old)/Q_H(youngest); confirms ionizing photon emission drops to near-zero with age
    'old_nodes.max_ratio': 0.01,
# Physical evaluation range for measured Lyman continuum escape fraction; bounds coverage evaluation
    'fesc.range': (0.0, 0.9),
# Maximum gap between adjacent sampled f_esc values; avoids unpopulated intervals for smooth emulation
    'fesc.max_gap': 0.1,
# Measured f_esc above which a model has absorbed nothing (a cloud cut at its face): with N_stop >= 0.1 N_S
# at least ~5% of the ionizing photons is absorbed even in hot gas, so f_esc_meas > 0.99 is a stopping fault
    'fesc.face_measured': 0.99,
# Minimum KS p-value for depletion parameter F*; verifies uniform sampling across gas depletion states
    'fstar.min_p_value': 0.01,
# Minimum Habing-band FUV transmittance (T_FUV > 0.01); selects models that illuminate an adjacent PDR
    'fuv.region_min_transmittance': 1e-2,
# Maximum spectral tilt (dex) across Habing band covered by the PDR beta axis: -BETA_MIN x log10(2066/911.76)
    'fuv.max_tilt': 0.888,
# Tolerance on transmission: print precision of the .con (4 significant digits, ~1e-3 on a ratio); T = 1
# outside the SED support and in [0, 1] inside
    'transparency.tolerance': 1e-3,
# Number of gas-phase elements in CLOUDY echo (H to Zn); verifies all 30 tracked species are initialized
    'echo.elements': 30,
# Print precision (0.0001 dex) of abundances in CLOUDY outputs; sets resolution floor for checks
    'echo.print_step': 1e-4,
# Maximum discrepancy (dex) in gas He/H; validates that undepleted He matches galactic enrichment models
    'helium.tolerance': 0.01,
# Maximum deviation (dex) of abundance differences at fixed F*; verifies relative element patterns
    'pattern.max_dev': 1e-3,
# Relative tolerance for incident ionizing photons per unit energy; verifies input SED shape in CLOUDY
    'incident.ionizing_ratio_rtol': 0.05,
# Relative tolerance for incident energy fraction in FUV Habing band; checks stellar illumination shape
    'incident.band_fraction_rtol': 0.05,
# Projected HPC budget (core-hours) for 432,000 models at ~45 s/model; prevents resource exhaustion
    'runtime.budget_core_hours': 5400.0,
    # Absolute tolerance between stored and recomputed imbalance; guards against HDF5 parsing quadrature bugs
    'raw_balance.attribute_atol': 2e-3,
}
T = THRESHOLDS
#=======================================================================================================================
########################################################################################################################

# HELPER functions

def _decode(value):
    """
    Decodes the given value based on its data type.

    If the value is of type bytes or numpy.bytes_, the function decodes it using the
    default string encoding. Otherwise, it converts the value to a string.

    Parameters:
    value : Any
        The input value to decode. It may be of type bytes, numpy.bytes_, or any other
        type that can be converted to a string.

    Returns:
    str
        The decoded or converted string representation of the input value.
    """
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)

def _as_float(value):
    """
    Converts the given value to a scalar float if it is a numeric attribute with
    a single element. Returns None if the conversion is not possible.

    Parameters:
        value: The input value to be converted to a scalar float.

    Returns:
        float: Scalar float representation of the value if conversion is successful.
        None: If the value cannot be converted to a scalar float.
    """
    try:
        array = np.asarray(value)
        if array.dtype.kind in 'iuf' and array.size == 1:
            return float(array.ravel()[0])
    except (TypeError, ValueError):
        pass
    return None

def _clean(obj):
    """
    Creates a JSON-serializable copy of the given object by converting incompatible types such as numpy types,
    NaNs, and pathlib.Path objects into forms that can be serialized.

    Parameters:
    obj: Any
        The object to be processed for JSON serialization.

    Returns:
    Any
        A JSON-serializable copy of the input object.
    """
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_clean(v) for v in (obj.tolist() if isinstance(obj, np.ndarray) else obj)]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, pathlib.Path):
        return str(obj)
    return obj

def _round(value, digits=4):
    """
    Rounds a numeric value to a specified number of digits.

    This function converts the given value to a float and rounds it to
    the specified number of decimal places. If the value is not finite,
    it returns None.

    Parameters:
    value : float
        The numeric value to be rounded.

    digits : int, optional
        The number of decimal places to round to. Defaults to 4.

    Returns:
    float or None
        The rounded value if the input is finite, otherwise None.
    """
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None

def _percentiles(values, qs=(5, 50, 95), digits=4):
    """
    Compute specified percentiles of a numeric dataset.

    This function calculates the percentiles for a given sequence of numeric
    values using numpy. It ignores any non-finite values in the sequence
    (e.g., NaN, infinity). Percentiles are calculated using the corresponding
    quantiles provided as input.

    Parameters:
    values : array-like
        A sequence of numeric values from which percentiles need to be computed.
    qs : tuple of int, optional
        A tuple of integers representing the percentiles to compute. Default is (5, 50, 95).
    digits : int, optional
        The number of decimal places to round the percentile values. Default is 4.

    Returns:
    dict or None
        A dictionary where keys are strings of the form 'pX' (where X is the
        percentile) and values are the corresponding rounded percentile values.
        Returns None if the input sequence contains no finite values.
    """
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return {f'p{q}': _round(np.percentile(values, q), digits) for q in qs} if values.size else None

def _combine(statuses):
    """
    Calculates the combined status from a list of individual statuses.

    This function determines the overall status based on the statuses provided.
    It supports statuses such as SKIP, FAIL, ERROR, PASS, and PARTIAL, applying
    the following logic:
    - If all statuses are SKIP, the combined status is SKIP.
    - If any status is FAIL or ERROR, the combined status is FAIL.
    - If all statuses are PASS, the combined status is PASS.
    - Otherwise, the combined status is PARTIAL.

    Parameters:
    statuses: list
        A list of individual statuses to be combined.

    Returns:
    str
        The overall combined status, which can be SKIP, FAIL, PASS, or PARTIAL.
    """
    if not statuses or all(s == SKIP for s in statuses):
        return SKIP
    if any(s in (FAIL, ERROR) for s in statuses):
        return FAIL
    return PASS if all(s == PASS for s in statuses) else PARTIAL

def _reason(exc):
    """
    Returns the reason for a failure as a formatted string.

    This function generates a string representation of the exception passed to
    it, including the type of the exception and the exception message.

    Parameters:
    exc: Exception
        The exception instance to extract the reason from.

    Returns:
    str
        The formatted string containing the type and message of the exception.
    """
    return f'{type(exc).__name__}: {exc}'

def _scan_key(value):
    """
    Transforms and formats the given value into a manageable scan key.

    This function takes a numerical value, ensures it is a finite
    floating-point number, and formats it to a precision of up to
    10 significant digits. If the value is not finite, the function
    returns None.

    Parameters:
    value: Any
        The input value to be processed into a scan key. It is expected
        to be convertible to a float.

    Returns:
    float or None
        The formatted scan key as a floating-point number with up to
        10 significant digits, or None if the input value is not finite.
    """
    value = float(value)
    return float(f'{value:.10g}') if np.isfinite(value) else None

@functools.lru_cache(maxsize=1)
def _chemistry():
    """
    Provides a cached instance of the Chemistry class, used for accessing the
    abundance database. The method utilizes an LRU cache to store the instance
    for efficient reuse. Raises an exception if the Chemistry module is
    unavailable.

    Returns:
        Chemistry: An instance of the Chemistry class from the abundance module.

    Raises:
        Exception: If the Chemistry module cannot be imported or is unavailable.
    """
    from galapy.spectroscopy.utils.physics.abundances import Chemistry
    return Chemistry()

def _lyman_series_mask(wavelength_A, relative_half_width=3e-3, n_max=40):
    """
    Compute a mask for the Lyman series, identifying wavelength regions that
    are near the series frequencies based on a specified relative half-width.

    This function computes a boolean mask over the provided wavelength values,
    marking positions that are close to the theoretical wavelengths of the
    Lyman series transitions.

    Parameters:
    wavelength_A : array-like
        An array of wavelength values (in Angstroms) to be analyzed.
    relative_half_width : float, optional
        The relative half-width around each Lyman series frequency within
        which a point is considered near. Default is 3e-3.
    n_max : int, optional
        The maximum principal quantum number n to include in the series.
        Default is 40.

    Returns:
    ndarray
        A boolean array of the same shape as `wavelength_A`, where True values
        indicate wavelength positions near the Lyman series transitions.

    """
    near = np.zeros(np.shape(wavelength_A), dtype=bool)
    for n in range(2, n_max + 1):
        near |= np.abs(np.asarray(wavelength_A) / (CONST.LyLimit / (1.0 - 1.0 / n ** 2)) - 1.0) < relative_half_width
    return near

def _integrate_log(values, wavelength, mask):
    """
    Computes the integral over d ln(lambda) along the last axis for the provided
    values and wavelength array, constrained by the given mask. This function
    calculates the energy within a column where nu * F_nu is defined.

    Arguments:
        values: Array of numerical values representing F_nu.
        wavelength: Array of wavelengths corresponding to the values.
        mask: Boolean array specifying the elements to include in the integration.

    Returns:
        Result of the integration as a numerical array.
    """
    x = np.log(np.asarray(wavelength, float)[mask])
    order = np.argsort(x)
    return _trapezoid(np.asarray(values, float)[..., mask][..., order], x[order], axis=-1)

def _spearman(a, b):
    """
    Computes the Spearman rank correlation coefficient for two input arrays.

    The Spearman rank correlation assesses how well the relationship between two variables
    can be described using a monotonic function. This function returns None if the correlation
    is undefined, such as when there are fewer than three valid points or one of the inputs is
    a constant array.

    Parameters:
    a : list or numpy.ndarray
        The first input array to compute the Spearman correlation.
    b : list or numpy.ndarray
        The second input array to compute the Spearman correlation.

    Returns:
    float or None
        The Spearman rank correlation coefficient. Returns None if computation is
        undefined due to insufficient data (< 3 points) or constant input arrays.
    """
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3 or np.ptp(a[good]) == 0 or np.ptp(b[good]) == 0:
        return None
    return float(stats.spearmanr(a[good], b[good])[0])

def _ks_uniform(values, lower, upper, n_effective=None):
    """
    Calculates the Kolmogorov-Smirnov (KS) distance to a uniform distribution U(lower, upper)
    and its associated p-value. The p-value computation considers `n_effective` independent
    draws if provided, otherwise uses the total number of draws.

    Parameters:
    values: list or numpy.ndarray
        A sequence of values to test against the uniform distribution.
    lower: float
        The lower bound of the uniform distribution.
    upper: float
        The upper bound of the uniform distribution.
    n_effective: int, optional
        The effective number of independent draws for p-value computation.

    Returns:
    tuple
        A tuple containing:
        - distance (float): The KS distance between the input values and the uniform distribution.
        - p-value (float): The p-value associated with the KS test.
    """
    from scipy import stats
    u = np.sort((np.asarray(values, float) - lower) / (upper - lower))
    n = u.size
    distance = max(np.max(np.arange(1, n + 1) / n - u), np.max(u - np.arange(n) / n))
    return float(distance), float(stats.kstwo.sf(distance, int(n_effective or n)))

def _below_hyperbola(x, y, a, x0, b):
    """
    Determine whether points lie below a hyperbolic curve on a specific branch.

    This function evaluates whether given points (x, y) in a coordinate system lie
    below the hyperbolic curve defined by the equation y = a / (x - x0) + b,
    specifically on the branch where x < x0.

    Parameters:
    x : array-like
        X-coordinates of the points to evaluate.
    y : array-like
        Y-coordinates of the points to evaluate.
    a : float
        The numerator of the hyperbolic term.
    x0 : float
        The x-coordinate of the vertical asymptote of the hyperbolic curve.
    b : float
        The constant offset added to the hyperbolic term.

    Returns:
    ndarray of bool
        Boolean array indicating whether each point lies below the hyperbolic curve
        on the specified branch.

    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.zeros(x.shape, dtype=bool)
    left = x < x0
    out[left] = y[left] < a / (x[left] - x0) + b
    return out

def _log10_ratio(numerator, denominator):
    """
    Computes the base-10 logarithm of the ratio of two arrays element-wise.

    This function calculates the logarithmic ratio between the provided numerator
    and denominator arrays. It handles invalid or non-positive values in the
    arrays by assigning NaN to the corresponding elements in the output.

    Parameters:
    numerator : array_like
        The array containing the numerator for the ratio calculation. All
        elements are converted to a float type.
    denominator : array_like
        The array containing the denominator for the ratio calculation. All
        elements are converted to a float type.

    Returns:
    ndarray
        An array containing the base-10 logarithm of the ratios, with the
        same shape as the input arrays. Elements corresponding to non-positive
        numerator or denominator values are replaced with NaN.
    """
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.log10(numerator / denominator)
    out[~((numerator > 0) & (denominator > 0))] = np.nan
    return out


########################################################################################################################

# CLOUDY raw files helper functions and inspects products

def _read_text(path):
    """
    Read text file, replacing invalid characters with U+FFFD.

    This function reads the content of a text file at the specified path.
    Any invalid characters encountered during reading are replaced by
    the replacement character U+FFFD.

    Args:
        path (str): The path to the text file to be read.

    Returns:
        str: The content of the text file as a string, with invalid characters replaced.
    """
    return pathlib.Path(path).read_text(errors='replace')

def read_sed_file(path):
    """
    Reads the spectral energy distribution (SED) data from a CLOUDY table-SED file.

    This function processes the input file line by line, extracting the wavelength and
    flux values from non-commented lines. Comment lines, lines with insufficient tokens,
    or lines with invalid numerical values are ignored.

    Parameters:
    path : str
        The file system path to the CLOUDY table-SED file.

    Returns:
    tuple
        A tuple containing two numpy arrays:
        - The first array contains the wavelength values in Angstroms.
        - The second array contains nu*F_nu values.
    """
    wavelength, flux = [], []
    for line in _read_text(path).splitlines():
        tokens = line.split()
        if len(tokens) < 2 or tokens[0].startswith('#'):
            continue
        try:
            wavelength.append(float(tokens[0]))
            flux.append(float(tokens[1]))
        except ValueError:
            continue
    return np.asarray(wavelength), np.asarray(flux)

def incident_shape(wavelength_A, nufnu, lyman_A, band_A):
    """
    Calculates the ionising photon rate per unit energy and the energy fraction
    in a specified band for a given nu*F_nu spectrum.

    This function computes two key quantities from a nu*F_nu spectrum:
    1. The ionising photon rate per unit energy.
    2. The energy fraction within a specified wavelength band.

    The calculations are performed using trapezoidal integration over the
    spectrum, and the results are independent of the normalization of the input
    spectrum. The input parameters must align with the expected formatting and
    values required by the model.

    Parameters:
    wavelength_A : array-like
        Array of wavelengths corresponding to the nu*F_nu values.
    nufnu : array-like
        Array of nu*F_nu values representing the spectrum.
    lyman_A : float
        Wavelength threshold (in Ångströms) below which photons are considered
        ionising.
    band_A : tuple of two floats
        Tuple specifying the lower and upper bounds (in Ångströms) of the
        wavelength band of interest.

    Returns:
    tuple of two floats
        A tuple containing:
        - Ionising photon rate per unit energy.
        - Energy fraction in the specified wavelength band.

    Raises:
    ValueError
        If the input arrays do not have the same length.

    Notes:
    The input wavelengths and nu*F_nu are expected to be positive. Negative
    values in the nu*F_nu array are clipped to zero. If the computed total
    energy of the spectrum is non-positive, the function returns NaN values.
    """
    order = np.argsort(np.asarray(wavelength_A, float))
    w = np.asarray(wavelength_A, float)[order]
    y = np.clip(np.asarray(nufnu, float)[order], 0.0, None)
    energy = _trapezoid(y, np.log(w))
    if not energy > 0:
        return np.nan, np.nan
    ionising = w < lyman_A
    in_band = (w >= band_A[0]) & (w <= band_A[1])
    photons = _trapezoid(y[ionising], w[ionising]) if ionising.sum() > 1 else 0.0
    band = _trapezoid(y[in_band], np.log(w[in_band])) if in_band.sum() > 1 else 0.0
    return float(photons / energy), float(band / energy)

def read_cloudy_banner(out_text, max_lines=200):
    """
    Reads the first line of a CLOUDY output that starts with 'Cloudy', referred to as the version banner.

    This function scans through the given output text up to a specified maximum number of lines
    and returns the first line that begins with the string 'Cloudy'. If no such line is found, it
    returns None.

    Parameters:
        out_text (str): The text content to search for the 'Cloudy' version banner.
        max_lines (int): The maximum number of lines to examine in the provided text. The
            default value is 200.

    Returns:
        str or None: The first line in the output text that starts with 'Cloudy', stripped of
            leading and trailing whitespace. If no line is found, None is returned.
    """
    return next((line.strip() for line in out_text.splitlines()[:max_lines]
                 if line.strip().startswith('Cloudy')), None)

def read_exec_time(out_text):
    """
    Extract the execution time from CLOUDY's final summary line.

    This function parses the provided output text to locate the
    execution time specified in the final summary line. If the
    execution time is found, it is extracted and returned as a
    float. If no execution time is identified, the function
    returns None.

    Parameters:
    out_text: str
        The output text from which execution time is to be parsed.

    Returns:
    float or None
        The extracted execution time as a floating-point number if
        found, otherwise None.
    """
    matches = EXEC_TIME_PATTERN.findall(out_text)
    return float(matches[-1]) if matches else None

#####
_ELEMENT_ENTRY = re.compile(r'([A-Z][a-z]?)\s*:\s*([-+]?\d+\.\d+)')
#####

def read_gas_phase_abundances(out_path):
    """
    Extracts the log10(X/H) values from the 'Gas Phase Chemical Composition'
    block in a given file. If the block is not found, returns None.

    Parameters:
    out_path: str
        The file path to the output containing the 'Gas Phase Chemical
        Composition' data.

    Returns:
    dict or None
        A dictionary with chemical element symbols as keys and their log10(X/H)
        values as float values, extracted from the specified file. Returns None
        if the 'Gas Phase Chemical Composition' block is not present.
    """
    lines = _read_text(out_path).splitlines()
    for i, line in enumerate(lines):
        if 'Gas Phase Chemical Composition' not in line:
            continue
        block = {}
        for following in lines[i + 1:]:
            entries = _ELEMENT_ENTRY.findall(following)
            if not entries:
                if block or following.strip():
                    break
                continue
            block.update({element: float(value) for element, value in entries})
        return block
    return None


def inspect_grain_abundance_file(path):
    """
    Analyze the structure and content of a file containing grain abundance data. The file
    typically contains headers and rows of numeric data, and this function inspects
    those elements to determine specific characteristics of the file.

    Parameters
    ----------
    path : str
        Path to the file to analyze.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - 'depth_headers': int. Count of headers starting with '#Depth'.
        - 'last_header_token': str or None. The last token of the last depth header,
          converted to lowercase, or None if no depth headers exist.
        - 'zones': int. Number of numeric data rows (zones) in the file.
        - 'unreadable_rows': int. Count of rows that couldn't be parsed into numeric data.
        - 'column_counts': list of int. Sorted set of unique column counts across
          numeric rows.
        - 'depth_increasing': bool. Indicates if the depth values in numeric rows
          are strictly increasing.
    """
    headers, rows = [], []
    for line in _read_text(path).splitlines():
        if line.startswith('#'):
            headers.append(line)
        elif line.strip():
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                rows.append(None)
    depth_headers = [h for h in headers if h.lower().startswith('#depth')]
    numeric = [r for r in rows if r is not None]
    depth = np.array([r[0] for r in numeric])
    return {'depth_headers': len(depth_headers),
            'last_header_token': depth_headers[-1].split()[-1].lower() if depth_headers else None,
            'zones': len(numeric), 'unreadable_rows': len(rows) - len(numeric),
            'column_counts': sorted({len(r) for r in numeric}),
            'depth_increasing': bool(depth.size >= 1 and np.all(np.diff(depth) > 0))}

# LINES in the CLOUDY outputs and input

_WAVELENGTH_SUFFIX = {'A': 1.0, 'm': 1.0e4, 'c': 1.0e8}          # Angstrom, micron, cm

def parse_line_label(label):
    """
    Parses a CLOUDY spectral line label and extracts its identifier and wavelength.

    The function processes a CLOUDY label such as 'H  1 6562.80A', extracting the
    element or ion identifier ('H 1') and the wavelength (6562.8) in the proper units,
    based on the suffix of the wavelength. It returns these as a tuple. If the input
    is invalid or cannot be read, the function returns None.

    Args:
        label (str): A string containing the CLOUDY label to be parsed.

    Returns:
        tuple[str, float] or None: A tuple containing the spectral line identifier as
        a string and the wavelength as a float, or None if the label is invalid.
    """
    tokens = str(label).split()
    if len(tokens) < 2 or tokens[-1][-1:] not in _WAVELENGTH_SUFFIX:
        return None
    try:
        return ' '.join(tokens[:-1]), float(tokens[-1][:-1]) * _WAVELENGTH_SUFFIX[tokens[-1][-1]]
    except ValueError:
        return None

class LineCatalogue:
    """Lines of the master list (galapy-dataset) by species and wavelength (relative tolerance 1e-3: air and
    vacuum labels match, distinct lines never do). Blends ('Blnd') never match an ionic species."""

    def __init__(self, labels, wavelengths=None):
        self.labels = [str(label) for label in labels]
        self.species, wavelength = [], []
        for k, label in enumerate(self.labels):
            parsed = parse_line_label(label)
            # a label without wavelength (line_names of the data, e.g. 'H 1') is the species,
            # and the wavelength comes from wavelengths_rest
            species = parsed[0] if parsed else (' '.join(label.split()) or None)
            self.species.append(species.casefold() if species else None)  # CLOUDY labels: case-insensitive
            fallback = float(wavelengths[k]) if wavelengths is not None and k < len(wavelengths) else np.nan
            wavelength.append(parsed[1] if parsed else fallback)
        self.wavelength = np.asarray(wavelength, float)
    def find(self, species, wavelength_A, rtol=1e-3):
        """
        Find the closest matching spectral line index for a given species and wavelength.

        This method searches through the list of species and their corresponding wavelengths
        to find the index of the spectral line that most closely matches the requested species
        and its wavelength. The search is case-insensitive for species names, and the
        closeness of the wavelength match is assessed within a specified relative tolerance.

        Parameters:
        species: str
            The name of the species to search for (case-insensitive).
        wavelength_A: float
            The wavelength in Angstroms to match against.
        rtol: float, optional
            The relative tolerance within which to accept a wavelength match (default is 1e-3).

        Returns:
        int or None
            The index of the closest matching spectral line if found; None otherwise.
        """
        species, best, best_distance = species.casefold(), None, rtol
        for k, sp in enumerate(self.species):
            if sp == species and abs(self.wavelength[k] / wavelength_A - 1.0) <= best_distance:
                best, best_distance = k, abs(self.wavelength[k] / wavelength_A - 1.0)
        return best

# LINES UNDER ANALYSIS FOR CHECKS
LINES = {'H_alpha': ('H 1', 6562.8), 'H_beta': ('H 1', 4861.3), 'OIII_5007': ('O 3', 5006.8),
         'OIII_4959': ('O 3', 4958.9), 'NII_6584': ('N 2', 6583.5)}

def line_flux(table, name):
    """
    Flux of a diagnostic line for every model; None if not in the master list.

    Parameters:
    table (Table): A table object containing the catalogue and line data.
    name (str): The name of the diagnostic line to search for in the catalogue.

    Returns:
    Optional[numpy.ndarray]: An array of flux values (as float) for the
    diagnostic line across all models if the line is found in the master list;
    otherwise, returns None.
    """
    index = table.catalogue.find(*LINES[name])
    return None if index is None else table.lines[:, index].astype(float)

def _missing_lines(table, names):
    """
    Constructs a list of missing line names from the given table.

    This function iterates through the provided list of names and checks if the
    corresponding line flux values in the table are None. If a name's flux value
    is missing, the name is added to the resulting list. This function serves as
    a utility to identify missing line data.

    Args:
        table (Any): The data structure containing line flux values. It should
            expose a way to retrieve flux values associated with names.
        names (list): A list of strings representing the line names to be checked
            for missing flux values.

    Returns:
        list: A list of strings containing the names of lines for which the flux
            values are missing.
    """
    return [name for name in names if line_flux(table, name) is None]

#######################################################################################################################

# the .h5 merge file

#============================== CLASS DataTable ==============================
class DataTable:
    """Column view of a merged data, read: grid parameters, flags,
    scalars, line fluxes and the transmission inside the Habing band; full
    spectra on demand (`read_spectra`)."""

    def __init__(self, path):
        self.path = str(path)
        with h5py.File(self.path, 'r') as f:
            self.root_attrs = {k: _decode(v) if isinstance(v, (bytes, str)) else _as_float(v)
                               for k, v in f.attrs.items()}
            # units declared by the parser: legend of the grid_point members, root datasets
            self.units = {k: _decode(v) for k, v in f['units'].attrs.items()} if 'units' in f else {}
            self.root_dataset_attrs = {name: {k: _decode(v) for k, v in f[name].attrs.items()}
                                       for name in CLOUDY.Root_Shared_HII if name in f}
            self.lines_vacuum = (np.asarray(f['lines_emergent/wavelengths_rest_vacuum'][()], float)
                                 if 'lines_emergent/wavelengths_rest_vacuum' in f else None)
            self.wave = np.asarray(f['continuum/wave_grid'][()], float) if 'continuum/wave_grid' in f else None
            labels = [_decode(x) for x in f['line_names'][()]] if 'line_names' in f else []
            rest = (np.asarray(f['lines_emergent/wavelengths_rest'][()], float)
                    if 'lines_emergent/wavelengths_rest' in f else None)
            self.catalogue = LineCatalogue(labels, rest)
            self.group_names = sorted(k for k in f.keys() if k.startswith(GRID_POINT_PREFIX))
            self.job_ids = [name[len(GRID_POINT_PREFIX):] for name in self.group_names]
            self.index = {job: i for i, job in enumerate(self.job_ids)}
            n = len(self.group_names)
            self.params = {}
            self.not_converged = np.zeros(n, dtype=bool)
            self.energy_balance = np.full(n, np.nan)
            self.reprocessing = np.full((n, 3), np.nan)  # excess, absorbed_fraction, lyc_emit_fraction
            self.scalars = {key: np.full(n, np.nan) for key in SCALAR_DATASETS}
            self.sed_support = np.full((n, 2), np.nan)
            self.lines = np.full((n, len(labels)), np.nan, dtype=np.float32)
            self.band_index = (np.flatnonzero((self.wave >= CONST.FUV_Lo_A) & (self.wave <= CONST.FUV_Hi_A))
                               if self.wave is not None else np.array([], dtype=int))
            self.band_wavelength = self.wave[self.band_index] if self.band_index.size else np.array([])
            self.band_transmission = (np.full((n, self.band_index.size), np.nan, dtype=np.float32)
                                      if self.band_index.size else None)
            band = slice(int(self.band_index.min()), int(self.band_index.max()) + 1) if self.band_index.size else None
            numeric = []
            if n:
                for key, value in f[self.group_names[0]].attrs.items():
                    if key not in NON_PARAMETER_ATTRIBUTES and _as_float(value) is not None:
                        numeric.append(key)
                        self.params[key] = np.full(n, np.nan)
            for i, name in enumerate(self.group_names):
                group, attrs = f[name], f[name].attrs
                for key in numeric:
                    value = _as_float(attrs.get(key))
                    if value is not None:
                        self.params[key][i] = value
                self.not_converged[i] = bool(attrs.get('not_converged', False))
                if attrs.get('energy_balance_rel') is not None:
                    self.energy_balance[i] = float(attrs['energy_balance_rel'])
                for k, key in enumerate(('reprocessing_excess', 'absorbed_fraction', 'lyc_emit_fraction')):
                    if attrs.get(key) is not None:
                        self.reprocessing[i, k] = float(attrs[key])
                support = attrs.get('sed_support_A')
                if support is not None and np.size(support) == 2:
                    self.sed_support[i] = np.ravel(support)
                for key, dataset in SCALAR_DATASETS.items():
                    source = group.get(dataset)
                    value = source[()] if source is not None else attrs.get(dataset)
                    if value is not None:
                        self.scalars[key][i] = float(np.ravel(value)[0])
                fluxes = group.get('lines_emergent/fluxes')
                if fluxes is not None and fluxes.size == len(labels):
                    self.lines[i] = np.ravel(fluxes[()])
                transmission = group.get('continuum/transmission')
                if band is not None and transmission is not None and transmission.shape == self.wave.shape:
                    self.band_transmission[i] = transmission[band]
            # the parser writes the target escape fraction as f_esc_target, the checks read f_esc
            if 'f_esc' not in self.params and 'f_esc_target' in self.params:
                self.params['f_esc'] = self.params['f_esc_target']

    @property
    def n(self):
        return len(self.job_ids)

    @property
    def converged(self):
        return ~self.not_converged

    def param(self, key):
        return self.params.get(key)

    def read_spectra(self, indices, dataset):
        """
        Reads the spectral data for specified indices and dataset.

        The method retrieves the spectral data for the given indices from an HDF5 file.
        If the specified dataset exists for the provided indices and matches the expected
        shape, it populates the returned array with the corresponding data. Missing rows
        are filled with NaN values. If no data is found, the method returns None.

        Parameters:
        indices (list or array-like): List of indices specifying the groups to retrieve
            the dataset from.
        dataset (str): The name of the dataset to be retrieved.

        Returns:
        numpy.ndarray or None: A 2D NumPy array of shape (len(indices), n_wave) containing
            the spectral data for the given dataset, where missing rows are filled with
            NaN values. Returns None if the dataset is not found or if `wave` is not set.
        """
        if self.wave is None:
            return None
        out, found = np.full((len(indices), self.wave.size), np.nan), False
        with h5py.File(self.path, 'r') as f:
            for row, i in enumerate(indices):
                group = f[self.group_names[i]]
                if dataset in group and group[dataset].shape == self.wave.shape:
                    out[row], found = group[dataset][()], True
        return out if found else None

#==============================================================================

########################################################################################################################

# Organize the checks (Register a check, context of check...)

@dataclass(frozen=True)
class CheckSpec:
    """specific of a registered check"""
    name: str
    function: object
    sectors: tuple
    blocking: bool

CHECKS = []

def check(name, sectors=('hii',), blocking=False):
    """Register a check function."""
    def register(function):
        CHECKS.append(CheckSpec(name, function, tuple(sectors), blocking))
        return function
    return register


#============================== CLASS Context ==============================
@dataclass
class Context:
    """Every input a check may read """
    sector: str
    data: DataTable
    charter: dict = None
    lhs: np.ndarray = None
    ssp_meta: list = None
    raw_workdirs: str = None
    n_expected: int = None
    max_spectra: int = 2000
    seed: int = 0

    def raw_runs(self):
        """
        Provides a list of job IDs and associated directories for kept working directories.

        Returns
        -------
        list of tuple
            A list of tuples where each tuple contains the job ID (str) and the directory
            (pathlib.Path object) of each kept working directory. Returns an empty list
            if the root directory does not exist or is not a directory.
        """
        root = pathlib.Path(self.raw_workdirs) if self.raw_workdirs else None
        return [(d.name, d) for d in sorted(root.iterdir()) if d.is_dir()] if root and root.is_dir() else []

    def raw_file(self, job, workdir, extension):
        """
        Generates the path to a specific file within a maintained working directory.

        This method constructs a file path based on the given working directory, job information,
        file extension, and an instance-specific sector attribute. The resulting path points to
        a file that corresponds to the provided details.

        Parameters:
        job: str
            Name or identifier of the job used to construct the filename.
        workdir: str
            Path to the working directory where the file is expected to reside.
        extension: str
            File extension to append to the constructed filename.

        Returns:
        pathlib.Path
            A Path object representing the constructed file path.
        """
        return pathlib.Path(workdir) / f'{self.sector}_{job}{extension}'

    def sample(self, mask):
        """
        Reproducible random subsampling of models based on the given mask.

        This function selects a random subsample of models from the provided `mask`.
        The size of the subsample will not exceed the value of `max_spectra`.
        The randomization process is reproducible using the specified random seed.

        Parameters:
            mask (numpy.ndarray): A boolean array indicating eligible elements
                for subsampling. True values correspond to candidates.

        Returns:
            numpy.ndarray: An array of indices representing the selected subsample
                of models from the `mask`.

        Raises:
            ValueError: If `mask` is not a one-dimensional numpy array.
        """
        candidates = np.flatnonzero(mask)
        if candidates.size <= self.max_spectra:
            return candidates
        return np.sort(np.random.default_rng(self.seed).choice(candidates, self.max_spectra, replace=False))

    def charter_bounds(self, attribute):
        """
        Returns the bounds of the given attribute from the charter data if available.

        The method searches for the specified attribute within the charter's axes
        and retrieves its minimum and maximum bounds.

        Parameters:
        attribute : str
            The name of the attribute for which bounds are requested.

        Returns:
        tuple[float, float] | None
            A tuple containing the minimum and maximum bounds of the given attribute
            in the form of (min, max). If the attribute is not found, None is returned.
        """
        for axis in (self.charter or {}).get('axes', []):
            if CHARTER_TO_ATTRIBUTE.get(axis['name'], axis['name']) == attribute:
                return float(axis['min']), float(axis['max'])
        return None
#===============================================================================

_NO_WORKDIRS = {'reason': 'no --raw-workdirs (kept CLOUDY working directories)'}

########################################################################################################################

# <checks on the data (merged .h5)

@check('units_legend', blocking=True)
def check_units_legend(ctx):
    """
    Checks the consistency of declared units in the data with the units expected by the HII parser.

    This function validates the units declared in the grid point groups and root datasets,
    comparing these with the predefined unit mappings available in the galapy.internal.cloudy
    module. It ensures that the units meet the expected standards necessary for further checks
    and processing. Any inconsistencies or missing attributes are recorded as problems.

    Attributes:
        ctx (Context): A context object which encapsulates the data table that contains
        units, and attributes required for validating compliance.

    Returns:
        Tuple[Status, Dict]: A tuple containing the result status, which could be either
        PASS, FAIL, or SKIP, along with a detailed dictionary that includes the number of
        problems identified, a summary of the findings, and a list of detected problems.

    Raises:
        No exceptions are explicitly raised as part of this function, but dependent
        attributes in ctx must be at least partially initialized for it to function
        correctly.
    """
    table = ctx.data
    if not table.units:
        return SKIP, {'reason': 'data without units (written by an older parser)'}
    problems = []
    expected = {name: unit for name, (unit, _description) in CLOUDY.Points_Units_HII.items() if unit is not None}
    for name in sorted(set(expected) | set(table.units)):
        if table.units.get(name) != expected.get(name):
            problems.append(f'{name}: {table.units.get(name)!r} instead of {expected.get(name)!r}')
    for name, (unit, _description, extra) in CLOUDY.Root_Units_HII.items():
        attrs = table.root_dataset_attrs.get(name)
        if attrs is None:
            problems.append(f'root dataset {name} missing')
            continue
        wanted = dict(extra, **({'units': unit} if unit is not None else {}))
        problems += [f'{name}.{key}: {attrs.get(key)!r} instead of {value!r}'
                     for key, value in wanted.items() if attrs.get(key) != value]
    detail = {'members': len(table.units), 'problems': problems[:20],
              'summary': f'{len(problems)} differences from the units of the parser'}
    return (PASS if not problems else FAIL), detail

@check('convergence', blocking=True)
def check_convergence(ctx):
    """
    Check if planned runs match converged runs and validate convergence flags.

    This function compares the number of converged runs against the planned
    runs and examines convergence-related flags for all relevant outputs.
    The planned runs are derived from the provided charter or the --n-expected
    argument. Crashed runs are considered failures. The function verifies
    convergence data for the working directories and ensures agreement between
    the output and the convergence flags. Outputs without the "Cloudy exited OK"
    marker are separately listed for further inspection.

    Parameters:
        ctx: The execution context containing relevant job data, working
             directories, and configuration, including planned runs.

    Returns:
        A tuple containing:
            - A status string indicating the result ('PASS' or 'FAIL').
            - A dictionary providing detailed information about the convergence
              state, including counts of converged, planned, and fragment runs,
              any disagreements between flags and output, and outputs without
              an exit marker.

    Raises:
        This function may raise exceptions depending on errors encountered
        within the execution context or data handling, not explicitly documented.
    """
    table = ctx.data
    converged = int(table.converged.sum())
    denominator = ctx.n_expected or table.n
    fraction = converged / denominator if denominator else 0.0
    detail = {'converged': converged, 'fragments': table.n, 'planned_runs': ctx.n_expected,
              'fraction': _round(fraction, 6), 'summary': f'{converged}/{denominator} converged ({fraction:.4%})'}
    if not ctx.n_expected:
        detail['warning'] = 'planned runs unknown (no --config or --n-expected): crashed runs are not counted'
    statuses = [PASS if fraction > T['convergence.min_fraction'] else FAIL]
    disagreements, no_exit_line, outputs = [], [], 0
    for job, workdir in ctx.raw_runs():
        out = ctx.raw_file(job, workdir, '.out')
        if not out.is_file() or (workdir / 'RUN_FAILED').is_file():
            continue
        outputs += 1
        text = _read_text(out)
        ok = converged_from_out(text)
        if CLOUDY_EXIT_OK not in text:
            no_exit_line.append(job)
        flags = {}
        if (workdir / 'converged.flag').is_file():
            flags['converged.flag'] = (workdir / 'converged.flag').read_text().strip() != '0'
            flags['NOT_CONVERGED marker'] = not (workdir / 'NOT_CONVERGED').is_file()
        if job in table.index:
            flags['not_converged attribute'] = not bool(table.not_converged[table.index[job]])
        disagreements += [{'job_id': job, 'flag': k} for k, v in flags.items() if v != ok]
    if outputs:
        detail['kept_outputs'] = {'outputs': outputs, 'disagreements': disagreements[:20],
                                  'without_exit_line': no_exit_line[:20]}
        statuses.append(FAIL if disagreements else PASS)
    return _combine(statuses), detail

@check('ionizing_photon_budget',  blocking=True)
def check_ionizing_photon_budget(ctx):
    """
    Checks the ionizing photon budget for the written SED files against the SSP table.

    This function compares the Q_H values of the written SED files (from node metadata) to the
    values found in the SSP table, ensuring consistency between them. The check passes if the
    worst relative difference between the SED file and table values is within 5%.

    Parameters:
        ctx: The context object containing metadata and configuration information.

    Returns:
        tuple: A tuple containing the result (PASS, FAIL, or SKIP) and a detail dictionary
        with information about the check, such as the worst relative difference, the node
        causing the largest discrepancy, and nodes with zero table Q_H values.

    Raises:
        None
    """
    if ctx.ssp_meta is None:
        return SKIP, {'reason': 'no --ssp-meta'}
    from galapy.spectroscopy.utils.extract_spectra import load_ssp_cube, cloudy_sed_QH_reference
    library = ctx.data.root_attrs.get('ssp_lib') or (ctx.charter or {}).get('ssp_library') or 'parsec22.NT'
    cube = load_ssp_cube(library)
    worst, worst_node, zero = 0.0, None, []
    for node in ctx.ssp_meta:
        q_table = float(cloudy_sed_QH_reference(cube, int(node['it']), int(node['iz'])))
        if q_table <= 0:
            zero.append([node['tau_SSP'], node['Z_star']])
            continue
        relative = abs(float(node['Qh_unit']) - q_table) / q_table
        if relative >= worst:
            worst, worst_node = relative, {'tau_SSP': node['tau_SSP'], 'Z_star': node['Z_star']}
    detail = {'ssp_library': library, 'nodes': len(ctx.ssp_meta), 'worst_rel_diff': _round(worst, 6),
              'worst_node': worst_node, 'nodes_with_zero_table_Q_H': zero,
              'summary': f'worst node differs by {worst:.3%}'}
    return (PASS if worst < T['ionizing_budget.max_rel_diff'] else FAIL), detail

@check('energy_balance',  blocking=True)
def check_energy_balance(ctx):
    """
    Emitted versus absorbed energy, per model.

    This function evaluates the energy balance of models by analyzing the energy
    reprocessing excess and the LyC (Lyman continuum) emission fraction. It focuses
    on models having a significant absorbed energy fraction and excludes those
    dominated by computational noise or cosmic microwave background (CMB) effects.
    The grading process determines models that meet stringent observable range
    excess criteria, while the function computes relevant statistics and identifies
    outlier models for detailed analysis.

    Parameters:
    ctx (object): The context which includes the input data and configurations used
                  for checking energy balance. The required attributes include
                  model properties and threshold values for energy fraction grading.

    Returns:
    tuple: A tuple containing the following elements:
        - A pass/fail/skip status as determined by the evaluation process.
        - A dictionary with detailed results including:
            - Total number of graded models.
            - Fraction of models passing the criteria.
            - Percentile statistics of observable reprocessing excess, total excess,
              and LyC emission fraction.
            - List of identifiers for models with the highest outlier excess.
            - Summary of the evaluation results.
    """
    table = ctx.data
    excess, absorbed, lyc = table.reprocessing.T
    graded = table.converged & np.isfinite(excess) & (absorbed >= T['energy_balance.min_absorbed_fraction'])
    if not graded.any():
        return SKIP, {'reason': 'no converged model with reprocessing_excess above the absorbed-fraction threshold'}
    observable = excess[graded] - lyc[graded]
    fraction = float(np.mean(observable <= T['energy_balance.max_observable_excess']))
    worst = np.flatnonzero(graded)[np.argsort(observable)[::-1][:10]]
    detail = {'n': int(graded.sum()), 'fraction_passing': _round(fraction),
              'observable_excess': _percentiles(observable, (50, 95, 99), 6),
              'total_excess': _percentiles(excess[graded], (50, 95, 99), 6),
              'lyc_emit_fraction': _percentiles(lyc[graded], (50, 95, 99), 6),
              'worst_models': [table.job_ids[i] for i in worst],
              'summary': f'{fraction:.1%} of {int(graded.sum())} models without excess in the observable range'}
    return (PASS if fraction >= T['energy_balance.min_pass_fraction'] else FAIL), detail

@check('balmer_decrement')
def check_balmer_decrement(ctx):
    """
    Check the emergent H-alpha to H-beta ratio against the predicted Case B value under
    specific conditions of log U and redshift.

    This function evaluates whether the median ratio of H-alpha to H-beta over selected
    data entries deviates significantly from the theoretical Case B ratio. The conditions
    for evaluation include the presence of relevant spectral lines and specific values of
    logU and redshift parameters.

    Parameters:
        ctx (Context): The context object containing the dataset to evaluate.

    Returns:
        tuple: A tuple containing the result of the check (PASS, FAIL, or SKIP) and a
        dictionary with detailed information regarding the evaluation.

    Raises:
        None
    """
    table = ctx.data
    if _missing_lines(table, ('H_alpha', 'H_beta')) or table.param('logU') is None or table.param('z_CMB') is None:
        return SKIP, {'reason': 'H-alpha, H-beta, logU or z_CMB missing'}
    halpha, hbeta = line_flux(table, 'H_alpha'), line_flux(table, 'H_beta')
    window = (table.converged & (np.abs(table.param('logU') - T['case_b.logU_centre']) <= T['case_b.logU_half_width'])
              & (table.param('z_CMB') <= T['case_b.max_z_cmb']) & (halpha > 0) & (hbeta > 0))
    if not window.any():
        return SKIP, {'reason': 'no converged model inside the evaluation window'}
    median = float(np.median(halpha[window] / hbeta[window]))
    deviation = abs(median / T['case_b.reference'] - 1.0)
    detail = {'n': int(window.sum()), 'median_ratio': _round(median), 'relative_deviation': _round(deviation),
              'summary': f'median H-alpha/H-beta {median:.3f} ({deviation:.2%} from Case B)'}
    return (PASS if deviation < T['case_b.max_rel_dev'] else FAIL), detail

@check('bpt_star_forming_locus')
def check_bpt_locus(ctx):
    """
    Models the behavior of galaxy classification in the [N II]/H-alpha versus [O III]/H-beta diagram.

    Evaluates model adherence to specific astrophysical thresholds defined by the maximum-starburst
    line using log U constraints and emission line ratios.

    Parameters:
        ctx (Context): The context object providing input data and parameters required for the check.

    Returns:
        tuple: A tuple containing the result ('PASS', 'FAIL', or 'SKIP') and a dictionary with
        details of the evaluation process.

    Raises:
        None
    """
    table = ctx.data
    if _missing_lines(table, ('NII_6584', 'H_alpha', 'OIII_5007', 'H_beta')) or table.param('logU') is None:
        return SKIP, {'reason': 'BPT lines or logU missing'}
    x = _log10_ratio(line_flux(table, 'NII_6584'), line_flux(table, 'H_alpha'))
    y = _log10_ratio(line_flux(table, 'OIII_5007'), line_flux(table, 'H_beta'))
    good = table.converged & np.isfinite(x) & np.isfinite(y)
    region = good & (table.param('logU') <= T['bpt.max_logU'])
    if not region.any():
        return SKIP, {'reason': 'no converged model with log U <= -2 and the four lines'}
    below_kewley = _below_hyperbola(x, y, *T['bpt.kewley01'])
    below_kauffmann = _below_hyperbola(x, y, *T['bpt.kauffmann03'])
    fraction = float(below_kewley[region].mean())
    detail = {'n_graded': int(region.sum()), 'below_kewley_graded': _round(fraction),
              'below_kauffmann_graded': _round(below_kauffmann[region].mean()),
              'below_kewley_all': _round(below_kewley[good].mean()),
              'summary': f'{fraction:.1%} of models with log U <= -2 below the maximum-starburst line'}
    return (PASS if fraction >= T['bpt.min_fraction_below'] else FAIL), detail

@check('sampling_design_uniformity')
def check_sampling_design(ctx):
    """
    Evaluates the Kolmogorov-Smirnov test for the uniformity of every marginal
    distribution in the continuous sample provided.

    This function checks the uniformity of sample distributions along specified
    axes with the Kolmogorov-Smirnov test. It also verifies whether sample values
    are within the specified bounds and summarizes the findings for each axis.
    A "pass" result is given if every axis satisfies the following conditions:
    - All p-values from the test are greater than 0.01.
    - No sample values exist outside the defined bounds.

    Parameters:
    ctx : Context object
        A context object that contains the left-hand sample (lhs) and the
        configuration of axes (charter) defining the bounds.

    Returns:
    tuple
        A tuple where the first element is either "PASS", "FAIL", or "SKIP", and
        the second element is a dictionary detailing the summary of the test results
        or the reason for skipping the check.

    Raises:
        KeyError: If necessary keys are missing in the axis configuration in ctx.charter.
        TypeError: If ctx.lhs cannot be converted to a numerical array.
    """
    if ctx.lhs is None or ctx.charter is None:
        return SKIP, {'reason': 'needs --lhs and --config (axis bounds)'}
    from scipy import stats
    axes, sample = ctx.charter.get('axes', []), np.atleast_2d(np.asarray(ctx.lhs, float))
    if sample.shape[1] != len(axes):
        return FAIL, {'summary': f'sample has {sample.shape[1]} columns, charter {len(axes)} axes'}
    per_axis = []
    for column, axis in enumerate(axes):
        lower, upper = float(axis['min']), float(axis['max'])
        values = sample[:, column]
        outside = int(np.sum((values < lower) | (values > upper)))
        p_value = float(stats.kstest((values - lower) / (upper - lower), 'uniform').pvalue)
        per_axis.append({'axis': axis['name'], 'p_value': _round(p_value, 6), 'outside_bounds': outside,
                         'pass': p_value > T['lhs.min_p_value'] and outside == 0})
    worst = min(per_axis, key=lambda item: item['p_value'])
    detail = {'n_samples': int(sample.shape[0]), 'axes': per_axis,
              'summary': f"lowest p-value {worst['p_value']} ({worst['axis']})"}
    return (PASS if all(a['pass'] for a in per_axis) else FAIL), detail

@check('grain_continuum_sanity')
def check_grain_continuum(ctx):
    """
    Validates the diagnostic grain continuum for a subsample of converged models.

    This function checks a series of conditions to ensure the physical accuracy of the
    diagnostic grain continuum data. The checks include validating that:
    (a) The grain continuum contains no negative values.
    (b) The peak of the spectrum (nu*F_nu) lies within the range of 5 to 500 micrometers.
    (c) Below 0.3 micrometers, the continuum is less than 1e-3 of the peak value.
    (d) The far-infrared (> 20 micrometers) energy of the continuum is comparable to the
        nebular emission continuum, with a median ratio in the range [0.1, 10].

    The function reports whether the conditions are met for at least 99% of the sampled
    models and provides detailed diagnostic data.

    Args:
        ctx: The context object that provides access to input data and configuration.

    Returns:
        tuple: A two-element tuple where the first element is a status flag indicating
        whether the checks passed (`PASS`), failed (`FAIL`), or were skipped (`SKIP`),
        and the second element is a dictionary providing details on the validation results.

    Raises:
        None
    """
    table = ctx.data
    if table.wave is None:
        return SKIP, {'reason': 'data has no continuum/wave_grid'}
    indices = ctx.sample(table.converged)
    grain = table.read_spectra(indices, 'continuum/grain_diag_per_Msun')
    if grain is None:
        return SKIP, {'reason': 'no continuum/grain_diag_per_Msun'}
    valid = np.all(np.isfinite(grain), axis=1)
    grain, rows = grain[valid], indices[valid]
    wavelength_um = table.wave * 1e-4
    peak, peak_um = np.max(grain, axis=1), wavelength_um[np.argmax(grain, axis=1)]
    lower, upper = T['grain.peak_window_um']
    uv = wavelength_um < T['grain.uv_limit_um']
    with np.errstate(divide='ignore', invalid='ignore'):
        uv_ratio = np.max(grain[:, uv], axis=1) / peak if uv.any() else np.zeros(rows.size)
    ok = (~np.any(grain < 0, axis=1) & (peak > 0) & (peak_um >= lower) & (peak_um <= upper)
          & (uv_ratio <= T['grain.max_uv_to_peak']))
    fraction = float(ok.mean()) if ok.size else 0.0
    detail = {'sampled': int(rows.size), 'fraction_passing_a_b_c': _round(fraction), 'peak_um': _percentiles(peak_um),
              'failing_examples': [table.job_ids[i] for i in rows[~ok][:10]]}
    status_fir = SKIP
    own = table.read_spectra(indices, 'continuum/nebular_emission_per_Msun')
    if own is not None:
        fir = wavelength_um >= T['grain.fir_min_um']
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = _integrate_log(grain, table.wave, fir) / _integrate_log(own[valid], table.wave, fir)
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if ratio.size:
            low, high = T['grain.fir_ratio_median_window']
            detail['fir_ratio'] = _percentiles(ratio)
            status_fir = PASS if low <= np.median(ratio) <= high else FAIL
    detail['summary'] = f'{fraction:.1%} of sampled models with correct sign, peak and UV; far-IR ratio {status_fir}'
    return _combine([PASS if fraction >= T['grain.min_pass_fraction'] else FAIL, status_fir]), detail

def _spike_flags(spectra, bins, excess, half=3, gap=2):
    """
    Identifies spikes in spectral data that exceed local baselines by a defined excess. Flags the
    corresponding bins where the spike condition is met. Local baselines are calculated on both
    sides of the target bin.

    Parameters:
    spectra: numpy.ndarray
        A 2D array of shape (n_models, n_bins) containing the spectral data.
    bins: list[int]
        A list of bin indices to evaluate for spikes.
    excess: float
        The percentage by which the peak in a bin must exceed the local baseline for it to be
        flagged.
    half: int, optional
        The width in bins on either side of the local baseline region for median calculation.
        Default is 3.
    gap: int, optional
        The gap in bins between the analyzed bin and the local baseline region. Default is 2.

    Returns:
    numpy.ndarray
        A 2D boolean array of the same shape as `(n_models, len(bins))`, where True indicates a
        spike in the corresponding bin of each model.
    """
    flags = np.zeros((spectra.shape[0], len(bins)), dtype=bool)
    for column, b in enumerate(bins):
        centre = np.max(spectra[:, b - 1:b + 2], axis=1)
        baseline = np.maximum(np.median(spectra[:, b - gap - half:b - gap], axis=1),
                              np.median(spectra[:, b + gap + 1:b + gap + 1 + half], axis=1))
        flags[:, column] = (baseline > 0) & (centre > (1.0 + excess) * baseline)
    return flags

@check('line_free_continua')
def check_line_free_continua(ctx):
    """
    Checks for the presence of residual line in the own continuum or the grain continuum.

    At each line of the master list, bin values are compared against baselines a few bins
    away on both sides. This ensures that edges are treated as gradual steps rather than spikes.
    The check is considered successful if no more than 1% of the model-line pairs are flagged
    for both continua.

    Parameters:
        ctx : Context
            The context object containing the wavelength grid, catalogue data, and other parameters
            necessary for the line-free continuum check.

    Returns:
        result : Status
            One of SKIP, PASS, or FAIL based on the analysis of flagged line pairs in the spectra.
        detail : dict
            A dictionary detailing the results of the check. It includes lines tested, fraction of flagged
            lines, and summary information about flagged continua.

    Raises:
        KeyError
            If required datasets or configurations are missing from the context object.
    """
    table = ctx.data
    if table.wave is None or not table.catalogue.labels:
        return SKIP, {'reason': 'no wavelength grid or no line list'}
    log_wave = np.log(table.wave)
    bins, labels = [], []
    # the continuum mesh is in vacuum: lines go on it with their vacuum wavelengths
    positions = table.lines_vacuum if table.lines_vacuum is not None else table.catalogue.wavelength
    for label, wavelength in zip(table.catalogue.labels, positions):
        b = int(np.argmin(np.abs(log_wave - np.log(wavelength)))) if np.isfinite(wavelength) else -1
        if 6 <= b < table.wave.size - 6:
            bins.append(b)
            labels.append(label)
    if not bins:
        return SKIP, {'reason': 'no line inside the wavelength grid'}
    indices = ctx.sample(table.converged)
    detail, statuses = {'lines_tested': len(bins), 'sampled': int(indices.size)}, []
    for key, dataset in (('own_continuum', 'continuum/nebular_emission_per_Msun'),
                         ('grain_continuum', 'continuum/grain_diag_per_Msun')):
        spectra = table.read_spectra(indices, dataset)
        if spectra is None:
            detail[key] = 'dataset missing'
            statuses.append(SKIP)
            continue
        flags = _spike_flags(spectra[np.all(np.isfinite(spectra), axis=1)], bins, T['line_free.spike_excess'])
        fraction = float(flags.mean()) if flags.size else 0.0
        per_line = flags.mean(axis=0) if flags.size else np.zeros(len(bins))
        detail[key] = {'flagged_fraction': _round(fraction, 6),
                       'worst_lines': [labels[k] for k in np.argsort(per_line)[::-1][:5] if per_line[k] > 0]}
        statuses.append(PASS if fraction <= T['line_free.max_flagged_fraction'] else FAIL)
    detail['summary'] = ', '.join(f"{k}: {v['flagged_fraction']:.2%} flagged" for k, v in detail.items()
                                  if isinstance(v, dict)) or 'no continuum'
    return _combine(statuses), detail

@check('oiii_doublet_ratio')
def check_oiii_doublet(ctx):
    """
    Checks the ratio of [O III] 4959 / 5007 emission lines in converged models.

    This function evaluates whether the ratio of [O III] lines (4959 and 5007) in astrophysical
    data conforms to an expected intrinsic ratio, accounting for small differences due to internal
    dust absorption. The intrinsic ratio is determined by atomic physics to be 0.3459. The function
    only considers models where the [O III] 5007 line strength is above a specified threshold,
    ensures that all conditions are met for valid analysis, and calculates the deviation from
    the expected ratio.

    The measure passes if the ratio deviates by less than 2% from the intrinsic value for valid
    models.

    Parameters:
        ctx (object): The context object containing the astrophysical data required for the check.

    Returns:
        tuple: A status indicating whether the check passed (PASS, FAIL, SKIP), and a dictionary
        containing detailed information about the result. The dictionary includes the number of
        valid models examined, the worst relative deviation observed, details about the model
        with the worst deviation, and a summary of the worst-case deviation.
    """
    table = ctx.data
    if _missing_lines(table, ('OIII_4959', 'OIII_5007', 'H_beta')):
        return SKIP, {'reason': '[O III] lines not in the master list'}
    weak, strong = line_flux(table, 'OIII_4959'), line_flux(table, 'OIII_5007')
    good = table.converged & (weak > 0) & (strong > 0) & (strong >= T['oiii.min_hbeta'] * line_flux(table, 'H_beta'))
    if not good.any():
        return SKIP, {'reason': 'no converged model with both lines'}
    deviation = np.abs(weak[good] / strong[good] / T['oiii.reference'] - 1.0)
    worst = int(np.argmax(deviation))
    detail = {'n': int(good.sum()), 'worst_relative_deviation': _round(deviation[worst], 6),
              'worst_model': table.job_ids[np.flatnonzero(good)[worst]],
              'summary': f'worst deviation {deviation[worst]:.3%} from {T["oiii.reference"]:.3f}'}
    return (PASS if deviation[worst] < T['oiii.max_rel_dev'] else FAIL), detail

@check('dust_mass_scaling')
def check_dust_mass_scaling(ctx):
    """
    Check the scaling of dust mass per solar mass against the truncated Stromgren scaling.

    This function validates the relationship:
    M_dust / M_SSP ~ (1 - f_esc) * xi_d * m_Z(zeta_O) * Q_H / n_H, where a constant is fitted based
    on the median residual. The m_Z (metal mass per hydrogen) is derived using the chemistry module
    and is treated as the grain_scale of the deck. The method also captures the residual trend
    concerning log U.

    Summary of validation:
    - Pass condition: At least 90% of the models must fall within 0.2 dex of the expected scaling.

    Parameters:
    ctx : Context
        The context object containing the required data and metadata for the check.

    Returns:
    tuple
        A tuple consisting of:
        - A result status, either PASS, FAIL, or SKIP.
        - A dictionary with details about the validation including model count,
          fitted constant, fraction within tolerance, and trends with log U.

    Raises:
    KeyError
        If required parameters or metadata are missing.
    ModuleNotFoundError
        If the chemistry module is unavailable for the calculations.

    """
    table = ctx.data
    needed = ('f_esc', 'log_zeta_O', 'xi_d', 'lognH_HII', 'tau_SSP', 'Z_star', 'logU')
    if ctx.ssp_meta is None or any(table.param(k) is None for k in needed):
        return SKIP, {'reason': 'needs --ssp-meta and the grid parameters'}
    try:
        chemistry = _chemistry()
    except Exception as exc:  # noqa: BLE001 - any failure means "not available"
        return SKIP, {'reason': f'chemistry module unavailable: {_reason(exc)}'}
    lookup = {(_scan_key(m['tau_SSP']), _scan_key(m['Z_star'])): float(m['Qh_unit']) for m in ctx.ssp_meta}
    q_h = np.array([lookup.get((_scan_key(t), _scan_key(z)), np.nan)
                    for t, z in zip(table.param('tau_SSP'), table.param('Z_star'))])
    # the dust mass of the deck: xi_d x metal mass per H (grain_scale of gen_lhs_hii)
    dust_factor = table.param('xi_d') * np.array([chemistry.metal_mass_per_H(v) for v in table.param('log_zeta_O')])
    with np.errstate(divide='ignore', invalid='ignore'):
        residual = np.log10(table.scalars['dust_mass']) - (
                np.log10((1.0 - table.param('f_esc')) * dust_factor * q_h) - table.param('lognH_HII'))
    good = table.converged & np.isfinite(residual)
    if not good.any():
        return SKIP, {'reason': 'no converged model with dust_mass_per_Msun and a finite prediction'}
    deviation = residual[good] - np.median(residual[good])
    fraction = float(np.mean(np.abs(deviation) <= T['dust.max_abs_residual']))
    detail = {'n': int(good.sum()), 'fitted_log_constant': _round(np.median(residual[good])),
              'fraction_within_tolerance': _round(fraction), 'residual': _percentiles(deviation, (1, 50, 99)),
              'rank_correlation_residual_vs_logU': _round(_spearman(deviation, table.param('logU')[good]) or np.nan),
              'summary': f'{fraction:.1%} of models within 0.2 dex of the Stromgren scaling'}
    return (PASS if fraction >= T['dust.min_pass_fraction'] else FAIL), detail

@check('old_node_ionizing_output')
def check_old_nodes(ctx):
    """
    Check Q_H ratio of 70 and 100 Myr nodes to the youngest node for each stellar metallicity.

    This function evaluates stellar metallicity nodes and compares the Q_H ratio of
    older nodes (70 and 100 Myr) against the youngest node. The primary objective is to
    verify that the Q_H ratio for all older nodes remains below 1% of the youngest for every
    metallicity. If this condition is not met, the check fails.

    The function performs this operation by iterating over the nodes provided in the context
    and computing the ratios per metallicity group. The worst case of the ratio is reported,
    and the successful or failed status is returned accordingly.

    Parameters:
    ctx: Context
        The context object containing `ssp_meta` data with stellar metallicity nodes.

    Returns:
    tuple
        A status outcome (PASS, FAIL, or SKIP), accompanied by a dictionary containing
        per-node results and a summary of the largest Q_H ratio computed for the older
        nodes relative to the youngest.

    Raises:
        None
    """
    if ctx.ssp_meta is None:
        return SKIP, {'reason': 'no --ssp-meta'}
    by_z = {}
    for node in ctx.ssp_meta:
        by_z.setdefault(float(node['Z_star']), []).append((float(node['tau_SSP']), float(node['Qh_unit'])))
    rows, ok = [], True
    for z_star, nodes in sorted(by_z.items()):
        ages, rates = np.array(nodes).T
        reference = rates[np.argmin(ages)]
        for target in T['old_nodes.ages']:
            k = int(np.argmin(np.abs(np.log10(ages / target))))
            ratio = rates[k] / reference if reference > 0 else np.inf
            passed = abs(np.log10(ages[k] / target)) <= np.log10(1.5) and ratio < T['old_nodes.max_ratio']
            ok &= passed
            rows.append({'Z_star': z_star, 'age': ages[k], 'ratio_to_youngest': ratio, 'pass': passed})
    worst = max((r['ratio_to_youngest'] for r in rows), default=np.inf)
    return (PASS if ok else FAIL), {'per_node': rows, 'summary': f'largest old/young Q_H ratio {worst:.2e}'}

@check('escape_fraction_coverage')
def check_escape_fraction(ctx):
    """Evaluate the coverage of the measured escape fraction and detect clouds stopping at their face.

    This function ensures that the coverage of the measured escape fraction is within the required
    range and checks that no model indicates an escape fraction above a specified threshold.
    Additionally, the function calculates the rank correlation of the measured escape fraction
    with the target parameter and evaluates gaps in the coverage of the escape fraction range.

    Parameters:
        ctx (object): Context object containing data for the models, including measured and target
            escape fractions, associated metadata, and thresholds.

    Returns:
        tuple: A tuple where the first element is a status (PASS, FAIL, or SKIP), and the second
            element is a dictionary providing detailed information on the widest gap, rank correlation,
            and any models that stopped at the cloud face.

    Raises:
        None
    """
    table = ctx.data
    measured, target = table.scalars['f_esc_meas'], table.param('f_esc')
    good = table.converged & np.isfinite(measured)
    if target is None or not good.any():
        return SKIP, {'reason': 'no f_esc_meas or no f_esc attribute'}
    lower, upper = T['fesc.range']
    inside = measured[good][(measured[good] >= lower) & (measured[good] <= upper)]
    edges = np.concatenate(([lower], np.unique(inside), [upper]))
    k = int(np.argmax(np.diff(edges)))
    gap, rho = float(edges[k + 1] - edges[k]), _spearman(target[good], measured[good])
    face = good & (measured > T['fesc.face_measured'])
    detail = {'n': int(good.sum()), 'widest_gap': _round(gap), 'gap_between': [_round(edges[k]), _round(edges[k + 1])],
              'rank_correlation': _round(rho if rho is not None else np.nan), 'stopped_at_face': int(face.sum()),
              'examples': [table.job_ids[i] for i in np.flatnonzero(face)[:10]],
              'summary': f'widest gap {gap:.3f}, {int(face.sum())} models stopped at the face, '
                         f'rank correlation {rho if rho is not None else float("nan"):.3f}'}
    ok = gap <= T['fesc.max_gap'] and not face.any()
    return (PASS if ok else FAIL), detail

@check('depletion(jk)_axis_coverage')
def check_depletion_coverage(ctx):
    """
    KS uniformity of F* over the converged models (sample size: distinct draws, not models). The method
    evaluates the uniformity of F* values using the Kolmogorov-Smirnov (KS) test. Converged models with
    a sufficient number of distinct F* values are considered. The test is passed if the calculated
    p-value is greater than 0.01.

    Parameters:
        ctx: Context object containing required data and configurations for the method.

    Returns:
        Tuple: A 2-element tuple where the first element is a status string ('PASS', 'FAIL', or 'SKIP')
        based on the result of the uniformity test or conditions for skipping, and the second element is a
        dictionary containing test details or reasoning.
    """
    f_star = ctx.data.param('F_star')
    if f_star is None:
        return SKIP, {'reason': 'models carry no F_star'}
    values = f_star[ctx.data.converged & np.isfinite(f_star)]
    if values.size < 3:
        return SKIP, {'reason': 'fewer than three converged models'}
    n_distinct = int(np.unique(values).size)
    distance, p_value = _ks_uniform(values, *(ctx.charter_bounds('F_star') or (0.0, 1.0)), n_effective=n_distinct)
    detail = {'n_distinct_draws': n_distinct, 'ks_distance': _round(distance, 6), 'p_value': _round(p_value, 6),
              'summary': f'KS p-value {p_value:.3g} on {n_distinct} distinct draws'}
    return (PASS if p_value > T['fstar.min_p_value'] else FAIL), detail

@check('fuv_terms')
def check_fuv_terms(ctx):
    """
    Analyzes the FUV terms of the ionized skin, including positivity conditions
    and the spectral tilt of the transmission in the Habing band.

    This function ensures that the FUV transmission (T_FUV) and FUV count (N_FUV)
    adhere to expected sign and positivity constraints. Additionally, where valid,
    it computes the tilt of the transmission spectrum across the FUV band. The
    tilt is derived as the slope of the logarithmic transmission fitted across the
    valid wavelength range (excluding Lyman-series bins).

    The function passes if all models meet positivity constraints and the 99th
    percentile of the absolute tilt does not exceed 0.888 dex, which represents
    compliance with the beta axis on the PDR grid.

    Parameters:
        ctx: The computational context, which includes the data necessary
             for validation and analysis.

    Returns:
        Tuple containing:
        - status: Computation result as a string indicating overall pass, fail,
                  or skipped status based on validation checks.
        - detail: A dictionary with detailed checks and results for positivity
                  violations, tilt computation, and summary statistics.

    Raises:
        No explicit exceptions are documented; however, internal operations might
        raise exceptions if data is corrupted or computation fails.
    """
    table = ctx.data
    t_fuv, n_fuv = table.scalars['T_fuv'], table.scalars['N_fuv']
    detail, statuses = {}, []
    good = table.converged & np.isfinite(t_fuv) & np.isfinite(n_fuv)
    if good.any():
        bad = int(np.sum((t_fuv[good] <= 0) | (t_fuv[good] > 1.0 + 1e-6) | (n_fuv[good] <= 0)))
        detail['positivity_violations'] = bad
        statuses.append(PASS if bad == 0 else FAIL)
    region = table.converged & np.isfinite(t_fuv) & (t_fuv > T['fuv.region_min_transmittance'])
    if table.band_transmission is None or not region.any():
        detail['tilt'] = 'skipped: no continuum/transmission in the band or no model with T_FUV > 1e-2'
        statuses.append(SKIP)
    else:
        keep = ~_lyman_series_mask(table.band_wavelength)
        x = np.log10(table.band_wavelength[keep])
        x = x - x.mean()
        y = np.log10(np.clip(table.band_transmission[region][:, keep].astype(float), 1e-30, None))
        finite = np.all(np.isfinite(y), axis=1)
        tilt = ((y[finite] - y[finite].mean(axis=1, keepdims=True)) @ x / np.dot(x, x)
                * np.log10(CONST.FUV_Hi_A / CONST.FUV_Lo_A))
        worst = float(np.max(np.abs(tilt)))
        detail.update({'n_region': int(tilt.size), 'max_abs_tilt_dex': _round(worst),
                       'tilt': _percentiles(tilt, (1, 50, 99))})
        statuses.append(PASS if np.percentile(np.abs(tilt), 99) < T['fuv.max_tilt'] else FAIL)
    status = _combine(statuses)
    detail['summary'] = (f"{detail.get('positivity_violations', 'n/a')} sign violations, "
                         f"max |tilt| {detail.get('max_abs_tilt_dex', 'n/a')} dex")
    return status, detail

@check('transparency_outside_sed_support', blocking=True)
def check_transparency(ctx):
    """
    Ensures the transmission values meet expected criteria based on the SED support range in the data.

    Summary:
    This function evaluates whether the transmission values of a dataset conform to specified tolerances:
    - Outside of the SED support range (sed_support_A), transmission values should strictly equal 1.
    - Inside the SED support range, transmission values must fall within the range [0, 1].

    Pass Criteria:
    The function passes if no transmission value outside the SED support range deviates from 1 by more than 1e-3,
    and all values within the range fall between 0 and 1 (inclusive).

    Failure Details:
    If bins are identified as incorrect, the function details how many bins are wrong both outside and inside
    the range, as well as a list of up to 10 examples (job IDs) that exhibited these issues.

    Parameters:
    ctx (object): The execution context containing input data and configuration.

    Returns:
    A tuple containing:
    - PASS or FAIL (str): Indicates whether the data meets the transmission criteria.
    - detail (dict): A dictionary summarizing validation details, including:
        - sampled (int): The number of data points sampled.
        - bins_wrong_outside (int): The count of bins deviating from 1 outside the range.
        - bins_wrong_inside (int): The count of bins with values outside [0, 1] inside the range.
        - examples (list): Up to 10 job IDs of datasets with transmission issues.
        - summary (str): A short description of the issues detected.

    Raises:
    SKIP: If no converged models with valid "continuum/transmission" and "sed_support_A" data are found in the input.
    """
    table = ctx.data
    indices = ctx.sample(table.converged & np.all(np.isfinite(table.sed_support), axis=1))
    transmission = table.read_spectra(indices, 'continuum/transmission') if indices.size else None
    if transmission is None:
        return SKIP, {'reason': 'no converged model with continuum/transmission and sed_support_A'}
    tolerance, outside_bad, inside_bad, examples = T['transparency.tolerance'], 0, 0, []
    for row, i in enumerate(indices):
        outside = (table.wave < table.sed_support[i, 0]) | (table.wave > table.sed_support[i, 1])
        values = transmission[row]
        bad_out = int(np.sum(np.abs(values[outside] - 1.0) > tolerance))
        bad_in = int(np.sum((values[~outside] < -tolerance) | (values[~outside] > 1.0 + tolerance)))
        outside_bad, inside_bad = outside_bad + bad_out, inside_bad + bad_in
        if (bad_out or bad_in) and len(examples) < 10:
            examples.append(table.job_ids[i])
    detail = {'sampled': int(indices.size), 'bins_wrong_outside': outside_bad, 'bins_wrong_inside': inside_bad,
              'examples': examples, 'summary': f'{outside_bad} bins != 1 outside, {inside_bad} out of [0, 1] inside'}
    return (PASS if outside_bad == inside_bad == 0 else FAIL), detail


# CHECKS on CLOUDY working directory (if still present, raw models not processed in data merged .h5)

@check('rw-cloudy_version_in_outputs', blocking=True)
def check_cloudy_version(ctx):
    """The version banner of every kept output contains the required version.

    This function checks whether the outputs from a CLOUDY run include the
    required version specified in the charter, data attributes, or a default
    value defined by CLOUDY_REQUIRED. Each main output is scanned for the
    presence of a version token to ensure compatibility.

    Args:
        ctx: The context object for running the check, which provides access
        to charter information, raw runs, and file handling.

    Returns:
        tuple: A two-element tuple where the first element is the status of the
        check (e.g., PASS, FAIL, or SKIP), and the second element is a detailed
        dictionary containing information about the check results, such as the
        required version, number of outputs checked, and details of any outputs
        that failed the version check.
    """
    runs = ctx.raw_runs()
    if not runs:
        return SKIP, dict(_NO_WORKDIRS)
    expected = ((ctx.charter or {}).get('cloudy_version') or ctx.data.root_attrs.get('cloudy_version')
                or CLOUDY_REQUIRED)
    token, wrong, read = str(expected).lstrip('Cc'), [], 0
    for job, workdir in runs:
        out = ctx.raw_file(job, workdir, '.out')
        if out.is_file():
            read += 1
            banner = read_cloudy_banner(_read_text(out))
            if banner is None or token not in banner:
                wrong.append({'job_id': job, 'banner': banner})
    if read == 0:
        return SKIP, {'reason': 'no main output among the kept runs'}
    detail = {'required': expected, 'outputs': read, 'wrong': wrong[:20],
              'summary': f'{read - len(wrong)}/{read} outputs from CLOUDY {token}'}
    return (PASS if not wrong else FAIL), detail

@check('rw-abundance_echo', blocking=True)
def check_abundance_echo(ctx):
    """
    Checks the gas-phase composition echoed by CLOUDY in the main output of the runs kept.

    The primary function of this check is to validate the gas-phase abundance blocks echoed by CLOUDY
    in each simulation run. It ensures that the abundance blocks contain the expected number of
    elements, check that helium abundances align with the expected values at a given metallicity
    (log zeta_O), verify that no element exceeds its total abundance, and confirm that the echoed
    blocks correctly reflect differences in abundance patterns when comparing runs at the same F*
    but with different metallicities. The presence and quality of main output files are also checked.

    Pass Criteria:
    1. Thirty elements must be present in every run (absent ones are silently switched off).
    2. Helium-to-hydrogen (He/H) ratio must be as expected at the run's metallicity (log zeta_O)
       within 0.01 dex.
    3. No element should have an abundance above its total.
    4. For runs at the same F*, the echoed abundance difference compared to a reference run close to
       log zeta_O = 0 must equal the log pattern differences (within a deviation tolerance of 0.001
       dex plus one print step).

    A PASS outcome indicates that all checks are satisfied. A PARTIAL outcome is returned in cases
    where checks (2)-(4) cannot be performed. A FAIL outcome is issued if any problem is detected
    (e.g., missing or incomplete composition blocks, He/H deviations, presence of excess elements,
    etc.).

    Arguments:
        ctx (any): Context object containing details about the pipeline runs, data, and associated
            metadata.

    Returns:
        tuple: A tuple containing the outcome of the check and a detailed dictionary. The outcome will
        be one of PASS, PARTIAL, or FAIL, based on the results of the checks. The dictionary includes
        details such as the number of outputs analyzed, the count of He/H-graded runs, the count of
        pattern differences, the worst pattern deviation (in dex), a summary, and the first 20
        problems identified, if any.

    Raises:
        Does not raise any errors directly. The function handles exceptions where the chemistry module
        cannot be loaded.

    """
    runs = ctx.raw_runs()
    if not runs:
        return SKIP, dict(_NO_WORKDIRS)
    blocks = {job: read_gas_phase_abundances(ctx.raw_file(job, wd, '.out')) for job, wd in runs
              if ctx.raw_file(job, wd, '.out').is_file()}
    if not blocks:
        return SKIP, {'reason': 'no main output among the kept runs'}
    try:
        chemistry = _chemistry()
    except Exception:  # noqa: BLE001
        chemistry = None
    table, step, count = ctx.data, T['echo.print_step'], int(T['echo.elements'])
    log_zeta_all, f_star_all = table.param('log_zeta_O'), table.param('F_star')
    problems, helium_graded, groups = [], 0, {}
    for job, block in blocks.items():
        if block is None:
            problems.append({'job_id': job, 'problem': 'composition block not found'})
            continue
        if len(block) != count:
            problems.append({'job_id': job, 'problem': f'{len(block)} elements instead of {count}'})
        i = table.index.get(job)
        if chemistry is None or i is None or log_zeta_all is None:
            continue
        log_zeta = float(log_zeta_all[i])
        pattern = chemistry.abundance_pattern(log_zeta)
        if 'He' in block:
            helium_graded += 1
            off = block['He'] - np.log10(pattern['He'])
            if abs(off) > T['helium.tolerance']:
                problems.append({'job_id': job, 'problem': f'He/H off by {off:+.4f} dex'})
        excess = {x: _round(block[x] - np.log10(pattern[x]), 4) for x in block
                  if x in pattern and block[x] - np.log10(pattern[x]) > step}
        if excess:
            problems.append({'job_id': job, 'problem': f'gas above the total abundance: {excess}'})
        if f_star_all is not None:
            groups.setdefault(_scan_key(f_star_all[i]), []).append((log_zeta, job, block))
    compared, worst = 0, 0.0
    for members in groups.values():
        if len({round(m[0], 8) for m in members}) < 2:
            continue
        ref_zeta, ref_job, ref_block = min(members, key=lambda m: abs(m[0]))
        ref_pattern = chemistry.abundance_pattern(ref_zeta)
        for log_zeta, job, block in members:
            if job == ref_job:
                continue
            pattern = chemistry.abundance_pattern(log_zeta)
            for x in set(block) & set(ref_block) & set(pattern):
                deviation = abs((block[x] - ref_block[x]) - np.log10(pattern[x] / ref_pattern[x]))
                compared, worst = compared + 1, max(worst, deviation)
                if deviation > T['pattern.max_dev'] + step:
                    problems.append({'job_id': job, 'element': x, 'deviation_dex': _round(deviation, 5)})
    detail = {'outputs': len(blocks), 'helium_graded': helium_graded, 'pattern_differences': compared,
              'worst_pattern_deviation_dex': _round(worst, 5), 'problems': problems[:20],
              'summary': f'{len(blocks)} outputs, {helium_graded} He/H graded, {compared} pattern differences, '
                         f'{len(problems)} problems'}
    if chemistry is None:
        detail['chemistry'] = 'unavailable: only the element count was checked'
    return (FAIL if problems else (PASS if compared else PARTIAL)), detail

@check('rw-line_raster_closure')
def check_line_raster_closure(ctx):
    """
    The function performs a validation check on the line raster closure, ensuring that the
    fluxes provided in the line-only column of the spectrum data meet specific criteria.

    Summary:
    The function assesses the fidelity of the flux values in a column (col9) of a spectral
    dataset by comparing the calculated summed window fluxes to the listed theoretical fluxes.
    It considers continuum-pumped lines, strong lines, and non-graded regions based on specific
    thresholds and grades the results accordingly.

    Parameters:
    ctx: Context object providing access to raw run files, configurations, and helper methods.

    Returns:
    tuple: A tuple containing the grade result ('PASS', 'FAIL', or 'SKIP') and a details dictionary
    containing:
        - runs: Number of runs processed.
        - lines_evaluated: Number of spectral lines evaluated.
        - violations: List of up to 20 closure violations with details.
        - strong_lines_above: Count of strong lines outside the tolerance range.
        - pump_dominated_windows: Number of windows affected by continuum pumping.
        - summary: Text summary of the evaluation results.
    """
    runs = ctx.raw_runs()
    if not runs:
        return SKIP, dict(_NO_WORKDIRS)
    half, tolerance = int(T['closure.half_window_bins']), T['closure.max_rel_dev']
    violations, evaluated, n_runs, above, pumped_windows = [], 0, 0, 0, 0
    for job, workdir in runs:
        con, lines = ctx.raw_file(job, workdir, '.con'), ctx.raw_file(job, workdir, '.lines')
        if not (con.is_file() and lines.is_file()):
            continue
        columns = parse_cloudy_con(str(con))                  # (lambda, col2, col3, col4, col9)
        wave, col9 = np.asarray(columns[0], float), np.asarray(columns[4], float)
        listed = parse_cloudy_linelist(str(lines))
        raw_fluxes = np.array(listed['fluxes'], float)
        pump = ctx.raw_file(job, workdir, '.pump')
        if pump.is_file():  # continuum pumping of the pumped lines, removed as in the parser
            remove_pumping(listed, str(pump))
        names, fluxes = [_decode(n) for n in listed['names']], np.asarray(listed['fluxes'], float)
        with np.errstate(divide='ignore', invalid='ignore'):
            pumped = np.where(raw_fluxes > 0, 1.0 - fluxes / raw_fluxes, 0.0) > T['closure.max_pumped_fraction']
        catalogue = LineCatalogue(names, np.asarray(listed['wavelengths'], float))
        k_hbeta = catalogue.find(*LINES['H_beta'])
        hbeta = fluxes[k_hbeta] if k_hbeta is not None else np.nan
        log_wave = np.log(wave)
        energy = col9 * np.abs(np.gradient(log_wave))
        # the continuum mesh is in vacuum, the listed wavelengths in CLOUDY's print convention
        bins = np.array([int(np.argmin(np.abs(log_wave - np.log(w)))) if np.isfinite(w) else -10 ** 9
                         for w in air_to_vacuum_A(catalogue.wavelength)])
        n_runs += 1
        for k, b in enumerate(bins):
            if b < half or b >= wave.size - half or fluxes[k] <= 0 or catalogue.wavelength[k] > T['closure.max_graded_A']:
                continue
            window = np.abs(bins - b) <= half
            if pumped[window].any():
                pumped_windows += 1
                continue
            ratio = float(energy[b - half:b + half + 1].sum()) / float(fluxes[window & (fluxes > 0)].sum())
            evaluated += 1
            if ratio < 1.0 - tolerance:
                violations.append({'job_id': job, 'line': names[k], 'ratio': _round(ratio)})
            elif ratio > 1.0 + tolerance and fluxes[k] >= T['closure.strong_line_min_hbeta'] * hbeta:
                above += 1
    if n_runs == 0:
        return SKIP, {'reason': 'no kept run with a continuum and a line-list output'}
    detail = {'runs': n_runs, 'lines_evaluated': evaluated, 'violations': violations[:20],
              'strong_lines_above': above, 'pump_dominated_windows': pumped_windows,
              'summary': f'{len(violations)} closure violations over {evaluated} lines in {n_runs} runs '
                         f'({above} strong lines above, {pumped_windows} pump-dominated windows, not graded)'}
    return (PASS if not violations else FAIL), detail

@check('rw-grain_abundance_file', blocking=True)
def check_grain_abundance_file(ctx):
    """
    Inspects grain-abundance files from simulation runs and validates their format and integrity according to
    specific criteria.

    The function processes the grain-abundance output file, analyzing its structure for the following:
    - The file contains exactly one '#Depth ... total' header.
    - Rows have a consistent width across the file.
    - There is at least one zone present, and depth increases strictly across zones.
    - The data is integrated to ensure positive and finite values are returned.

    The validation includes examining headers, row structure, and integrated values. It reports files with issues
    and provides a summary of valid and invalid files.

    Parameters:
        ctx (object): Context object that provides access to raw runs and files.

    Returns:
        tuple: A tuple containing the result status and detailed summary. The result status can be SKIP, PASS,
        or FAIL. The detail includes the number of files analyzed, identified problems, and an overall summary.

    Raises:
        None directly, but issues during file parsing or invalid formats are captured and reported as problems.
    """
    runs = ctx.raw_runs()
    if not runs:
        return SKIP, dict(_NO_WORKDIRS)
    problems, files = [], 0
    for job, workdir in runs:
        path = ctx.raw_file(job, workdir, '.dusa')
        if not path.is_file():
            continue
        files += 1
        info, issues = inspect_grain_abundance_file(path), []
        if info['depth_headers'] != 1 or info['last_header_token'] != 'total':
            issues.append(f"{info['depth_headers']} '#Depth' headers, last column {info['last_header_token']!r}")
        if info['zones'] < 1 or len(info['column_counts']) > 1 or info['unreadable_rows']:
            issues.append(f"{info['zones']} zones, row widths {info['column_counts']}")
        if not info['depth_increasing']:
            issues.append('depth not strictly increasing: more than one block')
        if not issues:
            try:
                column = float(integrate_grain_abundance(str(path)))
                if not (np.isfinite(column) and column > 0):
                    issues.append(f'parser returned {column}')
            except Exception as exc:  # noqa: BLE001 - the refusal is the finding
                issues.append(f'parser refused the file: {_reason(exc)}')
        if issues:
            problems.append({'job_id': job, 'problems': issues})
    if files == 0:
        return SKIP, {'reason': 'no grain-abundance file among the kept runs'}
    detail = {'files': files, 'problems': problems[:20],
              'summary': f'{files - len(problems)}/{files} files with one block of zones'}
    return (PASS if not problems else FAIL), detail

@check('rw-incident_continuum_matches_sed', blocking=True)
def check_incident_continuum(ctx):
    """
    Checks the consistency of the incident continuum with the SED file specified in the
    CLOUDY deck across multiple runs.

    This function verifies that the SED file referenced using 'table SED' in the CLOUDY
    deck is correctly reflected in the incident continuum output of the simulation. The
    check is performed by comparing two key metrics: the ionizing photons per unit energy
    and the energy fraction in the Habing band. A pass is determined if both metrics differ
    by less than 5% between the incident continuum and the SED file for all runs.

    Parameters:
        ctx (object): Context object containing information about the simulation runs.
                      This context provides access to directories, input files, and
                      continuum outputs required for the consistency checks.

    Returns:
        tuple[str, dict]: A tuple representing the status of the check and detailed results.
                          If any issues are detected, the status will be 'FAIL', otherwise
                          'PASS'. The detailed results include the number of comparisons
                          performed, the worst relative difference found, a summary, and
                          details of any problems detected (limited to 20 entries).

    Raises:
        None
    """
    runs = ctx.raw_runs()
    if not runs:
        return SKIP, dict(_NO_WORKDIRS)
    band, lyman = (CONST.FUV_Lo_A, CONST.FUV_Hi_A), float(CONST.LymanA)
    problems, compared, worst = [], 0, 0.0
    for job, workdir in runs:
        deck, con = ctx.raw_file(job, workdir, '.in'), ctx.raw_file(job, workdir, '.con')
        match = DECK_TABLE_SED.search(_read_text(deck)) if deck.is_file() else None
        if match is None or not con.is_file():
            continue
        name = pathlib.Path(match.group(1)).name
        path = next((p for p in (workdir / 'SED' / name, workdir / name) if p.is_file()), None)
        if path is None:
            problems.append({'job_id': job, 'problem': f'SED file {name} not found'})
            continue
        wave, col2 = parse_cloudy_con(str(con))[:2]
        stellar = wave < CLOUDY.Stellar_Max_A  # above it lives the CMB of the run, not in the file
        file_wave, file_nufnu = read_sed_file(path)
        on_mesh = np.exp(np.interp(np.log(wave[stellar]), np.log(file_wave), np.log(np.maximum(file_nufnu, 1e-300))))
        cloudy = incident_shape(wave[stellar], col2[stellar], lyman, band)
        written = incident_shape(wave[stellar], on_mesh, lyman, band)
        d_ion, d_band = (abs(c / w - 1.0) if w > 0 and np.isfinite(c) else np.inf for c, w in zip(cloudy, written))
        compared, worst = compared + 1, max(worst, d_ion, d_band)
        if d_ion > T['incident.ionizing_ratio_rtol'] or d_band > T['incident.band_fraction_rtol']:
            problems.append({'job_id': job, 'sed_file': name, 'ionising_rel_diff': _round(d_ion, 5),
                             'band_rel_diff': _round(d_band, 5)})
    if compared == 0 and not problems:
        return SKIP, {'reason': 'no kept run with a deck naming a SED file and a continuum output'}
    detail = {'compared': compared, 'worst_rel_diff': _round(worst, 5), 'problems': problems[:20],
              'summary': f'{compared} runs compared, {len(problems)} problems (worst difference {worst:.2%})'}
    return (PASS if not problems else FAIL), detail

@check('rw-line_list_complete', blocking=True)
def check_line_list_complete(ctx):
    """
    Every line requested by the list file of the deck comes out of CLOUDY, with a valid flux.

    Lines of the main output mentioning a line not found are reported for inspection.

    Pass: requested = returned and fluxes finite, >= 0, in every run.

    Parameters:
        ctx (Context): Context object containing information about the raw runs and file management.

    Returns:
        tuple: A tuple where the first element is either SKIP, PASS, or FAIL, indicating the check status,
               and the second element is a dictionary containing details such as problems identified,
               summary statistics, and output lines requiring inspection.

    Raises:
        None
    """
    runs = ctx.raw_runs()
    if not runs:
        return SKIP, dict(_NO_WORKDIRS)
    problems, suspicious, checked = [], [], 0
    for job, workdir in runs:
        deck, lines = ctx.raw_file(job, workdir, '.in'), ctx.raw_file(job, workdir, '.lines')
        if not (deck.is_file() and lines.is_file()):
            continue
        match = DECK_LINE_LIST.search(_read_text(deck))
        list_file = workdir / match.group(1) if match else None
        if list_file is None or not list_file.is_file():
            problems.append({'job_id': job, 'problem': 'list file of the deck not found'})
            continue
        requested = sum(1 for line in _read_text(list_file).splitlines() if line.strip() and not line.startswith('#'))
        fluxes = np.asarray(parse_cloudy_linelist(str(lines))['fluxes'], float)
        checked += 1
        if fluxes.size != requested or not np.all(np.isfinite(fluxes)) or np.any(fluxes < 0):
            problems.append({'job_id': job, 'returned': int(fluxes.size), 'requested': requested})
        out = ctx.raw_file(job, workdir, '.out')
        if out.is_file():
            suspicious += [f'{job}: {text.strip()}' for text in _read_text(out).splitlines()
                           if 'line' in text.lower() and any(k in text.lower() for k in ('not find', 'not found',
                                                                                        'unknown'))]
    if checked == 0 and not problems:
        return SKIP, {'reason': 'no kept run with a deck and a line-list output'}
    detail = {'runs': checked, 'problems': problems[:20], 'output_lines_to_inspect': suspicious[:20],
              'summary': f'{checked - len(problems)}/{checked} runs return every requested line'}
    return (PASS if not problems else FAIL), detail

@check('rw-runtime_per_model', blocking=True)
def check_runtime(ctx):
    """
    Check the runtime and projected core hours for planned runs.

    This function calculates the mean execution time from the provided outputs of kept runs, estimates the
    projected core hours required for the planned runs, and evaluates if this projected cost is within
    the defined budget limit.

    Parameters:
    ctx (object): The runtime context containing relevant data, including job output files and expected
                  number of runs.

    Returns:
    tuple: A tuple containing a status ('PASS', 'FAIL', or 'SKIP') and a details dictionary with
           calculated metrics and additional information.
    """
    outputs = [ctx.raw_file(job, workdir, '.out') for job, workdir in ctx.raw_runs()]
    times = [read_exec_time(_read_text(out)) for out in outputs if out.is_file()]
    times = np.asarray([t for t in times if t is not None])
    if times.size == 0:
        return SKIP, {'reason': 'no execution time in the kept main outputs'}
    detail = {'runs': int(times.size), 'mean_s': _round(times.mean(), 2), 'median_s': _round(np.median(times), 2),
              'max_s': _round(times.max(), 2)}
    if not ctx.n_expected:
        detail['reason'] = 'planned runs unknown (no --config or --n-expected)'
        return SKIP, detail
    projected = float(times.mean() * ctx.n_expected / 3600.0)
    detail.update({'planned_runs': ctx.n_expected, 'projected_core_hours': _round(projected, 1),
                   'summary': f'mean {times.mean():.1f} s -> {projected:.0f} core hours for {ctx.n_expected} runs'})
    return (PASS if projected <= T['runtime.budget_core_hours'] else FAIL), detail

@check('rw-energy_balance_from_raw_output', blocking=True)
def check_raw_energy_balance(ctx):
    """
    Energy balance recomputed from the continuum outputs, and the attribute stored by the parser.

    This function validates the energy balance consistency by checking the recomputed energy
    from the continuum outputs against stored energy balance attributes in the parser. It compares
    the integral of nu*F_nu (energy density) over the wavelength range against the incident stellar
    energy, ensuring that the overall median imbalance is within acceptable thresholds and all
    stored attributes agree with the computed results.

    Parameters:
        ctx (Context): The context object that provides access to raw runs, raw files, and data
                       for performing the energy balance checks.

    Returns:
        tuple: A tuple consisting of the status of the check (PASS, FAIL, or SKIP) and a detailed
               dictionary containing:
               - The number of runs considered for the check.
               - Median imbalance percentage.
               - A list of up to 20 disagreements between stored attributes and recomputed values.
               - A list of up to 20 runs without incident energy.
               - A summary of the results.

    Raises:
        None
    """
    runs = ctx.raw_runs()
    if not runs:
        return SKIP, dict(_NO_WORKDIRS)
    table, energy, disagreements, unlit = ctx.data, [], [], []
    for job, workdir in runs:
        con = ctx.raw_file(job, workdir, '.con')
        if not con.is_file():
            continue
        wave, col2, col3, col4 = (np.asarray(c, float) for c in parse_cloudy_con(str(con))[:4])
        everywhere = np.ones(wave.shape, dtype=bool)
        e2, e3, e4 = (_integrate_log(c, wave, everywhere) for c in (col2, col3, col4))
        e2_stellar = _integrate_log(col2, wave, wave < CLOUDY.Stellar_Max_A)
        if not e2_stellar > 0:
            unlit.append(job)
            continue
        value = abs(e2 - (e3 + e4)) / e2_stellar
        energy.append(value)
        i = table.index.get(job)
        if i is not None and np.isfinite(table.energy_balance[i]) \
                and abs(table.energy_balance[i] - value) > T['raw_balance.attribute_atol']:
            disagreements.append({'job_id': job, 'stored': _round(table.energy_balance[i], 6),
                                  'energy_weighted': _round(value, 6)})
    if not energy and not unlit:
        return SKIP, {'reason': 'no continuum output among the kept runs'}
    median = float(np.median(energy)) if energy else np.nan
    detail = {'runs': len(energy), 'median_imbalance': _round(median, 6),
              'attribute_disagreements': disagreements[:20], 'without_incident_energy': unlit[:20],
              'summary': f'median imbalance {median:.3%} over {len(energy)} runs; '
                         f'{len(disagreements)} stored attributes disagree'}
    ok = median < T['energy_balance.max_median'] and not disagreements and not unlit
    return (PASS if ok else FAIL), detail

########################################################################################################################
########################################################################################################################
#####################àààààààààààààààààààààààà###########################################################################

# Interface

def run_checks(ctx, strict=False):
    """
    Executes predefined checks based on the provided context and returns the results.

    Summary:
        This function iterates through a list of checks applicable to the given context sector
        and executes their associated functions. It collects the results and determines which
        checks have failed, adhering to the strictness setting if provided. A progress bar is
        displayed during execution to track the performed checks.

    Parameters:
        ctx: The context object specifying the sector for which the checks should be executed.
            The checks are filtered based on the sector value.
        strict: bool (optional)
            If set to True, skipped checks are considered as failed. Defaults to False.

    Returns:
        tuple:
            A tuple containing:
            - results: A list of dictionaries where each dictionary details the name,
              status, blocking status, and additional details of the executed check.
            - failed: A list of names of the checks that have failed. Blocking checks
              with statuses of FAIL or ERROR, and skipped checks marked as strict, are included.
    """
    results = []
    specs = [spec for spec in CHECKS if ctx.sector in spec.sectors]
    pbar = tqdm(specs, desc=f'[validate {ctx.sector}]', unit='check', file=sys.stderr)
    for spec in pbar:
        pbar.set_postfix_str(spec.name)  # the check running now: the rw-* ones read every workdir
        try:
            status, detail = spec.function(ctx)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            status, detail = ERROR, {'error': _reason(exc), 'summary': f'raised {type(exc).__name__}',
                                     'traceback_tail': traceback.format_exc().strip().splitlines()[-6:]}
        results.append({'name': spec.name, 'status': status,
                        'blocking': spec.blocking, 'detail': detail})
    failed = [r['name'] for r in results
              if r['blocking'] and (r['status'] in (FAIL, ERROR) or (strict and r['status'] == SKIP))]
    return results, failed

def render_markdown(report):
    """
    Renders a markdown report based on the provided validation grid report.

    Parameters:
        report (dict): A dictionary containing the validation grid report. It must
                       include keys like 'sector', 'overall', 'data', 'models',
                       'converged', 'failed_blocking', 'strict', and 'checks' with
                       relevant data types for each.

    Returns:
        str: A string representing the generated markdown report.
    """
    lines = [f"# Grid validation - {report['sector'].upper()} - {report['overall']}", '',
             f"Data `{report['data']}`: {report['models']} models, {report['converged']} converged. "
             f"Blocking failures: {', '.join(report['failed_blocking']) or 'none'}"
             + (' (strict)' if report['strict'] else ''), '',
             '| Check | Status | Blocking | Summary |', '|---|---|---|---|']
    for c in report['checks']:
        summary = str(c['detail'].get('summary', c['detail'].get('reason', ''))).replace('|', '/')
        lines.append(f"| {c['name']} | {c['status']} | {'yes' if c['blocking'] else ''} | {summary} |")
    lines += ['', '## Details', '']
    for c in report['checks']:
        lines.append(f"### {c['name']} - {c['status']}")
        lines += [f'- **{k}**: `{json.dumps(v)}`' for k, v in c['detail'].items() if k != 'summary'] + ['']
    return '\n'.join(lines) + '\n'

def _write(path, text):
    """
    Writes the provided text to a file at the specified path. Creates any necessary
    parent directories if they do not already exist.

    Parameters:
    path (str or pathlib.Path): The file path where the text should be written.
                                Parent directories will be created if needed.
    text (str): The content to write into the file.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

def _load_yaml(path):
    """
    Loads and parses a YAML file, returning its contents as a Python object.

    Args:
        path (str): The file path to the YAML file to be loaded.

    Returns:
        Any: The parsed content of the YAML file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If an error occurs during parsing the YAML file.
    """
    import yaml
    with open(path) as handle:
        return yaml.safe_load(handle)

def _load_json(path):
    """
    Loads and parses a JSON file from the specified path.

    This function reads the content of a file from the given file path and
    parses it as JSON, returning the corresponding Python object.

    Parameters:
    path : str
        The file path to the JSON file.

    Returns:
    dict
        Parsed JSON content as a Python dictionary.
    """
    return json.loads(pathlib.Path(path).read_text())

def _expected_runs(charter):
    """
    Calculate the expected number of runs based on input data structure.

    This function attempts to compute the product of specific keys present in the
    nested dictionary `charter`. If any of the required keys are missing or their
    values are invalid, the function returns None.

    Arguments:
        charter (dict): A nested dictionary expected to contain keys 'lhs', 'nodes',
            and their respective subkeys 'n_samples', 'n_tau_ssp', and 'n_zstar'.
            Values for these keys must be convertible to integers.

    Returns:
        int or None: The computed product of the values if all keys are present
        and their values are valid integers. Otherwise, returns None.
    """
    try:
        return int(charter['lhs']['n_samples']) * int(charter['nodes']['n_tau_ssp']) * int(charter['nodes']['n_zstar'])
    except (KeyError, TypeError, ValueError):
        return None

def _default_config_path(sector):
    """
    Determines the default configuration file path for a specified sector.

    This function iterates through a set of predefined file paths and checks whether
    a valid configuration file exists. If no valid file is found in the predefined
    paths, it attempts to look up the file using the datapath settings from
    `galapy.configuration.rcParams` and the global configuration location defined in
    `galapy.internal.globs`. If neither approach succeeds, it returns None.

    Parameters:
    sector: str
        The name of the sector for which the configuration path is being determined.

    Returns:
    str or None
        The path to the configuration file if found, otherwise None.
    """
    for candidate in (
        f'data/grids/{sector}/grid_{sector}.yaml',
        f'data/grids/{sector}/{sector}.yaml',
        f'data/{sector}/grid_{sector}.yaml',
        f'grid_{sector}.yaml',
    ):
        if pathlib.Path(candidate).is_file():
            return candidate
    try:
        from galapy.configuration import rcParams
        import galapy.internal.globs as GLOBS
        for datapath in rcParams.get('datapath', []):
            cand = pathlib.Path(datapath, *GLOBS.CONFIG_NEB, f'grid_{sector}.yaml')
            if cand.is_file():
                return str(cand)
    except Exception:
        pass
    return None

def _default_lhs_path(sector):
    """
    Generate the default file path for a given sector.

    This function checks a predefined list of potential file paths for the specified
    sector and returns the first path that exists. If none of the paths exist,
    the function returns None.

    Parameters:
    sector: str
        The name of the sector used to construct the file paths.

    Returns:
    str | None
        The first valid file path that exists, or None if no valid paths are found.
    """
    for candidate in (
        f'data/grids/{sector}/lhs_{sector}.npy',
        f'data/grids/{sector}/lhs.npy',
        f'data/{sector}/lhs_{sector}.npy',
    ):
        if pathlib.Path(candidate).is_file():
            return candidate
    return None

def _default_ssp_meta_path(sector):
    """
    Determine the default metadata path for a given sector.

    This function checks for the existence of a metadata file in a predefined
    set of directory paths. It returns the first valid file path found or
    None if no valid path exists.

    Parameters:
    sector : str
        The name of the sector for which the metadata path is being determined.

    Returns:
    str or None
        The file path to the metadata if it exists, or None if no valid path
        is found.
    """
    for candidate in (
        'data/cloudy_seds/ssp_metadata.json',
        f'data/grids/{sector}/ssp_metadata.json',
        f'data/{sector}/ssp_metadata.json',
    ):
        if pathlib.Path(candidate).is_file():
            return candidate
    return None

def _default_raw_workdirs(sector):
    """
    Generate the default working directory path for a given sector.

    This function checks if a constructed directory path exists and is a directory. If the directory
    exists, the path is returned. Otherwise, the function returns None. It is specifically designed to
    work with directory structures associated with each sector.

    Args:
        sector (str): The sector name used to construct the directory path.

    Returns:
        Optional[str]: The directory path if it exists and is valid, or None if the directory does
        not exist.
    """
    candidate = f'data/{sector}/work'
    return candidate if pathlib.Path(candidate).is_dir() else None

#######################
def main(argv=None):
    """Entry point; returns the exit code."""
    ap = argparse.ArgumentParser(prog='galapy-validate-grid',
                                 description='Validate a CloudIA grid data (hii or pdr sectors).')
    ap.add_argument('--sector', required=True, choices=('hii', 'pdr'),
                    help='grid sector to validate: hii or pdr')
    ap.add_argument('--data', '--corpus', dest='data', default=None,
                    help='merged h5 data (default: data/products/{sector}_grid.h5)')
    ap.add_argument('--report', default=None,
                    help='JSON report (default: data/validation/report_{sector}.json)')
    ap.add_argument('--markdown', default=None,
                    help='Markdown report (default: data/validation/report_{sector}.md)')
    ap.add_argument('--config', default=None,
                    help='grid charter yaml (default: data/grids/{sector}/grid_{sector}.yaml or database)')
    ap.add_argument('--lhs', default=None,
                    help='continuous LHS sample .npy (default: data/grids/{sector}/lhs_{sector}.npy)')
    ap.add_argument('--ssp-meta', default=None,
                    help='SSP node metadata (default: data/cloudy_seds/ssp_metadata.json)')
    ap.add_argument('--raw-workdirs', default=None,
                    help='directory of kept CLOUDY working directories (default: data/{sector}/work)')
    ap.add_argument('--n-expected', type=int, default=None, help='planned runs (overrides the charter)')
    ap.add_argument('--max-spectra', type=int, default=2000, help='models sampled by the spectral checks')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--strict', action='store_true', help='a skipped blocking check fails the run')
    args = ap.parse_args(sys.argv[1:] if argv is None else [str(a) for a in argv])
    sector = args.sector
    tag = f'[validate/{sector}]'

    # Ensure validation directory exists for default reports
    val_dir = pathlib.Path('data/validation')
    val_dir.mkdir(parents=True, exist_ok=True)

    # defaults in case only sector specified
    if args.data is None:
        args.data = f'data/products/{sector}_grid.h5'
    if args.report is None:
        args.report = str(val_dir / f'report_{sector}.json')
    if args.markdown is None:
        args.markdown = str(val_dir / f'report_{sector}.md')
    if args.config is None:
        args.config = _default_config_path(sector)
    if args.lhs is None:
        args.lhs = _default_lhs_path(sector)
    if args.ssp_meta is None:
        args.ssp_meta = _default_ssp_meta_path(sector)
    if args.raw_workdirs is None:
        args.raw_workdirs = _default_raw_workdirs(sector)

    try:
        data = DataTable(args.data)
    except Exception as exc:
        _write(args.report, json.dumps({'sector': args.sector, 'data': args.data, 'overall': FAIL,
                                        'error': _reason(exc), 'failed_blocking': ['data_load'], 'checks': []},
                                       indent=2))
        if args.markdown:
            _write(args.markdown, f"# Grid validation - {args.sector.upper()} - FAIL\n\nData `{args.data}`: not readable ({_reason(exc)}).\n")
        print(f'[grid-val] {tag} FAIL - data not readable: {exc}', file=sys.stderr)
        return 1
    ctx = Context(sector=args.sector, data=data, raw_workdirs=args.raw_workdirs,
                  max_spectra=args.max_spectra, seed=args.seed)
    input_errors = []
    for attribute, path, loader in (('charter', args.config, _load_yaml), ('lhs', args.lhs, np.load),
                                    ('ssp_meta', args.ssp_meta, _load_json)):
        if path is not None:
            try:
                setattr(ctx, attribute, loader(path))
            except Exception as exc:
                input_errors.append(f'{attribute} ({path}): {_reason(exc)}')
    ctx.n_expected = args.n_expected or (_expected_runs(ctx.charter) if ctx.charter else None)
    results, failed = run_checks(ctx, strict=args.strict)
    report = _clean({'sector': args.sector, 'data': data.path, 'models': data.n,
                     'converged': int(data.converged.sum()), 'strict': args.strict,
                     'overall': FAIL if failed else PASS, 'failed_blocking': failed,
                     'input_errors': input_errors, 'thresholds': THRESHOLDS, 'checks': results})
    _write(args.report, json.dumps(report, indent=2))
    if args.markdown:
        _write(args.markdown, render_markdown(report))
    counts = {s: sum(r['status'] == s for r in results) for s in (PASS, PARTIAL, SKIP, FAIL, ERROR)}
    print(f'[grid-val] {tag} {report["overall"]} - ' + ', '.join(f'{v} {k}' for k, v in counts.items() if v)
          + (f' - blocking failures: {failed}' if failed else ''))
    for message in input_errors:
        print(f'[grid-val] {tag} input not loaded: {message}', file=sys.stderr)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
