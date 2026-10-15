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
from galapy.spectroscopy.utils import CLOUDY_REQUIRED
from galapy.spectroscopy.utils.run_core import converged_from_out
from galapy.spectroscopy.utils.hii.parse_one_hii import (parse_cloudy_con, parse_cloudy_linelist, air_to_vacuum_A,
                                                         integrate_grain_abundance)


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
# Maximum relative deviation between rasterised energy and tabulated flux; ensures flux conservation
    'closure.max_rel_dev': 0.02,
# Flux threshold (10% of H-beta); grades upper-side raster excess only on lines free of weak blends
    'closure.strong_line_min_hbeta': 0.1,
# Theoretical [O III] 4959/5007 ratio from C25 Einstein A-values;
    'oiii.reference': 0.346,
# Maximum relative deviation from [O III] 0.346; detects corrupted atomic data
    'oiii.max_rel_dev': 0.01,
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
# Minimum Spearman correlation between target and measured f_esc; checks monotonic regulation by cloud depth
    'fesc.min_spearman': 0.9,
# Minimum KS p-value for depletion parameter F*; verifies uniform sampling across gas depletion states
    'fstar.min_p_value': 0.01,
# Minimum Habing-band FUV transmittance (T_FUV > 0.01); selects models that illuminate an adjacent PDR
    'fuv.region_min_transmittance': 1e-2,
# Maximum spectral tilt (dex) across Habing band; checks if transmitted FUV is flat enough for scalar G0
    'fuv.max_tilt': 0.1,
# Numerical tolerance on transmission; enforces T = 1 outside SED support and T in [0, 1] inside
    'transparency.tolerance': 1e-6,
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
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)

def _as_float(value):
    """Scalar float of an attribute, or None."""
    try:
        array = np.asarray(value)
        if array.dtype.kind in 'iuf' and array.size == 1:
            return float(array.ravel()[0])
    except (TypeError, ValueError):
        pass
    return None

def _clean(obj):
    """JSON-serialisable copy (numpy types, NaN -> None)."""
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
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None

def _percentiles(values, qs=(5, 50, 95), digits=4):
    """compute percentiles"""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return {f'p{q}': _round(np.percentile(values, q), digits) for q in qs} if values.size else None

def _combine(statuses):
    """Total status of a check (summ of all status)"""
    if not statuses or all(s == SKIP for s in statuses):
        return SKIP
    if any(s in (FAIL, ERROR) for s in statuses):
        return FAIL
    return PASS if all(s == PASS for s in statuses) else PARTIAL

def _reason(exc):
    """return reason for failure"""
    return f'{type(exc).__name__}: {exc}'

def _scan_key(value):
    """return scan key """
    value = float(value)
    return float(f'{value:.10g}') if np.isfinite(value) else None

@functools.lru_cache(maxsize=1)
def _chemistry():
    """Chemistry instance of the abundance module (reads the database); raises if unavailable."""
    from galapy.spectroscopy.utils.physics.abundances import Chemistry
    return Chemistry()

def _lyman_series_mask(wavelength_A, relative_half_width=3e-3, n_max=40):
    """compute mask for lyman series"""
    near = np.zeros(np.shape(wavelength_A), dtype=bool)
    for n in range(2, n_max + 1):
        near |= np.abs(np.asarray(wavelength_A) / (CONST.LyLimit / (1.0 - 1.0 / n ** 2)) - 1.0) < relative_half_width
    return near

def _integrate_log(values, wavelength, mask):
    """Integral over d ln(lambda) along the last axis inside `mask` (energy of a nu*F_nu column)."""
    x = np.log(np.asarray(wavelength, float)[mask])
    order = np.argsort(x)
    return _trapezoid(np.asarray(values, float)[..., mask][..., order], x[order], axis=-1)

def _spearman(a, b):
    """Spearman rank correlation, None when undefined (< 3 points or a constant input)."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3 or np.ptp(a[good]) == 0 or np.ptp(b[good]) == 0:
        return None
    return float(stats.spearmanr(a[good], b[good])[0])

def _ks_uniform(values, lower, upper, n_effective=None):
    """KS (kolmogorov-smirnov) distance to U(lower, upper) and its p-value for `n_effective` independent draws."""
    from scipy import stats
    u = np.sort((np.asarray(values, float) - lower) / (upper - lower))
    n = u.size
    distance = max(np.max(np.arange(1, n + 1) / n - u), np.max(u - np.arange(n) / n))
    return float(distance), float(stats.kstwo.sf(distance, int(n_effective or n)))

def _below_hyperbola(x, y, a, x0, b):
    """True where (x, y) lies below y = a / (x - x0) + b on the branch x < x0."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.zeros(x.shape, dtype=bool)
    left = x < x0
    out[left] = y[left] < a / (x[left] - x0) + b
    return out

def _log10_ratio(numerator, denominator):
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.log10(numerator / denominator)
    out[~((numerator > 0) & (denominator > 0))] = np.nan
    return out


########################################################################################################################

# CLOUDY raw files helper functions and inspects products

def _read_text(path):
    """Read text file, replacing invalid characters with U+FFFD."""
    return pathlib.Path(path).read_text(errors='replace')

def read_sed_file(path):
    """(wavelength [A], nu*F_nu) of a CLOUDY table-SED file (unit keywords ignored)."""
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
    """(ionising photon rate per unit energy, energy fraction in `band_A`) of a nu*F_nu spectrum.

    Both are independent of the normalisation, so the SED file and CLOUDY's
    incident column must agree.
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
    """First line of a CLOUDY output that starts with 'Cloudy' (the version banner)."""
    return next((line.strip() for line in out_text.splitlines()[:max_lines]
                 if line.strip().startswith('Cloudy')), None)

def read_exec_time(out_text):
    """Execution time on CLOUDY's final summary line, or None."""
    matches = EXEC_TIME_PATTERN.findall(out_text)
    return float(matches[-1]) if matches else None


_ELEMENT_ENTRY = re.compile(r'([A-Z][a-z]?)\s*:\s*([-+]?\d+\.\d+)')


def read_gas_phase_abundances(out_path):
    """log10(X/H) of the 'Gas Phase Chemical Composition' block, or None."""
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
    """Structure of a 'save last grain abundance ... no hash' file: '#Depth <bins> total', one row per zone."""
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
            'depth_increasing': bool(depth.size >= 2 and np.all(np.diff(depth) > 0))}

# LINES in the CLOUDY outputs and input

_WAVELENGTH_SUFFIX = {'A': 1.0, 'm': 1.0e4, 'c': 1.0e8}          # Angstrom, micron, cm

def parse_line_label(label):
    """('H 1', 6562.8) from a CLOUDY label such as 'H  1 6562.80A', 'O  3 88.3323m'; None if unreadable."""
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
        species, best, best_distance = species.casefold(), None, rtol
        for k, sp in enumerate(self.species):
            if sp == species and abs(self.wavelength[k] / wavelength_A - 1.0) <= best_distance:
                best, best_distance = k, abs(self.wavelength[k] / wavelength_A - 1.0)
        return best

# LINES UNDER ANALYSIS FOR CHECKS
LINES = {'H_alpha': ('H 1', 6562.8), 'H_beta': ('H 1', 4861.3), 'OIII_5007': ('O 3', 5006.8),
         'OIII_4959': ('O 3', 4958.9), 'NII_6584': ('N 2', 6583.5)}

def line_flux(table, name):
    """Flux of a diagnostic line for every model; None if not in the master list."""
    index = table.catalogue.find(*LINES[name])
    return None if index is None else table.lines[:, index].astype(float)

def _missing_lines(table, names):
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
                                       for name in CONST.Root_Shared_HII if name in f}
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
        """(len(indices), n_wave) array of `dataset` (NaN rows where missing), or None if no model has it."""
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
        """(job_id, directory) of every kept working directory."""
        root = pathlib.Path(self.raw_workdirs) if self.raw_workdirs else None
        return [(d.name, d) for d in sorted(root.iterdir()) if d.is_dir()] if root and root.is_dir() else []

    def raw_file(self, job, workdir, extension):
        """Path to a file in a kept working directory."""
        return pathlib.Path(workdir) / f'{self.sector}_{job}{extension}'

    def sample(self, mask):
        """Reproducible random subsample (at most max_spectra) of the models in `mask`."""
        candidates = np.flatnonzero(mask)
        if candidates.size <= self.max_spectra:
            return candidates
        return np.sort(np.random.default_rng(self.seed).choice(candidates, self.max_spectra, replace=False))

    def charter_bounds(self, attribute):
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
    """The units the data declares are the ones the HII parser writes, i.e. the ones every check assumes.

    Compared with the tables of galapy.internal.constants: the root group 'units' (one entry per
    member of the grid_point groups) and the attributes of the root datasets.

    Pass: same unit for every member and root dataset, same wavelength media.
    """
    table = ctx.data
    if not table.units:
        return SKIP, {'reason': 'data without units (written by an older parser)'}
    problems = []
    expected = {name: unit for name, (unit, _description) in CONST.Points_Units_HII.items() if unit is not None}
    for name in sorted(set(expected) | set(table.units)):
        if table.units.get(name) != expected.get(name):
            problems.append(f'{name}: {table.units.get(name)!r} instead of {expected.get(name)!r}')
    for name, (unit, _description, extra) in CONST.Root_Units_HII.items():
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
    """Converged runs over planned runs, and the convergence flags against the kept outputs.

    The planned runs come from the charter or --n-expected, so crashed runs count as failures.
    On the kept working directories the convergence read from the main output must agree with
    converged.flag, the NOT_CONVERGED marker and the data attribute; outputs without
    'Cloudy exited OK' are listed.

    Pass: converged / planned > 99% and no flag disagrees with its output.
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
    """Q_H of the written SED files (node metadata) versus the SSP table.

    Pass: worst node within 5%.
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
    """Emitted versus absorbed energy, per model.

    Graded: converged models with absorbed_fraction above the threshold (below it the print
    precision of the continua dominates; this also leaves out the models dominated by the CMB).
    The excess in the observable range (lambda >= 911.76 A) is reprocessing_excess - lyc_emit_fraction:
    the diffuse Lyman continuum is reported apart, since with `sphere` CLOUDY counts its inward
    half twice.

    Pass: >= 95% of the graded models with an observable excess <= 2%.
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
    """Emergent H-alpha / H-beta against Case B, around log U = -3 and z ~ 0.

    Pass: |median / 2.86 - 1| < 5%.
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
    """Models in the [N II]/H-alpha versus [O III]/H-beta diagram.

    Graded against the maximum-starburst line on the models with log U <= -2 (the range it was
    derived for); the fraction below the Kauffmann et al. (2003) line is reported, not graded.

    Pass: >= 95% of the models with log U <= -2 below the Kewley et al. (2001) line.
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
    """Kolmogorov-Smirnov uniformity of every marginal of the continuous sample.

    Pass: every p-value > 0.01 and no value outside the bounds.
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
    """Shape and scale of the diagnostic grain continuum, on a subsample of converged models.

    (a) no negative value; (b) peak of nu*F_nu between 5 and 500 um; (c) below 0.3 um less than
    1e-3 of the peak; (d) far-infrared (> 20 um) energy of the same order as the own continuum
    (median ratio in [0.1, 10]).

    Pass: (a)-(c) for >= 99% of the sample, and (d).
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
    """(n_models, n_bins): the line bin exceeds both local baselines by `excess`."""
    flags = np.zeros((spectra.shape[0], len(bins)), dtype=bool)
    for column, b in enumerate(bins):
        centre = np.max(spectra[:, b - 1:b + 2], axis=1)
        baseline = np.maximum(np.median(spectra[:, b - gap - half:b - gap], axis=1),
                              np.median(spectra[:, b + gap + 1:b + gap + 1 + half], axis=1))
        flags[:, column] = (baseline > 0) & (centre > (1.0 + excess) * baseline)
    return flags

@check('line_free_continua')
def check_line_free_continua(ctx):
    """No residual line in the own continuum (col4 - col9) nor in the grain continuum.

    At every line of the master list the bin is compared with two baselines a few bins away on
    each side (edges are steps, not spikes).

    Pass: <= 1% of the (model, line) pairs flagged, for both continua.
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
    """[O III] 4959 / 5007 (same upper level), worst converged model.

    Pass: within 1% of 0.346.
    """
    table = ctx.data
    if _missing_lines(table, ('OIII_4959', 'OIII_5007')):
        return SKIP, {'reason': '[O III] lines not in the master list'}
    weak, strong = line_flux(table, 'OIII_4959'), line_flux(table, 'OIII_5007')
    good = table.converged & (weak > 0) & (strong > 0)
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
    """Dust mass per solar mass against the truncated Stromgren scaling.

    M_dust / M_SSP ~ (1 - f_esc) xi_d m_Z(zeta_O) Q_H / n_H, constant fitted (median residual);
    m_Z (metal mass per H) from the chemistry module, as the grain_scale of the deck. The residual trend with log U is reported.

    Pass: >= 90% of the models within 0.2 dex.
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
    """Q_H of the 70 and 100 Myr nodes over Q_H of the youngest, per stellar metallicity.

    Pass: both ratios < 1% for every metallicity.
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
    """Coverage of the measured escape fraction and its monotonicity with the target.

    Pass: widest empty interval inside [0, 0.9] <= 0.1 and Spearman(target, measured) > 0.9.
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
    detail = {'n': int(good.sum()), 'widest_gap': _round(gap), 'gap_between': [_round(edges[k]), _round(edges[k + 1])],
              'rank_correlation': _round(rho if rho is not None else np.nan),
              'summary': f'widest gap {gap:.3f}, rank correlation {rho if rho is not None else float("nan"):.3f}'}
    ok = gap <= T['fesc.max_gap'] and rho is not None and rho > T['fesc.min_spearman']
    return (PASS if ok else FAIL), detail

@check('depletion(jk)_axis_coverage')
def check_depletion_coverage(ctx):
    """KS uniformity of F* over the converged models (sample size: distinct draws, not models).

    Pass: p-value > 0.01.
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
    """FUV terms of the ionised skin: sign, and spectral tilt of the transmission in the Habing band.

    (a) 0 < T_FUV <= 1 and N_FUV > 0. (c) where T_FUV > 1e-2, log10 T is fitted as a line in
    log10 lambda (Lyman-series bins excluded); tilt = log10 T(red edge) - log10 T(blue edge).
    A tilt above 0.1 dex is a decision, not a bug: add a tilt axis to the PDR grid.

    Pass: (a) for every model and max |tilt| < 0.1 dex.
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
        statuses.append(PASS if worst < T['fuv.max_tilt'] else FAIL)
    status = _combine(statuses)
    detail['summary'] = (f"{detail.get('positivity_violations', 'n/a')} sign violations, "
                         f"max |tilt| {detail.get('max_abs_tilt_dex', 'n/a')} dex")
    return status, detail

@check('transparency_outside_sed_support', blocking=True)
def check_transparency(ctx):
    """Transmission = 1 outside the SED support (sed_support_A) and in [0, 1] inside, on a subsample.

    Pass: no bin off by more than 1e-6.
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

    Required version: charter 'cloudy_version', else data attribute, else CLOUDY_REQUIRED.

    Pass: every output from the required version.
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
    """The gas-phase composition CLOUDY echoes in the main output of the kept runs.

    (1) thirty elements in every run (absent ones are switched off silently). With the chemistry
    module: (2) He/H as expected at the run's log zeta_O, within 0.01 dex; (3) no element above
    its total abundance; (4) among runs at the same F*, the echo of each run minus that of the
    run nearest to log zeta_O = 0 equals the difference of log pattern_X (the depletion cancels
    at fixed F*, so CLOUDY's depletion table is not re-implemented).

    Pass: (1)-(4), (4) within 0.001 dex plus one print step; PARTIAL when (2)-(4) cannot run.
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
    """The line-only column (col9) contains the listed line fluxes.

    Around each listed line the column is summed cell by cell over +-3 bins (energy of a cell:
    value x cell width in ln lambda) and compared with the listed fluxes whose cell falls in the
    same window. The column also holds unlisted lines, so the upper side is graded only for
    strong lines.

    Pass: raster / listed >= 0.98 for every line, <= 1.02 for lines above 0.1 H-beta.
    """
    runs = ctx.raw_runs()
    if not runs:
        return SKIP, dict(_NO_WORKDIRS)
    half, tolerance = int(T['closure.half_window_bins']), T['closure.max_rel_dev']
    violations, evaluated, n_runs = [], 0, 0
    for job, workdir in runs:
        con, lines = ctx.raw_file(job, workdir, '.con'), ctx.raw_file(job, workdir, '.lines')
        if not (con.is_file() and lines.is_file()):
            continue
        columns = parse_cloudy_con(str(con))                  # (lambda, col2, col3, col4, col9)
        wave, col9 = np.asarray(columns[0], float), np.asarray(columns[4], float)
        listed = parse_cloudy_linelist(str(lines))
        names, fluxes = [_decode(n) for n in listed['names']], np.asarray(listed['fluxes'], float)
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
            if b < half or b >= wave.size - half or fluxes[k] <= 0:
                continue
            window = np.abs(bins - b) <= half
            ratio = float(energy[b - half:b + half + 1].sum()) / float(fluxes[window & (fluxes > 0)].sum())
            evaluated += 1
            if ratio < 1.0 - tolerance or (ratio > 1.0 + tolerance
                                           and fluxes[k] >= T['closure.strong_line_min_hbeta'] * hbeta):
                violations.append({'job_id': job, 'line': names[k], 'ratio': _round(ratio)})
    if n_runs == 0:
        return SKIP, {'reason': 'no kept run with a continuum and a line-list output'}
    detail = {'runs': n_runs, 'lines_evaluated': evaluated, 'violations': violations[:20],
              'summary': f'{len(violations)} closure violations over {evaluated} lines in {n_runs} runs'}
    return (PASS if not violations else FAIL), detail

@check('rw-grain_abundance_file', blocking=True)
def check_grain_abundance_file(ctx):
    """The grain-abundance output holds one block of zones, and the parser integrates it.

    One '#Depth ... total' header, rows of constant width, >= 2 zones, depth strictly increasing
    (a second block, i.e. a lost 'last', would double the dust mass); a positive integral.

    Pass: every file.
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
        if info['zones'] < 2 or len(info['column_counts']) > 1 or info['unreadable_rows']:
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
    """CLOUDY was illuminated by the SED file its deck names, read as intended.

    The file named by 'table SED' (under SED/ of the working directory, else in the working
    directory) is compared with the incident column through two normalisation-free numbers:
    ionising photons per unit energy and the energy fraction in the Habing band.

    Pass: both within 5% in every run.
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
        stellar = wave < CONST.Stellar_Max_A  # above it lives the CMB of the run, not in the file
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
    """Every line requested by the list file of the deck comes out of CLOUDY, with a valid flux.

    Lines of the main output mentioning a line not found are reported for inspection.

    Pass: requested = returned and fluxes finite, >= 0, in every run.
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
    """Mean execution time of the kept runs times the planned runs (charter or --n-expected).

    Pass: projected cost within 5400 core hours.
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
    """Energy balance recomputed from the continuum outputs, and the attribute the parser stored.

    E = integral of nu*F_nu over d ln lambda, referred (as in the parser) to the stellar incident
    energy, the incident column below Stellar_Max_A (above it lives the CMB of the run).

    Pass: median imbalance < 1%, every stored attribute within 2e-3 of the recomputed one, no
    output without incident energy.
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
        e2_stellar = _integrate_log(col2, wave, wave < CONST.Stellar_Max_A)
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
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

def _load_yaml(path):
    import yaml
    with open(path) as handle:
        return yaml.safe_load(handle)

def _load_json(path):
    return json.loads(pathlib.Path(path).read_text())

def _expected_runs(charter):
    try:
        return int(charter['lhs']['n_samples']) * int(charter['nodes']['n_tau_ssp']) * int(charter['nodes']['n_zstar'])
    except (KeyError, TypeError, ValueError):
        return None

def _default_config_path(sector):
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
    for candidate in (
        f'data/grids/{sector}/lhs_{sector}.npy',
        f'data/grids/{sector}/lhs.npy',
        f'data/{sector}/lhs_{sector}.npy',
    ):
        if pathlib.Path(candidate).is_file():
            return candidate
    return None

def _default_ssp_meta_path(sector):
    for candidate in (
        'data/cloudy_seds/ssp_metadata.json',
        f'data/grids/{sector}/ssp_metadata.json',
        f'data/{sector}/ssp_metadata.json',
    ):
        if pathlib.Path(candidate).is_file():
            return candidate
    return None

def _default_raw_workdirs(sector):
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
        print(f'{tag} FAIL - data not readable: {exc}', file=sys.stderr)
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
    print(f'{tag} {report["overall"]} - ' + ', '.join(f'{v} {k}' for k, v in counts.items() if v)
          + (f' - blocking failures: {failed}' if failed else ''))
    for message in input_errors:
        print(f'{tag} input not loaded: {message}', file=sys.stderr)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
