# Author: Enrico Veraldi
# training the HII emulator

import argparse
import json
import pathlib
import os

import gpytorch
import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
import copy

import galapy.internal.constants as CONST
import galapy.internal.cloudy as CLOUDY
import galapy.internal.emulators as EMUL

from galapy.spectroscopy.svgp import BatchedSVGPS, lines_to_ratio
from galapy.spectroscopy.hii_emulator import line_log_scale
from galapy.spectroscopy.svgp import select_outputs
from galapy.spectroscopy.hii_emulator import augment_inputs
from galapy.spectroscopy.svgp import target_transform
from galapy.spectroscopy.hii_emulator import incident_tables, _node_weights
from galapy.spectroscopy.utils.validate_emulator import (holdout_split, encode_targets,
                            holdout_scores, write_holdout, log_targets,
                            build_frozen_set, calibration_scale, coverage_scores,
                            per_output_scores)

trap = getattr(np, 'trapezoid', None) or np.trapz


def sed_hardness(wave, sed):
    """
    Q(He0)/Q(H) and Q(He+)/Q(H) computations for an incident spectral energy distribution (SED).

    This function calculates the photon rate ratios Q(He0)/Q(H) and Q(He+)/Q(H) from an incident
    spectral energy distribution (SED) given as nu L_nu across wavelengths defined by `wave`
    (in Angstrom). Photon rates Q(> E) are computed by integrating nu L_nu * lambda over
    logarithmic wavelength ranges up to the defined energy edges. The computation accounts for
    the edges at CONST.LyLimit for H, EMUL.SED_EDGES_A for He0 and He+, with values floored
    at EMUL.SED_HARDNESS_FLOOR.

    The integrals assume `h c` cancels in the ratio, simplifying the computation. The resulting
    ratios are compared to a floor value for stability, and the function outputs these ratios
    as a tuple.

    Parameters:
    wave : list[float] or numpy.ndarray
        Array of wavelength values in Angstrom.
    sed : list[float] or numpy.ndarray
        Array of spectral energy distribution values corresponding to the given wavelengths.

    Returns:
    tuple[float, float]
        A tuple containing the hardness ratios:
        - Q(He0)/Q(H)
        - Q(He+)/Q(H)
    """
    w = np.asarray(wave, dtype=np.float64)
    o = np.argsort(w)
    w, n = w[o], (np.asarray(sed, dtype=np.float64) * np.asarray(wave, dtype=np.float64))[o]
    def q(edge):
        m = w <= edge
        return float(trap(n[m], np.log(w[m]))) if m.sum() > 1 else 0.0
    qh = q(CONST.LyLimit)
    return tuple(max(q(e) / qh, EMUL.SED_HARDNESS_FLOOR) if qh > 0 else EMUL.SED_HARDNESS_FLOOR
                 for e in EMUL.SED_EDGES_A)

def load_corpus(path, keys=None):
    """
    Loads and processes a corpus file for use in the hii-train pipeline. This function validates the
    contents of the file, verifies conformity to expected formats and units, extracts metadata, and
    structures data into formats suitable for further analysis. It handles grid points, training families,
    pumping removal consistency, and Q_H grouping per grid node.

    ### Parameters
    path : str
        The file path of the corpus file to be loaded.

    ### Returns
    tuple
        A tuple containing:
          - A numpy 2D array of independent variables (X).
          - A dictionary of dependent variables split by training families (Y).
          - A dictionary with metadata:
              - 'lines_names': List of line names.
              - 'lines_vacuum': Vacuum wavelengths of emergent lines.
              - 'wave_grid': Wavelength grid used for continuum data.
              - 'units': A dictionary mapping variable names to their respective units.
              - 'n_used': Number of models used after validation.
              - 'pumping_removed': Boolean indicating if pumping removal was consistent.
              - 'qh_unit_nodes': Tabulated Q_H values grouped by grid nodes.

    ### Raises
    ValueError
        If no grid points are found, units mismatch, grid points are missing the 'pumping_removed'
        attribute, or pumping removal consistency is violated.
    """
    # read mode
    with h5py.File(path, 'r') as f:
        # keys given: read only those grid points (the holdout, for validate_emulator)
        if keys is None:
            keys = sorted(k for k in f.keys() if k.startswith('grid_point_'))
        if not keys:
            raise ValueError(f"[hii-train] No grid points found in the corpus file '{path}', postprocessing missing")

        # check units
        units = {k: (v.decode() if isinstance(v, bytes) else v) for k,v in f['units'].attrs.items()} if 'units' in f else {}
        expected = {k: u for k, (u, _d) in CLOUDY.Points_Units_HII.items() if u is not None}
        if units != expected:
            raise ValueError(f"[hii-train] Units mismatch in the corpus file '{path}', expected {expected} but found {units}")

        # global metadata
        line_names = [n.decode() if isinstance(n, bytes) else n for n in f['line_names'][:]]
        lines_vacuum = f['lines_emergent/wavelengths_rest_vacuum'][:] # need per mode of galapy spec with flux lines given
        wave_grid = f['continuum/wave_grid'][:]
        wave_inc = f['incident/wave_grid'][:] if 'incident/wave_grid' in f else None # wavelengths of the incident SED
        hard = {}
        sed = {} #i ncident SED of every node for the energy fractions
        # 7 families of the training
        src = {'lines': 'lines_emergent/fluxes',
               'cont': 'continuum/nebular_emission_per_Msun',
               'grain': 'continuum/grain_diag_per_Msun',
               'trans': 'continuum/transmission',
               'dust': 'dust_mass_per_Msun',
               'fuv_t': 'T_fuv_hii',
               'fuv_n': 'N_fuv_hii'}

        #cycle on models
        #   - X input (free parameters)
        #   - Y list per family
        #   - qh value of Q_H grouped per node
        #   - pumped, flags of pump lines
        X, Y, qh, pumped, used = [], {fam: [] for fam in src}, {}, set(), []
        # the wavelength families go into preallocated arrays of the dtype stored in the corpus
        g0 = f[keys[0]]
        Y.update({fam: np.empty((len(keys), g0[src[fam]].shape[0]), g0[src[fam]].dtype) for fam in EMUL.WAVE_FAMS})
        for k in keys:
            g = f[k]
            # excluding not converged models
            if g.attrs.get('not_converged', False):
                continue
            # be sure lines pumping removed
            if 'pumping_removed' not in g.attrs:
                raise ValueError(f"[hii-train] {path}: {k} without pumping removed")
            pumped.add(bool(g.attrs['pumping_removed']))
            used.append(k)   # key of each row of X: links the row to its cloud (holdout_split)

            X.append([np.log10(g.attrs[a]) if a in EMUL.LOG10_INPUTS_HII else g.attrs[a] for a in EMUL.INPUT_KEYS_HII])
            for fam, dset in src.items():
                v = g[dset]
                # row len(used) - 1 of the preallocated array
                if fam in EMUL.WAVE_FAMS:
                    Y[fam][len(used) - 1] = v[:]
                    continue
                Y[fam].append(v[()] if v.shape == () else v[:])

            # Q_H node unit
            node = (float(g.attrs['tau_SSP']), float(g.attrs['Z_star']))
            qh.setdefault(node, []).append(float(g['s_k'][()]) * 10.0 ** (g.attrs['logU'] + g.attrs['lognH_HII']) * CONST.clight['cm/s'])
            # hardness of the node, read once
            if wave_inc is not None and node not in hard and 'incident/sed_per_Msun' in g:
                hard[node] = sed_hardness(wave_inc, g['incident/sed_per_Msun'][:])
            if wave_inc is not None and node not in sed and 'incident/sed_per_Msun' in g:
                sed[node] = g['incident/sed_per_Msun'][:]

    # pumping choerence
    if len(pumped) > 1:
        raise ValueError(f"[hii-train] {path}: pumping removed for some models and not for others")

    # conversion on array numpy format
    # the wavelength families keep the dtype of the corpus
    Y = {fam: (v[:len(used)] if fam in EMUL.WAVE_FAMS else np.asarray(v, np.float64)) for fam, v in Y.items()}
    for fam in ('dust', 'fuv_t', 'fuv_n'):
        Y[fam] = Y[fam][:, None]
    # hardness tables next to Q_H
    qh_nodes = qh_table(qh, hard, sed, wave_inc)
    X = np.asarray(X, np.float64)
    if 'sed_inc' in qh_nodes:
        Y['energy'] = energy_targets(qh_nodes, wave_grid, X, Y['trans'], Y['cont'])

    return (X, Y,
            {'lines_names': line_names,
             'lines_vacuum': lines_vacuum,
             'wave_grid': wave_grid,
             'units': units,
             'n_used': len(X),
             'pumping_removed': pumped == {True},
             'qh_unit_nodes': qh_nodes,
             'keys': used})

def energy_targets(qh_nodes, wave_grid, X, T, C):
    """
    Computes targets of the 'energy' family, returning an array of shape (N, 2), where each row consists of
    two values: the odds of the absorbed fraction `a / (1 - a)` and `q = c / a`. The absorbed fraction `a` is
    calculated as `1 - int(T * L_inc) / int(L_inc)`, floored at `EMUL.ENERGY_A_FLOOR`. The value `c` is
    determined as `int(L_cont) / int(L_inc)`. Integrals are computed in `dln(lambda)` on the incident grid
    with `T` and `L_cont` interpolated linearly in `ln(lambda)`. This computation ensures that the constrained
    emulator closes the Energy check during `validate_emulator` as defined in `hii_emulator.incident_tables`.

    Parameters:
        qh_nodes (numpy.ndarray): Array of quasar halo nodes.
        wave_grid (numpy.ndarray): Wavelength grid for the incident spectrum.
        X (numpy.ndarray): Input array used for interpolation.
        T (numpy.ndarray): Incident luminosity at specific wavelengths.
        C (numpy.ndarray): Continuous source luminosity at specific wavelengths.

    Returns:
        numpy.ndarray: A 2D array of shape (N, 2), where each row contains:
            - The odds of the absorbed fraction `a / (1 - a)`.
            - The value of `q = c / a`.

    Raises:
        None
    """
    tabs = incident_tables(qh_nodes, wave_grid)
    it, lo, hi, w = _node_weights(qh_nodes, list(EMUL.INPUT_KEYS_HII), X)
    J, out = tabs['J'], []
    for i in range(0, len(X), EMUL.PCA_CHUNK):
        s = slice(i, i + EMUL.PCA_CHUNK)
        b = (1.0 - w[s])[:, None] * tabs['b'][it[s], lo[s]] + w[s][:, None] * tabs['b'][it[s], hi[s]]
        E = (1.0 - w[s]) * tabs['E'][it[s], lo[s]] + w[s] * tabs['E'][it[s], hi[s]]
        a = np.clip(1.0 - np.einsum('nj,nj->n', np.asarray(T[s], np.float64)[:, J], b) / E,
                    EMUL.ENERGY_A_FLOOR, 1.0 - 1e-12)
        c = np.asarray(C[s], np.float64)[:, J] @ tabs['d'] / E
        out.append(np.column_stack([a / (1.0 - a), c / a]))
    return np.concatenate(out)

def qh_table(qh, hard=None, sed=None, wave_inc=None):
    """
    Constructs a 2D table of Q_H values as a function of tau and Z_star based on the input
    dictionary and verifies consistency of Q_H values within each node.

    Parameters:
    qh : dict
        A dictionary where keys are tuples of (tau_SSP, Z_star) and values are lists of Q_H
        values from different models. Each entry corresponds to a specific stellar node.

    Returns:
    dict
        A dictionary containing:
        - 'log_tau': Logarithmic values of tau as a numpy array computed from distinct tau_SSP values.
        - 'Z': Distinct Z_star values as a numpy array.
        - 'Qh': A 2D numpy array where rows correspond to tau, columns to Z_star, and elements
          contain the mean Q_H value for each (tau, Z_star) pair.

    Raises:
    ValueError
        If Q_H values for a particular (tau_SSP, Z_star) node are not consistent (i.e., not
        approximately constant across models).
    """
    taus = sorted({t for t, _ in qh})
    Zs = sorted({Z for _, Z in qh})
    Q = np.full((len(taus), len(Zs)), np.nan)
    for (t, z), v in qh.items():
        v = np.asarray(v)
        if np.ptp(v) > 1e-5 * v.mean():
            raise ValueError(f"[hii-train] node (tau={t:g}, Z={z:g}): Q_H values are not consistent (not constant between points)")
        Q[taus.index(t), Zs.index(z)] = v.mean()
    out = {'log_tau': np.log10(taus), 'Z': np.asarray(Zs), 'Qh': Q}
    if hard:
        H = np.full((2,) + Q.shape, np.nan)
        for (t, z), v in hard.items():
            H[:, taus.index(t), Zs.index(z)] = v
        out['hard_he0'], out['hard_he1'] = H
    if sed:                                       # nodes absent from the keys read stay NaN
        S = np.full(Q.shape + (len(wave_inc),), np.nan)
        for (t, z), v in sed.items():
            S[taus.index(t), Zs.index(z)] = v
        out['sed_inc'], out['wave_inc'] = S, np.asarray(wave_inc, dtype=np.float64)
    return out

def parse_rel_floor(items, line_targets='ratio'):
    """
    Relative floor of each vector family from the --rel-floor items.

    Parameters:
        items (list of str): 'FAM=VALUE' sets one family, a bare 'VALUE' sets all the vector families;
            later items override earlier ones. Empty -> the defaults EMUL.REL_FLOOR.
        line_targets (str): 'ratio' or 'absolute'. With absolute line targets there is no ratio_floor, so the
            default of the lines is EMUL.REL_FLOOR_ABS_LINES instead of 0.

    Returns:
        dict: {family: rel_floor} for every family in EMUL.VECTOR_FAMS.
    """
    rel = dict(EMUL.REL_FLOOR)
    if line_targets == 'absolute':
        rel['lines'] = EMUL.REL_FLOOR_ABS_LINES
    return parse_family_values(items, rel, EMUL.VECTOR_FAMS, '--rel-floor')


def parse_rel_floor_lyc(items):
    """
    Ionizing floor (lambda < CONST.LyLimit) of each wavelength-grid family from the --rel-floor-lyc items,
    same syntax as --rel-floor. 0 means no ionizing floor (rel_floor everywhere).

    Returns:
        dict: {family: floor} for every family in EMUL.WAVE_FAMS (0.0 = none).
    """
    lyc = {f: EMUL.REL_FLOOR_LYC.get(f, 0.0) for f in EMUL.WAVE_FAMS}
    return parse_family_values(items, lyc, EMUL.WAVE_FAMS, '--rel-floor-lyc')


def wavelength_floor(rel, lyc, wave):
    """
    Relative floor per column of a wavelength-grid family: lyc below CONST.LyLimit, rel above.
    Returns the scalar rel when there is no ionizing floor (lyc = 0), so old behaviour and artifacts are unchanged.
    log_targets accepts both: the array broadcasts on the columns (rel_floor * max of each row).
    """
    if not lyc:
        return rel
    return np.where(np.asarray(wave, dtype=np.float64) < CONST.LyLimit, lyc, rel)


def parse_family_values(items, defaults, families, flag):
    """
    Update `defaults` ({family: value}) with the items of a CLI flag: 'FAM=VALUE' sets one family, a bare
    'VALUE' sets all of `families`; later items override earlier ones. Values must be >= 0.
    """
    rel = dict(defaults)
    for it in items:
        fam, sep, val = it.rpartition('=')
        try:
            v = float(val)
        except ValueError:
            raise ValueError(f"[hii-train] {flag}: '{it}' is not FAM=VALUE or VALUE") from None
        if v < 0:
            raise ValueError(f"[hii-train] {flag}: negative floor in '{it}'")
        if not sep:
            rel.update({f: v for f in families})
        elif fam in families:
            rel[fam] = v
        else:
            raise ValueError(f"[hii-train] {flag}: unknown family '{fam}' (allowed: {families})")
    return rel

def pca_rank(Y, var_target=0.9999, rel_floor=0.0, transform=None,
             offset=0.0, fit_max=EMUL.PCA_FIT_MAX, seed=42):
    """
    Computes the Principal Component Analysis (PCA) transformation for the target data, including
    normalization, random sampling, and determination of the rank required to reach a specified
    variance threshold.

    Args:
        Y (numpy.ndarray): Input data array containing target values.
        var_target (float, optional): Proportion of variance to retain in the PCA representation.
            Defaults to 0.9999.
        rel_floor (float, optional): Relative floor value for normalization to cap the relative
            range of values. Defaults to 0.0.
        transform (callable, optional): Transformation function to be applied to target values
            before computing PCA. Defaults to None.
        offset (float, optional): Offset value added to the transformation of targets. Defaults
            to 0.0.
        fit_max (int, optional): Maximum number of data points to use for fitting the PCA model.
            If None, all input data points are used. Defaults to EMUL.PCA_FIT_MAX.
        seed (int, optional): Random seed for reproducibility when sub-sampling data points for
            PCA fitting. Defaults to 42.

    Returns:
        tuple: A tuple containing the following elements:
            - fit (sklearn.decomposition.PCA): PCA model fitted to the transformed input data.
            - mu (numpy.ndarray): Mean vector used for normalizing the input data.
            - sd (numpy.ndarray): Standard deviation vector used for normalizing the input data.
            - k_var (int): Number of principal components required to achieve the specified
              variance target.
            - len(rows) (int): Effective number of data points used for PCA fitting.
    """
    ytil = lambda Yc: log_targets(target_transform(transform, Yc), offset, rel_floor)
    n, ch = len(Y), EMUL.PCA_CHUNK
    mu = sum(ytil(Y[i:i + ch]).sum(0) for i in range(0, n, ch)) / n
    sd = np.sqrt(sum(((ytil(Y[i:i + ch]) - mu) ** 2).sum(0) for i in range(0, n, ch)) / n)
    sd = np.maximum(sd, EMUL.SD_FLOOR)
    rows = (np.arange(n) if not fit_max or n <= fit_max
            else np.sort(np.random.default_rng(seed).choice(n, int(fit_max), replace=False)))
    z = np.concatenate([(ytil(Y[rows[i:i + ch]]) - mu) / sd for i in range(0, len(rows), ch)])
    fit = PCA().fit(z)    # compute PCA
    del z
    k_var = int(np.searchsorted(np.cumsum(fit.explained_variance_ratio_), var_target) + 1)
    return fit, mu, sd, k_var, len(rows)

def allocate_pca_budget(k_var, cap):
    """
    Divides a maximum number (cap) of PCA components among families in
    proportion to their needs.

    This function distributes a specified capacity of PCA components among
    the provided families proportionally based on their requirements. If the
    total need is less than or equal to the capacity, the exact requirements
    are returned. Otherwise, a proportional allocation is made while ensuring
    that the results are integers and sum up to the given capacity.

    Parameters
    ----------
    k_var : dict
        A dictionary where the keys are family names (str) and the values are
        their respective PCA component needs (int or float).

    cap : int or None
        The maximum capacity of PCA components that can be distributed.
        If None, the exact values from `k_var` will be used.

    Returns
    -------
    dict
        A dictionary where the keys are family names (str) and the values are
        the allocated PCA components (int). The sum of the returned values
        will not exceed the given capacity (`cap`).
    """
    # sum total components of PCA all families, if <=cap ok
    tot = sum(k_var.values())
    if cap is None or tot<=cap:
        return dict(k_var)

    # if not, compute exact to stay in cap. Until rest is zero give to the one necessary
    names = sorted(k_var)
    exact = {n: k_var[n] * cap / tot for n in names}
    out = {n: max(1, int(exact[n])) for n in names}
    rest = cap - sum(out.values())
    for n in sorted(names, key=lambda n: exact[n] - int(exact[n]), reverse=True):
        if rest <= 0:
            break
        out[n] += 1
        rest -= 1
    return out

def preprocess_targets(Y, do_pca, var_target=0.9999, k_granted=None, offset=0.0, rel_floor=0.0,
                       ratio_ref=None, ratio_floor=0.0, log_scale=None, transform=None,
                       fit_max=EMUL.PCA_FIT_MAX, seed=42):
    """
    Processes target data for use in modeling by applying optional PCA transformation, standardization,
    log scaling, offset adjustments, or other specified transformations. This function can scale the data
    to improve model training stability and reduce computational cost by reducing dimensionality.

    Parameters:
        Y (array-like): Input target data of shape (n_samples, n_features). This can be raw target values
            for later preprocessing.
        do_pca (bool): Indicates whether PCA transformation should be applied to the data.
        var_target (float, optional): Variance capture threshold for PCA dimensional reduction. Default is 0.9999.
        k_granted (int, optional): If specified, manually sets the number of PCA components to retain, overriding
            the variance target.
        offset (float, optional): Offset value added to the target data before any transformations. Default is 0.0.
        rel_floor (float, optional): Relative floor applied during log scaling to avoid instability. Default is 0.0.
        ratio_ref (int, optional): If provided, specifies the reference index for converting target values
            into ratios (log-transformed, optional). Default is None.
        ratio_floor (float, optional): Minimum floor value applied to computed ratios. Default is 0.0.
        log_scale (str, optional): Indicates the logarithmic scale used when converting the target values
            into ratios. Required when `ratio_ref` is provided. Default is None.
        transform (callable, optional): Optional transformation function applied to the data before any other
            processing. Default is None.
        fit_max (int, optional): Maximum number of rows used during PCA fitting if data size exceeds this
            limit. Default is EMUL.PCA_FIT_MAX.
        seed (int, optional): Random seed for reproducibility during PCA fitting operations. Default is 42.

    Returns:
        tuple:
            - Preprocessed target data as a NumPy array of shape (n_samples, n_final_features), where the
              number of features may vary depending on PCA configuration or other transformations.
            - Dictionary containing the preprocessing metadata, including transformations applied, PCA configuration,
              z-scoring statistics (mean, standard deviation), and additional related parameters.

    Raises:
        ValueError: If `ratio_ref` is specified but `log_scale` is not provided.
    """
    if not do_pca:
        Y = np.asarray(Y, dtype=np.float64) + offset # [N_models times n_outputs] put in array and add offset
    if not do_pca:
        ytil = log_targets(Y, rel_floor=rel_floor)
        ratio = None
        if ratio_ref is not None:
            if log_scale is None:
                raise ValueError("[hii-train] ratio form of the lines needs log_scale")
            ytil = lines_to_ratio(ytil, ratio_ref, log_scale, ratio_floor)
            ratio = {'ref': int(ratio_ref), 'floor': float(ratio_floor), 'scale': 'log10 Q_H (1 - f_esc)'}
        mu, sd = ytil.mean(0), np.maximum(ytil.std(0), EMUL.SD_FLOOR)
        return (ytil - mu) / sd, {'mu': mu, 'sd': sd, 'offset': offset, 'rel_floor': rel_floor,
                                  'ratio': ratio, 'pca': None}

    fit, mu, sd, k_var, n_fit = pca_rank(Y, var_target, rel_floor, transform, offset, fit_max, seed)
    k = int(k_granted) if k_granted else k_var
    # components PCA ordered for decreasing variance, so we can truncate directly
    comp = fit.components_[:k]
    prep = {
        'mu': mu, 'sd': sd, 'offset': offset, 'rel_floor': rel_floor, 'transform': transform, 'pca': {
            'components': comp, 'mean': fit.mean_,
            'n_components': k, 'n_components_var_target': k_var,
            'truncated_by_budget': k < k_var,
            'explained': float(fit.explained_variance_ratio_[:k].sum()),
            'n_fit': int(n_fit)
        }
    }
    scores = np.concatenate([encode_targets(prep, Y[i:i + EMUL.PCA_CHUNK]) for i in range(0, len(Y), EMUL.PCA_CHUNK)])
    # whitening: every PC score to unit variance, the scale the GP starts from
    # Raw scores have variance up to thousands and the hyperparameters cannot get there.
    score_sd = np.maximum(scores.std(0), 1e-12)
    prep['pca']['score_sd'] = score_sd
    return scores / score_sd, prep


def drop_pcs(prep, keep):
    """
    Adjusts a PCA preprocessing object by keeping only the specified principal components (PCs).

    This function modifies the PCA preprocessing object to include only the selected PCs, recalculating
    associated metadata such as explained variance, component scores, and the number of components.

    Args:
        prep (dict): The preprocessing dictionary containing the PCA-related data.
                     Expected to have a 'pca' entry, which includes:
                     - 'score_sd': Standard deviations of scores.
                     - 'explained': Total explained variance.
                     - 'components': Components of the PCA.
        keep (list of int): Indices of the PCs to retain.

    Returns:
        dict: A new dictionary with updated PCA data, including:
              - 'explained': Recomputed explained variance fraction.
              - 'dropped_pcs': List of indices of PCs that were removed.
              - 'n_components_trained': Total number of components trained initially.
              - 'components': Retained components after filtering.
              - 'score_sd': Retained scores' standard deviations.
              - 'n_components': Number of components retained.
    """
    pca = dict(prep['pca'])
    s2 = np.asarray(pca['score_sd'], dtype=np.float64) ** 2
    pca['explained'] = float(pca['explained'] * s2[keep].sum() / s2.sum())
    pca['dropped_pcs'] = [int(k) for k in np.setdiff1d(np.arange(len(s2)), keep)]
    pca['n_components_trained'] = int(len(s2))
    pca['components'] = np.asarray(pca['components'])[keep]
    pca['score_sd'] = np.asarray(pca['score_sd'])[keep]
    pca['n_components'] = int(len(keep))
    return dict(prep, pca=pca)


def pca_residual_var(prep, Y, chunk=4096):
    """
    Computes the unexplained variance per column in the z-scored log space of the data after projection onto
    retained principal components (PCs). This includes PCs removed by criteria such as truncation at a target
    variance or explicit exclusion of specific PCs, and accounts for additional transformations during training.

    Summary:
    Given a PCA-preprocessed dataset and its associated PCA model components, this function calculates the
    variance remaining in the data projections that is not captured by the retained PCs. The computation is
    split into chunks to manage memory for large datasets, particularly when handling wavelength families
    with a large number of columns.

    Args:
        prep (dict): A dictionary containing PCA preprocessing data, including the PCA model, means,
            scaling factors, and transformation parameters.
        Y (numpy.ndarray): The dataset for which residual variance must be computed. It is expected to
            have rows as observations and columns as variables.
        chunk (int): The chunk size for dataset processing. Defaults to 4096. This is necessary for
            memory-efficient processing of large datasets.

    Returns:
        numpy.ndarray: An array containing the unexplained variance for each column in the dataset.
    """
    pca = prep['pca']
    C, mean = np.asarray(pca['components']), np.asarray(pca['mean'])
    mu, sd = np.asarray(prep['mu']), np.asarray(prep['sd'])
    acc = np.zeros(C.shape[1])
    for i in range(0, len(Y), chunk):     # chunks: a wavelength family is ~1e4 columns
        # same transform as the training targets
        Yc = target_transform(prep.get('transform'), Y[i:i + chunk])
        z = (log_targets(Yc, prep.get('offset', 0.0), prep.get('rel_floor', 0.0)) - mu) / sd - mean
        acc += np.sum((z - (z @ C.T) @ C) ** 2, axis=0)
    return acc / len(Y)

# peak memory of the training, estimated before reading the corpus, against the memory of the job
################################# FUNCTIONS FOR SLURM PURPOSES ######################################
def estimate_train_memory(n_models, n_wave, n_train, n_test, fit_max,
                          n_fams=len(EMUL.WAVE_FAMS)):
    """
    Estimate the peak memory usage for the galapy-train-hii process.

    Summary:
    This function calculates the estimated peak memory in bytes required for the
    galapy-train-hii process. It considers the memory usage of wavelength families
    in float32, the training rows of one family with PCA fitting, and the gate set
    of the holdout. Additionally, a fixed overhead is included to reflect system-level
    requirements. Memory coefficients are based on measurements from a 50k dataset corpus.

    Arguments:
    n_models: int
        The number of models to be processed.
    n_wave: int
        The number of wavelengths per model.
    n_train: int
        The number of training rows per wavelength family.
    n_test: int
        The number of testing rows in the holdout set.
    fit_max: int or None
        Maximum number of rows considered for PCA fitting.
    n_fams: int, optional
        The number of wavelength families. Defaults to the length of EMUL.WAVE_FAMS.

    Returns:
    int
        The estimated peak memory usage in bytes.
    """
    raw = n_models * n_wave * n_fams * 4
    m = n_train if not fit_max else min(int(fit_max), n_train)
    fam = n_train * n_wave * 4 + EMUL.MEM_PCA_COPIES * m * n_wave * 8
    gate = n_test * n_wave * n_fams * 8
    return int(raw + max(fam, gate) + EMUL.MEM_OVERHEAD)


def available_memory():
    """
    Determines the available memory the job can use, considering multiple sources and constraints.

    This function calculates the minimum available memory among values retrieved from:
    - MemAvailable in /proc/meminfo
    - Memory limits defined in cgroup (v1 or v2)
    - SLURM_MEM_PER_NODE environment variable

    If none of these sources can be successfully read, it returns `None`.

    Returns
    -------
    int or None
        The available memory in bytes, or `None` if no memory limits can be determined.
    """
    vals = []
    try:
        with open('/proc/meminfo') as fh:
            vals += [int(l.split()[1]) * 1024 for l in fh if l.startswith('MemAvailable:')]
    except OSError:
        pass
    try:
        with open('/proc/self/cgroup') as fh:
            for l in fh:
                hid, ctrl, path = l.rstrip('\n').split(':', 2)
                f = (f'/sys/fs/cgroup{path}/memory.max' if hid == '0' else
                     f'/sys/fs/cgroup/memory{path}/memory.limit_in_bytes' if 'memory' in ctrl.split(',') else None)
                if f and pathlib.Path(f).is_file():
                    v = pathlib.Path(f).read_text().strip()
                    if v.isdigit() and int(v) < 1 << 60:
                        vals.append(int(v))
    except (OSError, ValueError):
        pass
    if os.environ.get('SLURM_MEM_PER_NODE', '').isdigit():
        vals.append(int(os.environ['SLURM_MEM_PER_NODE']) * 2 ** 20)
    return min(vals) if vals else None

def init_inducing(Xz, n_out, M=512, seed=42):
    """
    Decides the initial position of the M inducing points of the SVGP for a family.

    The initial positions are determined by using the centers of a k-means clustering performed
    on the input space. These positions are then replicated across the grid for each of the
    n_out Gaussian Processes (GPs).

    Parameters:
        Xz (torch.tensor): The input data used for k-means clustering. It should have
            shape [N x D] where N is the number of data points and D is the dimensionality.
        n_out (int): The number of Gaussian Processes (GPs) in the family. For each GP, a
            grid of inducing points will be generated.
        M (int, optional): The number of inducing points to be used. Defaults to 512.
        seed (int, optional): The random seed used for initializing k-means. Defaults to 42.

    Returns:
        torch.tensor: A tensor of shape [n_out x M x D] containing the positions of the
            inducing points. Each Gaussian Process in the family will have identical inducing
            point positions.
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=M, n_init=4, random_state=seed).fit(Xz)  # train, initializ 4 different, take one with minimum inertia
    Z = torch.as_tensor(km.cluster_centers_, dtype=torch.float32)   # tensor [M x D]
    return Z.unsqueeze(0).repeat(n_out, 1, 1)                       # [n_out x M x D]


def train_family(Xz, Yz, M, epochs, batch, lr, device, seed=42, test=None, eval_every=10,
                 progress=None, cost=1.0, select=None):
    """
    Trains all Gaussian Processes (GPs) of a family together using a batched model composed of independent Stochastic Variational
    Gaussian Processes (SVGP). This function facilitates the simultaneous training of multiple output dimensions by leveraging
    a batched approach.

    Parameters:
        Xz: array-like
            Input data used for training the model. Must be convertible to a PyTorch tensor of dtype torch.float32.
        Yz: array-like
            Target data corresponding to the input features Xz. Must have the same number of rows as Xz and be convertible
            to a PyTorch tensor of dtype torch.float32.
        M: int
            The number of inducing points to initialize in the SVGP model.
        epochs: int
            The number of iterations through the full training dataset for model optimization.
        batch: int
            The batch size used for sampling mini-batches during training.
        lr: float
            The initial learning rate of the Adam optimizer (cosine schedule down to lr * EMUL.LR_MIN_FACTOR).
        device: str
            The computation device on which the training is conducted (e.g., "cpu" or "cuda").
        seed: int, optional
            The seed value for the random number generator to ensure reproducible results. Default is 42.
        test: (Xz_test, Yz_test), optional
            The holdout in the training space, scored every eval_every epochs (TE lines).
        progress: tqdm, optional
            Progress bar of the whole training, advanced by `cost` at each epoch.
        select: (Xz_sel, Yz_sel), optional
            Calibration clouds: every eval_every epochs the rmse_z on them is computed and the model of the best
            evaluation is returned (not the last epoch). Never the holdout, which stays an independent check.

    Returns:
        tuple:
            A tuple containing:
                - model: The trained batched SVGP model.
                - likelihood: The trained GaussianLikelihood corresponding to the batched SVGP model.
                - hist: A list containing the history of ELBO losses recorded per epoch.
                - epoch_times: A list of times taken to complete each epoch during training.
                - test_hist: holdout scores every eval_every epochs.
                - best_epoch: epoch of the returned model (chosen on `select`; the last epoch without it).

    """
    torch.manual_seed(seed)
    n_out = Yz.shape[1]
    Xt = torch.as_tensor(Xz, dtype=torch.float32)
    Yt = torch.as_tensor(Yz, dtype=torch.float32)

    model = BatchedSVGPS(init_inducing(Xz, n_out, M, seed), n_out).to(device) # create n_out independent SVGP, put togheder in a batched model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_out])).to(device) # likelihood for n_out independent SVGP
    # mode train for likelihood and model
    model.train()
    likelihood.train()

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Xt.size(0)) # what maximize (ELBO)
    # Adam instead of AdamW. AdamW has weight_decay = 0.01 by default, which shrinks every parameter towards 0
    # (variational mean and Cholesky, inducing points, raw lengthscale/outputscale/noise): with lr 5e-3 for 8000 steps a
    # factor ~exp(-0.4) where the gradient does not resist. A GP has no reason for it.
    opt = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=lr) # optimizer
    # cosine decay of the learning rate over the epochs: large steps first, fine convergence at the end
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * EMUL.LR_MIN_FACTOR)
    best = None   # (rmse_z on select, epoch, model state, likelihood state)

    ds = torch.utils.data.TensorDataset(Xt, Yt)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True) # mini batch of batch points

    hist, test_hist = [], [] #history
    # rmse_z of a constant predictor (the training mean, i.e. z = 0)
    ref = float(np.sqrt(np.mean(np.asarray(test[1]) ** 2))) if test is not None else None
    import time
    epoch_times = []
    # training loop
    for epoch in range(epochs):
        _t0 = time.perf_counter()
        tot = 0.0
        for xb, yb in loader: # for each mini batch
            xb, yb = xb.to(device), yb.to(device) # batch on device
            opt.zero_grad() # zero grad from previous step
            out = model(xb)
            loss = -mll(out, yb.T).sum() # ELBO, became a loss to minimize
            loss.backward()
            opt.step()
            tot += loss.item()
        hist.append(tot / max(len(loader), 1))
        epoch_times.append(time.perf_counter() - _t0)
        # learning rate schedule, and best checkpoint on the calibration clouds
        sched.step()
        evaluate = (epoch + 1) % eval_every == 0 or epoch == epochs - 1
        if select is not None and evaluate:
            sc = holdout_scores(model, select[0], select[1], device, likelihood=likelihood)['rmse_z']
            model.train()
            if best is None or sc < best[0]:
                best = (sc, epoch + 1, copy.deepcopy(model.state_dict()), copy.deepcopy(likelihood.state_dict()))
        # holdout scores (cov68) with the likelihood noise in the variance
        if test is not None and evaluate:
            test_hist.append({'epoch': epoch+1, **holdout_scores(model, test[0], test[1], device, likelihood=likelihood)})
            model.train()
            tqdm.write(f"[hii-train] TE epoch {epoch + 1:4d}/{epochs}  rmse_z {test_hist[-1]['rmse_z']:.4f}"
                       f" (constant predictor {ref:.4f}, good << it)  explained {100 * test_hist[-1]['explained']:.1f}%"
                       f"  cov68 {test_hist[-1]['cov68']:.3f} (calibrated 0.68)")
        if epoch % 20 == 0 or epoch == epochs - 1:
            tqdm.write(f"[hii-train] TR epoch {epoch + 1:4d}/{epochs}  ELBO loss {hist[-1]: .4f} (must decrease)")
        if progress is not None:
            progress.update(cost)

    # back to the best checkpoint
    best_epoch = epochs
    if best is not None:
        model.load_state_dict(best[2])
        likelihood.load_state_dict(best[3])
        best_epoch = best[1]
        tqdm.write(f"[hii-train] best checkpoint on the calibration clouds: epoch {best_epoch} (rmse_z {best[0]:.4f})")
    return model, likelihood, hist, epoch_times, test_hist, best_epoch

def report_per_pc(fam, model, Xz_te, Yz_te, score_sd, device='cpu'):
    """
    Prints, at the end of the training of a PCA family, the holdout rmse_z and explained fraction of
    every PC, and the explained fraction weighted by the variance each PC carries in the spectrum.

    The scores are whitened, so the TE lines average all PCs with equal weight: the last PCs, close
    to noise and badly predicted, weigh as much as PC 1. Undoing the whitening, PC k carries a
    variance score_sd_k^2 of the (normalized log) spectrum, and since the PCA components are
    orthonormal the squared error of the reconstructed spectrum is sum_k score_sd_k^2 MSE_k. Hence

        weighted explained = 1 - sum_k score_sd_k^2 MSE_k / sum_k score_sd_k^2 MSE0_k ,

    with MSE0_k the error of the constant predictor. It is the fraction of the spectral variance
    (within the retained PCs) that the emulator reproduces on the holdout.

    Returns:
        Dict[str, Any]: per-PC rmse_z, explained and variance share, and the weighted explained.
    """
    rmse, expl = per_output_scores(model, Xz_te, Yz_te, device)
    w = np.asarray(score_sd, dtype=np.float64) ** 2
    share = w / w.sum()
    mse0 = np.maximum(np.mean(np.asarray(Yz_te) ** 2, axis=0), 1e-30)
    weighted = float(1.0 - np.sum(w * rmse ** 2) / np.sum(w * mse0))
    tqdm.write(f"[hii-train] holdout per PC, {fam}: rmse_z (constant predictor ~1), explained, "
               f"share of the spectral variance carried by the PC")
    for k in range(len(rmse)):
        tqdm.write(f"[hii-train]   PC {k + 1:3d}  rmse_z {rmse[k]:.4f}  explained {100 * expl[k]:6.2f}%"
                   f"  variance share {100 * share[k]:7.3f}%")
    tqdm.write(f"[hii-train] {fam}: explained {100 * float(np.mean(expl)):.1f}% (equal weight per PC), "
               f"{100 * weighted:.2f}% (weighted by the variance of each PC)")
    return {'rmse_z': rmse, 'explained': expl, 'variance_share': share, 'explained_weighted': weighted}


def _delta_m_budget(path, budget_mib):
    """
    read emulator .pt and compute how many inducing points can be added to the model without exceeding the budget

    Parameters:
    path: str
        Path to the .pt file containing the serialized emulator model.
    budget_mib: int
        Maximum allowed memory budget in MiB for the additional inducing points.

    Returns:
    int
        Maximum number of inducing points that can be added to the model without
        exceeding the specified memory budget.
    """
    import torch
    blob = torch.load(path, map_location='cpu', weights_only=False) # load .pt in cpu
    # for each family read number of GPs D, inducing points per GP M, dimensions of inputs
    fams = [f['model']['variational_strategy.inducing_points'].shape for f in blob['families'].values()]

    # estimation of the bytes with increase of inducing points of dM
    def size(dM):
        return sum(D * ((M + dM) ** 2 + (M + dM) * dim + (M + dM)) * 4 for D, M, dim in fams)

    # convert budget in bytes, then give if space avaiable number of dM can be added
    lim = budget_mib * 2**20
    if size(0) > lim:
        return 0
    dM = 0
    while size(dM + 1) <= lim and dM < 4096:
        dM += 1
    return dM

def assert_size_budget(paths, budget_mib):
    """
    Checks if the files in a given list of paths exceed a specified size budget and raises an error if any do.

    This utility function verifies whether the sizes of the files provided in the `paths` list surpass the
    given `budget_mib` limit. If one or more files exceed the limit, it terminates execution with an error
    message listing these files and their respective sizes.

    Args:
        paths (list[str]): A list of file paths to be checked.
        budget_mib (float): The size budget in MiB (mebibytes) that no file should exceed.

    Raises:
        SystemExit: Raised with an error message if one or more files exceed the given size budget.
    """
    import os
    over = [(f, os.path.getsize(f) / 2**20) for f in paths if os.path.getsize(f) > budget_mib * 2**20]
    if over:
        det = ', '.join(f"{f}: {s:.2f} MiB" for f, s in over)
        raise SystemExit(f"[hii-train] Files exceeding the size budget {budget_mib} MiB og : {det}")


###################################### MAIN ########################################

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus', default='data/products/hii_grid.h5',
                    help='merged HII corpus (default: data/products/hii_grid.h5)')
    ap.add_argument('--out', default='data/emulators/hii.pt',
                    help='stem of the emulators: writes hii_{lines,cont,grain,trans,scalars}.pt '
                         '(default: data/emulators/hii.pt)')
    ap.add_argument('--meta', default='data/emulators/hii_meta.json',
                    help='JSON manifest of the training (default: data/emulators/hii_meta.json)')
    ap.add_argument('--inducing', type=int, default=512, help='number of inducing points for the'
                                                              'families using PCA (default: 512)')
    ap.add_argument('--inducing-lines', type=int, default=256, help='number of inducing points'
                                                            'for the lines family and scalars (no PCA) (default: 256)')
    ap.add_argument('--var-target', type=float, default=0.9999, help='target variance explained'
                                                                    ' by PCA components (default: 0.9999)')
    ap.add_argument('--max-pca-components', type=int, default=100, help="maximum number of "
                                                                        "PCA components to use (default=100)")
    ap.add_argument('--rel-floor', nargs='+', default=[], metavar='[FAM=]VALUE', help='floor of the vector '
                    'families before the log10, relative to the peak of each model: zeros go there instead of '
                    '1e-35 (0 disables it). FAM=VALUE sets one family (e.g. trans=1e-5 cont=1e-10), a bare VALUE '
                    'sets all of them (default: ' + ' '.join(f'{k}={v:g}' for k, v in EMUL.REL_FLOOR.items())
                    + f', lines={EMUL.REL_FLOOR_ABS_LINES:g} with --line-targets absolute)')
    ap.add_argument('--rel-floor-lyc', nargs='+', default=[], metavar='[FAM=]VALUE', help='relative floor that '
                    f'replaces --rel-floor below the Lyman limit ({CONST.LyLimit} A) for the families on the wavelength '
                    f'grid {EMUL.WAVE_FAMS}, same syntax as --rel-floor; 0 = none (default: '
                    + ' '.join(f'{f}={EMUL.REL_FLOOR_LYC.get(f, 0.0):g}' for f in EMUL.WAVE_FAMS) + ')')
    ap.add_argument('--line-targets', choices=('ratio', 'absolute'), default='ratio',
                    help='targets of the lines family: ratio = H-beta / [Q_H (1 - f_esc)] and the other '
                         'lines as ratios to H-beta; absolute = log10 of the fluxes (default: ratio)')
    ap.add_argument('--ratio-floor', type=float, default=EMUL.RATIO_FLOOR, help='minimum of L_i / H-beta '
                    'with --line-targets ratio (0 disables it) (default: %(default)g)')
    ap.add_argument('--size-budget-mib', type=int, default=100, help='maximum size budget in MiB'
                                                                     ' (default: 100)')
    ap.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    ap.add_argument('--batch-size', type=int, default=1024, help='batch size (default: 1024)')
    ap.add_argument('--lr', type=float, default=5e-3, help='learning rate (default: 5e-3)')
    ap.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    ap.add_argument('--holdout-frac', type=float, default=0.1, help='fraction of data to hold out for validation (default: 0.1)')
    ap.add_argument('--holdout-seed', type=int, default=42, help='random seed for holdout split (default: 42)')
    ap.add_argument('--eval-every', type=int, default=10, help='evaluate model every n epochs (default: 10)')
    ap.add_argument('--calib-frac', type=float, default=EMUL.CALIB_FRAC, help='fraction of the clouds kept out of '
                    'training (and disjoint from the holdout) to calibrate the predictive sigma, one scale s_k '
                    'per output; 0 disables the calibration (default: %(default)g)')
    ap.add_argument('--calib-families', default=','.join(EMUL.CALIB_FAMS_HII), help='comma-separated families whose '
                    'sigma is calibrated (default: %(default)s)')
    ap.add_argument('--calib-quantile', type=float, default=EMUL.CALIB_QUANTILE, help='quantile q of |r| used for the '
                    'scale, s_k = quantile_q(|r|) / Phi^-1((1+q)/2): 0.95 calibrates the tails (cov95), 0.5 is '
                    'the median estimator (cov68) (default: %(default)g)')
    ap.add_argument('--no-sed-inputs', action='store_true', help='do not append the SED descriptors of the node '
                    f'{EMUL.SED_INPUT_KEYS_HII} to the inputs of the GP (default: appended when the corpus has the '
                    'incident SED)')
    ap.add_argument('--pca-fit-max', type=int, default=EMUL.PCA_FIT_MAX, help='maximum number of training models '
                    'the PCA of cont, grain and trans is fitted on (drawn at random with --seed; 0 = all); mean, sd '
                    f'and scores use every model (default: {EMUL.PCA_FIT_MAX})')
    ap.add_argument('--no-memory-check', action='store_true', help='train even if the estimated peak memory exceeds '
                    f'{EMUL.MEM_SAFETY:.0%}% of the memory available to the job')
    ap.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto', help='device '
                                                                                      'to use (default: auto)')
    args=ap.parse_args()


    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    device = (('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    # memory check before reading the corpus (all the grid points counted, converged or not)
    with h5py.File(args.corpus, 'r') as f:
        n_models = sum(1 for k in f.keys() if k.startswith('grid_point_'))
        n_wave = len(f['continuum/wave_grid'])
    n_test = int(round(n_models * args.holdout_frac))
    n_train = n_models - n_test - int(round(n_models * args.calib_frac))
    need, avail = estimate_train_memory(n_models, n_wave, n_train, n_test, args.pca_fit_max), available_memory()
    print(f"[hii-train] memory: estimated peak {need / 2 ** 30:.1f} GiB ({n_models} models x {n_wave} wavelengths, "
          f"PCA on {min(args.pca_fit_max or n_train, n_train)} models), available "
          + (f"{avail / 2 ** 30:.1f} GiB" if avail else "unknown"))
    if avail and need > EMUL.MEM_SAFETY * avail and not args.no_memory_check:
        raise SystemExit(f"[hii-train] ERROR: estimated peak memory {need / 2 ** 30:.1f} GiB > {EMUL.MEM_SAFETY:.0%} of "
                         f"the {avail / 2 ** 30:.1f} GiB available: lower --pca-fit-max ({args.pca_fit_max}), use a "
                         f"node with more memory, or --no-memory-check")
    X, Y, meta = load_corpus(args.corpus)
    print(f"[hii-train] {meta['n_used']} converged models, {X.shape[1]} input, device={device}")
    # SED descriptors of the node appended to the inputs (X keeps the physical inputs in the first columns)
    sed_inputs = 'hard_he0' in meta['qh_unit_nodes'] and not args.no_sed_inputs
    if sed_inputs:
        X = augment_inputs(meta['qh_unit_nodes'], list(EMUL.INPUT_KEYS_HII), X)
        print(f"[hii-train] SED inputs {EMUL.SED_INPUT_KEYS_HII} appended: {X.shape[1]} inputs of the GP")
    else:
        print("[hii-train] no SED inputs (" + ("--no-sed-inputs" if args.no_sed_inputs else
              "corpus without incident/sed_per_Msun") + ")")

    # train-test
    te = holdout_split(meta['keys'], X, EMUL.AXES_DOMINANT_HII, args.holdout_frac, args.holdout_seed)
    # calibration set for sigma: a further fraction of the clouds,
    # taken from the non-holdout ones with the same stratified split, so that it is unseen by the training
    # (avoid the GP sigma on training points to be too optimistic) and disjoint from the holdout (which then stays an
    # independent check of the calibrated sigma). --calib-frac is a fraction of ALL the clouds.
    cal_fams = tuple(f for f in args.calib_families.split(',') if f) if args.calib_frac > 0 else ()
    cal = np.zeros_like(te)
    if cal_fams:
        frac_rest = args.calib_frac / (1.0 - args.holdout_frac)
        if not 0.0 < frac_rest < 1.0:
            raise ValueError(f"[hii-train] --calib-frac {args.calib_frac} too large with --holdout-frac {args.holdout_frac}")
        cal[~te] = holdout_split(np.asarray(meta['keys'])[~te], X[~te], EMUL.AXES_DOMINANT_HII, frac_rest, args.holdout_seed + 1)
    tr = ~te & ~cal
    n_clouds = lambda m: len({k.split('_')[2] for k in np.asarray(meta['keys'])[m]})
    print(f"[hii-train] holdout: {int(te.sum())} models of {n_clouds(te)} clouds, calibration: {int(cal.sum())} models"
          f" of {n_clouds(cal)} clouds, training: {int(tr.sum())}")

    # lines in ratio form: index of H-beta and the analytic scale log10 Q_H (1 - f_esc) of every model
    ratio_ref, log_scale = None, None
    if args.line_targets == 'ratio':
        names = [' '.join(str(n).split()).casefold() for n in meta['lines_names']]
        cand = [i for i, (n, w) in enumerate(zip(names, np.asarray(meta['lines_vacuum'], float)))
                if n == EMUL.RATIO_REF[0] and abs(w - EMUL.RATIO_REF[1]) < 2.0]
        if len(cand) != 1:
            raise ValueError(f"[hii-train] reference line {EMUL.RATIO_REF} not found once in the line list ({cand})")
        ratio_ref = cand[0]
        log_scale = line_log_scale(meta['qh_unit_nodes'], list(EMUL.INPUT_KEYS_HII), X)

    # input standardization on everything but the holdout: it uses only the input POSITIONS of the calibration
    # set, never its targets, and keeps the holdout completely untouched
    xmu, xsd = X[~te].mean(0), X[~te].std(0)
    xsd = np.where(xsd > 0, xsd, 1.0)
    Xz = (X - xmu) / xsd

    trained, Yz_by_fam, Yz_te_by_fam = {}, {}, {}
    epoch_times = []
    models, extras = {}, {'x_mean': xmu, 'x_std': xsd, 'line_names': meta['lines_names'], 'lines_vacuum':
                         meta['lines_vacuum'], 'wave_grid': meta['wave_grid'], 'input_keys': list(EMUL.INPUT_KEYS_HII),
                         'log10_inputs': list(EMUL.LOG10_INPUTS_HII), 'qh_unit_nodes': meta['qh_unit_nodes'],
                         'pumping_removed': meta['pumping_removed'], 'ard_num_dims': X.shape[1],
                         'n_inducing': args.inducing, 'n_inducing_lines': args.inducing_lines}
    # SED descriptors in the inputs; predictive variance = latent + likelihood noise
    extras['sed_inputs'] = list(EMUL.SED_INPUT_KEYS_HII) if sed_inputs else []
    extras['predictive_noise'] = True

    # families of emulators (name, PCA, offset)
    FAM = tuple(f for f in EMUL.FAMILIES_HII if f[0] in Y)
    cal_fams = tuple(f for f in cal_fams if f != 'energy' or 'energy' in Y)
    unknown = set(cal_fams) - {f for f, _, _ in FAM}
    if unknown:
        raise ValueError(f"[hii-train] --calib-families: unknown families {sorted(unknown)}")

    # computation of required PCA components for each family
    rel = parse_rel_floor(args.rel_floor, args.line_targets)
    rel.update({fam: 0.0 for fam, _, _ in FAM if fam not in EMUL.VECTOR_FAMS})
    lyc = parse_rel_floor_lyc(args.rel_floor_lyc)
    # floor actually applied: a per-wavelength array for the families with an ionizing floor, rel otherwise
    floor = {fam: (wavelength_floor(rel[fam], lyc[fam], meta['wave_grid']) if fam in EMUL.WAVE_FAMS else rel[fam])
             for fam, _, _ in FAM}
    # PCA on at most --pca-fit-max models
    k_var = {fam: pca_rank(Y[fam][tr], args.var_target, floor[fam], fit_max=args.pca_fit_max, seed=args.seed)[3]
             for fam, do_pca, _ in FAM if do_pca}
    k_granted = allocate_pca_budget(k_var, args.max_pca_components)
    # transform of trans (k_var above is on log10 T on purpose)
    trans_transform = ({'kind': 'tau', 'tau_min': EMUL.TRANS_TAU_RANGE[0], 'tau_max': EMUL.TRANS_TAU_RANGE[1]}
                       if EMUL.TRANS_TARGET == 'tau' else None)
    # energy constrain correction if yes
    if 'energy' not in Y:
        print("[hii-train] corpus without incident/sed_per_Msun: no energy family, no energy constraints")
    print(f"[hii-train] constraints: trans as " + ('log10 tau, tau in [%g, %g]' % EMUL.TRANS_TAU_RANGE if trans_transform
          else 'log10 T') + "; energy family (odds of the absorbed fraction, c/a) for the energy balance; grain <= cont")
    print(f"[hii-train] settings: epochs {args.epochs}, batch {args.batch_size}, lr {args.lr:g}, "
          f"M {args.inducing} (PCA families) / {args.inducing_lines} (lines, scalars), seed {args.seed}")
    print(f"[hii-train] preprocessing: var-target {args.var_target:g}, max PCA comp {args.max_pca_components}, "
          f"rel-floor {' '.join(f'{f}={rel[f]:g}' for f in EMUL.VECTOR_FAMS)} (below {CONST.LyLimit} A: "
          f"{' '.join(f'{f}={lyc[f]:g}' for f in EMUL.WAVE_FAMS if lyc[f]) or 'none'}), sd-floor {EMUL.SD_FLOOR:g} dex, PCA scores whitened")
    print(f"[hii-train] line targets: " + (f"ratio (H-beta / [Q_H (1 - f_esc)], other lines / H-beta, "
          f"floor {args.ratio_floor:g})" if ratio_ref is not None else "absolute (log10 fluxes)"))
    print(f"[hii-train] PCA components: requested {k_var}, granted {k_granted}"
          + (" (cut by --max-pca-components)" if k_granted != k_var else ""))

    # one progress bar for the whole training
    n_batches = -(-int(tr.sum()) // args.batch_size)
    cost = {}
    for fam, do_pca, _ in FAM:
        d = k_granted[fam] if do_pca else Y[fam].shape[1]
        M = args.inducing if do_pca else args.inducing_lines
        cost[fam] = float(d * (M ** 3 + args.batch_size * M ** 2) * n_batches)
    pbar = tqdm(total=args.epochs * sum(cost.values()), desc='[hii-train] training',
                bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {elapsed} elapsed, {remaining} left')

    # training with k_granted
    for fam, do_pca, off in FAM:
        lr_kw = ({'ratio_ref': ratio_ref, 'ratio_floor': args.ratio_floor, 'log_scale': log_scale[tr]}
                 if fam == 'lines' and ratio_ref is not None else {})
        tr_kw = {'transform': trans_transform} if fam == 'trans' and trans_transform else {}
        Yz, prep = preprocess_targets(Y[fam][tr], do_pca, args.var_target, k_granted.get(fam) if do_pca else None,
                                      offset=off, rel_floor=0.0 if tr_kw else floor[fam], **lr_kw, **tr_kw,
                                      fit_max=args.pca_fit_max, seed=args.seed)
        Yz_te = encode_targets(prep, Y[fam][te], log_scale=log_scale[te] if lr_kw else None)
        M_fam = args.inducing if do_pca else args.inducing_lines
        tqdm.write(f"[hii-train] Training {fam}: {Yz.shape[1]} output, M={M_fam}"
                + (f" (PCA {prep['pca']['n_components']} PC, "f"{prep['pca']['explained']:.4f} var)" if prep['pca'] else ""))

        # calibration clouds encoded before the training, to choose the best checkpoint on them
        Yz_cal = encode_targets(prep, Y[fam][cal], log_scale=log_scale[cal] if lr_kw else None) if cal.any() else None
        m, lik, hist, ep_t, te_hist, best_epoch = train_family(Xz[tr], Yz, M_fam, args.epochs, args.batch_size, args.lr,
                                                   device, args.seed, test=(Xz[te], Yz_te), eval_every=args.eval_every,
                                                   progress=pbar, cost=cost[fam],
                                                   select=(Xz[cal], Yz_cal) if Yz_cal is not None else None)
        extras[f'{fam}_best_epoch'] = best_epoch
        epoch_times.extend(ep_t)

        if prep['pca'] is not None:
            extras[f'{fam}_holdout_per_pc'] = report_per_pc(fam, m, Xz[te], Yz_te, prep['pca']['score_sd'], device)

        # PCs that the GP does not predict better than their mean (explained <= EMUL.PC_MIN_EXPLAINED on the
        # calibration clouds) are removed: they cost M^2 each in file size and evaluation time
        # and their prediction is almost 0. Their variance, and the truncation at var_target, are residual
        # variance per wavelength that decode adds to the sigma
        if prep['pca'] is not None:
            if Yz_cal is not None:
                _, expl_cal = per_output_scores(m, Xz[cal], Yz_cal, device)
                keep = np.flatnonzero(expl_cal > EMUL.PC_MIN_EXPLAINED)
                if 0 < len(keep) < len(expl_cal):
                    m, lik = select_outputs(m, lik, keep)
                    prep = drop_pcs(prep, keep)
                    Yz, Yz_te, Yz_cal = Yz[:, keep], Yz_te[:, keep], Yz_cal[:, keep]
                    tqdm.write(f"[hii-train] {fam}: removed {len(expl_cal) - len(keep)} of {len(expl_cal)} PCs "
                               f"(explained <= {EMUL.PC_MIN_EXPLAINED:g} on the calibration clouds): "
                               f"{prep['pca']['dropped_pcs']}")
            prep['pca']['resid_var'] = pca_residual_var(prep, Y[fam][tr])
            tqdm.write(f"[hii-train] {fam}: residual variance outside the PCs, median "
                       f"{np.median(prep['pca']['resid_var']):.2e}, max {np.max(prep['pca']['resid_var']):.2e} (z units)")

        # post-hoc sigma calibration: s_k per output from the calibration clouds .
        # Stored in the preprocessing, applied by fast_predict as var -> s_k^2 var.
        # The holdout coverage is recorded before and after
        if fam in cal_fams:
            # calibration on latent + likelihood noise, the variance that fast_predict returns
            # s_min: latent + noise can be too wide, so the calibration may also narrow sigma
            s = calibration_scale(m, Xz[cal], Yz_cal, device, s_min=EMUL.SIGMA_SCALE_MIN_NOISE,
                                  quantile=args.calib_quantile, likelihood=lik)
            prep['sigma_scale'] = s
            before = coverage_scores(m, Xz[te], Yz_te, device=device, likelihood=lik)
            after = coverage_scores(m, Xz[te], Yz_te, scale=s, device=device, likelihood=lik)
            extras[f'{fam}_sigma_calibration'] = {
                's_median': float(np.median(s)), 's_max': float(s.max()),
                # clip at SIGMA_SCALE_MIN_NOISE; n_shrunk = outputs whose sigma was narrowed (s_k < 1)
                'n_clipped': int(np.sum(s <= EMUL.SIGMA_SCALE_MIN_NOISE)), 's_min': float(np.min(s)),
                'n_shrunk': int(np.sum(s < 1.0)),
                'holdout_cov68_before': before['cov68'], 'holdout_cov68_after': after['cov68'],
                'holdout_cov95_before': before['cov95'], 'holdout_cov95_after': after['cov95'],
                'holdout_out5_before': before['out5'], 'holdout_out5_after': after['out5']}
            tqdm.write(f"[hii-train] sigma calibration {fam}: s_k median {np.median(s):.2f}, max {s.max():.2f}"
                       f" | holdout cov95 {before['cov95']:.3f} -> {after['cov95']:.3f} (target 0.95),"
                       f" cov68 {before['cov68']:.3f} -> {after['cov68']:.3f},"
                       f" |r|>5 {before['out5']:.3f} -> {after['out5']:.3f}")

        # dictionary of state on CPU
        models[fam] = {'model': {k: v.cpu() for k, v in m.state_dict().items()},
                       'likelihood': {k: v.cpu() for k, v in lik.state_dict().items()}}
        extras[f'{fam}_prep'] = prep
        extras[f'{fam}_elbo_history'] = hist
        extras[f'{fam}_holdout_history'] = te_hist
        trained[fam] = (m,lik)
        Yz_by_fam[fam] = Yz
        Yz_te_by_fam[fam] = Yz_te
    pbar.close()

    # split emulators file .pt per family
    stem = str(args.out)[:-3] if str(args.out).endswith('.pt') else str(args.out)
    split = {'lines': ['lines'],
             'cont': ['cont'],
             'grain': ['grain'],
             'trans': ['trans'],
             'scalars': ['dust', 'fuv_t', 'fuv_n', 'energy']}

    # only the trained families (no energy without the incident SED)
    split = {p: [f for f in fs if f in models] for p, fs in split.items()}
    assert sorted(f for fs in split.values() for f in fs) == sorted(models), ("[hii-train] incomplete split: "
                                                                 "every trained family must finish on a file")
    out_paths = {}
    for part, fams in split.items():
        out_paths[part] = f"{stem}_{part}.pt"
        torch.save({'families': {f: models[f] for f in fams}, 'extras': extras}, out_paths[part])

    assert_size_budget(list(out_paths.values()), args.size_budget_mib)

    holdout_path = write_holdout(f"{stem}_holdout.json", np.asarray(meta['keys'])[te], args.holdout_frac,
                                 args.holdout_seed, EMUL.AXES_DOMINANT_KEYS_HII, corpus=args.corpus,
                                 n_models = int(meta['n_used']))
    # keys of the calibration clouds, same format as the holdout file (reproducibility of the s_k)
    calib_path = (write_holdout(f"{stem}_calib.json", np.asarray(meta['keys'])[cal], args.calib_frac,
                                args.holdout_seed + 1, EMUL.AXES_DOMINANT_KEYS_HII, corpus=args.corpus,
                                n_models=int(meta['n_used'])) if cal.any() else None)
    gate_set_path = f"{stem}_gate_set_frozen.npz"   # one name, written here and recorded in the meta below
    build_frozen_set(gate_set_path, Xz[te], np.concatenate([Yz_te_by_fam[f]
                        for f in sorted(Yz_te_by_fam)], axis=1), np.concatenate([Y[f][te] for f in sorted(Y)], axis=1),
                          axes_idx = EMUL.AXES_DOMINANT_HII, families = [(f, Yz_te_by_fam[f].shape[1]) for f in sorted(Yz_te_by_fam)])

    # Part of emulator update (CloudIA AI AGENT, WIP)
    try:
        from galapy.spectroscopy.emulator_update import update_emulator
    except ImportError:
        print("[hii-train] emulator update not available (under development)")

    # metadata production
    pathlib.Path(args.meta).write_text(json.dumps({
                'n_models': int(meta['n_used']),
                'holdout': {'file': holdout_path, 'gate_set': gate_set_path,
                    'frac': args.holdout_frac, 'seed': args.holdout_seed,
                    'n_train': int(tr.sum()), 'n_test': int(te.sum()),
                    'final': {k: extras[f'{k}_holdout_history'][-1]
                              for k, _, _ in FAM},
                    'explained_weighted': {k: extras[f'{k}_holdout_per_pc']['explained_weighted']
                                           for k, do_pca, _ in FAM if do_pca}},
                # holdout.final is the RAW sigma (during training); the calibrated one is below
                'sigma_calibration': {'file': calib_path, 'frac': args.calib_frac, 'n_calib': int(cal.sum()),
                    'families': list(cal_fams), 's_min': EMUL.SIGMA_SCALE_MIN_NOISE,
                    'estimator': f"quantile_{args.calib_quantile:g}(|r|) / Phi^-1((1+q)/2)",
                    'quantile': args.calib_quantile,
                    'per_family': {f: extras[f'{f}_sigma_calibration'] for f in cal_fams}},
                'input_keys': list(EMUL.INPUT_KEYS_HII),
                'sed_inputs': extras['sed_inputs'],
                'predictive_noise': extras['predictive_noise'],
                'best_epoch': {k: extras[f'{k}_best_epoch'] for k, _, _ in FAM},
                'dropped_pcs': {k: extras[f'{k}_prep']['pca'].get('dropped_pcs', [])
                                for k, do_pca, _ in FAM if do_pca},
                'constraints': {'trans_target': EMUL.TRANS_TARGET, 'tau_range': list(EMUL.TRANS_TAU_RANGE),
                                'energy_family': True, 'k_range': list(EMUL.ENERGY_K_RANGE), 'grain_le_cont': True,
                                'energy_rescale': EMUL.ENERGY_RESCALE},
                'x_min': X[tr].min(0).tolist(),
                'x_max': X[tr].max(0).tolist(),
                'n_inducing': args.inducing,
                'n_inducing_lines': args.inducing_lines,
                'artifacts': out_paths,
                'delta_M_allowed': {p: _delta_m_budget(out_paths[p], args.size_budget_mib) for p in out_paths},
                'update_mode': 'base',
                'pumping_removed': meta['pumping_removed'],
                'var_target': args.var_target,
                'pca_fit_max': args.pca_fit_max,
                'pca_n_fit': {k: extras[f'{k}_prep']['pca']['n_fit'] for k, do_pca, _ in FAM if do_pca},
                'rel_floor': {f: rel[f] for f in EMUL.VECTOR_FAMS},
                'rel_floor_lyc': {f: lyc[f] for f in EMUL.WAVE_FAMS},
                'lyman_limit_A': CONST.LyLimit,
                'sd_floor': EMUL.SD_FLOOR,
                'line_targets': args.line_targets,
                'ratio_floor': args.ratio_floor if ratio_ref is not None else None,
                'max_pca_components': args.max_pca_components,
                'size_budget_mib': args.size_budget_mib,
                'timing': {'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
                            'torch': torch.__version__,
                            'tf32': bool(getattr(torch.backends.cuda.matmul,'allow_tf32', False)),
                            'batch': args.batch_size,
                            'epochs_run': len(hist),
                            's_per_epoch_median': float(np.median(epoch_times)),
                            's_per_epoch_iqr': float(np.subtract(*np.percentile(epoch_times, [75, 25]))),
                            'peak_vram_gib': (torch.cuda.max_memory_allocated() / 2 ** 30 if device == 'cuda' else None)},
                'epochs': args.epochs,
                'batch': args.batch_size, 'lr': args.lr,
                'seed': args.seed,
                'families': {k: (extras[f'{k}_prep']['pca']['n_components']
                                 if extras[f'{k}_prep']['pca'] else int(Y[k].shape[1])) for k, _, _ in FAM},
                'final_elbo': {k: extras[f'{k}_elbo_history'][-1] for k, _, _ in FAM},
                }, indent = 2, default = str))

    print(f"[hii-train] -> {args.out} | meta -> {args.meta}")

if __name__ == '__main__':
    main()














