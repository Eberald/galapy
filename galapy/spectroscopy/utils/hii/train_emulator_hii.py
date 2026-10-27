# Author: Enrico Veraldi
# training the HII emulator

import argparse
import json
import pathlib

import gpytorch
import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from galapy.internal.constants import clight
from galapy.spectroscopy.svgp import BatchedSVGPS, lines_to_ratio
from galapy.spectroscopy.hii_emulator import line_log_scale
from galapy.spectroscopy.utils.validate_emulator import (holdout_split, encode_targets,
                            holdout_scores, write_holdout, log_targets,
                            build_frozen_gate_set, calibration_scale, coverage_scores)

# input free parameters of the training on HII
INPUT_KEYS = ('logU', 'lognH_HII', 'log_zeta_O', 'z_CMB', 'xi_d', 'f_esc_meas', 'F_star', 'tau_SSP', 'Z_star')
LOG10_INPUTS = ('tau_SSP', 'Z_star')

AXES_DOMINANT_KEYS = ('logU', 'lognH_HII', 'log_zeta_O') # axes that are dominant in the training, for PCA more impact
LOG_FLOOR = 1e-35   # avoid -inf on null fluxes
# Default floor of each vector family before the log10, relative to the peak of each model (row): values below
# rel_floor * max(row) are set to it. The floor fixes the depth where a spectrum goes to ~0: in log
# space a transition transparent -> opaque is a jump of -log10(rel_floor) dex, which a stationary GP kernel smooths
# over, spilling error onto the cells near the jump. So the floor should be as HIGH as the physics allows:
#   - trans: T = transmitted/incident <= 1, so T < 1e-5 is an opaque cloud, indistinguishable from 0. Floor 1e-5
#     vs 1e-10 (holdout, 40 epochs): error below 912 A p50 0.20 -> 0.12 dex, p90 0.76 -> 0.49 dex, cells with
#     |dT| > 0.1 35% -> 27%, nothing changes above 912 A.
#   - cont, grain: the spectrum spans ~10 dex (free-free radio tail far below the peak), a higher floor clips
#     cells that are needed (at 1e-5: 32% of the cells > 100 um, error p95 0.2 -> 2.3 dex), so they keep 1e-10
#     (except below 912 A, see REL_FLOOR_LYC).
#   - lines: with ratio targets the floor is ratio_floor (on L_i / H-beta), so no relative floor: it would be
#     relative to the brightest line (Ly-alpha ~ 20-40 H-beta) and clip at ~2e-9 H-beta, above ratio_floor.
#     With absolute targets they get REL_FLOOR_ABS_LINES.
REL_FLOOR = {'lines': 0.0, 'cont': 1e-10, 'grain': 1e-10, 'trans': 1e-5}
REL_FLOOR_ABS_LINES = 1e-10  # relative floor of the lines with --line-targets absolute
# Floor that REPLACES rel_floor at the ionizing wavelengths (lambda < LYMAN_LIMIT_A) of the families on the
# continuum wavelength grid. The Lyman break is where the transparent/opaque cliff lives, while the deep tails that
# need the low floor (free-free radio) are all at lambda > 912 A, so the two regimes get separate floors:
#   - cont, lyc 1e-5 + rest 1e-10 vs 1e-10 everywhere (holdout, 40 epochs): error below 912 A p50 0.39 -> 0.18 dex,
#     p90 2.1 -> 0.83 dex, |err|/peak p95 0.078 -> 0.046; above 912 A unchanged (radio tail p95 0.21 -> 0.20 dex).
#     Same gain below 912 A as a flat 1e-5, without its loss above (radio tail p95 2.3 dex).
#   - grain: no ionizing floor. Below 912 A the grain emission is already < 1e-10 of the peak in every model (a
#     constant column), so the ionizing floor changes nothing (identical holdout errors). A flat 1e-7 / 1e-5
#     gains <= 10% at 1-100 um but loses on the weak cells (1e-5: 100 um - 1 cm p90 0.18 -> 0.48 dex): 1e-10.
# Families not listed have no ionizing floor (their rel_floor holds at every wavelength).
REL_FLOOR_LYC = {'cont': 1e-5}
LYMAN_LIMIT_A = 911.76      # Lyman limit (vacuum, Angstrom): the wave grid is in Angstrom
WAVE_FAMS = ('cont', 'grain', 'trans')  # families defined on the continuum wavelength grid
SD_FLOOR = 0.01     # minimum sd (dex) of the z-score: near-constant column otherwise blow numerical noise up to unit variance and inflate the PCA rank
VECTOR_FAMS = ('lines', 'cont', 'grain', 'trans')   # families that get the relative floor
RATIO_REF = ('h 1', 4862.7)  # reference line of the ratio form (species, vacuum wavelength in Angstrom): H-beta
RATIO_FLOOR = 1e-10  # default floor of L_i / H-beta in the ratio form: low enough that the weak lines are emulated
# Post-hoc sigma calibration
CALIB_FRAC = 0.05   # default fraction of the CLOUDS kept out of training to estimate the scale s_k of sigma
CALIB_FAMS = ('cont', 'grain', 'trans')  # default families calibrated: the PCA ones, overconfident on the holdout
SIGMA_SCALE_MIN = 1.0  # lower clip of s_k: the calibration only widens sigma, never shrinks it
FUV_N_OFFSET = 1e-6 # N_FUV >=0, can be zero and in this case we add a small offset to avoid log(0)

AXES_DOMINANT = tuple(INPUT_KEYS.index(k) for k in AXES_DOMINANT_KEYS)

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

        #check units
        from galapy.internal.constants import Points_Units_HII
        units = {k: (v.decode() if isinstance(v, bytes) else v) for k,v in f['units'].attrs.items()} if 'units' in f else {}
        expected = {k: u for k, (u, _d) in Points_Units_HII.items() if u is not None}
        if units != expected:
            raise ValueError(f"[hii-train] Units mismatch in the corpus file '{path}', expected {expected} but found {units}")

        #global metadata
        line_names = [n.decode() if isinstance(n, bytes) else n for n in f['line_names'][:]]
        lines_vacuum = f['lines_emergent/wavelengths_rest_vacuum'][:] # need per mode of galapy spec with flux lines given
        wave_grid = f['continuum/wave_grid'][:]
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

            X.append([np.log10(g.attrs[a]) if a in LOG10_INPUTS else g.attrs[a] for a in INPUT_KEYS])
            for fam, dset in src.items():
                v = g[dset]
                Y[fam].append(v[()] if v.shape == () else v[:])

            # Q_H node unit
            node = (float(g.attrs['tau_SSP']), float(g.attrs['Z_star']))
            qh.setdefault(node, []).append(float(g['s_k'][()]) * 10.0 ** (g.attrs['logU'] + g.attrs['lognH_HII']) * clight['cm/s'])

    # pumping choerence
    if len(pumped) > 1:
        raise ValueError(f"[hii-train] {path}: pumping removed for some models and not for others")

    #conversion on array numpy format
    Y = {fam: np.asarray(v, np.float64) for fam, v in Y.items()}
    for fam in ('dust', 'fuv_t', 'fuv_n'):
        Y[fam] = Y[fam][:, None]

    return (np.asarray(X, np.float64), Y,
            {'lines_names': line_names,
             'lines_vacuum': lines_vacuum,
             'wave_grid': wave_grid,
             'units': units,
             'n_used': len(X),
             'pumping_removed': pumped == {True},
             'qh_unit_nodes': qh_table(qh),
             'keys': used})

def qh_table(qh):
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
    return {'log_tau': np.log10(taus), 'Z': np.asarray(Zs), 'Qh': Q}

def parse_rel_floor(items, line_targets='ratio'):
    """
    Relative floor of each vector family from the --rel-floor items.

    Parameters:
        items (list of str): 'FAM=VALUE' sets one family, a bare 'VALUE' sets all the vector families;
            later items override earlier ones. Empty -> the defaults REL_FLOOR.
        line_targets (str): 'ratio' or 'absolute'. With absolute line targets there is no ratio_floor, so the
            default of the lines is REL_FLOOR_ABS_LINES instead of 0.

    Returns:
        dict: {family: rel_floor} for every family in VECTOR_FAMS.
    """
    rel = dict(REL_FLOOR)
    if line_targets == 'absolute':
        rel['lines'] = REL_FLOOR_ABS_LINES
    return parse_family_values(items, rel, VECTOR_FAMS, '--rel-floor')


def parse_rel_floor_lyc(items):
    """
    Ionizing floor (lambda < LYMAN_LIMIT_A) of each wavelength-grid family from the --rel-floor-lyc items,
    same syntax as --rel-floor. 0 means no ionizing floor (rel_floor everywhere).

    Returns:
        dict: {family: floor} for every family in WAVE_FAMS (0.0 = none).
    """
    lyc = {f: REL_FLOOR_LYC.get(f, 0.0) for f in WAVE_FAMS}
    return parse_family_values(items, lyc, WAVE_FAMS, '--rel-floor-lyc')


def wavelength_floor(rel, lyc, wave):
    """
    Relative floor per column of a wavelength-grid family: lyc below LYMAN_LIMIT_A, rel above.
    Returns the scalar rel when there is no ionizing floor (lyc = 0), so old behaviour and artifacts are unchanged.
    log_targets accepts both: the array broadcasts on the columns (rel_floor * max of each row).
    """
    if not lyc:
        return rel
    return np.where(np.asarray(wave, dtype=np.float64) < LYMAN_LIMIT_A, lyc, rel)


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


def pca_rank(Y, var_target=0.9999, rel_floor=0.0):
    """
    Transforms the input data by applying a logarithmic scale, normalizes
    each column, performs Principal Component Analysis (PCA), and determines
    the minimum number of components required to explain the specified variance.

    Parameters:
    Y : numpy.ndarray
        A 2-dimensional array of input data with rows as observations
        and columns as features.
    var_target : float, default 0.9999
        The target proportion of variance to be explained by the selected
        principal components. Must be a value between 0 and 1.
    rel_floor : float or np.ndarray (n_columns,), default 0.0
        Floor relative to the peak of each model before the log10 (see log_targets); an array gives one floor per
        column (e.g. per wavelength, see wavelength_floor).

    Returns:
    tuple
        fit : sklearn.decomposition.PCA
            An instance of PCA after fitting to the normalized data.
        z : numpy.ndarray
            The z-scored input data used for PCA.
        mu : numpy.ndarray
            The mean values of each column in the log-transformed data.
        sd : numpy.ndarray
            The standard deviations of each column in the log-transformed data.
        k_var : int
            The minimum number of principal components required to explain
            at least the specified target variance.
    """
    ytil = log_targets(Y, rel_floor=rel_floor)
    mu, sd = ytil.mean(0), np.maximum(ytil.std(0), SD_FLOOR)
    z = (ytil - mu) / sd  # z-score
    fit = PCA().fit(z)    # compute PCA
    k_var = int(np.searchsorted(np.cumsum(fit.explained_variance_ratio_), var_target) + 1)
    return fit, z, mu, sd, k_var

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
                       ratio_ref=None, ratio_floor=0.0, log_scale=None):
    """
    Transforms the physical targets (e.g., continuum, lines, scalars) of a family into a format suitable
    for training Sparse Variational Gaussian Processes (SVGP). The transformation includes logarithmic
    scaling, normalization, and optionally Principal Component Analysis (PCA). Returns the transformed
    targets and a dictionary of preprocessing details that can be utilized to revert the changes and
    recover the original physical units at runtime.

    Parameters:
        Y (np.ndarray): Input physical targets of shape (N_models, n_outputs).
        do_pca (bool): Flag indicating whether to apply PCA for dimensionality reduction.
        var_target (float): Proportion of target variance to retain when applying PCA. Default is 0.9999.
        k_granted (Optional[int]): Maximum number of PCA components to retain. If None, k is determined
            automatically based on var_target.
        offset (float): Offset to be added to the input targets before transformation. Default is 0.0.
        rel_floor (float or np.ndarray): Floor relative to the peak of each model before the log10, scalar or one
            per column (see wavelength_floor). Default is 0.0 (absolute LOG_FLOOR only).
        ratio_ref (Optional[int]): lines only (no PCA). Index of the reference line (H-beta): the targets
            become the ratio form of galapy.spectroscopy.svgp.lines_to_ratio. None = log10 of the fluxes.
        ratio_floor (float): minimum of L_i / L_ref in the ratio form (0 = none).
        log_scale (Optional[np.ndarray]): log10 Q_H (1 - f_esc) of each model, required with ratio_ref.

    Returns:
        Tuple[Any, Dict[str, Any]]: A tuple where the first element is the transformed targets (z-scored
        and optionally PCA-transformed), and the second element is a dictionary containing preprocessing
        details such as mean, standard deviation, offset, and PCA configurations.

    Raises:
        ValueError: If invalid inputs are provided for PCA-related calculations.
    """
    # offset is zero for all families except N_FUV, saved in prepr and removed at runtime
    Y = np.asarray(Y, dtype=np.float64) + offset # [N_models times n_outputs] put in array and add offset
    if not do_pca:
        ytil = log_targets(Y, rel_floor=rel_floor)
        ratio = None
        if ratio_ref is not None:
            if log_scale is None:
                raise ValueError("[hii-train] ratio form of the lines needs log_scale")
            ytil = lines_to_ratio(ytil, ratio_ref, log_scale, ratio_floor)
            ratio = {'ref': int(ratio_ref), 'floor': float(ratio_floor), 'scale': 'log10 Q_H (1 - f_esc)'}
        mu, sd = ytil.mean(0), np.maximum(ytil.std(0), SD_FLOOR)
        return (ytil - mu) / sd, {'mu': mu, 'sd': sd, 'offset': offset, 'rel_floor': rel_floor,
                                  'ratio': ratio, 'pca': None}

    fit, z, mu, sd, k_var = pca_rank(Y, var_target, rel_floor)
    k = int(k_granted) if k_granted else k_var
    # components PCA ordered for decreasing variance, so we can truncate directly
    comp = fit.components_[:k]
    scores = (z - fit.mean_) @ comp.T
    # whitening: every PC score to unit variance, the scale the GP starts from (outputscale and noise
    # ~0.7). Raw scores have variance up to thousands and the hyperparameters cannot get there.
    score_sd = np.maximum(scores.std(0), 1e-12)
    return scores / score_sd, {
        'mu': mu, 'sd': sd, 'offset': offset, 'rel_floor': rel_floor, 'pca': {
            'components': comp, 'mean': fit.mean_, 'score_sd': score_sd,
            'n_components': k, 'n_components_var_target': k_var,
            'truncated_by_budget': k < k_var,
            'explained': float(fit.explained_variance_ratio_[:k].sum())
        }
    }

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
                 progress=None, cost=1.0):
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
            The learning rate for the AdamW optimizer.
        device: str
            The computation device on which the training is conducted (e.g., "cpu" or "cuda").
        seed: int, optional
            The seed value for the random number generator to ensure reproducible results. Default is 42.
        test: (Xz_test, Yz_test), optional
            The holdout in the training space, scored every eval_every epochs (TE lines).
        progress: tqdm, optional
            Progress bar of the whole training, advanced by `cost` at each epoch.

    Returns:
        tuple:
            A tuple containing:
                - model: The trained batched SVGP model.
                - likelihood: The trained GaussianLikelihood corresponding to the batched SVGP model.
                - hist: A list containing the history of ELBO losses recorded per epoch.
                - epoch_times: A list of times taken to complete each epoch during training.

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
    opt = torch.optim.AdamW(list(model.parameters()) + list(likelihood.parameters()), lr=lr) # optimizer

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
        if test is not None and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1):
            test_hist.append({'epoch': epoch+1, **holdout_scores(model, test[0], test[1], device)})
            model.train()
            tqdm.write(f"[hii-train] TE epoch {epoch + 1:4d}/{epochs}  rmse_z {test_hist[-1]['rmse_z']:.4f}"
                       f" (constant predictor {ref:.4f}, good << it)  cov68 {test_hist[-1]['cov68']:.3f}"
                       f" (calibrated 0.68)")
        if epoch % 20 == 0 or epoch == epochs - 1:
            tqdm.write(f"[hii-train] TR epoch {epoch + 1:4d}/{epochs}  ELBO loss {hist[-1]: .4f} (must decrease)")
        if progress is not None:
            progress.update(cost)

    return model, likelihood, hist, epoch_times, test_hist

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
                    'sets all of them (default: ' + ' '.join(f'{k}={v:g}' for k, v in REL_FLOOR.items())
                    + f', lines={REL_FLOOR_ABS_LINES:g} with --line-targets absolute)')
    ap.add_argument('--rel-floor-lyc', nargs='+', default=[], metavar='[FAM=]VALUE', help='relative floor that '
                    f'replaces --rel-floor below the Lyman limit ({LYMAN_LIMIT_A} A) for the families on the wavelength '
                    f'grid {WAVE_FAMS}, same syntax as --rel-floor; 0 = none (default: '
                    + ' '.join(f'{f}={REL_FLOOR_LYC.get(f, 0.0):g}' for f in WAVE_FAMS) + ')')
    ap.add_argument('--line-targets', choices=('ratio', 'absolute'), default='ratio',
                    help='targets of the lines family: ratio = H-beta / [Q_H (1 - f_esc)] and the other '
                         'lines as ratios to H-beta; absolute = log10 of the fluxes (default: ratio)')
    ap.add_argument('--ratio-floor', type=float, default=RATIO_FLOOR, help='minimum of L_i / H-beta '
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
    ap.add_argument('--calib-frac', type=float, default=CALIB_FRAC, help='fraction of the clouds kept out of '
                    'training (and disjoint from the holdout) to calibrate the predictive sigma, one scale s_k '
                    'per output; 0 disables the calibration (default: %(default)g)')
    ap.add_argument('--calib-families', default=','.join(CALIB_FAMS), help='comma-separated families whose '
                    'sigma is calibrated (default: %(default)s)')
    ap.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto', help='device '
                                                                                      'to use (default: auto)')
    args=ap.parse_args()


    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    device = (('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    X, Y, meta = load_corpus(args.corpus)
    print(f"[hii-train] {meta['n_used']} converged models, {X.shape[1]} input, device={device}")

    # train-test
    te = holdout_split(meta['keys'], X, AXES_DOMINANT, args.holdout_frac, args.holdout_seed)
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
        cal[~te] = holdout_split(np.asarray(meta['keys'])[~te], X[~te], AXES_DOMINANT, frac_rest, args.holdout_seed + 1)
    tr = ~te & ~cal
    n_clouds = lambda m: len({k.split('_')[2] for k in np.asarray(meta['keys'])[m]})
    print(f"[hii-train] holdout: {int(te.sum())} models of {n_clouds(te)} clouds, calibration: {int(cal.sum())} models"
          f" of {n_clouds(cal)} clouds, training: {int(tr.sum())}")

    # lines in ratio form: index of H-beta and the analytic scale log10 Q_H (1 - f_esc) of every model
    ratio_ref, log_scale = None, None
    if args.line_targets == 'ratio':
        names = [' '.join(str(n).split()).casefold() for n in meta['lines_names']]
        cand = [i for i, (n, w) in enumerate(zip(names, np.asarray(meta['lines_vacuum'], float)))
                if n == RATIO_REF[0] and abs(w - RATIO_REF[1]) < 2.0]
        if len(cand) != 1:
            raise ValueError(f"[hii-train] reference line {RATIO_REF} not found once in the line list ({cand})")
        ratio_ref = cand[0]
        log_scale = line_log_scale(meta['qh_unit_nodes'], list(INPUT_KEYS), X)

    # input standardization on everything but the holdout: it uses only the input POSITIONS of the calibration
    # set, never its targets, and keeps the holdout completely untouched
    xmu, xsd = X[~te].mean(0), X[~te].std(0)
    xsd = np.where(xsd > 0, xsd, 1.0)
    Xz = (X - xmu) / xsd

    trained, Yz_by_fam, Yz_te_by_fam = {}, {}, {}
    epoch_times = []
    models, extras = {}, {'x_mean': xmu, 'x_std': xsd, 'line_names': meta['lines_names'], 'lines_vacuum':
                         meta['lines_vacuum'], 'wave_grid': meta['wave_grid'], 'input_keys': list(INPUT_KEYS),
                         'log10_inputs': list(LOG10_INPUTS), 'qh_unit_nodes': meta['qh_unit_nodes'],
                         'pumping_removed': meta['pumping_removed'], 'ard_num_dims': X.shape[1],
                         'n_inducing': args.inducing, 'n_inducing_lines': args.inducing_lines}

    # families of emulators (name, PCA, offset)
    FAM = (('lines', False, 0.0), ('cont', True, 0.0), ('grain', True, 0.0), ('trans', True, 0.0),
           ('dust', False, 0.0), ('fuv_t', False, 0.0), ('fuv_n', False, FUV_N_OFFSET))
    unknown = set(cal_fams) - {f for f, _, _ in FAM}
    if unknown:
        raise ValueError(f"[hii-train] --calib-families: unknown families {sorted(unknown)}")

    # computation of required PCA components for each family
    rel = parse_rel_floor(args.rel_floor, args.line_targets)
    rel.update({fam: 0.0 for fam, _, _ in FAM if fam not in VECTOR_FAMS})
    lyc = parse_rel_floor_lyc(args.rel_floor_lyc)
    # floor actually applied: a per-wavelength array for the families with an ionizing floor, rel otherwise
    floor = {fam: (wavelength_floor(rel[fam], lyc[fam], meta['wave_grid']) if fam in WAVE_FAMS else rel[fam])
             for fam, _, _ in FAM}
    k_var = {fam: pca_rank(Y[fam][tr], args.var_target, floor[fam])[4] for fam, do_pca, _ in FAM if do_pca}
    k_granted = allocate_pca_budget(k_var, args.max_pca_components)
    print(f"[hii-train] settings: epochs {args.epochs}, batch {args.batch_size}, lr {args.lr:g}, "
          f"M {args.inducing} (PCA families) / {args.inducing_lines} (lines, scalars), seed {args.seed}")
    print(f"[hii-train] preprocessing: var-target {args.var_target:g}, max PCA comp {args.max_pca_components}, "
          f"rel-floor {' '.join(f'{f}={rel[f]:g}' for f in VECTOR_FAMS)} (below {LYMAN_LIMIT_A} A: "
          f"{' '.join(f'{f}={lyc[f]:g}' for f in WAVE_FAMS if lyc[f]) or 'none'}), sd-floor {SD_FLOOR:g} dex, PCA scores whitened")
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
        Yz, prep = preprocess_targets(Y[fam][tr], do_pca, args.var_target, k_granted.get(fam) if do_pca else None,
                                      offset=off, rel_floor=floor[fam], **lr_kw)
        Yz_te = encode_targets(prep, Y[fam][te], log_scale=log_scale[te] if lr_kw else None)
        M_fam = args.inducing if do_pca else args.inducing_lines
        tqdm.write(f"[hii-train] Training {fam}: {Yz.shape[1]} output, M={M_fam}"
                + (f" (PCA {prep['pca']['n_components']} PC, "f"{prep['pca']['explained']:.4f} var)" if prep['pca'] else ""))

        m, lik, hist, ep_t, te_hist = train_family(Xz[tr], Yz, M_fam, args.epochs, args.batch_size, args.lr, device, args.seed,
                                                   test=(Xz[te], Yz_te), eval_every=args.eval_every,
                                                   progress=pbar, cost=cost[fam])
        epoch_times.extend(ep_t)

        # post-hoc sigma calibration: s_k per output from the calibration clouds .
        # Stored in the preprocessing, applied by fast_predict as var -> s_k^2 var.
        # The holdout coverage is recorded before and after
        if fam in cal_fams:
            Yz_cal = encode_targets(prep, Y[fam][cal], log_scale=log_scale[cal] if lr_kw else None)
            s = calibration_scale(m, Xz[cal], Yz_cal, device, s_min=SIGMA_SCALE_MIN)
            prep['sigma_scale'] = s
            before = coverage_scores(m, Xz[te], Yz_te, device=device)
            after = coverage_scores(m, Xz[te], Yz_te, scale=s, device=device)
            extras[f'{fam}_sigma_calibration'] = {
                's_median': float(np.median(s)), 's_max': float(s.max()),
                'n_clipped': int(np.sum(s <= SIGMA_SCALE_MIN)),
                'holdout_cov68_before': before['cov68'], 'holdout_cov68_after': after['cov68'],
                'holdout_out5_before': before['out5'], 'holdout_out5_after': after['out5']}
            tqdm.write(f"[hii-train] sigma calibration {fam}: s_k median {np.median(s):.2f}, max {s.max():.2f}"
                       f" | holdout cov68 {before['cov68']:.3f} -> {after['cov68']:.3f} (calibrated 0.68),"
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
             'scalars': ['dust', 'fuv_t', 'fuv_n']}

    assert sorted(f for fs in split.values() for f in fs) == sorted(models), ("[hii-train] incomplete split: "
                                                                 "every trained family must finish on a file")
    out_paths = {}
    for part, fams in split.items():
        out_paths[part] = f"{stem}_{part}.pt"
        torch.save({'families': {f: models[f] for f in fams}, 'extras': extras}, out_paths[part])

    assert_size_budget(list(out_paths.values()), args.size_budget_mib)

    holdout_path = write_holdout(f"{stem}_holdout.json", np.asarray(meta['keys'])[te], args.holdout_frac,
                                 args.holdout_seed, AXES_DOMINANT_KEYS, corpus=args.corpus,
                                 n_models = int(meta['n_used']))
    # keys of the calibration clouds, same format as the holdout file (reproducibility of the s_k)
    calib_path = (write_holdout(f"{stem}_calib.json", np.asarray(meta['keys'])[cal], args.calib_frac,
                                args.holdout_seed + 1, AXES_DOMINANT_KEYS, corpus=args.corpus,
                                n_models=int(meta['n_used'])) if cal.any() else None)
    build_frozen_gate_set(f"{stem}_gate_set_frozen.npz", Xz[te], np.concatenate([Yz_te_by_fam[f]
                        for f in sorted(Yz_te_by_fam)], axis=1), np.concatenate([Y[f][te] for f in sorted(Y)], axis=1),
                          axes_idx = AXES_DOMINANT, families = [(f, Yz_te_by_fam[f].shape[1]) for f in sorted(Yz_te_by_fam)])

    # Part of emulator update (CloudIA AI AGENT, WIP)
    try:
        from galapy.spectroscopy.emulator_update import update_emulator
    except ImportError:
        print("[hii-train] emulator update not available (under development)")

    # metadata production
    pathlib.Path(args.meta).write_text(json.dumps({
                'n_models': int(meta['n_used']),
                'holdout': {'file': holdout_path, 'gate_set': f"{stem}_gate_set_frozen.npz",
                    'frac': args.holdout_frac, 'seed': args.holdout_seed,
                    'n_train': int(tr.sum()), 'n_test': int(te.sum()),
                    'final': {k: extras[f'{k}_holdout_history'][-1]
                              for k, _, _ in FAM}},
                # holdout.final is the RAW sigma (during training); the calibrated one is below
                'sigma_calibration': {'file': calib_path, 'frac': args.calib_frac, 'n_calib': int(cal.sum()),
                    'families': list(cal_fams), 's_min': SIGMA_SCALE_MIN,
                    'estimator': 'median|r| / 0.6745',
                    'per_family': {f: extras[f'{f}_sigma_calibration'] for f in cal_fams}},
                'input_keys': list(INPUT_KEYS),
                'x_min': X[tr].min(0).tolist(),
                'x_max': X[tr].max(0).tolist(),
                'n_inducing': args.inducing,
                'n_inducing_lines': args.inducing_lines,
                'artifacts': out_paths,
                'delta_M_allowed': {p: _delta_m_budget(out_paths[p], args.size_budget_mib) for p in out_paths},
                'update_mode': 'base',
                'pumping_removed': meta['pumping_removed'],
                'var_target': args.var_target,
                'rel_floor': {f: rel[f] for f in VECTOR_FAMS},
                'rel_floor_lyc': {f: lyc[f] for f in WAVE_FAMS},
                'lyman_limit_A': LYMAN_LIMIT_A,
                'sd_floor': SD_FLOOR,
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














