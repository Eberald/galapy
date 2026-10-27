# Author: Enrico Veraldi
# Holdout (validation) and emulators Gates
"""
This file:
  1. Define the holdouts (split trainig/test). holdout_split, encode_targets, holdout_scores,
     write_holdout and build_frozen_gate_set. Called by training files
     (train_emulator_{hii,pdr}.py): holdout is the 10% (default) of the clouds
  2. Use the holdout. Gate tests on the test data. Report
     JSON + Markdown, state PASS / FAIL / SKIP / ERROR as in validate_grid.

    galapy-validate-emulator --sector hii                 # default in data/

Exit code: 1 if a gate FAIL or ERROR (with --strict also SKIP return 1).
"""

import argparse
import importlib
import json
import pathlib
import subprocess
import sys
import time
import traceback

import numpy as np

PASS, FAIL, SKIP, ERROR = 'PASS', 'FAIL', 'SKIP', 'ERROR'
LOG_FLOOR = 1e-35                    # ame floor of preprocessing in training emulator


# HOLDOUT = defined by training, here readed by validation

def cloud_id(key):
    """
    Extract the cloud identifier from a grid point identifier.

    This function takes a grid point identifier string formatted as
    'grid_point_<lhs>_<tau>_<Z>' and extracts the '<lhs>' portion,
    which represents the unique cloud identifier. The cloud identifier
    is typically the second component in the underscore-separated string.

    Parameters:
    key: str
        The grid point identifier string from which to extract the cloud
        identifier. Must follow the specific format
        'grid_point_<lhs>_<tau>_<Z>'.

    Returns:
    str
        The extracted cloud identifier '<lhs>'.
    """
    return str(key).split('_')[2]


def holdout_split(keys, X, axes_idx, frac=0.1, seed=20240601):
    """
    Performs a holdout split to validate the emulator by separating data per cloud
    and distributing it uniformly.

    This function allocates clouds to octants based on the primary axis and
    performs a stratified sampling to extract the test set for model validation.
    It ensures that the split is consistent and balanced based on the cloud
    distribution across different axes.

    Arguments:
        keys: List of unique identifiers for each data point.
        X: Numeric data array containing the feature values used for the split.
        axes_idx: Indices of the relevant axes used for stratification.
        frac: Proportion of the data to allocate to the holdout set. Must be a value
              between 0 and 1 (exclusive).
        seed: Random seed to ensure reproducibility of the split.

    Returns:
        A boolean array indicating the holdout membership for each data point in 'keys'.
        True values correspond to the holdout set, and False values correspond to the
        training set.

    Raises:
        ValueError: If the fraction 'frac' is not in the range (0, 1).
        ValueError: If the corpus size is too small to accommodate the required holdout set.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"[emul-val] holdout_frac = {frac}: mandatory 0 < frac < 1")
    clouds = np.array([cloud_id(k) for k in keys])
    uniq, inv = np.unique(clouds, return_inverse=True)
    # A = one row per cloud with the position on the axes (same cloud parameters, different others)
    A = np.zeros((uniq.size, len(axes_idx)))
    np.add.at(A, inv, np.asarray(X, float)[:, list(axes_idx)])
    A /= np.bincount(inv)[:, None]
    # assing every cloud to a octant (octant because 3 main axes logn logU logzeta)
    octant = ((A > np.median(A, axis=0)) * (2 ** np.arange(len(axes_idx)))).sum(axis=1)
    # stratified extraction
    rng = np.random.default_rng(seed)
    test = np.zeros(uniq.size, bool)
    for o in np.unique(octant):
        idx = np.flatnonzero(octant == o)
        test[rng.choice(idx, size=int(round(frac * idx.size)), replace=False)] = True
    # limit
    if not test.any():
        test[rng.integers(uniq.size)] = True
    if test.all():
        raise ValueError("[emul-val] corpus too small")
    return test[inv]


def log_targets(Y, offset=0.0, rel_floor=0.0):
    """
    Transforms target values to their logarithmic scale, preventing infinity by
    applying an offset and managing minimum allowable values.

    Parameters:
    Y : array-like
        Input array containing target values to be transformed. Values must be
        convertible to float64.
    offset : float, optional
        A constant offset added to the input values to avoid taking the logarithm of zero.
        Default is 0.0.
    rel_floor : float or array-like, optional
        Relative floor value(s), used to define a minimum cutoff based on a fraction of
        the maximum value in each row of the input. If an array is provided, its shape
        must be compatible with the rows of the input array. If set to 0, this feature
        is disabled. Default is 0.0.

    Returns:
    array-like
        Transformed array with the logarithm (base 10) of the input values. Values are
        floored to the greater of `LOG_FLOOR` or a relative cutoff when `rel_floor` is
        specified.

    Raises:
    ValueError
        If `rel_floor` is specified as an array and its shape is not compatible with
        the input array when `Y` is 2-dimensional.
    """
    Y = np.asarray(Y, dtype=np.float64) + float(offset)
    # minimum admitted value in log
    floor = LOG_FLOOR
    rel_floor = np.asarray(rel_floor, dtype=np.float64)
    # if rel floor is positive, use it as a relative floor for each row (e.g. per wavelength)
    if np.any(rel_floor > 0) and Y.ndim == 2:
        floor = np.maximum(rel_floor * Y.max(axis=1, keepdims=True), LOG_FLOOR)
    return np.log10(np.maximum(Y, floor))


def encode_targets(prep, Y, log_scale=None):
    """
    Converts physical values into a transformed format for emulator usage. The transformation
    process involves applying logarithmic scaling to the physical values, optionally converting
    them to ratios relative to Hb, normalizing the data, applying PCA transformations if specified,
    and projecting onto the GP space.

    Parameters:
    prep (dict): Dictionary containing transformation settings such as offsets, relative floors,
                 ratio configuration, normalization means and standard deviations, and optional
                 PCA-related parameters.
    Y (ndarray): Physical value data to be transformed.
    log_scale (float, optional): Logarithmic scale factor used for ratio transformation.

    Returns:
    ndarray: Transformed data ready for emulator processing.

    Raises:
    ValueError: If ratio transformation is requested in `prep` and `log_scale` is not provided.
    """
    from galapy.spectroscopy.svgp import lines_to_ratio
    ytil = log_targets(Y, prep.get('offset', 0.0), prep.get('rel_floor', 0.0))   #log
    # lines treatment
    # if prep ratio exist, prediction is in ratio between lines and Hb
    if prep.get('ratio') is not None:
        if log_scale is None:
            raise ValueError("encode_targets: righe in forma a rapporti, serve log_scale")
        ytil = lines_to_ratio(ytil, prep['ratio']['ref'], log_scale, prep['ratio']['floor'])
    # normalization
    z = (ytil - np.asarray(prep['mu'])) / np.asarray(prep['sd'])
    pca = prep['pca']
    # optional pca
    if pca is None:
        return z
    # if pca, whitening of scores, so that variance is 1 for each PCA comp
    s = (z - np.asarray(pca['mean'])) @ np.asarray(pca['components']).T
    return s / np.asarray(pca['score_sd']) if 'score_sd' in pca else s


def latent_predict(model, Xz, device='cpu', chunk=1024):
    """LATENT mean and variance of the GP (no likelihood noise, as in predict_batch), each (N, d),
    on the already standardized input Xz."""
    import torch
    model.eval()
    mu, var = [], []
    with torch.no_grad():
        for i in range(0, len(Xz), chunk):
            post = model(torch.as_tensor(Xz[i:i + chunk], dtype=torch.float32, device=device))
            mu.append(post.mean.cpu().numpy())
            var.append(post.variance.cpu().numpy())
    return np.concatenate(mu, axis=1).T, np.concatenate(var, axis=1).T


def holdout_scores(model, Xz, Yz, device='cpu', chunk=1024):
    """RMSE nello spazio del GP e copertura a 1 sigma (varianza latente, quella che
    predict_batch propaga), medie sugli output. E' la misura che il training
    registra ogni --eval-every epoche."""
    if len(Xz) == 0:
        return {'rmse_z': float('nan'), 'cov68': float('nan')}
    mu, var = latent_predict(model, Xz, device, chunk)
    r = np.asarray(Yz) - mu
    sd = np.sqrt(var)
    return {'rmse_z': float(np.sqrt(np.mean(r ** 2))), 'cov68': float(np.mean(np.abs(r) <= sd))}


# Median of |r| for r ~ N(0, 1): Phi^-1(0.75). Half of a standard normal lies within +-0.6745.
MEDIAN_ABS_STD_NORMAL = 0.6744897501960817


def calibration_scale(model, Xz, Yz, device='cpu', s_min=1.0, chunk=1024):
    """
    Post-hoc calibration of the predictive sigma of a GP family: one scale factor s_k per output.

    Why
    ---
    The SVGP sigma is not guaranteed to be honest. On the HII holdout the PCA families (cont, grain,
    trans) are overconfident: only ~40-50% of the true values fall within 1 sigma instead of 68%, and
    the effect grows with the PCA index (the last, low-variance components are close to noise and the
    GP cannot predict them, yet it still reports a small sigma). When the emulator sigma enters the
    likelihood of a fit as model error, a sigma that is too small gives posteriors that are too narrow
    and possibly biased. Rescaling sigma fixes the stated error without touching the mean.

    Math
    ----
    For each calibration point i and output k (in the GP space: z-scored, whitened PCA scores),
    the normalized residual is

        r_ik = (y_ik - mu_ik) / sigma_ik .

    If sigma is honest, r_k ~ N(0, 1). If instead the true error is s_k times the stated one,
    r_k ~ N(0, s_k^2), and replacing sigma_k -> s_k sigma_k restores r_k / s_k ~ N(0, 1). For
    r ~ N(0, s^2) the median of |r| is s * Phi^-1(0.75) = 0.6745 s, hence the estimator

        s_k = median_i |r_ik| / 0.6745 .

    By construction the corrected sigma then holds exactly 50% of the calibration points within
    0.6745 sigma, and about 68% within 1 sigma when the residuals are roughly Gaussian. The median
    is used instead of the maximum-likelihood estimate s_k^2 = mean_i r_ik^2 because the residuals
    have heavy tails (a few % of points beyond 5 sigma): the mean of r^2 would be dominated by them
    and would inflate sigma for every other point.

    s_k is clipped from below at s_min (default 1): the calibration only ever WIDENS sigma. Outputs
    with s_k < 1 are already conservative, and on the HII emulator that happens mostly where the
    residual distribution is very peaked with heavy tails (lines, T_FUV) or where truth and
    prediction both sit on a floor: shrinking sigma there would hide the outliers in the tails.

    The variance is then multiplied by s_k^2 at prediction time (svgp.fast_predict, via
    prep['sigma_scale']); the PCA decoding propagates it to physical units as before.

    Limits
    ------
    - One number per output: if the GP is overconfident only in a region of the input space, there
      it stays overconfident and elsewhere it becomes conservative. It corrects the average, not
      the shape.
    - It does not remove the tails: points beyond 5 sigma become fewer, not zero.
    - The calibration points must be unseen by the training AND distinct from the holdout, or the
      holdout coverage after calibration is 0.68 by construction and checks nothing.

    Parameters:
        model: trained BatchedSVGPS of the family (d outputs).
        Xz (np.ndarray): standardized inputs of the calibration set, (N, D).
        Yz (np.ndarray): targets of the calibration set in the GP space (encode_targets), (N, d).
        s_min (float): lower clip of s_k (1 = never shrink sigma).

    Returns:
        np.ndarray: s_k, (d,).
    """
    mu, var = latent_predict(model, Xz, device, chunk)
    r = (np.asarray(Yz) - mu) / np.sqrt(np.maximum(var, 1e-30))
    s = np.median(np.abs(r), axis=0) / MEDIAN_ABS_STD_NORMAL
    return np.maximum(s, s_min)


def coverage_scores(model, Xz, Yz, scale=None, device='cpu', chunk=1024):
    """cov68 (fraction with |r| <= 1) and fraction with |r| > 5 of the normalized residuals, averaged
    over points and outputs. With `scale` (s_k) the sigma is the calibrated one, s_k sigma_k."""
    mu, var = latent_predict(model, Xz, device, chunk)
    r = (np.asarray(Yz) - mu) / np.sqrt(np.maximum(var, 1e-30))
    if scale is not None:
        r = r / np.asarray(scale)
    return {'cov68': float(np.mean(np.abs(r) <= 1.0)), 'out5': float(np.mean(np.abs(r) > 5.0))}


def write_holdout(path, keys, frac, seed, axes_keys, corpus=None, n_models=None):
    """<stem>_holdout.json: le chiavi dell'holdout e come sono state scelte."""
    keys = [str(k) for k in keys]
    payload = {'keys': keys, 'n_test': len(keys), 'n_clouds': len({cloud_id(k) for k in keys}),
               'frac': frac, 'seed': seed, 'stratified_on': list(axes_keys),
               'corpus': str(corpus) if corpus else None, 'n_models_corpus': n_models}
    pathlib.Path(path).write_text(json.dumps(payload, indent=1))
    return str(path)


def read_holdout(path):
    return json.loads(pathlib.Path(path).read_text())


def build_frozen_gate_set(path, X, Y_gp, Y_phys, axes_idx, n_primary=8192,
                          n_physical=512, seed=20240601, families=None):
    """Sottoinsieme CONGELATO dell'holdout, che viaggia senza corpus (12.3-ter).

    X, Y_gp, Y_phys sono GIA' l'holdout (input standardizzato, target nello spazio
    del GP e fisici). Due componenti, perche' i gate misurano cose diverse:
      - primario  (8192): spazio del GP, dove vive la calibrazione della varianza.
        sqrt(p(1-p)/n) = 0.0052 a p = 0.68: +-0.5% sulla copertura nominale;
      - fisico     (512): spazio osservabile non compresso, per vedere un degrado
        IN BANDA, che la PCA congelata del primario non mostra.
    Stratificazione per ottanti sui tre assi dominanti, seed fisso. families =
    [(nome, n colonne)] di Y_gp, nell'ordine della concatenazione."""
    rng = np.random.default_rng(seed)
    n_primary = min(n_primary, X.shape[0])
    A = X[:, list(axes_idx)]
    octant = ((A > np.median(A, axis=0)) * np.array([1, 2, 4])).sum(axis=1)      # 0..7
    sel = []
    for o in range(8):
        idx = np.flatnonzero(octant == o)
        if idx.size:
            sel.append(rng.choice(idx, size=min(n_primary // 8, idx.size), replace=False))
    sel = np.sort(np.concatenate(sel)) if sel else np.array([], int)
    if sel.size < n_primary:                # ottanti sbilanciati: completa a caso
        rest = np.setdiff1d(np.arange(X.shape[0]), sel)
        sel = np.sort(np.concatenate([sel, rng.choice(rest, size=n_primary - sel.size,
                                                      replace=False)]))
    phys = np.sort(rng.choice(sel, size=min(n_physical, sel.size), replace=False))
    fam = families or []
    np.savez_compressed(
        path, idx_primary=sel, idx_physical=phys,
        X=X[sel], Y_gp=Y_gp[sel], X_phys=X[phys], Y_phys=Y_phys[phys],
        seed=seed, axes_idx=np.asarray(axes_idx),
        families=np.array([f for f, _ in fam]), widths=np.array([w for _, w in fam], int))
    return path


# =============================================================================
# 2. SETTORI: cosa leggere e con quale emulatore (il PDR si aggiunge in 3.5-quater)
# =============================================================================

SECTORS = {
    'hii': {'train': 'galapy.spectroscopy.utils.hii.train_emulator_hii',
            'emulator': ('galapy.spectroscopy.hii_emulator', 'HIIEmulator'),
            'predict': 'predict_batch',
            'parts': ('lines', 'cont', 'grain', 'trans', 'scalars'),
            'lines': 'lines'},
}


def _sector(name):
    s = dict(SECTORS[name])
    s['train_mod'] = importlib.import_module(s['train'])
    mod, cls = s['emulator']
    s['emulator_cls'] = getattr(importlib.import_module(mod), cls)
    return s


class Context:
    """Holdout + emulatore + predizioni, calcolate una volta e condivise dai gate."""

    def __init__(self, args):
        self.args = args
        self.sector = _sector(args.sector)
        self.stem = str(args.emulator)
        self.emu = self.sector['emulator_cls']().load(self.stem)
        self.meta = json.loads(pathlib.Path(args.meta).read_text()) if args.meta else {}
        hold = read_holdout(args.holdout)
        keys = hold['keys']
        if args.max_points and len(keys) > args.max_points:
            rng = np.random.default_rng(args.seed)
            keys = sorted(rng.choice(keys, size=args.max_points, replace=False).tolist())
        self.holdout = {k: hold[k] for k in hold if k != 'keys'}
        self.holdout['n_evaluated'] = len(keys)
        self.X, self.Y, self.corpus_meta = self.sector['train_mod'].load_corpus(args.corpus, keys=keys)
        self.keys = self.corpus_meta['keys']
        self.mu, self.var = predict_gp(self.emu, self.X)
        self.Yz = {f: encode_targets(self.emu.prep[f], self.Y[f], log_scale_of(self.emu, f, self.X))
                   for f in self.mu}
        self._noise = None

    def col(self, key):
        return self.X[:, self.emu.input_keys.index(key)]

    def phys(self, fam, idx=None):
        """(mediana, sigma) fisiche dell'emulatore sulla famiglia, righe idx."""
        return physical(self.emu, fam, self.mu[fam] if idx is None else self.mu[fam][idx],
                        self.var[fam] if idx is None else self.var[fam][idx],
                        self.X if idx is None else self.X[idx])

    def noise(self):
        """Rumore della GaussianLikelihood di ogni famiglia (diagnostica di T-2.4)."""
        if self._noise is None:
            self._noise = likelihood_noise(self.stem, self.sector['parts'])
        return self._noise

    def lines(self):
        from galapy.spectroscopy.utils.validate_grid import LineCatalogue
        return LineCatalogue(self.emu.line_names, self.emu.line_wavelengths_rest)


def predict_gp(emu, X, chunk=1024):
    """Predizione latente di ogni famiglia sull'input FISICO X: {fam: (mu, var)},
    ciascuno (N, d), nello spazio del GP. E' la forma chiusa di predict_batch."""
    import torch
    from galapy.spectroscopy.svgp import fast_predict
    mu, var = {}, {}
    x = (torch.as_tensor(X, dtype=torch.float32) - emu.x_mean) / emu.x_std
    with torch.no_grad():
        for fam, cache in emu.cache.items():
            m, v = zip(*(fast_predict(cache, x[i:i + chunk]) for i in range(0, len(x), chunk)))
            mu[fam] = torch.cat(m, dim=1).T.numpy().astype(float)
            var[fam] = torch.cat(v, dim=1).T.numpy().astype(float)
    return mu, var


def log_scale_of(emu, fam, X):
    """log10 della scala analitica che serve a encode/decode della famiglia (righe in forma a
    rapporti), None per le altre famiglie, per gli artefatti vecchi e per gli emulatori senza."""
    f = getattr(emu, 'log_scale', None)
    return None if f is None or X is None else f(fam, X)


def physical(emu, fam, mu, var, X=None):
    """(mediana, sigma) fisiche. X = input FISICI degli stessi punti: serve per le righe in forma
    a rapporti (Q_H e f_esc), per le altre famiglie si puo' omettere."""
    import torch
    from galapy.spectroscopy.svgp import decode
    return decode(emu.prep[fam], torch.as_tensor(mu.T), torch.as_tensor(var.T),
                  log_scale=log_scale_of(emu, fam, X))


def likelihood_noise(stem, parts):
    import gpytorch
    import torch
    from galapy.spectroscopy.svgp import load_artifacts
    fam, _ = load_artifacts([f"{stem}_{p}.pt" for p in parts])
    out = {}
    for name, f in fam.items():
        d = f['model']['variational_strategy.inducing_points'].shape[0]
        lik = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([d]))
        lik.load_state_dict(f['likelihood'])
        out[name] = lik.noise.detach().reshape(-1).numpy().astype(float)
    return out


# =============================================================================
# 3. GATE: registro e misure comuni ai due settori
# =============================================================================

GATES = {}            # id -> (funzione, settore, bloccante, titolo)


def register(gid, function, sector, blocking=False, title=''):
    GATES[gid] = (function, sector, blocking, title)


def gate(gid, sector, blocking=False, title=''):
    def deco(function):
        register(gid, function, sector, blocking, title)
        return function
    return deco


def _ks_p(r, n, rng):
    from scipy.stats import kstest
    r = r[np.isfinite(r)]
    if r.size > n:
        r = rng.choice(r, size=n, replace=False)
    return float(kstest(r, 'norm').pvalue) if r.size >= 20 else float('nan')


def calibration(ctx, rows=None, families=None, alpha=0.05):
    """Residui normalizzati (y - mu)/sigma nello spazio del GP contro N(0,1).

    KS su n_ks residui per output (sottocampione a seed fisso): con decine di
    migliaia di residui KS rifiuterebbe scarti di qualche per mille nella CDF,
    senza effetto sulla likelihood. Correzione di Bonferroni sul numero di output:
    PASS se ogni output ha p > alpha / n_output. La sigma e' quella latente,
    cioe' quella che predict_batch propaga nella likelihood; la copertura con il
    rumore della likelihood aggiunto e' riportata a parte, come diagnostica."""
    rng = np.random.default_rng(ctx.args.seed)
    fams = families or list(ctx.mu)
    rows = slice(None) if rows is None else rows
    per, n_out, p_all = {}, 0, []
    noise = ctx.noise() if ctx.args.noise_diagnostic else {}
    for f in fams:
        y, m, v = ctx.Yz[f][rows], ctx.mu[f][rows], ctx.var[f][rows]
        r = (y - m) / np.sqrt(np.maximum(v, 1e-30))
        p = np.array([_ks_p(r[:, j], ctx.args.n_ks, rng) for j in range(r.shape[1])])
        n_out += r.shape[1]
        p_all.append(p)
        det = {'min_p': float(np.nanmin(p)) if np.isfinite(p).any() else None,
               'cov68': float(np.mean(np.abs(r) <= 1)), 'cov95': float(np.mean(np.abs(r) <= 1.96))}
        if f in noise:
            rn = (y - m) / np.sqrt(v + noise[f])
            det['cov68_with_noise'] = float(np.mean(np.abs(rn) <= 1))
        per[f] = det
    p_all = np.concatenate(p_all)
    if not np.isfinite(p_all).any():
        return SKIP, {'reason': 'meno di 20 residui per output'}
    thr = alpha / n_out
    worst = float(np.nanmin(p_all))
    return (PASS if worst > thr else FAIL), {
        'n_outputs': n_out, 'threshold_p': thr, 'min_p': worst,
        'n_failing_outputs': int(np.sum(p_all <= thr)), 'families': per,
        'summary': f"min p = {worst:.3g} su {n_out} output (soglia {thr:.2g})"}


def rmse_dex(pred, true, graded):
    d = np.log10(pred[graded]) - np.log10(true[graded])
    return float(np.sqrt(np.mean(d ** 2))) if d.size else float('nan')


def ood_growth(ctx, n_probe=20, steps=(0.25, 0.5, 1.0, 2.0)):
    """La sigma latente cresce lontano dal dominio di training (T-2.7 / T-3.17b).

    Da n_probe punti dell'holdout ci si sposta, un asse continuo alla volta, oltre
    il massimo del training di steps x (larghezza del range). PASS se in ogni
    famiglia la sigma mediana sugli output cresce con la distanza e, al passo piu'
    lontano, supera quella nel dominio in almeno il 95% delle sonde."""
    import torch
    from galapy.spectroscopy.svgp import fast_predict
    emu = ctx.emu
    lo = np.asarray(ctx.meta.get('x_min') or ctx.X.min(0), float)
    hi = np.asarray(ctx.meta.get('x_max') or ctx.X.max(0), float)
    nodes = {'tau_SSP', 'Z_star'}
    axes = [i for i, k in enumerate(emu.input_keys) if k not in nodes]
    rng = np.random.default_rng(ctx.args.seed)
    base = ctx.X[rng.choice(len(ctx.X), size=min(n_probe, len(ctx.X)), replace=False)]

    def sig(X):
        x = (torch.as_tensor(X, dtype=torch.float32) - emu.x_mean) / emu.x_std
        with torch.no_grad():
            return {f: np.median(np.sqrt(fast_predict(c, x)[1].numpy()), axis=0)
                    for f, c in emu.cache.items()}
    s0 = sig(base)
    out, ok = {}, True
    for f in emu.cache:
        med, beyond = [], []
        for k in steps:
            sk = []
            for a in axes:
                Xk = base.copy()
                Xk[:, a] = hi[a] + k * (hi[a] - lo[a])
                sk.append(sig(Xk)[f])
            sk = np.concatenate(sk)
            med.append(float(np.median(sk)))
            beyond = sk > np.tile(s0[f], len(axes))
        mono = bool(np.all(np.diff(med) >= 0))
        frac = float(np.mean(beyond))
        out[f] = {'median_sigma_in_domain': float(np.median(s0[f])), 'median_sigma_by_step': med,
                  'monotone': mono, 'fraction_above_at_farthest': frac}
        ok &= mono and frac >= 0.95
    return (PASS if ok else FAIL), {'steps': list(steps), 'families': out}


def throughput(ctx, n_rep=200):
    """Valutazioni batch-8 al secondo su CPU, a 8 e a 1 thread (T-2.8 / T-3.17c)."""
    import torch
    emu, call = ctx.emu, getattr(ctx.emu, ctx.sector['predict'])
    theta = ctx.X[:8] if len(ctx.X) >= 8 else np.repeat(ctx.X[:1], 8, axis=0)
    res, old = {}, torch.get_num_threads()
    try:
        for nt in (8, 1):
            torch.set_num_threads(nt)
            for _ in range(5):
                call(theta)
            t0 = time.perf_counter()
            for _ in range(n_rep):
                call(theta)
            res[f'eval_per_s_{nt}_threads'] = n_rep / (time.perf_counter() - t0)
    finally:
        torch.set_num_threads(old)
    return res


def footprint(ctx, budget_rss_gib=1.4):
    """Peso dei file dell'artefatto e RAM di un processo che lo carica e predice
    (T-2.10 / T-3.17d). Il processo e' separato: la RAM del corpus letto qui non
    entra nella misura."""
    stem, parts = ctx.stem, ctx.sector['parts']
    sizes = {p: pathlib.Path(f"{stem}_{p}.pt").stat().st_size / 2 ** 20 for p in parts}
    mod, cls = ctx.sector['emulator']
    code = (f"import resource, numpy as np\nfrom {mod} import {cls}\n"
            f"e = {cls}().load({stem!r})\nx = np.tile(e.x_mean.numpy(), (8, 1))\n"
            f"[e.{ctx.sector['predict']}(x) for _ in range(20)]\n"
            f"print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    rss = int(out.stdout.split()[-1]) / 2 ** 20                       # kB (Linux) -> GiB
    budget = float(ctx.meta.get('size_budget_mib', 85))
    ok = all(s <= budget for s in sizes.values()) and rss <= budget_rss_gib
    return (PASS if ok else FAIL), {'file_mib': {p: round(s, 2) for p, s in sizes.items()},
                                    'total_mib': round(sum(sizes.values()), 2),
                                    'budget_mib_per_file': budget, 'peak_rss_gib': round(rss, 3),
                                    'budget_rss_gib': budget_rss_gib}


def per_node_bin(ctx, line_gate):
    """L'accuratezza (line_gate) e la calibrazione per bin di tau (T-2.13 / T-3.17e)."""
    lt = ctx.col('tau_SSP')
    bins, ok = {}, True
    for t in np.unique(np.round(lt, 4)):
        rows = np.flatnonzero(np.isclose(lt, t, atol=1e-4))
        s_acc, d_acc = line_gate(ctx, rows)
        s_cal, d_cal = calibration(ctx, rows=rows)
        bins[f"{10 ** t:.3g} yr"] = {'n': int(rows.size), 'accuracy': s_acc,
                                     'calibration': s_cal, 'min_p': d_cal.get('min_p'),
                                     'worst_line_dex': d_acc.get('worst_dex')}
        ok &= s_acc != FAIL and s_cal != FAIL
    return (PASS if ok else FAIL), {'bins': bins}


def learning_curve(ctx):
    """Learning curve su una validazione INTERNA al training (T-2.9 / T-3.17f).

    Si riaddestra la sola famiglia --lc-family per ogni M e per frazioni crescenti
    delle nubi di training, e si misura su un 10% di nubi tolte al TRAINING: la
    scelta di M non si fa sull'holdout, che resta fuori campione per gli altri gate.
    PASS se, al M dell'artefatto, raddoppiare i dati riduce l'RMSE di meno del 5%."""
    a, T = ctx.args, ctx.sector['train_mod']
    hold = set(read_holdout(a.holdout)['keys'])
    X, Y, meta = T.load_corpus(a.corpus)
    keys = np.asarray(meta['keys'])
    tr = np.array([k not in hold for k in keys])
    X, keys = X[tr], keys[tr]
    fam = a.lc_family or ctx.sector['lines']
    Yf = Y[fam][tr]
    val = holdout_split(keys, X, T.AXES_DOMINANT, 0.1, a.seed + 1)
    clouds = np.array([cloud_id(k) for k in keys])
    uc = np.unique(clouds[~val])
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(uc)
    epochs = a.lc_epochs or int(ctx.meta.get('epochs', 200))
    table = {}
    for M in a.lc_M:
        for fr in sorted(a.lc_fractions):
            sub = np.isin(clouds, order[:max(1, int(round(fr * uc.size)))]) & ~val
            mu, sd = X[sub].mean(0), np.where(X[sub].std(0) > 0, X[sub].std(0), 1.0)
            ratio = ctx.emu.prep[fam].get('ratio')       # stessa forma dei target dell'artefatto
            rkw = ({'ratio_ref': ratio['ref'], 'ratio_floor': ratio['floor'],
                    'log_scale': log_scale_of(ctx.emu, fam, X[sub])} if ratio else {})
            Yz, prep = T.preprocess_targets(Yf[sub], False, ctx.meta.get('var_target', 0.995),
                                            rel_floor=ctx.emu.prep[fam].get('rel_floor', 0.0) if Yf.ndim == 2 else 0.0,
                                            **rkw)
            m = T.train_family((X[sub] - mu) / sd, Yz, M, epochs, a.lc_batch, ctx.meta.get('lr', 1e-3),
                               'cpu', a.seed)[0]
            sc = holdout_scores(m, (X[val] - mu) / sd,
                                encode_targets(prep, Yf[val], log_scale_of(ctx.emu, fam, X[val])))
            table[f"M={M}, f={fr:g}"] = {'n_train': int(sub.sum()), **sc}
    M0 = int(ctx.meta.get('n_inducing_lines' if fam == ctx.sector['lines'] else 'n_inducing', a.lc_M[0]))
    fr_hi, fr_lo = max(a.lc_fractions), max(f for f in a.lc_fractions if f <= max(a.lc_fractions) / 2)
    hi_r, lo_r = (table.get(f"M={M0}, f={f:g}", {}).get('rmse_z') for f in (fr_hi, fr_lo))
    if hi_r is None or lo_r is None:
        return SKIP, {'reason': f"M dell'artefatto ({M0}) non in --lc-M", 'table': table}
    gain = 1.0 - hi_r / lo_r
    return (PASS if gain < 0.05 else FAIL), {'family': fam, 'M_artifact': M0,
                                             'gain_doubling_data': round(gain, 4), 'table': table}


def gpu_matches_cpu(ctx, rtol=1e-4):
    """(c) di T-2.15: lo stesso artefatto su GPU e su CPU, stesse predizioni."""
    import torch
    if not torch.cuda.is_available():
        return SKIP, {'reason': 'nessuna GPU: (c) e\' manuale'}
    gpu = ctx.sector['emulator_cls']().load(ctx.stem, device='cuda')
    call_c, call_g = (getattr(e, ctx.sector['predict']) for e in (ctx.emu, gpu))
    a, b = call_c(ctx.X[:64]), call_g(ctx.X[:64])
    worst = 0.0
    for k in a:
        va, vb = (np.asarray(v[0] if isinstance(v, tuple) else v) for v in (a[k], b[k]))
        worst = max(worst, float(np.max(np.abs(va - vb) / np.maximum(np.abs(va), 1e-300))))
    return (PASS if worst <= rtol else FAIL), {'max_rel_diff': worst, 'tolerance': rtol}


def acceptance_gates(candidate, gate_set, base=None, alpha=0.05, max_degradation=0.01, n_ks=1000):
    """Promozione di un emulatore raffinato (11.4.5, 12.3): T-10.4 e T-10.5 sul gate
    set CONGELATO del base (<stem>_gate_set_frozen.npz), senza corpus.

    T-10.4 = la calibrazione di T-2.4 sul primario; T-10.5 = l'RMSE nello spazio del
    GP non peggiora oltre max_degradation rispetto al base (se dato). Il candidato
    deve avere il preprocessing del base: gli aggiornamenti L1-L3 non lo cambiano."""
    import torch
    from scipy.stats import kstest
    from galapy.spectroscopy.svgp import fast_predict
    z = np.load(gate_set)
    X, Yg = torch.as_tensor(z['X'], dtype=torch.float32), z['Y_gp']
    names, widths = list(z['families']), list(z['widths'])
    cols = dict(zip(names, np.split(np.arange(Yg.shape[1]), np.cumsum(widths)[:-1])))
    rng = np.random.default_rng(0)

    def score(emu):
        p, sq = [], []
        with torch.no_grad():
            for f in names:
                m, v = (t.T.numpy() for t in fast_predict(emu.cache[f], X))
                y = Yg[:, cols[f]]
                r = (y - m) / np.sqrt(np.maximum(v, 1e-30))
                sq.append(((y - m) ** 2).ravel())
                for j in range(r.shape[1]):
                    rj = r[:, j] if len(r) <= n_ks else rng.choice(r[:, j], n_ks, replace=False)
                    p.append(kstest(rj, 'norm').pvalue)
        return float(np.min(p)), float(np.sqrt(np.mean(np.concatenate(sq)))), len(p)

    p_min, rmse, n_out = score(candidate)
    gates = {'T-10.4_calibration': {'value': p_min, 'threshold': alpha / n_out,
                                    'pass': p_min > alpha / n_out}}
    if base is not None:
        rmse_b = score(base)[1]
        gates['T-10.5_no_degradation'] = {'value': rmse, 'base': rmse_b,
                                          'threshold': rmse_b * (1 + max_degradation),
                                          'pass': rmse <= rmse_b * (1 + max_degradation)}
    return {'gates': gates, 'all_pass': all(g['pass'] for g in gates.values()),
            'gate_set': str(gate_set), 'n_points': int(len(X))}


# =============================================================================
# 4. GATE HII (T-2.x, 2.5)
# =============================================================================

# Righe critiche di T-2.1: (specie, lambda in aria) come LINES di validate_grid.
CRITICAL_HII = {'H_alpha': ('H 1', 6562.8), 'H_beta': ('H 1', 4861.3),
                'OIII_5007': ('O 3', 5006.8), 'NII_6584': ('N 2', 6583.5),
                'OII_3726': ('O 2', 3726.0), 'OII_3729': ('O 2', 3728.8),
                'CIII_1909': ('Blnd', 1909.0)}
FRAC_5PCT_DEX = np.log10(1.05)        # "5% in log-space": RMSE di log10(pred/true)
FRAC_10PCT_DEX = np.log10(1.10)


def _line_index(ctx, name, table=CRITICAL_HII):
    return ctx.lines().find(*table[name])


def _hbeta_graded(ctx, rows):
    """Le righe si valutano dove la verita' supera 1e-3 Hbeta (soglia di T-1.13)."""
    from galapy.spectroscopy.utils.validate_grid import T as TG
    hb = ctx.Y['lines'][rows, _line_index(ctx, 'H_beta')]
    return TG['oiii.min_hbeta'] * hb


@gate('T-2.1', 'hii', title='righe critiche: RMSE(log10) sull\'holdout < 5%')
def t21_lines(ctx, rows=None):
    if _line_index(ctx, 'H_beta') is None:
        return SKIP, {'reason': 'H-beta non nella master list: nessuna soglia di flusso'}
    rows = np.arange(len(ctx.X)) if rows is None else rows
    pred = ctx.phys('lines', rows)[0]
    true = ctx.Y['lines'][rows]
    floor = _hbeta_graded(ctx, rows)
    out, worst = {}, 0.0
    for name in CRITICAL_HII:
        j = _line_index(ctx, name)
        if j is None:
            out[name] = 'assente dalla master list'
            continue
        g = (true[:, j] > 0) & (pred[:, j] > 0) & (true[:, j] >= floor)
        r = rmse_dex(pred[:, j], true[:, j], g)
        out[name] = {'n': int(g.sum()), 'rmse_dex': round(r, 5)}
        worst = max(worst, r) if np.isfinite(r) else worst
    return (PASS if worst < FRAC_5PCT_DEX else FAIL), {
        'threshold_dex': round(FRAC_5PCT_DEX, 5), 'worst_dex': round(worst, 5), 'lines': out}


@gate('T-2.2', 'hii', title='[O III] 4959/5007 emulato: criterio di T-1.13')
def t22_oiii(ctx):
    from galapy.spectroscopy.utils.validate_grid import T as TG
    j5, j4 = _line_index(ctx, 'OIII_5007'), ctx.lines().find('O 3', 4958.9)
    if j5 is None or j4 is None or _line_index(ctx, 'H_beta') is None:
        return SKIP, {'reason': '[O III] non nella master list'}
    pred, true = ctx.phys('lines')[0], ctx.Y['lines']
    g = (true[:, j5] >= _hbeta_graded(ctx, slice(None))) & (pred[:, j4] > 0) & (pred[:, j5] > 0)
    r_emu = pred[g, j4] / pred[g, j5]
    dev = np.abs(r_emu / TG['oiii.reference'] - 1.0)
    vs_true = np.abs(r_emu / (true[g, j4] / true[g, j5]) - 1.0)
    return (PASS if dev.max() < TG['oiii.max_rel_dev'] else FAIL), {
        'n': int(g.sum()), 'worst_rel_dev_from_reference': round(float(dev.max()), 5),
        'reference': TG['oiii.reference'], 'tolerance': TG['oiii.max_rel_dev'],
        'median_rel_dev_from_grid': round(float(np.median(vs_true)), 5)}


@gate('T-2.3', 'hii', title='decremento di Balmer emulato: criterio di T-1.1')
def t23_balmer(ctx):
    from galapy.spectroscopy.utils.validate_grid import T as TG
    ja, jb = _line_index(ctx, 'H_alpha'), _line_index(ctx, 'H_beta')
    if ja is None or jb is None:
        return SKIP, {'reason': 'H-alpha o H-beta non nella master list'}
    pred, true = ctx.phys('lines')[0], ctx.Y['lines']
    w = ((np.abs(ctx.col('logU') - TG['case_b.logU_centre']) <= TG['case_b.logU_half_width'])
         & (ctx.col('z_CMB') <= TG['case_b.max_z_cmb']) & (true[:, jb] > 0))
    if not w.any():
        return SKIP, {'reason': 'nessun punto dell\'holdout nella finestra di T-1.1'}
    r_emu, r_true = pred[w, ja] / pred[w, jb], true[w, ja] / true[w, jb]
    dev = abs(np.median(r_emu) / TG['case_b.reference'] - 1.0)
    pct = lambda v: [round(float(x), 4) for x in np.percentile(v, (5, 50, 95))]
    return (PASS if dev < TG['case_b.max_rel_dev'] else FAIL), {
        'n': int(w.sum()), 'median_emulated': round(float(np.median(r_emu)), 4),
        'relative_deviation': round(float(dev), 4), 'p5_50_95_emulated': pct(r_emu),
        'p5_50_95_grid': pct(r_true)}


@gate('T-2.4', 'hii', blocking=True, title='calibrazione della sigma SVGP (bloccante)')
def t24_calibration(ctx):
    return calibration(ctx)


@gate('T-2.5', 'hii', title='bilancio energetico emulato contro quello della griglia')
def t25_energy(ctx, n_max=2048, tol=0.01, min_fraction=0.95):
    """R = (int T L_inc + int L_neb) / int L_inc, in d ln(lambda) sulla griglia della
    SED incidente, emulato contro vero sullo stesso punto; T emulata in [0, 1]."""
    import h5py
    from galapy.spectroscopy.utils.validate_grid import T as TG
    rng = np.random.default_rng(ctx.args.seed)
    rows = np.sort(rng.choice(len(ctx.X), size=min(n_max, len(ctx.X)), replace=False))
    with h5py.File(ctx.args.corpus, 'r') as f:
        if 'incident/wave_grid' not in f:
            return SKIP, {'reason': 'il corpus non porta la SED incidente (incident/)'}
        w_inc = f['incident/wave_grid'][:].astype(float)
        L_inc = np.array([f[ctx.keys[i]]['incident/sed_per_Msun'][:] for i in rows], float)
    w = np.asarray(ctx.emu.wave_grid, float)
    o = np.argsort(w)
    lw, lwi = np.log(w[o]), np.log(w_inc)
    trap = getattr(np, 'trapezoid', None) or np.trapz

    def closure(T, Ln):
        Ti = np.array([np.interp(lwi, lw, t[o]) for t in T])
        Li = np.array([np.interp(lwi, lw, l[o], left=0.0, right=0.0) for l in Ln])
        E = trap(L_inc, lwi, axis=1)
        return (trap(Ti * L_inc, lwi, axis=1) + trap(Li, lwi, axis=1)) / E
    T_emu, L_emu = ctx.phys('trans', rows)[0], ctx.phys('cont', rows)[0]
    R_emu = closure(T_emu, L_emu)
    R_true = closure(ctx.Y['trans'][rows], ctx.Y['cont'][rows])
    dev = np.abs(R_emu / R_true - 1.0)
    frac = float(np.mean(dev <= tol))
    t_tol = TG['transparency.tolerance']
    t_ok = bool(T_emu.min() >= -t_tol and T_emu.max() <= 1.0 + t_tol)
    return (PASS if frac >= min_fraction and t_ok else FAIL), {
        'n': int(rows.size), 'fraction_within_1pct': round(frac, 4),
        'rel_dev_p50_p95_p99': [round(float(x), 5) for x in np.percentile(dev, (50, 95, 99))],
        'T_min': float(T_emu.min()), 'T_max': float(T_emu.max()), 'T_tolerance': t_tol}


@gate('T-2.6', 'hii', title='massa di polvere: RMSE(log10) < 10% e scaling di T-1.14')
def t26_dust(ctx):
    from galapy.spectroscopy.utils.validate_grid import T as TG, _chemistry
    pred, true = ctx.phys('dust')[0][:, 0], ctx.Y['dust'][:, 0]
    g = (pred > 0) & (true > 0)
    r = rmse_dex(pred, true, g)
    try:
        chem = _chemistry()
    except Exception as exc:  # noqa: BLE001 - come validate_grid: non disponibile -> SKIP
        return SKIP, {'reason': f'modulo di chimica non disponibile: {exc}', 'rmse_dex': r}
    q_h = ctx.emu.Qh_unit_lookup(ctx.X)
    dust = ctx.col('xi_d') * np.array([chem.metal_mass_per_H(v) for v in ctx.col('log_zeta_O')])
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.log10(pred) - (np.log10((1.0 - ctx.col('f_esc_meas')) * dust * q_h)
                                - ctx.col('lognH_HII'))
    ok = np.isfinite(res)
    dev = res[ok] - np.median(res[ok])
    frac = float(np.mean(np.abs(dev) <= TG['dust.max_abs_residual']))
    return (PASS if r < FRAC_10PCT_DEX and frac >= TG['dust.min_pass_fraction'] else FAIL), {
        'rmse_dex': round(r, 5), 'threshold_dex': round(FRAC_10PCT_DEX, 5),
        'scaling_fraction_within_0.2dex': round(frac, 4), 'n': int(ok.sum())}


@gate('T-2.7', 'hii', title='sigma cresce fuori dominio (OOD)')
def t27_ood(ctx):
    return ood_growth(ctx)


@gate('T-2.8', 'hii', title='throughput batch-8 su CPU >= 100 eval/s a 8 thread')
def t28_throughput(ctx):
    res = throughput(ctx)
    return (PASS if res['eval_per_s_8_threads'] >= 100 else FAIL), {
        k: round(v, 1) for k, v in res.items()}


@gate('T-2.9', 'hii', title='learning curve e scelta di M')
def t29_learning(ctx):
    if not ctx.args.learning_curve:
        return SKIP, {'reason': 'richiede --learning-curve (riaddestra: ore su HPC)'}
    return learning_curve(ctx)


@gate('T-2.10', 'hii', title='footprint dei file e RAM a runtime')
def t210_footprint(ctx):
    return footprint(ctx)


# Righe degli scan di T-2.11 (T-1.17, T-1.18).
SCAN_HII = {'SII_6716': ('S 2', 6716.4), 'SII_6731': ('S 2', 6730.8),
            'SIII_9069': ('S 3', 9068.6), 'SIII_9531': ('S 3', 9530.6),
            'OI_6300': ('O 1', 6300.3), 'H_alpha': ('H 1', 6562.8),
            'OIII_5007': ('O 3', 5006.8), 'OII_3726': ('O 2', 3726.0), 'OII_3729': ('O 2', 3728.8)}


def _scan(ctx, axis, n_points=20, n_steps=5):
    """Scan emulato su un asse a theta fissato: log10 dei rapporti e loro sigma."""
    rng = np.random.default_rng(ctx.args.seed)
    i = ctx.emu.input_keys.index(axis)
    lo = np.asarray(ctx.meta.get('x_min') or ctx.X.min(0), float)[i]
    hi = np.asarray(ctx.meta.get('x_max') or ctx.X.max(0), float)[i]
    base = ctx.X[rng.choice(len(ctx.X), size=min(n_points, len(ctx.X)), replace=False)]
    X = np.repeat(base, n_steps, axis=0)
    X[:, i] = np.tile(np.linspace(lo, hi, n_steps), len(base))
    mu, var = predict_gp(ctx.emu, X, chunk=4096)
    mean, sig = physical(ctx.emu, 'lines', mu['lines'], var['lines'], X)
    cat = ctx.lines()
    idx = {k: cat.find(*v) for k, v in SCAN_HII.items()}
    s = lambda *names: sum(mean[:, idx[n]] for n in names)
    sl = lambda *names: np.sqrt(sum((sig[:, idx[n]] / mean[:, idx[n]]) ** 2 for n in names)) / np.log(10)
    ratios = {'SII/Ha': (np.log10(s('SII_6716', 'SII_6731') / s('H_alpha')),
                         sl('SII_6716', 'SII_6731', 'H_alpha')),
              'SIII/Ha': (np.log10(s('SIII_9069', 'SIII_9531') / s('H_alpha')),
                          sl('SIII_9069', 'SIII_9531', 'H_alpha')),
              'OI/Ha': (np.log10(s('OI_6300') / s('H_alpha')), sl('OI_6300', 'H_alpha')),
              'O32': (np.log10(s('OIII_5007') / s('OII_3726', 'OII_3729')),
                      sl('OIII_5007', 'OII_3726', 'OII_3729'))}
    return {k: (v.reshape(len(base), n_steps), e.reshape(len(base), n_steps))
            for k, (v, e) in ratios.items()}


def _monotone_fraction(ratio, err, sign):
    """Frazione di scan monotoni nel verso atteso entro le barre GP: un passo nel
    verso sbagliato e' ammesso se sta sotto la sigma combinata dei due punti."""
    step = np.diff(ratio, axis=1) * sign
    tol = np.sqrt(err[:, 1:] ** 2 + err[:, :-1] ** 2)
    return float(np.mean(np.all(step >= -tol, axis=1)))


@gate('T-2.11', 'hii', title='sanita\' ARD e monotonie di T-1.17/T-1.18 emulate')
def t211_ard(ctx, min_fraction=0.9, max_ls_ratio=10.0):
    missing = [k for k, v in SCAN_HII.items() if ctx.lines().find(*v) is None]
    if missing:
        return SKIP, {'reason': f'righe degli scan non nella master list: {missing}'}
    keys = ctx.emu.input_keys
    # lengthscale in unita' della larghezza del range di training di ciascun asse:
    # e' la scala su cui "comparabili" ha senso fra assi con unita' diverse
    lo = np.asarray(ctx.meta.get('x_min') or ctx.X.min(0), float)
    hi = np.asarray(ctx.meta.get('x_max') or ctx.X.max(0), float)
    ls = (ctx.emu.cache['lines']['lengthscale'][:, 0, :].numpy() * ctx.emu.x_std.numpy()
          / np.where(hi > lo, hi - lo, 1.0))
    cat = ctx.lines()
    active = [keys.index(k) for k in ('logU', 'lognH_HII', 'log_zeta_O')]
    f_star = keys.index('F_star')
    sulfur = [cat.find(*SCAN_HII[k]) for k in ('SII_6716', 'SII_6731', 'SIII_9069', 'SIII_9531')]
    ratio_ls = [float(ls[j, f_star] / np.median(ls[j, active])) for j in sulfur if j is not None]
    a_ok = bool(ratio_ls) and all(np.isfinite(r) and r <= max_ls_ratio for r in ratio_ls)
    expect = {'F_star': {'SII/Ha': -1, 'SIII/Ha': -1},
              'f_esc_meas': {'O32': +1, 'SII/Ha': -1, 'OI/Ha': -1}}
    scans, ok = {}, a_ok
    for axis, want in expect.items():
        sc = _scan(ctx, axis)
        scans[axis] = {k: round(_monotone_fraction(*sc[k], s), 3) for k, s in want.items()}
        ok &= all(v >= min_fraction for v in scans[axis].values())
    xi = _scan(ctx, 'xi_d')          # nessun verso atteso dichiarato: si riporta
    scans['xi_d (riportato)'] = {k: round(float(np.median(v[:, -1] - v[:, 0])), 4)
                                 for k, (v, _e) in xi.items()}
    return (PASS if ok else FAIL), {'lengthscale_ratio_Fstar_over_active_sulfur': ratio_ls,
                                    'max_ratio': max_ls_ratio, 'monotone_fraction': scans,
                                    'min_fraction': min_fraction}


@gate('T-2.12', 'hii', title='TF32: ri-supera T-2.4 e RMSE entro 1% del riferimento')
def t212_tf32(ctx):
    if not ctx.args.reference:
        return SKIP, {'reason': 'richiede --reference <stem fp32> (artefatto valutato: TF32)'}
    s_cal, d_cal = calibration(ctx)
    ref = ctx.sector['emulator_cls']().load(ctx.args.reference)
    mu_r, _ = predict_gp(ref, ctx.X)
    f = ctx.sector['lines']
    rm = lambda m: float(np.sqrt(np.mean((ctx.Yz[f] - m) ** 2)))
    r_tf32, r_ref = rm(ctx.mu[f]), rm(mu_r[f])
    ok = s_cal == PASS and r_tf32 <= 1.01 * r_ref
    return (PASS if ok else FAIL), {'calibration': s_cal, 'min_p': d_cal.get('min_p'),
                                    'rmse_z_tf32': r_tf32, 'rmse_z_reference': r_ref}


@gate('T-2.13', 'hii', title='T-2.1 e T-2.4 per bin di tau')
def t213_bins(ctx):
    return per_node_bin(ctx, t21_lines)


@gate('T-2.14', 'hii', title='interpolazione in Z_star fra i nodi (run CLOUDY a meta\')')
def t214_midnode(ctx):
    if not ctx.args.midnode_corpus:
        return SKIP, {'reason': 'richiede --midnode-corpus (50 run CLOUDY a meta\' fra nodi)'}
    from scipy.stats import kstest
    T = ctx.sector['train_mod']
    Xm, Ym, _ = T.load_corpus(ctx.args.midnode_corpus)
    mu, var = predict_gp(ctx.emu, Xm)
    f = ctx.sector['lines']
    z = encode_targets(ctx.emu.prep[f], Ym[f], log_scale_of(ctx.emu, f, Xm))
    r = ((z - mu[f]) / np.sqrt(np.maximum(var[f], 1e-30))).ravel()
    p = float(kstest(r[np.isfinite(r)], 'norm').pvalue)
    pred = physical(ctx.emu, f, mu[f], var[f], Xm)[0]
    good = (Ym[f] > 0) & (pred > 0)
    r_mid = rmse_dex(pred, Ym[f], good)
    hold = ctx.phys(f)[0]
    r_hold = rmse_dex(hold, ctx.Y[f], (ctx.Y[f] > 0) & (hold > 0))
    ok = p > 0.05 and r_mid <= 1.5 * r_hold
    return (PASS if ok else FAIL), {'n': int(len(Xm)), 'ks_p': p, 'rmse_dex_midnode': r_mid,
                                    'rmse_dex_holdout': r_hold, 'max_ratio': 1.5}


@gate('T-2.15c', 'hii', title='stesse predizioni su GPU e su CPU')
def t215c_gpu(ctx):
    return gpu_matches_cpu(ctx)


# =============================================================================
# 5. ESECUZIONE E REPORT
# =============================================================================

def run_gates(ctx, only=None):
    report = {}
    for gid, (fn, sector, blocking, title) in GATES.items():
        if sector != ctx.args.sector or (only and gid not in only):
            continue
        t0 = time.perf_counter()
        try:
            status, detail = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - un gate che alza e' un ERROR, non un crash
            status, detail = ERROR, {'reason': f"{type(exc).__name__}: {exc}",
                                     'traceback': traceback.format_exc(limit=3)}
        report[gid] = {'status': status, 'blocking': blocking, 'title': title,
                       'seconds': round(time.perf_counter() - t0, 2), 'detail': detail}
        print(f"[validate-emulator] {gid:8s} {status:5s} {title}", flush=True)
    return report


def render_markdown(rep):
    rows = [f"# Validazione emulatore {rep['sector'].upper()}", '',
            f"Artefatto `{rep['emulator']}`; holdout {rep['holdout'].get('n_evaluated')} punti "
            f"di {rep['holdout'].get('n_test')} ({rep['holdout'].get('n_clouds')} nubi).", '',
            '| Gate | Stato | Bloccante | Che cosa | Sintesi |', '|---|---|---|---|---|']
    for gid, g in rep['gates'].items():
        d = g['detail']
        summ = d.get('summary') or d.get('reason') or ''
        rows.append(f"| {gid} | {g['status']} | {'si' if g['blocking'] else ''} | {g['title']} | {summ} |")
    rows += ['', '```json', json.dumps(rep['gates'], indent=1, default=str), '```']
    return '\n'.join(rows) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sector', required=True, choices=('hii', 'pdr'))
    ap.add_argument('--emulator', help='stem dell\'artefatto (default data/emulators/<s>)')
    ap.add_argument('--meta', help='meta del training (default <stem>_meta.json)')
    ap.add_argument('--corpus', help='corpus h5 (default data/products/<s>_grid.h5)')
    ap.add_argument('--holdout', help='chiavi dell\'holdout (default <stem>_holdout.json)')
    ap.add_argument('--report', help='report JSON (default data/validation/report_emulator_<s>.json)')
    ap.add_argument('--markdown', help='report Markdown (default: il JSON con .md)')
    ap.add_argument('--gates', help='solo questi gate, separati da virgole')
    ap.add_argument('--max-points', type=int, default=8192,
                    help='punti dell\'holdout valutati (0 = tutti; 8192 = taglia del gate set)')
    ap.add_argument('--n-ks', type=int, default=1000, help='residui per output nel KS di T-2.4')
    ap.add_argument('--noise-diagnostic', action='store_true',
                    help='T-2.4: riporta anche la copertura con il rumore della likelihood')
    ap.add_argument('--midnode-corpus', help='corpus dei run a meta\' fra nodi di Z_star (T-2.14)')
    ap.add_argument('--reference', help='stem dell\'artefatto di riferimento fp32 (T-2.12)')
    ap.add_argument('--learning-curve', action='store_true', help='esegue T-2.9 / T-3.17f')
    ap.add_argument('--lc-M', type=int, nargs='+', default=[256, 512, 1024])
    ap.add_argument('--lc-fractions', type=float, nargs='+', default=[0.25, 0.5, 1.0])
    ap.add_argument('--lc-epochs', type=int, default=None)
    ap.add_argument('--lc-batch', type=int, default=1024)
    ap.add_argument('--lc-family', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--strict', action='store_true', help='un gate bloccante in SKIP fa fallire')
    a = ap.parse_args(argv)
    s = a.sector
    a.emulator = a.emulator or f"data/emulators/{s}"
    a.meta = a.meta or f"{a.emulator}_meta.json"
    a.corpus = a.corpus or f"data/products/{s}_grid.h5"
    a.holdout = a.holdout or f"{a.emulator}_holdout.json"
    a.report = a.report or f"data/validation/report_emulator_{s}.json"
    a.markdown = a.markdown or str(pathlib.Path(a.report).with_suffix('.md'))

    ctx = Context(a)
    gates = run_gates(ctx, set(a.gates.split(',')) if a.gates else None)
    rep = {'sector': s, 'emulator': a.emulator, 'corpus': a.corpus, 'holdout': ctx.holdout,
           'gates': gates, 'summary': {st: sum(g['status'] == st for g in gates.values())
                                       for st in (PASS, FAIL, SKIP, ERROR)}}
    pathlib.Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.report).write_text(json.dumps(rep, indent=1, default=str))
    pathlib.Path(a.markdown).write_text(render_markdown(rep))
    bad = [g for g, v in gates.items() if v['blocking']
           and (v['status'] in (FAIL, ERROR) or (a.strict and v['status'] == SKIP))]
    print(f"[validate-emulator] {rep['summary']} -> {a.report}")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())