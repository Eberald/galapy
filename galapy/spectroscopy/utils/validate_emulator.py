# Author: Enrico Veraldi
# Holdout (validation) and emulators Gates
"""
This file:
  1. Define the holdouts (split training/test). holdout_split, encode_targets, holdout_scores,
     write_holdout and build_frozen_set. Called by training files
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

import galapy.internal.emulators as EMUL

PASS, FAIL, SKIP, ERROR = 'PASS', 'FAIL', 'SKIP', 'ERROR'


#========================================== HOLDOUT = defined by training, here readed by validation

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
        floored to the greater of `EMUL.LOG_FLOOR` or a relative cutoff when `rel_floor` is
        specified.

    Raises:
    ValueError
        If `rel_floor` is specified as an array and its shape is not compatible with
        the input array when `Y` is 2-dimensional.
    """
    Y = np.asarray(Y, dtype=np.float64) + float(offset)
    # minimum admitted value in log
    floor = EMUL.LOG_FLOOR
    rel_floor = np.asarray(rel_floor, dtype=np.float64)
    # if rel floor is positive, use it as a relative floor for each row (e.g. per wavelength)
    if np.any(rel_floor > 0) and Y.ndim == 2:
        floor = np.maximum(rel_floor * Y.max(axis=1, keepdims=True), EMUL.LOG_FLOOR)
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
    """
    Compute the mean and variance of the a posteriori distribution of a Gaussian Process (GP) for each input
    point, excluding likelihood noise. The input points are standardized before computation. The function
    processes the input in chunks to manage memory constraints.

    Parameters:
    model : object
        A fitted GP model that supports evaluation using standardized inputs.
    Xz : array_like
        The standardized input data points.
    device : str, optional
        Device on which to perform computations ('cpu' or 'cuda').
    chunk : int, optional
        The size of chunks used for batch processing of input data.

    Returns:
    tuple
        A tuple containing the mean and variance of the GP's a posteriori distribution for the input points.
        Both returned arrays have a shape of (N, d), where N is the number of input points, and d is the
        dimensionality of the GP output.
    """
    import torch
    model.eval()          # model put in evaluation mode
    mu, var = [], []
    with torch.no_grad():                      # no gradient computation, we do not need it (memory saving)
        for i in range(0, len(Xz), chunk):     # working on the input points in chunks (matrix size of GP)
            post = model(torch.as_tensor(Xz[i:i + chunk], dtype=torch.float32, device=device))  # model posterior
            mu.append(post.mean.cpu().numpy())
            var.append(post.variance.cpu().numpy())
    return np.concatenate(mu, axis=1).T, np.concatenate(var, axis=1).T


def holdout_scores(model, Xz, Yz, device='cpu', chunk=1024):
    """
    Compute performance metrics for a predictive model on a holdout dataset.
    This function evaluates prediction error, coverage, and explained variance
    for model predictions.

    Parameters:
    model: The predictive model used for generating predictions.
    Xz: Input features of the holdout dataset.
    Yz: Ground truth target values corresponding to Xz.
    device: Specifies the device for computation, default is 'cpu'.
    chunk: Batch size for splitting computations, default is 1024.

    Returns:
    dict: A dictionary with the following keys:
        - 'rmse_z': Root Mean Squared Error (RMSE) of the predictions.
        - 'cov68': Prediction coverage at 68% confidence interval.
        - 'explained': Explained variance of the model predictions.
    """
    # no holdout case (no test set)
    if len(Xz) == 0:
        return {'rmse_z': float('nan'), 'cov68': float('nan'), 'explained': float('nan')}
    mu, var = latent_predict(model, Xz, device, chunk)
    Yz = np.asarray(Yz)
    r = Yz - mu
    sd = np.sqrt(var)
    # explained = 1 - MSE / MSE of the constant predictor (z = 0, the training mean): an R^2
    # averaged over the outputs with equal weight
    return {'rmse_z': float(np.sqrt(np.mean(r ** 2))), 'cov68': float(np.mean(np.abs(r) <= sd)),
            'explained': float(1.0 - np.mean(r ** 2) / max(np.mean(Yz ** 2), 1e-30))}


def per_output_scores(model, Xz, Yz, device='cpu', chunk=1024):
    """
    Calculate holdout scorers for each output separately.

    This function computes error metrics for model predictions on holdout data,
    assessing the model's performance for each output dimension independently.
    The calculated metrics include the root mean square error (RMSE) and the
    coefficient of determination (R2 score) for each output.

    Parameters:
        model: Any
            The trained model used for predictions.
        Xz: Any
            Input data used for generating predictions.
        Yz: Any
            Ground truth output data corresponding to the input data.
        device: str
            Device identifier for computation (e.g., 'cpu' or 'cuda').
        chunk: int
            Chunk size to process input data in segments for efficient computation.

    Returns:
        tuple:
            A tuple containing the following:
            - Array of RMSE values for each output dimension.
            - Array of R2 scores for each output dimension.
    """
    mu, _ = latent_predict(model, Xz, device, chunk)
    Yz = np.asarray(Yz)
    mse = np.mean((Yz - mu) ** 2, axis=0)
    ref = np.maximum(np.mean(Yz ** 2, axis=0), 1e-30)
    return np.sqrt(mse), 1.0 - mse / ref


def calibration_scale(model, Xz, Yz, device='cpu', s_min=1.0, chunk=1024, quantile=None):
    """
    Post-hoc calibration of the predictive sigma of a GP family: one scale factor s_k per output.

    The SVGP sigma is not guaranteed to be honest. When it enters the likelihood of a fit as model
    error, a sigma that is too small gives posteriors that are too narrow and possibly biased.
    Rescaling sigma fixes the stated error without touching the mean.

    For each calibration point i and output k (in the GP space: z-scored, whitened PCA scores),
    the normalized residual is r_ik = (y_ik - mu_ik) / sigma_ik. If sigma is honest r_k ~ N(0, 1);
    if the true error is s_k times the stated one, r_k ~ N(0, s_k^2). For r ~ N(0, s^2) the
    q-quantile of |r| is s * z_q with z_q = Phi^-1((1 + q) / 2), hence the estimator

        s_k = quantile_q(|r_ik|) / z_q ,

    after which a fraction q of the calibration points lies within z_q sigma.

    Which quantile. The residuals have heavy tails (peaked core, a few % beyond 5 sigma), so no
    single factor gives both 68% within 1 sigma and 95% within 1.96 sigma. On the HII holdout:
      - q = 0.5 (median / 0.6745): cov68 ~ 0.65 but cov95 only 0.80-0.89 and up to 3% beyond
        5 sigma: the tails, where the emulator is wrong, stay overconfident;
      - q = 0.95 (default): cov95 = 0.95 in every family and 0.1-0.6% beyond 5 sigma, with a
        conservative core (cov68 0.77-0.91).
    For a model error in the likelihood an overconfident tail biases the posterior, a conservative
    core only widens it a little: hence q = 0.95.

    s_k is clipped from below at s_min (default 1): the calibration only ever WIDENS sigma.
    The variance is multiplied by s_k^2 at prediction time (svgp.fast_predict, via
    prep['sigma_scale']); the PCA decoding propagates it to physical units as before.

    Limits: one number per output, so it corrects the average over the input space, not regions
    where the GP is locally worse; and it does not remove the tails, it makes them rarer.

    Parameters:
        model: trained BatchedSVGPS of the family (d outputs).
        Xz (np.ndarray): standardized inputs of the calibration set, (N, D).
        Yz (np.ndarray): targets of the calibration set in the GP space (encode_targets), (N, d).
        s_min (float): lower clip of s_k (1 = never shrink sigma).
        quantile (Optional[float]): q of the estimator, in (0, 1). Default EMUL.CALIB_QUANTILE (0.95).

    Returns:
        np.ndarray: s_k, (d,).
    """
    from statistics import NormalDist
    q = EMUL.CALIB_QUANTILE if quantile is None else float(quantile)
    if not 0.0 < q < 1.0:
        raise ValueError(f"calibration_scale: quantile {q} must be in (0, 1)")
    mu, var = latent_predict(model, Xz, device, chunk)                  # prediction
    r = (np.asarray(Yz) - mu) / np.sqrt(np.maximum(var, 1e-30))         # normalised residuals
    s = np.quantile(np.abs(r), q, axis=0) / NormalDist().inv_cdf(0.5 * (1.0 + q))
    return np.maximum(s, s_min)                                         # clip, with s_min=1 never narrowed sigma


def coverage_scores(model, Xz, Yz, scale=None, device='cpu', chunk=1024):
    """
    Calculates coverage metrics for predictions and normalized residuals of a model.

    This function computes the percentage of predictions falling within
    one standard deviation (68% coverage) and the fraction of predictions
    having residuals greater than 5 standard deviations (outliers). The
    residuals can be optionally calibrated by scaling.

    Args:
        model: The predictive model from which latent predictions are generated.
        Xz: The input data used for predictions by the model.
        Yz: The true output values to compare against the model's predictions.
        scale: Optional; array of scaling factors for residual calibration.
        device: The device type ('cpu' or 'gpu') for computation.
        chunk: Number of samples processed per chunk during predictions
            (used for memory optimization).

    Returns:
        dict: Dictionary with three keys:
            - 'cov68': Fraction of residuals within 1 standard deviation.
            - 'cov95': Fraction of residuals within 1.96 standard deviations.
            - 'out5': Fraction of residuals greater than 5 standard deviations.
    """
    # predictions and normalized residuals
    mu, var = latent_predict(model, Xz, device, chunk)
    r = (np.asarray(Yz) - mu) / np.sqrt(np.maximum(var, 1e-30))
    # application of calibration (optional)
    if scale is not None:
        r = r / np.asarray(scale)
    return {'cov68': float(np.mean(np.abs(r) <= 1.0)), 'cov95': float(np.mean(np.abs(r) <= 1.96)),
            'out5': float(np.mean(np.abs(r) > 5.0))}


def write_holdout(path, keys, frac, seed, axes_keys, corpus=None, n_models=None):
    """
    save <stem>_holdout.json with the models not used for training and how the holdout
    has been selected for reproducibility.

    Parameters:
    path: str
        The file path where the holdout JSON file will be saved.
    keys: list[str]
        A list of model keys to be used for creating the holdout dataset.
    frac: float
        The fraction of the total dataset reserved for the holdout.
    seed: int
        The random seed value used for reproducibility of the holdout creation process.
    axes_keys: list[str]
        A list of axis keys on which stratification is applied for holdout selection.
    corpus: str or None, optional
        The name of the corpus (if any) related to the models in the holdout dataset.
    n_models: int or None, optional
        The total number of models available in the specified corpus.

    Returns:
    str
        The path of the JSON file where the holdout data was saved.

    Raises:
    None
    """
    keys = [str(k) for k in keys]
    payload = {'keys': keys, 'n_test': len(keys), 'n_clouds': len({cloud_id(k) for k in keys}),
               'frac': frac, 'seed': seed, 'stratified_on': list(axes_keys),
               'corpus': str(corpus) if corpus else None, 'n_models_corpus': n_models}
    pathlib.Path(path).write_text(json.dumps(payload, indent=1))
    return str(path)


def read_holdout(path):
    """
    Read the holdout data from a JSON file.

    Args:
        path (str): The path to the JSON file.

    Returns:
        dict: The holdout data loaded from the JSON file.
    """
    return json.loads(pathlib.Path(path).read_text())


def build_frozen_set(path, X, Y_gp, Y_phys, axes_idx, n_primary=8192,
                          n_physical=512, seed=42, families=None):
    """
    Creates and saves a subset of data extracted from the given dataset in a compressed .npz file
    format. The subset includes points distributed across octants in the GP input space and a further
    subset for validation or testing consistency. The function is particularly useful for evaluating
    emulator calibration and error distribution on physical observables.

    This function ensures a balanced representation by assigning points to octants based on median
    splits along specified axes and ensures additional random selection for unbalanced distributions
    or filling insufficient selections. It also allows optional inclusion of family and width data.

    Parameters:
        path (str): File path where the .npz file will be saved.
        X (numpy.ndarray): Input data array. Each row represents a data point.
        Y_gp (numpy.ndarray): Target data corresponding to the input X in GP space.
        Y_phys (numpy.ndarray): Target data corresponding to physical observables.
        axes_idx (sequence of int): Indices of axes to define octants in GP input space.
        n_primary (int, optional): Number of points to include in the primary subset. Default is 8192.
        n_physical (int, optional): Number of points to include in the physical subset. Default is 512.
        seed (int, optional): Random seed for reproducibility. Default is 42.
        families (list of tuples, optional): List of (family, width) mapping for metadata. Default is None.

    Returns:
        str: The path to the saved .npz file.

    Raises:
        ValueError: If input arrays X, Y_gp, or Y_phys have inconsistent shapes.
    """
    # random seed for extraction of sub-group
    rng = np.random.default_rng(seed)
    n_primary = min(n_primary, X.shape[0])
    # assign every point to an octant for the extraction
    A = X[:, list(axes_idx)]
    octant = ((A > np.median(A, axis=0)) * np.array([1, 2, 4])).sum(axis=1)
    sel = []
    for o in range(8):
        idx = np.flatnonzero(octant == o)
        if idx.size:
            sel.append(rng.choice(idx, size=min(n_primary // 8, idx.size), replace=False))
    sel = np.sort(np.concatenate(sel)) if sel else np.array([], int)
    # in case unbalanced octants: complete randomly
    if sel.size < n_primary:
        rest = np.setdiff1d(np.arange(X.shape[0]), sel)
        sel = np.sort(np.concatenate([sel, rng.choice(rest, size=n_primary - sel.size,
                                                      replace=False)]))

    # physical sub-sample
    phys = np.sort(rng.choice(sel, size=min(n_physical, sel.size), replace=False))
    fam = families or []
    np.savez_compressed(
        path, idx_primary=sel, idx_physical=phys,
        X=X[sel], Y_gp=Y_gp[sel], X_phys=X[phys], Y_phys=Y_phys[phys],
        seed=seed, axes_idx=np.asarray(axes_idx),
        families=np.array([f for f, _ in fam]), widths=np.array([w for _, w in fam], int))
    return path


#============================================= SECTORS (HII or PDR)

SECTORS = {
    'hii': {'train': 'galapy.spectroscopy.utils.hii.train_emulator_hii',
            'emulator': ('galapy.spectroscopy.hii_emulator', 'HIIEmulator'),
            'predict': 'predict_batch',
            'parts': ('lines', 'cont', 'grain', 'trans', 'scalars'),
            'lines': 'lines'},
}


def _sector(name):
    """
    Fetches and processes sector-specific configuration and components.

    This function retrieves a dictionary of configuration details for a specified
    sector by its name. It dynamically imports the specified modules for training
    and emulation and appends the corresponding class definitions to the config
    dictionary for further use.

    Parameters:
        name (str): The name of the sector whose configuration is to be loaded.

    Returns:
        dict: A dictionary containing the sector configuration and dynamically
            imported components.
    """
    s = dict(SECTORS[name])
    s['train_mod'] = importlib.import_module(s['train'])
    mod, cls = s['emulator']
    s['emulator_cls'] = getattr(importlib.import_module(mod), cls)
    return s

################# CONTEXT CLASS ##################
class Context:
    """
    Load emulator+holdout+predictions, so that are avaiable for the tests of validate_emulator
    """

    def __init__(self, args):
        # sector (hii/pdr) and emulator
        self.args = args
        self.sector = _sector(args.sector)
        self.stem = str(args.emulator)
        self.emu = self.sector['emulator_cls']().load(self.stem)
        # metadata json of training
        self.meta = json.loads(pathlib.Path(args.meta).read_text()) if args.meta else {}
        # holdout and eventually sub-sample
        hold = read_holdout(args.holdout)
        keys = hold['keys']
        if args.max_points and len(keys) > args.max_points:
            rng = np.random.default_rng(args.seed)
            keys = sorted(rng.choice(keys, size=args.max_points, replace=False).tolist())
        # holdout json metadata
        self.holdout = {k: hold[k] for k in hold if k != 'keys'}
        self.holdout['n_evaluated'] = len(keys)
        # corpus .h5 file of CLOUDY points
        self.X, self.Y, self.corpus_meta = self.sector['train_mod'].load_corpus(args.corpus, keys=keys)
        self.keys = self.corpus_meta['keys']
        # predictions
        self.mu, self.var = predict_gp(self.emu, self.X)
        # targets (observations true) on GP space
        self.Yz = {f: encode_targets(self.emu.prep[f], self.Y[f], log_scale_of(self.emu, f, self.X))
                   for f in self.mu}
        # eventually noise
        self._noise = None

    def col(self, key):
        """
        Fetches a specific column from the dataset based on its key.
        Ex: context.col("logU")

        Parameters:
        key : str
            The string identifying the key of the column to extract.

        Returns:
        numpy.ndarray
            The column of the dataset corresponding to the provided key.
        """
        return self.X[:, self.emu.input_keys.index(key)]

    def phys(self, fam, idx=None):
        """
        Reports the median and variance of the specified family in physical units. This is achieved
        by decoding the family data and returning the computed median and standard deviation (sigma).

        Parameters:
        fam: str
            The identifier of the family for which the median and variance should be computed.
        idx: int, optional
            The index of the specific member within the family. If None, calculations are performed
            for the entire family.

        Returns:
        tuple
            A tuple containing the median and sigma (standard deviation) in physical units.
        """
        return physical(self.emu, fam, self.mu[fam] if idx is None else self.mu[fam][idx],
                        self.var[fam] if idx is None else self.var[fam][idx],
                        self.X if idx is None else self.X[idx])

    def noise(self):
        """
        Returns the noise value of the GaussianLikelihood for each family in the diagnostic.

        This method computes the noise value for the GaussianLikelihood associated with
        the specified diagnostic and its corresponding families. The computation is
        performed using a utility function if the noise value has not been previously
        calculated.

        Returns:
            float: The computed noise value.
        """
        if self._noise is None:
            self._noise = likelihood_noise(self.stem, self.sector['parts'])
        return self._noise

    def lines(self):
        """
        Create a catalog of lines consisting of names and wavelengths.

        Returns
        -------
        LineCatalogue
            An instance of `LineCatalogue` containing the line names and rest-frame
            wavelengths.
        """
        from galapy.spectroscopy.utils.validate_grid import LineCatalogue
        return LineCatalogue(self.emu.line_names, self.emu.line_wavelengths_rest)
##################################################

def predict_gp(emu, X, chunk=1024):
    """
    Predict latent mean and variance for each family of the emulator on physical inputs.

    This function computes the latent mean and variance of a Gaussian Process (GP) emulator
    for each family (e.g., lines, continuum, etc.) using the provided physical inputs X.
    The computation is performed in chunks of point blocks to optimize memory usage. It
    utilizes pre-computed matrices for efficiency instead of relying on a torch-based model.
    The result includes dictionaries containing the mean and variance for each family, each
    with dimensions of (N, d), where N is the number of input samples and d is the dimension
    of the output for each family.

    Parameters:
      emu: The emulator object, which contains families and cached matrices required
           for prediction.
      X: ndarray
           Input array of physical parameters with shape (N, p), where N is the number
           of samples and p is the number of input features.
      chunk: int, optional
           Number of samples to process in each chunk for memory efficiency. Default is 1024.

    Returns:
      tuple[dict[str, ndarray], dict[str, ndarray]]
           A tuple containing two dictionaries:
             - The first dictionary maps each family name to its corresponding latent means
               with dimensions (N, d).
             - The second dictionary maps each family name to its corresponding latent
               variances with dimensions (N, d).
    """
    import torch
    from galapy.spectroscopy.svgp import fast_predict
    mu, var = {}, {}
    x = (torch.as_tensor(X, dtype=torch.float32) - emu.x_mean) / emu.x_std # normalise input
    with torch.no_grad():
        for fam, cache in emu.cache.items():
            m, v = zip(*(fast_predict(cache, x[i:i + chunk]) for i in range(0, len(x), chunk))) #(m1,v2) -> (m1, m2..), (v1, v2..)
            mu[fam] = torch.cat(m, dim=1).T.numpy().astype(float)   # (d,N), then transpose (N,d), float64
            var[fam] = torch.cat(v, dim=1).T.numpy().astype(float)
    return mu, var

def log_scale_of(emu, fam, X):
    """
    Calculate the base-10 logarithm of the analytic scale for encoding and decoding the
    specified family. The result depends on the emulator object and the provided family
    and input values. If the emulator does not have a 'log_scale' attribute, or if the
    input is None, the function returns None.

    Parameters:
    emu : object
        The emulator object, which may optionally define a 'log_scale' attribute
        representing the scaling function.
    fam : Any
        The family identifier for which the scale should be calculated, used as an
        input to the scaling function if applicable.
    X : Any
        The input value or parameter passed to the scaling function in combination
        with the family identifier.

    Returns:
    float or None
        Returns the calculated log10 scaling value if both the 'log_scale' attribute
        of the emulator and the input value X are provided. Returns None otherwise.
    """
    f = getattr(emu, 'log_scale', None)
    return None if f is None or X is None else f(fam, X)


def physical(emu, fam, mu, var, X=None):
    """
    Performs physical decoding of input parameters.

    This function computes the physical properties (e.g., median and sigma) for given inputs
    using Gaussian Process decoding. It optionally allows the processing of inputs for specific
    families in forms of ratios (e.g., Q_H and f_esc).

    Parameters:
        emu: The emulator object containing preprocessed data and configurations.
        fam: The family of models or dataset identifier to process.
        mu: The mean values as input for decoding, formatted as numpy or tensor-like array.
        var: The variance values as input for decoding, formatted as numpy or tensor-like array.
        X: Optional physical input corresponding to the same points being evaluated. It is
           particularly useful for ratio-based families (default is None).

    Returns:
        The result of the decoding process as defined by the Gaussian Process decoder.

    Raises:
        Any exception raised originates either from the decoding process or incompatible inputs.
    """
    import torch
    from galapy.spectroscopy.svgp import decode
    return decode(emu.prep[fam], torch.as_tensor(mu.T), torch.as_tensor(var.T),
                  log_scale=log_scale_of(emu, fam, X))

def likelihood_noise(stem, parts):
    """
    Calculate the likelihood noise values for each family based on the provided emulator file.

    This function reads from pre-saved emulator artifacts and retrieves the noise values
    for each family by loading GaussianLikelihood instances and extracting the
    noise parameters.

    Args:
        stem (str): The base name of the emulator file without extensions.
        parts (list of str): List of file suffixes corresponding to different parts of the
            emulator data.

    Returns:
        dict: A dictionary where keys are family names and values are numpy arrays of
        noise values for each family.
    """
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


#============================= CHECKS (register, common checks to HII and PDR sectors)

CHECKS = {}            # id -> (function, sector, blocking/not blocking, description)


def register(gid, function, sector, blocking=False, title=''):
    """
    Registers a gate with the specified parameters.

    Adds a gate to the CHECK registry with the provided gate ID, function, sector,
    blocking option, and title.

    Parameters:
    gid: str
        The unique identifier for the check being registered.
    function: Callable
        The function to be executed by the check.
    sector: str
        The sector to which the check belongs.
    blocking: bool, optional
        Specifies whether the check operation is blocking. Defaults to False.
    title: str, optional
        A descriptive title for the check. Defaults to an empty string.
    """
    CHECKS[gid] = (function, sector, blocking, title)


def check(gid, sector, blocking=False, title=''):
    """
    Decorates a function to register it with a check system for the specified configuration.

    This decorator wraps a function for use with a configurable gate system, allowing it to
    be associated with a specific check ID, sector, and other optional attributes. The
    registration binds the function to the check ID and sector while modifying its runtime
    behavior if required by its blocking status or title.

    Parameters:
    gid: str
        The unique identifier for the check.
    sector: str
        The specific sector or category associated with the check.
    blocking: bool, optional
        Determines whether the check will block other operations until completion
        of the decorated function. Defaults to False.
    title: str, optional
        An optional descriptive title for the registration. Defaults to an
        empty string.

    Returns:
    function
        The wrapped function after it has been registered with the check system.
    """
    def deco(function):
        register(gid, function, sector, blocking, title)
        return function
    return deco

def _ks_p(r, n, rng):
    """
    Performs the Kolmogorov-Smirnov test to verify whether the residuals are
    normalized and follow a standard normal distribution (N(0, 1)).

    A subset of the residuals is used for testing, and if the size of the residuals
    is less than 20, the function returns NaN.

    Parameters:
    r (np.ndarray): The array of residuals to be tested.
    n (int): The size of the subset of residuals to be used for testing.
    rng (np.random.Generator): A random number generator to sample residuals.

    Returns:
    float: The p-value from the Kolmogorov-Smirnov test if the residual subset
           size is at least 20; otherwise, returns NaN.
    """
    from scipy.stats import kstest  # KS test
    r = r[np.isfinite(r)] # only finite values
    if r.size > n:
        r = rng.choice(r, size=n, replace=False)
    return float(kstest(r, 'norm').pvalue) if r.size >= 20 else float('nan')


def calibration(ctx, rows=None, families=None, alpha=0.05):
    """
    Checks that the (calibrated) sigma of the emulator is honest on the holdout, family by family.

    Criterion: in every family the fraction of normalized residuals |r| = |y - mu| / sigma within
    1.96 is in EMUL.COV95_RANGE (0.93-0.97), and the fraction within 1 is at least EMUL.COV68_MIN (0.60).
    It matches the calibration of the training (s_k from the 95% quantile, see calibration_scale):
    the residuals have heavy tails, so a calibrated cov95 comes with a conservative cov68 and no
    single scale can give both 0.68 and 0.95.

    A Kolmogorov-Smirnov test of r against N(0, 1) per output (Bonferroni threshold alpha / n_output)
    is still reported, as information: with heavy tails and thousands of residuals it rejects
    gaussianity for any sigma, so it is not the criterion.

    Parameters:
    ctx: validation context (targets, mean and variance in the GP space, noise if requested).
    rows: optional indices of the holdout rows to use (default all), e.g. one tau_SSP bin.
    families: optional list of families (default all those of the emulator).
    alpha: significance level of the informative KS test.

    Return:
    (status, detail): PASS/FAIL/SKIP and, per family, cov68, cov95, fraction beyond 5 sigma, KS min p
    and whether the family passes; plus the failing families and a summary string.
    """
    rng = np.random.default_rng(ctx.args.seed)
    fams = families or list(ctx.mu)
    rows = slice(None) if rows is None else rows
    per, n_out, p_all, bad = {}, 0, [], []
    noise = ctx.noise() if getattr(ctx.args, 'noise_diagnostic', False) else {}
    lo, hi = EMUL.COV95_RANGE
    for f in fams:
        y, m, v = ctx.Yz[f][rows], ctx.mu[f][rows], ctx.var[f][rows]  # targets, mean and variance
        r = (y - m) / np.sqrt(np.maximum(v, 1e-30))                    # normalised residuals
        if r.shape[0] < 20:
            continue
        p = np.array([_ks_p(r[:, j], ctx.args.n_ks, rng) for j in range(r.shape[1])])   # informative KS
        n_out += r.shape[1]
        p_all.append(p)
        c68, c95 = float(np.mean(np.abs(r) <= 1)), float(np.mean(np.abs(r) <= 1.96))
        ok = lo <= c95 <= hi and c68 >= EMUL.COV68_MIN
        det = {'cov68': round(c68, 4), 'cov95': round(c95, 4), 'out5': round(float(np.mean(np.abs(r) > 5)), 4),
               'ks_min_p': float(np.nanmin(p)) if np.isfinite(p).any() else None, 'pass': ok}
        if f in noise:
            rn = (y - m) / np.sqrt(v + noise[f])
            det['cov68_with_noise'] = float(np.mean(np.abs(rn) <= 1))
        per[f] = det
        if not ok:
            bad.append(f)
    if not per:
        return SKIP, {'reason': 'less than 20 residuals per output'}
    p_all = np.concatenate(p_all)
    thr = alpha / n_out
    return (FAIL if bad else PASS), {
        'criterion': f"cov95 in [{lo}, {hi}] and cov68 >= {EMUL.COV68_MIN} in every family",
        'families': per, 'failing_families': bad, 'n_outputs': n_out,
        'ks_info': {'threshold_p': thr, 'min_p': float(np.nanmin(p_all)),
                    'n_outputs_below': int(np.sum(p_all <= thr))},
        'summary': (f"cov95 {min(d['cov95'] for d in per.values()):.3f}-{max(d['cov95'] for d in per.values()):.3f} "
                    f"(need {lo}-{hi}), cov68 min {min(d['cov68'] for d in per.values()):.3f} (need >= {EMUL.COV68_MIN})"
                    + (f"; failing: {', '.join(bad)}" if bad else ''))}


def rmse_dex(pred, true, graded):
    """
    Calculates the root mean square error (RMSE) in dex logarithm space
    between predicted and true values for the provided graded indices.

    This function computes the element-wise difference of the base-10
    logarithm of the predicted and true values for the subset of indices
    defined by the graded parameter, and returns the square root of the
    mean squared error of these differences. If the subset is empty, the
    function returns NaN.

    Args:
        pred (np.ndarray): An array of predicted values.
        true (np.ndarray): An array of true values.
        graded (np.ndarray): A boolean mask or index array specifying the
            subset of values to include in the calculation.

    Returns:
        float: The root mean square error in decadic logarithm space, or NaN
            if the subset is empty.
    """
    d = np.log10(pred[graded]) - np.log10(true[graded])
    return float(np.sqrt(np.mean(d ** 2))) if d.size else float('nan')


def ood_growth(ctx, n_probe=20, steps=(0.25, 0.5, 1.0, 2.0)):
    """
    Check that the emulator's latent sigma grows as we move away from the training domain.

    This function selects points from the holdout dataset and evaluates how the emulator's uncertainty
    (latent sigma) evolves as one axis at a time moves progressively further out of the training regime.
    It checks that the median sigma increases with distance and ensures that at the highest distance,
    the median sigma is at least 95% higher than within the training domain.

    Parameters:
    ctx : Context object
        The context containing emulator, training data, and metadata information.
    n_probe : int, optional
        Number of points to probe from the holdout dataset. Default is 20.
    steps : tuple of float, optional
        Steps representing the relative distances to move outside the training domain. Default is
        (0.25, 0.5, 1.0, 2.0).

    Returns:
    tuple
        A tuple containing:
        - A status indicating whether the test passed or failed (PASS or FAIL).
        - A dictionary containing detailed results, including:
            - Steps with calculated latent sigmas by step.
            - Whether the median sigma increases monotonically.
            - Fraction of samples above the reference sigma at the maximum distance.

    Raises:
    None
    """
    import torch
    from galapy.spectroscopy.svgp import fast_predict
    emu = ctx.emu
    # limits of the training domainm and axes of move (not the discrete nodes of SSPs)
    lo = np.asarray(ctx.meta.get('x_min') or ctx.X.min(0), float)
    hi = np.asarray(ctx.meta.get('x_max') or ctx.X.max(0), float)
    nodes = {'tau_SSP', 'Z_star'}
    axes = [i for i, k in enumerate(emu.input_keys) if k not in nodes]
    rng = np.random.default_rng(ctx.args.seed)
    # choose n_probe points from holdout
    base = ctx.X[rng.choice(len(ctx.X), size=min(n_probe, len(ctx.X)), replace=False)]

    def sig(X):
        """
        compute the median of the latent sigma for a given set of points
        """
        x = (torch.as_tensor(X, dtype=torch.float32) - emu.x_mean) / emu.x_std
        with torch.no_grad():
            return {f: np.median(np.sqrt(fast_predict(c, x)[1].numpy()), axis=0)
                    for f, c in emu.cache.items()}
    s0 = sig(base) # reference sigma
    out, ok = {}, True
    for f in emu.cache:
        med, beyond = [], []
        # main cycle, steps moving away from the training domain
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


def _available_cpus():
    """
    Determines the number of available CPU cores on the system.

    This function attempts to calculate the number of CPU cores available
    for use by the program. It first tries to use the `os.sched_getaffinity`
    method to determine the count of available cores, which provides accurate
    information about CPU affinity. If that method is not supported on the
    operating system, it falls back to `os.cpu_count` to get a total count
    of CPUs. If no valid count is determined, it defaults to returning 1.

    Returns:
        int: The number of available CPU cores.
    """
    import os
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def throughput(ctx, threads=None, batches=None, min_time=0.5, min_calls=5):
    """
    Measures the prediction speed of the emulator on a grid of torch thread counts x batch sizes.

    One spectrum of a galaxy needs the 8 nodes of tau_SSP, so the reference call is a batch of 8 rows
    (the acceptance criterion of the check). Larger batches are measured as well because a vectorized
    sampler can evaluate several points of the parameter space in a single call: the GP matrices are
    read from memory once per call, so up to some size the cost grows much less than the batch. Where
    the gain stops depends on the CPU (cache, memory bandwidth) and on M, so the grid is meant to be
    run on the machine of the fits, to choose threads and batch when integrating the emulator in galapy.

    Each configuration runs 3 warm-up calls, then repeats the call for at least `min_time` seconds and
    `min_calls` calls. Inputs are holdout points (repeated if the holdout is smaller than the batch).

    Parameters:
        ctx: validation context (emulator, sector, holdout inputs ctx.X).
        threads (Optional[Iterable[int]]): torch thread counts. Default 1, 2, 4, 8 and all the available
            CPUs, capped at the available CPUs.
        batches (Optional[Iterable[int]]): rows per call. Default 8, 16, 32, 64, 128.
        min_time (float): minimum seconds of measurement per configuration.
        min_calls (int): minimum calls per configuration.

    Returns:
        dict: 'grid' {threads: {batch: ms_per_call, calls_per_s, rows_per_s, galaxies_per_s}}, 'eval_per_s_8_threads'
        (calls per second of the reference batch of 8 at 8 threads, or at the most threads available
        if fewer than 8), 'best' (configuration with most rows per second), 'cpus_available' and
        'rows_per_galaxy' (rows of one galaxy spectrum: one per tau_SSP node).
    """
    import torch
    call = getattr(ctx.emu, ctx.sector['predict'])
    n_cpu = _available_cpus()
    threads = sorted({t for t in (threads or (1, 2, 4, 8, n_cpu)) if 1 <= t <= n_cpu})
    batches = sorted(set(batches or (8, 16, 32, 64, 128)))
    pool = ctx.X if len(ctx.X) >= max(batches) else np.resize(ctx.X, (max(batches), ctx.X.shape[1]))
    # one galaxy = one row per tau_SSP node (8 in the HII grid)
    rows_gal = len(getattr(ctx.emu, 'qh_nodes', {}).get('log_tau', ())) or 8
    grid, old = {}, torch.get_num_threads()
    try:
        for nt in threads:
            torch.set_num_threads(nt)
            grid[nt] = {}
            for b in batches:
                theta = pool[:b]
                for _ in range(3):
                    call(theta)
                n, t0 = 0, time.perf_counter()
                while n < min_calls or time.perf_counter() - t0 < min_time:
                    call(theta)
                    n += 1
                dt = (time.perf_counter() - t0) / n
                grid[nt][b] = {'ms_per_call': 1e3 * dt, 'calls_per_s': 1.0 / dt, 'rows_per_s': b / dt,
                               'galaxies_per_s': b / dt / rows_gal}
    finally:
        torch.set_num_threads(old)
    ref_t = 8 if 8 in grid else max(grid)
    best_t, best_b = max(((t, b) for t in grid for b in grid[t]), key=lambda tb: grid[tb[0]][tb[1]]['rows_per_s'])
    return {'grid': grid, 'eval_per_s_8_threads': grid[ref_t][8]['calls_per_s'] if 8 in grid[ref_t] else float('nan'),
            'reference_threads': ref_t, 'cpus_available': n_cpu, 'rows_per_galaxy': rows_gal,
            'best': {'threads': best_t, 'batch': best_b, **grid[best_t][best_b]}}


def throughput_tables(res, threshold=100):
    """
    The throughput grid as (Markdown for the report, aligned text lines for the log): galaxies per
    second and milliseconds per call, threads x batch. The reference cell of the check (batch 8 at the
    reference threads) is marked with *.
    """
    grid, rg = res['grid'], res['rows_per_galaxy']
    batches = sorted({b for row in grid.values() for b in row})
    ref_t, best = res['reference_threads'], res['best']
    def cell(t, b, key, fmt):
        v = format(grid[t][b][key], fmt)
        return v + ('*' if (t, b) == (ref_t, 8) else '')
    intro = (f"{res['cpus_available']} CPUs available; 1 galaxy = {rg} rows (one per tau_SSP node), "
             f"batch B = rows per call = B/{rg} galaxies.")
    crit = f"check: >= {threshold} galaxies/s at batch 8 with {ref_t} threads (*)"
    best_line = (f"best: {best['galaxies_per_s']:.0f} galaxies/s ({best['rows_per_s']:.0f} rows/s) "
                 f"with {best['threads']} threads and batch {best['batch']}")
    tables = (('Galaxies per second', 'galaxies_per_s', '.0f'), ('Milliseconds per call', 'ms_per_call', '.1f'))

    md = [intro, '', crit + '; ' + best_line + '.']
    for title, key, fmt in tables:
        md += ['', f'{title}:', '', '| threads | ' + ' | '.join(f'B={b}' for b in batches) + ' |',
               '|---|' + '---|' * len(batches)]
        md += [f"| {t} | " + ' | '.join(cell(t, b, key, fmt) for b in batches) + ' |' for t in grid]

    w = max(7, max(len(f'B={b}') for b in batches) + 1)
    head = f"{'threads':>8} |" + ''.join(f"{f'B={b}':>{w}}" for b in batches)
    txt = ['Throughput: ' + intro]
    for title, key, fmt in tables:
        txt += [f"{title}" + (f"  [{crit}]" if key == 'galaxies_per_s' else ''), head, '-' * 9 + '+' + '-' * (w * len(batches))]
        txt += [f"{t:>8} |" + ''.join(f"{cell(t, b, key, fmt):>{w}}" for b in batches) for t in grid]
    txt.append(best_line[0].upper() + best_line[1:])
    return '\n'.join(md), txt


def footprint(ctx, budget_rss_gib=1.5):
    """
    Checks the size and memory footprint of the emulator in the context, verifying compliance with the specified
    budgets for file sizes (default 85 MiB) and peak memory usage.

    Loads the emulator from .pt files, measures the sizes of these files, and ensures their individual sizes
    do not exceed the specified budget. Additionally, emulates prediction by loading the emulator model and
    performing 20 batch predictions of 8 spectra each, while monitoring peak memory usage against the defined
    RAM budget. Returns pass or fail status along with the relevant measured statistics.

    Args:
        ctx: A context object containing metadata and configuration for the specific emulator. It must include
             attributes such as `stem` for file naming and `sector` with details like `parts`, `emulator`,
             and `predict`.
        budget_rss_gib (float, optional): The RAM budget in GiB for the peak memory usage during loading
                                          and predictions. Defaults to 1.5.

    Returns:
        tuple: A tuple comprising:
            - str: 'PASS' if the budget constraints are adhered to, otherwise 'FAIL'.
            - dict: A dictionary of statistics including individual file sizes in MiB, total file size,
                    file size budget per individual file, measured peak memory in GiB, and the RAM budget.
    """
    stem, parts = ctx.stem, ctx.sector['parts']
    sizes = {p: pathlib.Path(f"{stem}_{p}.pt").stat().st_size / 2 ** 20 for p in parts} # size in MiB
    mod, cls = ctx.sector['emulator']
    # script to load emulator and predict 20 batches of 8 spectra, print peak memory usage
    # 8 holdout inputs, as in throughput: the mean input is not valid (tau_SSP must sit on a grid node)
    theta = ctx.X[:8] if len(ctx.X) >= 8 else np.repeat(ctx.X[:1], 8, axis=0)
    code = (f"import resource, numpy as np\nfrom {mod} import {cls}\n"
            f"e = {cls}().load({str(pathlib.Path(stem).resolve())!r})\nx = np.array({theta.tolist()!r})\n"
            f"[e.{ctx.sector['predict']}(x) for _ in range(20)]\n"
            f"try:\n"
            f"    print(int([l for l in open('/proc/self/status') if l.startswith('VmHWM')][0].split()[1]) * 1024)\n"
            f"except OSError:\n"
            f"    import sys; r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
            f"    print(r if sys.platform == 'darwin' else r * 1024)")
    # Peak RSS in bytes. On Linux ru_maxrss of a child started by fork+exec also counts the peak of the
    # parent (the validation process, several GiB with the holdout loaded): VmHWM is the child's own.
    # `python -c` puts the current directory first on sys.path: run from the repository root it would
    # import the galapy source tree (no compiled *_core extensions) instead of the installed package.
    # Hence a neutral working directory, and the absolute path of the emulator above.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=tmp)
    if out.returncode != 0:
        raise RuntimeError(f"footprint subprocess failed:\n{out.stderr[-2000:]}")
    rss = int(out.stdout.split()[-1]) / 2 ** 30   # bytes -> GiB
    # budget and check
    budget = float(ctx.meta.get('size_budget_mib', 85))
    ok = all(s <= budget for s in sizes.values()) and rss <= budget_rss_gib
    return (PASS if ok else FAIL), {'file_mib': {p: round(s, 2) for p, s in sizes.items()},
                                    'total_mib': round(sum(sizes.values()), 2),
                                    'budget_mib_per_file': budget, 'peak_rss_gib': round(rss, 3),
                                    'budget_rss_gib': budget_rss_gib}


def per_node_bin(ctx, line_gate):
    """
    Divides holdout data per `tau_SSP` and, in every bin, repeats the line gate
    and calibration gate. This process is performed to identify local problems
    where the emulator may perform well on average but poorly for a specific
    population of ages.

    Parameters:
    ctx : Context object
        The context instance containing the dataset and relevant functionalities.
    line_gate : callable
        Function that evaluates the line gating process for a given set of rows.

    Returns:
    tuple
        A tuple where the first element is an overall status (`PASS` or `FAIL`),
        and the second element is a dictionary with detailed bin-wise evaluation
        results, including:
        - `n`: Number of rows in each bin.
        - `accuracy`: Accuracy metric achieved through line gating.
        - `calibration`: Calibration metric achieved.
        - `failing_families`: Families failing the calibration criterion in the bin.
        - `worst_line_dex`: Index of the worst-performing line in the bin.
    """
    lt = ctx.col('tau_SSP')
    bins, ok = {}, True
    for t in np.unique(np.round(lt, 4)):
        rows = np.flatnonzero(np.isclose(lt, t, atol=1e-4))
        s_acc, d_acc = line_gate(ctx, rows)
        s_cal, d_cal = calibration(ctx, rows=rows)
        bins[f"{10 ** t:.3g} yr"] = {'n': int(rows.size), 'accuracy': s_acc,
                                     'calibration': s_cal, 'failing_families': d_cal.get('failing_families'),
                                     'worst_line_dex': d_acc.get('worst_dex')}
        ok &= s_acc != FAIL and s_cal != FAIL
    return (PASS if ok else FAIL), {'bins': bins}

def gpu_matches_cpu(ctx, rtol=1e-4):
    """
    Checks if the results from GPU and CPU computations match within a relative tolerance.

    Evaluates whether the predictions generated by an emulator running on a GPU device and
    its CPU counterpart are identical within a specified relative tolerance. If a GPU resource
    is unavailable, the function will skip the check.

    Parameters:
    ctx : Context object
        The context containing data, model emulator definitions, and configuration details.
    rtol : float, optional
        The relative tolerance value for comparison, default is 1e-4. Specifies the maximum
        allowed difference between GPU and CPU results, relative to their magnitude.

    Returns:
    tuple
        A tuple where the first element is either PASS, FAIL, or SKIP, denoting the result
        of the comparison. The second element is a dictionary with additional details
        including the reason for skipping (if relevant), the worst relative difference, and
        the tolerance value.
    """
    """
    check the results form gpu and cpu are the same
    """
    import torch
    if not torch.cuda.is_available():
        return SKIP, {'reason': 'no GPU found'}
    gpu = ctx.sector['emulator_cls']().load(ctx.stem, device='cuda')
    call_c, call_g = (getattr(e, ctx.sector['predict']) for e in (ctx.emu, gpu))
    a, b = call_c(ctx.X[:64]), call_g(ctx.X[:64])
    worst = 0.0
    for k in a:
        va, vb = (np.asarray(v[0] if isinstance(v, tuple) else v) for v in (a[k], b[k]))
        worst = max(worst, float(np.max(np.abs(va - vb) / np.maximum(np.abs(va), 1e-300))))
    return (PASS if worst <= rtol else FAIL), {'max_rel_diff': worst, 'tolerance': rtol}


def acceptance_gates(candidate, gate_set, base=None, alpha=0.05, max_degradation=0.01, n_ks=1000):
    """
    Decides whether a retrained emulator is acceptable as a substitute for the base emulator. The function
    evaluates the emulator's performance using the Kolmogorov-Smirnov (KS) test and Root Mean Square Error
    (RMSE) applied to a frozen dataset created during the initial training of the base emulator.

    Parameters:
        candidate (object): The retrained emulator to evaluate.
        gate_set (str): Path to the frozen dataset used for testing. Must be in a format compatible with numpy
            loading (e.g., .npz file).
        base (object, optional): The base emulator for comparison. If provided, ensures no significant
            degradation in performance.
        alpha (float, optional): Significance level for the KS test (default is 0.05).
        max_degradation (float, optional): Maximum allowable RMSE degradation in the retrained emulator
            relative to the base (default is 0.01).
        n_ks (int, optional): Maximum number of data points to sample for the KS test to improve performance
            (default is 1000).

    Returns:
        dict: A dictionary containing the following keys:
            - gates: A dictionary of individual evaluation gates with their conditions and pass status.
            - all_pass: A boolean indicating whether all gates have been passed.
            - gate_set: The path to the frozen dataset.
            - n_points: The number of points in the dataset used for evaluation.

    Raises:
        None
    """
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
    checks = {'check calibration': {'value': p_min, 'threshold': alpha / n_out,
                                    'pass': p_min > alpha / n_out}}
    if base is not None:
        rmse_b = score(base)[1]
        checks['check no_degradation'] = {'value': rmse, 'base': rmse_b,
                                          'threshold': rmse_b * (1 + max_degradation),
                                          'pass': rmse <= rmse_b * (1 + max_degradation)}
    return {'checks': checks, 'all_pass': all(g['pass'] for g in checks.values()),
            'gate_set': str(gate_set), 'n_points': int(len(X))}


#=============================== HII SECTOR SPECIFIC CHECKS


def _line_index(ctx, name, table=EMUL.CRITICAL_HII):
    """
    Finds the index of a specific line in the context based on a given table.

    This function searches for a line in the context using a specified lookup
    table and returns the index of the line. The table contains mappings that
    are used to identify the target line in the context.

    Args:
        ctx: The context object containing lines to search.
        name: The key used to retrieve values from the lookup table.
        table: A dictionary-like object representing the lookup table.
            Defaults to EMUL.CRITICAL_HII.

    Returns:
        int: The index of the line found in the context or -1 if not found.
    """
    return ctx.lines().find(*table[name])


def _hbeta_graded(ctx, rows):
    """
    Evaluates where the truth condition exceeds the threshold of 1e-3 Hβ

    Args:
        ctx: Context object, which includes data used for evaluation.
        rows: Array or iterable of rows to evaluate.

    Returns:
        Array or iterable of evaluation results where the condition is met.
    """
    from galapy.spectroscopy.utils.validate_grid import T as TG
    hb = ctx.Y['lines'][rows, _line_index(ctx, 'H_beta')]
    return TG['oiii.min_hbeta'] * hb

@check('crit_l', 'hii', title='critical lines: RMSE(log10) on the holdout < 5%')
def critical_lines(ctx, rows=None):
    if _line_index(ctx, 'H_beta') is None:
        return SKIP, {'reason': 'H-beta not in the master list, no threshold of flux'}
    rows = np.arange(len(ctx.X)) if rows is None else rows
    pred = ctx.phys('lines', rows)[0]
    true = ctx.Y['lines'][rows]
    floor = _hbeta_graded(ctx, rows)
    out, worst = {}, 0.0
    for name in EMUL.CRITICAL_HII:
        j = _line_index(ctx, name)
        if j is None:
            out[name] = 'absent from the master list'
            continue
        g = (true[:, j] > 0) & (pred[:, j] > 0) & (true[:, j] >= floor)
        r = rmse_dex(pred[:, j], true[:, j], g)
        out[name] = {'n': int(g.sum()), 'rmse_dex': round(r, 5)}
        worst = max(worst, r) if np.isfinite(r) else worst
    return (PASS if worst < EMUL.FRAC_5PCT_DEX else FAIL), {
        'threshold_dex': round(EMUL.FRAC_5PCT_DEX, 5), 'worst_dex': round(worst, 5), 'lines': out}


@check('OIII r', 'hii', title='[O III] 4959/5007 emulate')
def oiii(ctx):
    from galapy.spectroscopy.utils.validate_grid import T as TG
    j5, j4 = _line_index(ctx, 'OIII_5007'), ctx.lines().find('O 3', 4958.9)
    if j5 is None or j4 is None or _line_index(ctx, 'H_beta') is None:
        return SKIP, {'reason': '[O III] not in the master list'}
    pred, true = ctx.phys('lines')[0], ctx.Y['lines']
    g = (true[:, j5] >= _hbeta_graded(ctx, slice(None))) & (pred[:, j4] > 0) & (pred[:, j5] > 0)
    r_emu = pred[g, j4] / pred[g, j5]
    dev = np.abs(r_emu / TG['oiii.reference'] - 1.0)
    vs_true = np.abs(r_emu / (true[g, j4] / true[g, j5]) - 1.0)
    return (PASS if dev.max() < TG['oiii.max_rel_dev'] else FAIL), {
        'n': int(g.sum()), 'worst_rel_dev_from_reference': round(float(dev.max()), 5),
        'reference': TG['oiii.reference'], 'tolerance': TG['oiii.max_rel_dev'],
        'median_rel_dev_from_grid': round(float(np.median(vs_true)), 5)}

@check('Balmer', 'hii', title='balmer decrement emulation')
def balmer(ctx):
    from galapy.spectroscopy.utils.validate_grid import T as TG
    ja, jb = _line_index(ctx, 'H_alpha'), _line_index(ctx, 'H_beta')
    if ja is None or jb is None:
        return SKIP, {'reason': 'H-alpha or H-beta not present in the master list'}
    pred, true = ctx.phys('lines')[0], ctx.Y['lines']
    w = ((np.abs(ctx.col('logU') - TG['case_b.logU_centre']) <= TG['case_b.logU_half_width'])
         & (ctx.col('z_CMB') <= TG['case_b.max_z_cmb']) & (true[:, jb] > 0))
    if not w.any():
        return SKIP, {'reason': 'no holdout point in the tested parameters range'}
    r_emu, r_true = pred[w, ja] / pred[w, jb], true[w, ja] / true[w, jb]
    dev = abs(np.median(r_emu) / TG['case_b.reference'] - 1.0)
    pct = lambda v: [round(float(x), 4) for x in np.percentile(v, (5, 50, 95))]
    return (PASS if dev < TG['case_b.max_rel_dev'] else FAIL), {
        'n': int(w.sum()), 'median_emulated': round(float(np.median(r_emu)), 4),
        'relative_deviation': round(float(dev), 4), 'p5_50_95_emulated': pct(r_emu),
        'p5_50_95_grid': pct(r_true)}

@check('Calibr.', 'hii', blocking=True, title='calibration of the sigma SVGP (blocking)')
def c_calibration(ctx):
    return calibration(ctx)

@check('Energy', 'hii', title='emulated energy balance against the one from the grid')
def energy(ctx, n_max=2048, tol=0.01, min_fraction=0.95):
    import h5py
    from galapy.spectroscopy.utils.validate_grid import T as TG
    rng = np.random.default_rng(ctx.args.seed)
    rows = np.sort(rng.choice(len(ctx.X), size=min(n_max, len(ctx.X)), replace=False))
    with h5py.File(ctx.args.corpus, 'r') as f:
        if 'incident/wave_grid' not in f:
            return SKIP, {'reason': 'corpus does not have the incident radiation (incident/)'}
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

@check('Dust', 'hii', title='dust mass: RMSE(log10) < 10% and scaling')
def dust(ctx):
    from galapy.spectroscopy.utils.validate_grid import T as TG, _chemistry
    pred, true = ctx.phys('dust')[0][:, 0], ctx.Y['dust'][:, 0]
    g = (pred > 0) & (true > 0)
    r = rmse_dex(pred, true, g)
    try:
        chem = _chemistry()
    except Exception as exc:  # noqa: BLE001 - come validate_grid: non disponibile -> SKIP
        return SKIP, {'reason': f'abbundances module not avaiable: {exc}', 'rmse_dex': r}
    q_h = ctx.emu.Qh_unit_lookup(ctx.X)
    dust = ctx.col('xi_d') * np.array([chem.metal_mass_per_H(v) for v in ctx.col('log_zeta_O')])
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.log10(pred) - (np.log10((1.0 - ctx.col('f_esc_meas')) * dust * q_h)
                                - ctx.col('lognH_HII'))
    ok = np.isfinite(res)
    dev = res[ok] - np.median(res[ok])
    frac = float(np.mean(np.abs(dev) <= TG['dust.max_abs_residual']))
    return (PASS if r < EMUL.FRAC_10PCT_DEX and frac >= TG['dust.min_pass_fraction'] else FAIL), {
        'rmse_dex': round(r, 5), 'threshold_dex': round(EMUL.FRAC_10PCT_DEX, 5),
        'scaling_fraction_within_0.2dex': round(frac, 4), 'n': int(ok.sum())}

@check('OOD', 'hii', title='sigma of SVGP increase out of domain')
def ood(ctx):
    return ood_growth(ctx)

@check('Th.put', 'hii', title='throughput batch-8 on CPU >= 100 eval/s at 8 thread (+ grid threads x batch)')
def c_throughput(ctx):
    a = ctx.args
    res = throughput(ctx, getattr(a, 'tp_threads', None), getattr(a, 'tp_batches', None),
                     getattr(a, 'tp_min_time', 0.5))
    md, txt = throughput_tables(res)
    print('\n'.join(f"[emul-val]   {line}" for line in txt), flush=True)
    ref, best = res['eval_per_s_8_threads'], res['best']
    grid = {f"{t} threads": {f"B={b}": {k: round(v, 2) for k, v in r.items()} for b, r in row.items()}
            for t, row in res['grid'].items()}
    return (PASS if ref >= 100 else FAIL), {
        'eval_per_s_8_threads': round(ref, 1), 'reference_threads': res['reference_threads'],
        'cpus_available': res['cpus_available'], 'best': {k: round(v, 2) for k, v in best.items()},
        'grid': grid, 'markdown': md,
        'summary': (f"batch 8, {res['reference_threads']} thr: {ref:.0f} galaxies/s (>= 100); best "
                    f"{best['galaxies_per_s']:.0f} galaxies/s at batch {best['batch']}, {best['threads']} thr")}

@check('Size', 'hii', title='footprint of files and RAM at runtime')
def c_footprint(ctx):
    return footprint(ctx)


def _scan(ctx, axis, n_points=20, n_steps=5):
    """
    Scan the emulator along a specified input parameter axis. For each axis, the function evaluates
    20 points sampled from the holdout data. It calculates 4 diagnostic line ratios (in log10) along
    scan lines, including their respective uncertainties.

    Parameters:
    ctx: Context object
        The context containing all required data, including emulator, input keys, metadata, and lines data.
    axis: str
        The input parameter axis along which to perform the scan. Must match one of the emulator input keys.
    n_points: int, optional
        Number of holdout points to sample. Default is 20.
    n_steps: int, optional
        Number of sampling steps along the axis. Default is 5.

    Returns:
    dict
        A dictionary where each key represents a diagnostic ratio ('SII/Ha', 'SIII/Ha', 'OI/Ha', 'O32').
        Each value is a tuple of two numpy arrays:
        - The first array contains the calculated diagnostic ratio values reshaped for the specified
          number of points and steps.
        - The second array contains the corresponding uncertainties (standard deviances in log10 space),
          reshaped similarly.

    """
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
    idx = {k: cat.find(*v) for k, v in EMUL.SCAN_HII.items()}
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
    """
    Computes the fraction of monotonic scans in the expected direction within
    the GP error margins. A single step in the wrong direction is allowed if it
    stays below the combined sigma of the two adjacent points.

    Args:
        ratio: Array-like input of shape (n, m), where each row represents a
            sequence of values. The monotonic behavior is assessed for the
            columns of the input.
        err: Array-like input of the same shape as `ratio`. Represents the
            associated errors for each element, used to calculate the
            tolerance for monotonicity.
        sign: Integer indicating the expected direction of monotonicity. A
            positive `sign` checks for monotonic increase, while a
            negative `sign` checks for monotonic decrease.

    Returns:
        float: The fraction of sequences (rows in the input) that satisfy the
            monotonicity condition within the calculated tolerance.
    """
    step = np.diff(ratio, axis=1) * sign
    tol = np.sqrt(err[:, 1:] ** 2 + err[:, :-1] ** 2)
    return float(np.mean(np.all(step >= -tol, axis=1)))


@check('ARD', 'hii', title='sanity check in ARD and monotonic')
def ard(ctx, min_fraction=0.9, max_ls_ratio=10.0):
    missing = [k for k, v in EMUL.SCAN_HII.items() if ctx.lines().find(*v) is None]
    if missing:
        return SKIP, {'reason': f'lines to scan are not in master list: {missing}'}
    keys = ctx.emu.input_keys
    lo = np.asarray(ctx.meta.get('x_min') or ctx.X.min(0), float)
    hi = np.asarray(ctx.meta.get('x_max') or ctx.X.max(0), float)
    ls = (ctx.emu.cache['lines']['lengthscale'][:, 0, :].numpy() * ctx.emu.x_std.numpy()
          / np.where(hi > lo, hi - lo, 1.0))
    cat = ctx.lines()
    active = [keys.index(k) for k in ('logU', 'lognH_HII', 'log_zeta_O')]
    f_star = keys.index('F_star')
    sulfur = [cat.find(*EMUL.SCAN_HII[k]) for k in ('SII_6716', 'SII_6731', 'SIII_9069', 'SIII_9531')]
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
    scans['xi_d (reported)'] = {k: round(float(np.median(v[:, -1] - v[:, 0])), 4)
                                 for k, (v, _e) in xi.items()}
    return (PASS if ok else FAIL), {'lengthscale_ratio_Fstar_over_active_sulfur': ratio_ls,
                                    'max_ratio': max_ls_ratio, 'monotone_fraction': scans,
                                    'min_fraction': min_fraction}


@check('Bins', 'hii', title='check per tau_ssp bins')
def bins(ctx):
    return per_node_bin(ctx, critical_lines)


@check('Midnodes', 'hii', title='interpolation on Z_star between nodes (CLOUDY runs)')
def midnode(ctx):
    if not ctx.args.midnode_corpus:
        return SKIP, {'reason': 'need --midnode-corpus (50 CLOUDY runs between nodes)'}
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


@check('GPU', 'hii', title='same predictions on GPU and CPU')
def gpu(ctx):
    return gpu_matches_cpu(ctx)


#========================================= EXECUTION AND REPORT GENERATION

def run_checks(ctx, only=None):
    """
    Executes a series of gate checks based on the given context and generates a report.

    The function iterates through the predefined checks in the `CHECKS` dictionary, applying
    each check that meets the specified conditions. It measures the execution time of each
    check and collects the results into a report. If an exception occurs during the execution
    of a check, the error information is recorded in the report.

    Parameters:
    ctx (Context): The current context containing the necessary data for the checks, including
        the sector information.
    only (Optional[Iterable[str]]): A set or list of gate IDs to restrict the execution to.
        If not provided, all gates applicable to the given sector are executed.

    Returns:
    dict: A report containing the results of the executed checks. Each entry in the dictionary
        corresponds to a gate ID and includes:
        - status: The result of the check, indicating success or failure.
        - blocking: A boolean indicating whether the gate is blocking.
        - title: A descriptive title of the gate check.
        - seconds: Execution time of the gate check in seconds.
        - detail: Additional information or error details, if applicable.
    """
    report = {}
    for gid, (fn, sector, blocking, title) in CHECKS.items():
        if sector != ctx.args.sector or (only and gid not in only):
            continue
        t0 = time.perf_counter()
        try:
            status, detail = fn(ctx)
        except Exception as exc:
            status, detail = ERROR, {'reason': f"{type(exc).__name__}: {exc}",
                                     'traceback': traceback.format_exc(limit=3)}
        report[gid] = {'status': status, 'blocking': blocking, 'title': title,
                       'seconds': round(time.perf_counter() - t0, 2), 'detail': detail}
        print(f"[emul-val] {gid:8s} {status:5s} {title}", flush=True)
    return report


def render_markdown(rep):
    """
    Converts a validation report into a formatted Markdown string.

    The function takes a dictionary containing details of an emulator validation
    report and processes its information into a human-readable Markdown format.
    The resulting Markdown includes a title, various details about the validation,
    and tabular data summarizing the results. The gates information is also
    embedded as a formatted JSON block for reference.

    Parameters:
        rep (dict): A dictionary containing the emulator validation report details.
            The dictionary should include information about the emulator, sector,
            holdout data, and gates.

    Returns:
        str: A string formatted as Markdown, including a tabular representation
            of the report and the JSON-encoded gate details.
    """
    rows = [f"# Emulator Validation {rep['sector'].upper()}", '',
            f"Artefatto `{rep['emulator']}`; holdout {rep['holdout'].get('n_evaluated')} punti "
            f"di {rep['holdout'].get('n_test')} ({rep['holdout'].get('n_clouds')} nubi).", '',
            '| Check | State | Blocking | Description | Summary |', '|---|---|---|---|---|']
    for gid, g in rep['gates'].items():
        d = g['detail']
        summ = d.get('summary') or d.get('reason') or ''
        rows.append(f"| {gid} | {g['status']} | {'si' if g['blocking'] else ''} | {g['title']} | {summ} |")
    # checks with their own tables (e.g. the throughput grid) get a section before the raw JSON
    for gid, g in rep['gates'].items():
        if isinstance(g['detail'], dict) and g['detail'].get('markdown'):
            rows += ['', f"## {gid}: {g['title']}", '', g['detail']['markdown']]
    gates = {gid: {**g, 'detail': {k: v for k, v in g['detail'].items() if k != 'markdown'}}
             if isinstance(g['detail'], dict) else g for gid, g in rep['gates'].items()}
    rows += ['', '```json', json.dumps(gates, indent=1, default=str), '```']
    return '\n'.join(rows) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sector', required=True, choices=('hii', 'pdr'))
    ap.add_argument('--emulator', help='emulators (default data/emulators/<s>)')
    ap.add_argument('--meta', help='training metadata (default <stem>_meta.json)')
    ap.add_argument('--corpus', help='corpus h5 (default data/products/<s>_grid.h5)')
    ap.add_argument('--holdout', help='holdout metadata (default <stem>_holdout.json)')
    ap.add_argument('--report', help='report JSON (default data/validation/report_emulator_<s>.json)')
    ap.add_argument('--markdown', help='report Markdown (default: data/validation/report_emulator_<s>.md)')
    ap.add_argument('--max-points', type=int, default=8192,
                    help='holdout points evaluated (0 = all; 8192 = cut of set checks)')
    ap.add_argument('--n-ks', type=int, default=1000, help='residuals per output in KS test')
    ap.add_argument('--noise-diagnostic', action='store_true',
                    help='report the coverege with the likelihood noise included')
    ap.add_argument('--midnode-corpus', help='activate CLOUDY check with models between nodes Z_star')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--strict', action='store_true', help='gate blocking if SKIP gives failure')
    ap.add_argument('--tp-threads', type=int, nargs='+', default=None,
                    help='torch thread counts of the throughput grid (default 1 2 4 8 and all available CPUs)')
    ap.add_argument('--tp-batches', type=int, nargs='+', default=None,
                    help='rows per call of the throughput grid (default 8 16 32 64 128)')
    ap.add_argument('--tp-min-time', type=float, default=0.5,
                    help='seconds of measurement per configuration of the throughput grid')
    ap.add_argument('--checks', '--gates', dest='checks', default=None,
                    help='run only these checks, comma-separated ids (default: all checks of the sector)')
    a = ap.parse_args(argv)
    s = a.sector
    a.emulator = a.emulator or f"data/emulators/{s}"
    a.meta = a.meta or f"{a.emulator}_meta.json"
    a.corpus = a.corpus or f"data/products/{s}_grid.h5"
    a.holdout = a.holdout or f"{a.emulator}_holdout.json"
    a.report = a.report or f"data/validation/report_emulator_{s}.json"
    a.markdown = a.markdown or str(pathlib.Path(a.report).with_suffix('.md'))

    ctx = Context(a)
    checks = run_checks(ctx, set(a.checks.split(',')) if a.checks else None)
    rep = {'sector': s, 'emulator': a.emulator, 'corpus': a.corpus, 'holdout': ctx.holdout,
           'gates': checks, 'summary': {st: sum(c['status'] == st for c in checks.values())
                                       for st in (PASS, FAIL, SKIP, ERROR)}}
    pathlib.Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.report).write_text(json.dumps(rep, indent=1, default=str))
    pathlib.Path(a.markdown).write_text(render_markdown(rep))
    bad = [g for g, v in checks.items() if v['blocking']
           and (v['status'] in (FAIL, ERROR) or (a.strict and v['status'] == SKIP))]
    print(f"[emul-val] {rep['summary']} -> {a.report}")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())