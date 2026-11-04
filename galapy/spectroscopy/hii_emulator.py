# Author: Enrico Veraldi
# the implementation of the emulator for the HII regions (inference class)

import numpy as np
import torch
import gpytorch

import galapy.internal.emulators as EMUL

from galapy.spectroscopy.svgp import (BatchedSVGPS, load_artifacts, predictive_cache, fast_predict, decode)



def qh_unit_lookup(qh_nodes, input_keys, theta_batch):
    """
    Q_H of a 1 Msun SSP for each row of theta_batch (physical inputs, tau_SSP and Z_star in log10):
    nearest node in tau, linear interpolation in Z_star. See HIIEmulator.Qh_unit_lookup.
    """
    th = np.atleast_2d(np.asarray(theta_batch, dtype=float))
    q = qh_nodes
    lt = th[:, input_keys.index('tau_SSP')]                         # take column of tau_SSP
    it = np.abs(lt[:, None] - q['log_tau'][None, :]).argmin(1)      # take nearest node
    if np.any(np.abs(q['log_tau'][it] - lt) > 1e-4):
        raise ValueError(f"[hii-emulator] log10 tau_SSP {lt} out of nodes range {q['log_tau']}")
    Zn = q['Z']
    Z = np.clip(10.0 ** th[:, input_keys.index('Z_star')], Zn[0], Zn[-1])              # clip Zstar to valid range
    hi = np.clip(np.searchsorted(Zn, Z), 1, len(Zn) - 1)                                 # two nodes Zstar used for interpolation
    w = (Z - Zn[hi - 1]) / (Zn[hi] - Zn[hi - 1])                                         # weight of interpolation
    w = np.where(np.isclose(w, 0.0), 0.0, np.where(np.isclose(w, 1.0), 1.0, w))          # attach the weight
    lo_q, hi_q = q['Qh'][it, hi - 1], q['Qh'][it, hi]                                    # two values of Qh used for interpolation
    out = np.where(w == 0.0, lo_q, np.where(w == 1.0, hi_q, (1.0 - w) * lo_q + w * hi_q))  # interpolation
    if not np.all(np.isfinite(out)):
        raise ValueError("[hii-emulator] Q_H, unit absent for the requested node: corpus does not cover the 48 nodes"
                         "(tau_SSP, Z_star)")
    return out


def line_log_scale(qh_nodes, input_keys, theta_batch):
    """
    log10[Q_H (1 - f_esc)]: the analytic scale of H-beta, removed from the lines in ratio form
    (see galapy.spectroscopy.svgp.lines_to_ratio). Same function at training and at inference.
    """
    th = np.atleast_2d(np.asarray(theta_batch, dtype=float))
    one_m_fesc = np.clip(1.0 - th[:, input_keys.index('f_esc_meas')], EMUL.ONE_MINUS_FESC_FLOOR, None)
    return np.log10(qh_unit_lookup(qh_nodes, input_keys, th)) + np.log10(one_m_fesc)


def sed_features(qh_nodes, input_keys, theta_batch):
    """
    Computes specific hardness features for each input row based on provided SSP node data.

    The function calculates log10 Q_H per Msun, log10 Q(He0)/Q(H), and log10 Q(He+)/Q(H)
    for each entry in the SSP node using provided Q_H tables and hardness tables.
    The lookup for Q_H is based on the nearest node in tau and linear interpolation
    in Z_star.

    Parameters:
    qh_nodes : dict
        A dictionary of Q_H and hardness values per Msun, structured as required
        by the qh_unit_lookup function. It must include the keys 'hard_he0' and
        'hard_he1' for accessing He0 and He+ hardness data.

    input_keys : list or array
        Keys specifying the dimensions or SSP parameters used in the lookup for Q_H
        and hardness values.

    theta_batch : array-like
        A batch of SSP parameter values for which calculations are performed.
        Each row corresponds to one set of parameters.

    Returns:
    numpy.ndarray
        A 2D array where each row corresponds to the calculated features
        [log10 Q_H per Msun, log10 Q(He0)/Q(H), log10 Q(He+)/Q(H)] for the
        corresponding input row.
    """
    th = np.atleast_2d(np.asarray(theta_batch, dtype=float))
    cols = [np.log10(qh_unit_lookup(qh_nodes, input_keys, th))]
    for k in ('hard_he0', 'hard_he1'):
        cols.append(np.log10(qh_unit_lookup(dict(qh_nodes, Qh=qh_nodes[k]), input_keys, th)))
    return np.column_stack(cols)


def augment_inputs(qh_nodes, input_keys, theta_batch):
    """
    Appends Spectral Energy Distribution (SED) descriptors to the input features for use in Gaussian Processes (GP).
    This function is applicable during both training and inference stages.

    Parameters:
    qh_nodes (ndarray): Array containing quantum harmonic nodes data.
    input_keys (list): List of keys or features used as input.
    theta_batch (ndarray or list): Batch of input features to which SED descriptors will be appended.

    Returns:
    ndarray: Combined array containing the original input features and appended SED descriptors.
    """
    th = np.atleast_2d(np.asarray(theta_batch, dtype=float))[:, :len(input_keys)]
    return np.hstack([th, sed_features(qh_nodes, input_keys, th)])

def _node_weights(qh_nodes, input_keys, th):
    """
    Computes the index of the tau_SSP node for each row and calculates the
    linear weights in Z_star. Outputs the results as qh_unit_lookup.

    Parameters:
    qh_nodes (dict): Dictionary containing node information with keys:
                     'log_tau' and 'Z'.
    input_keys (list): List of strings representing the input parameter names.
    th (numpy.ndarray): Array of input parameters where each row corresponds
                        to a set of parameters.

    Returns:
    tuple: A tuple containing the following:
        - numpy.ndarray: Indices of the tau_SSP node for each row.
        - numpy.ndarray: Low index positions for the Z_star interpolation.
        - numpy.ndarray: High index positions for the Z_star interpolation.
        - numpy.ndarray: Interpolation weights for Z_star.
    """
    lt = th[:, input_keys.index('tau_SSP')]
    it = np.abs(lt[:, None] - np.asarray(qh_nodes['log_tau'])[None, :]).argmin(1)
    Zn = np.asarray(qh_nodes['Z'], dtype=float)
    Z = np.clip(10.0 ** th[:, input_keys.index('Z_star')], Zn[0], Zn[-1])
    if len(Zn) == 1:
        zero = np.zeros(len(th), int)
        return it, zero, zero, np.zeros(len(th))
    hi = np.clip(np.searchsorted(Zn, Z), 1, len(Zn) - 1)
    w = np.clip((Z - Zn[hi - 1]) / (Zn[hi] - Zn[hi - 1]), 0.0, 1.0)
    return it, hi - 1, hi, w


def incident_tables(qh_nodes, wave_grid):
    """
    Computes the weights of the energy integrals for validating the energy balance of an emulator on a continuum grid.
    It calculates the integral of a quantity, X(lambda), weighted by the incident Spectral Energy Distribution (SED)
    over a logarithmic wavelength grid. The function uses a trapezoid rule for integration and interpolates
    linearly in the logarithmic space of wavelengths. Only columns within the incident grid are assigned weights.
    If the input does not contain the incident SED data, the function returns None.

    Parameters:
    qh_nodes: dict
        Dictionary containing the incident spectral energy distributions with keys such as 'sed_inc' and
        'wave_inc'. 'sed_inc' represents nu L_nu per Msun.

    wave_grid: array-like
        Continuum wavelength grid in physical units.

    Returns:
    dict or None
        If incident SED data is available, the function returns a dictionary with the following keys:
            'J': ndarray
                Indices of the continuum grid used in the interpolation.
            'd': ndarray
                Integration weights without considering the incident luminosity.
            'b': ndarray
                Integration weights including the incident luminosity.
            'E': ndarray
                Total incident luminosity defined as the integral of the incident SED over ln(lambda).

        Returns None if 'sed_inc' is not found in qh_nodes or if qh_nodes is None.
    """
    if qh_nodes is None or 'sed_inc' not in qh_nodes:
        return None
    w = np.asarray(wave_grid, dtype=float)
    o = np.argsort(w)                                         # the continuum grid is not sorted
    lw, lwi = np.log(w[o]), np.log(np.asarray(qh_nodes['wave_inc'], dtype=float))
    tw = np.zeros_like(lwi)                                   # trapezoid weights in ln(lambda)
    tw[:-1] += 0.5 * np.diff(lwi)
    tw[1:] += 0.5 * np.diff(lwi)
    j = np.clip(np.searchsorted(lw, lwi), 1, len(lw) - 1)     # lw[j-1] <= lwi <= lw[j]
    t = np.clip((lwi - lw[j - 1]) / (lw[j] - lw[j - 1]), 0.0, 1.0)
    J = np.unique(np.concatenate([o[j - 1], o[j]]))           # continuum columns used by the interpolation
    W = np.zeros((len(lwi), len(J)))                          # interpolation matrix, incident <- continuum (J)
    rows = np.arange(len(lwi))
    np.add.at(W, (rows, np.searchsorted(J, o[j - 1])), 1.0 - t)
    np.add.at(W, (rows, np.searchsorted(J, o[j])), t)
    L = np.asarray(qh_nodes['sed_inc'], dtype=float)          # (n_tau, n_Z, n_inc)
    return {'J': J, 'd': tw @ W, 'b': (L * tw) @ W, 'E': L @ tw}


def apply_constraints(tabs, qh_nodes, input_keys, theta_batch, out):
    """
    Adjusts and applies physical constraints to the decoded outputs in place.

    This function modifies the decoded outputs represented as a dictionary of
    output families, applying corrections for energy consistency. It adjusts
    transmission, continuum, and grain emission values to ensure physical
    correctness for attributes related to energy preservation and emission across
    different families.

    Parameters:
        tabs: dict
            Lookup tables for energy-related corrections, including scaling factors
            and energy integration values used for adjustments.
        qh_nodes: numpy.ndarray
            Quantized nodes for hierarchical interpolation of energy correction values.
        input_keys: list of str
            Keys corresponding to the input parameters that drive predictions.
        theta_batch: numpy.ndarray or list
            Batch of input parameter values for which the constraints are to be
            applied.
        out: dict
            Outputs decoded from the model, represented as a dictionary where each key
            corresponds to an output family and each value contains a tuple of mean
            values and uncertainties (may contain None for uncertainties).

    Returns:
        dict:
            The same `out` dictionary after in-place modifications, where transmission
            values are rescaled, continuum values are adjusted to preserve energy
            consistency, and grain emission values are restricted to adhere to
            continuum limits.

    Raises:
        (Not explicitly documented; exceptions may be raised during computation, such
        as ValueError for invalid inputs or IndexError for array access errors.)
    """
    if tabs is None or 'energy' not in out:
        return out
    import galapy.internal.emulators as EMUL
    th = np.atleast_2d(np.asarray(theta_batch, dtype=float))
    it, lo, hi, w = _node_weights(qh_nodes, input_keys, th)
    b = (1.0 - w)[:, None] * tabs['b'][it, lo] + w[:, None] * tabs['b'][it, hi]       # (n, nJ)
    E = (1.0 - w) * tabs['E'][it, lo] + w * tabs['E'][it, hi]                          # (n,)
    J = tabs['J']
    # energy scalars: odds of a and q = c / a
    (odds, q), sig_e = np.asarray(out['energy'][0], float).T, out['energy'][1]
    a = np.clip(odds / (1.0 + odds), EMUL.ENERGY_A_FLOOR, 1.0 - 1e-12)
    c = q * a
    # T -> T^k only with EMUL.ENERGY_RESCALE True (default false)
    if EMUL.ENERGY_RESCALE and 'trans' in out:
        T, sT = out['trans']
        lnT = np.log(np.maximum(T[:, J], 1e-300))
        ln_target = np.log((1.0 - a) * E)
        k_lo, k_hi = np.log(EMUL.ENERGY_K_RANGE[0]), np.log(EMUL.ENERGY_K_RANGE[1])
        u = np.zeros(len(th))                                 # u = ln k, from k = 1
        for _ in range(50):
            P = b * np.exp(np.exp(u)[:, None] * lnT)          # b T^k
            S = P.sum(1)
            f = np.log(np.maximum(S, 1e-300)) - ln_target
            fp = np.exp(u) * (P * lnT).sum(1) / np.maximum(S, 1e-300)   # d ln S / d ln k (<= 0)
            step = np.where(fp < -1e-300, -f / np.where(fp < -1e-300, fp, -1.0), 0.0)
            u = np.clip(u + np.clip(step, -2.0, 2.0), k_lo, k_hi)
            if np.all(np.abs(f) < 1e-10):
                break
        k = np.exp(u)[:, None]
        Tk = np.exp(k * np.log(np.maximum(T, 1e-300)))
        out['trans'] = (Tk, None if sT is None else k * Tk / np.maximum(T, 1e-300) * sT)
    if 'cont' in out:
        C, sC = out['cont']
        #  cont -> s cont only with EMUL.ENERGY_RESCALE True (default False); else: s = 1, grain <= cont applied
        s = ((c * E / np.maximum(C[:, J] @ tabs['d'], 1e-300))[:, None] if EMUL.ENERGY_RESCALE
             else np.ones((len(C), 1)))
        out['cont'] = (s * C, None if sC is None else s * sC)
        if 'grain' in out:
            G, sG = out['grain']
            out['grain'] = (np.minimum(s * G, s * C), None if sG is None else s * sG)
    sa = None if sig_e is None else np.asarray(sig_e, float)[:, 0] / (1.0 + odds) ** 2
    sc = None if sig_e is None else c * np.hypot(np.asarray(sig_e, float)[:, 1] / q, sa / a)
    out['energy'] = (np.column_stack([a, c]), None if sig_e is None else np.column_stack([sa, sc]))
    return out

def zero_lyc_floor(wave_grid, out):
    """
    Continuum below the Lyman limit training floor is adjusted to zero in-place. Operates on
    a dictionary containing data families mapped to their respective mean and sigma values.
    This function modifies the mean values for wavelengths below the Lyman limit according to
    a threshold based on predefined constants and factors.

    Args:
        wave_grid (array-like): Array of wavelength grid values to determine positions relative
                                to the Lyman limit.
        out (dict): Dictionary where keys are data family names and values are tuples of mean
                    and sigma arrays. This will be updated in-place.

    Returns:
        dict: The updated dictionary where mean values below the Lyman limit are adjusted
              to zero, depending on thresholds and conditions.
    """
    import galapy.internal.constants as CONST
    if EMUL.LYC_ZERO_FACTOR <= 0:
        return out
    lyc = np.asarray(wave_grid, float) < CONST.LyLimit
    for f, rel in EMUL.REL_FLOOR_LYC.items():
        if f not in out or not rel:
            continue
        m, s = out[f]
        thr = EMUL.LYC_ZERO_FACTOR * rel * m.max(1, keepdims=True)
        m = m.copy()
        m[:, lyc] = np.where(m[:, lyc] < thr, 0.0, m[:, lyc])
        out[f] = (m, s)
    if 'cont' in out and 'grain' in out:
        out['grain'] = (np.minimum(out['grain'][0], out['cont'][0]), out['grain'][1])
    return out

def lines_below_floor(prep, L):
    """
    Flag lines below the floor threshold based on the training ratio factor.

    This function evaluates if decoded fluxes, represented by matrix L, fall below a floor threshold relative
    to a training ratio factor and reference line, as defined in the 'ratio' parameter of the 'prep' dictionary.
    The result is a boolean matrix where flagged lines are set to True, except for the reference line,
    which is always False. If the 'ratio' parameter in 'prep' is None, the function will return a matrix of
    all False values. This is relevant for absolute targets (non-ratio forms).

    Parameters:
        prep (dict): A dictionary containing ratio information. The 'ratio' key should provide a
                     dictionary with the following keys:
                     - 'floor': Factor defining the floor ratio.
                     - 'ref': Index of the reference line (H-beta).
        L (ndarray): A 2D array (N, n_lines) representing decoded fluxes for N samples
                     and multiple lines, where n_lines is the number of lines.

    Returns:
        ndarray: A boolean array of the same shape as L, where True indicates lines flagged as
                 below the floor, and False means not flagged. The reference line (specified in
                 'ref') is always set to False.
    """
    r = prep.get('ratio')
    if r is None:
        return np.zeros(np.shape(L), dtype=bool)
    L = np.asarray(L, float)
    flag = L < EMUL.LINE_FLOOR_FLAG_FACTOR * float(r['floor']) * L[:, [int(r['ref'])]]
    flag[:, int(r['ref'])] = False
    return flag


class HIIEmulator:
    """
    Emulator HII:
        - One SVGP for each spectral line
        - SVGPs for continuum with PCA
        - SVGP for dust mass
        - FUV scalars (T_FUV and N_FUV)

    Input - 9 inputs from HII
    """
    WIRING = {'lines': ('line_gp', 'lines_emergent'),
              'cont': ('nebular_gp', 'continuum_nebular'),
              'grain': ('grain_gp', 'continuum_grain'),
              'trans': ('transmission_gp', 'transmission'),
              'dust': ('dust_mass_gp', 'dust_mass_cloud'),
              'fuv_t': ('fuv_t_gp', 'T_fuv_hii'),
              'fuv_n': ('fuv_n_gp', 'N_fuv_hii'),
              'energy': ('energy_gp', 'energy_fractions')}

    def __init__(self):
        self.line_gp = None             # d_out = n_lines (master lines file)
        self.nebular_gp = None          # d_out = n_pca (nebular continuum + grains)
        self.grain_gp = None            # d_out = n_pca (grain continuum)
        self.transmission_gp = None     # d_out = n_pca (T = transmitted/incident)
        self.dust_mass_gp = None        # d_out = 1 (dust mass)
        self.fuv_t_gp = None            # d_out = 1 (log10 T_FUV)
        self.fuv_n_gp = None            # d_out = 1 (log10 N_FUV)
        self.prep = {}                  # preprocessing of every family (mu, sd, offset, pca)
        self.cache = {}                 # predictive_cache
        self.x_mean = None              # normalization of input
        self.x_std = None
        self.input_keys = None
        self.qh_nodes = None            # QH, unit per node (tau, Zstar). QH_unit_lookup
        self.device = torch.device('cpu')
        self.line_names = None
        self.line_wavelengths_rest = None
        self.wave_grid = None
        self.sed_inputs = False         # True when the GP was trained with the SED descriptors appended to the inputs
        self.energy_tabs = None         # weights of the energy integrals on the continuum grid (None for artifacts without the incident SED)
        self.energy_gp = None

    def load(self, artifact_stem, device = None):
        """
        Loads a pre-trained HII model from a .pt file, reconstructing SVGP models, precomputing predictive
        caches for evaluation, and maintaining preprocessing steps.

        Parameters:
        artifact_stem (str): Path stem to the .pt file(s) containing the trained model artifacts.
        device (str, optional): Specifies the device for computation ('cpu' or 'cuda'). Defaults to None,
                                which maps to 'cpu'.

        Returns:
        self: The instance of the class with the loaded model and required metadata.

        Raises:
        FileNotFoundError: If none of the specified artifact files exist at the provided path.
        """
        import pathlib as _pl
        stem = str(artifact_stem)                                                           # path without suffix
        stem = stem[:-3] if stem.endswith('.pt') else stem                                  # remove suffix in case
        parts = [f"{stem}_{p}.pt" for p in ('lines', 'cont', 'grain', 'trans', 'scalars')]  # names of all pt

        #uncomplete set never loaded
        if not all(_pl.Path(f).is_file() for f in parts):
            if _pl.Path(F"{stem}.pt").is_file():
                parts = [f"{stem}.pt"]
            else:
                raise FileNotFoundError(f"[hii-emulator] None of the files {parts} exist")

        # reading
        fam, ex = load_artifacts(parts)
        self.device = torch.device(device or 'cpu')
        ard = ex['ard_num_dims']
        # cycle on all the families of the emulator
        for name, (attr, _key) in self.WIRING.items():
            if name not in fam:
                continue
            sd = fam[name]['model']                                 # state dict of SVGP (name -> tensor)
            Z = sd['variational_strategy.inducing_points']          # (d_out, M, ard)
            # prior mean of the artifact (linear or constant)
            mean = 'linear' if 'mean_module.weights' in sd else 'constant'
            gp = BatchedSVGPS(torch.zeros_like(Z), Z.shape[0], ard, mean=mean) # empty SVGP
            gp.load_state_dict(sd)                                  # load it with emulator data
            setattr(self, attr, gp.eval())                          # model in inference mode
            # likelihood noise in the predictive variance
            lik = None
            if ex.get('predictive_noise', False):
                lik = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([Z.shape[0]]))
                lik.load_state_dict(fam[name]['likelihood'])

            # cache the predictions of the emulator
            self.cache[name] = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                                for k, v in predictive_cache(gp, lik).items()}
            self.prep[name] = ex[f'{name}_prep']
            # squared PCA components for the propagation of the variance, once (not at every call)
            if self.prep[name]['pca'] is not None:
                self.prep[name]['pca']['components_sq'] = np.asarray(self.prep[name]['pca']['components']) ** 2
            # sigma calibration of the training (s_k per output):
            # fast_predict multiplies the latent variance by s_k^2.
            s = self.prep[name].get('sigma_scale')
            if s is not None:
                self.cache[name]['var_scale'] = torch.as_tensor(np.asarray(s) ** 2, dtype=torch.float32,
                                                                device=self.device)

        # GP trained on (X - x_mean) / x_std, so needed for the correct rescale
        self.x_mean = torch.as_tensor(ex['x_mean'], dtype=torch.float32, device=self.device)
        self.x_std = torch.as_tensor(ex['x_std'], dtype=torch.float32, device=self.device)

        # common informations
        self.input_keys = list(ex['input_keys'])
        self.qh_nodes = ex['qh_unit_nodes']
        self.line_names = ex.get('line_names')
        self.line_wavelengths_rest = ex.get('lines_vacuum')
        self.wave_grid = ex.get('wave_grid')
        self.sed_inputs = bool(ex.get('sed_inputs'))
        self.energy_tabs = incident_tables(self.qh_nodes, self.wave_grid) if 'energy' in self.cache else None
        return self

    def standardize(self, theta_batch):
        """
        Standardize input data for processing in the Gaussian Process (GP) model.

        This function takes physical input data and standardizes it based on the mean
        and standard deviation of the relevant inputs. If the artifact was trained
        with Spectral Energy Distribution (SED) descriptors, these are appended during
        the process.

        Args:
            theta_batch (array-like): Input data of shape (N, 9); a batch of physical
                inputs that needs to be standardized.

        Returns:
            torch.Tensor: Standardized input data of shape (N, D), where D depends on
                whether SED descriptors are appended (D = 12 if SED inputs are used;
                otherwise D = 9). The standardized data is on the same device as
                defined by self.device.
        """
        th = np.atleast_2d(np.asarray(theta_batch, dtype=float))
        if self.sed_inputs:
            th = augment_inputs(self.qh_nodes, self.input_keys, th)
        x = torch.as_tensor(th, dtype=torch.float32, device=self.device)
        return (x - self.x_mean) / self.x_std

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "[hii-emulator] Training lives in a separate script file, galapy-train-hii. This class is for inference only.")

    def Qh_unit_lookup(self, theta_batch):
        """
        Determine the QH unit for 1 solar mass SSP for each set of input parameters in theta_batch. The function
        interpolates over metallicity and identifies the appropriate node for tau. This is used to scale predictions
        for QH_bin/QH_unit. If tau does not correspond to a valid node or if any output value is not finite, an
        error is raised.

        Parameters:
            theta_batch (np.ndarray): A batch of input parameters, where each row corresponds to a set of
                                      SSP parameters (tau_SSP, Z_star, etc.).

        Returns:
            np.ndarray: An array of QH unit values corresponding to each input parameter set.

        Raises:
            ValueError: If log10 tau_SSP is out of the valid range of nodes.
            ValueError: If QH unit values are absent or not finite for the requested nodes.
        """
        return qh_unit_lookup(self.qh_nodes, self.input_keys, theta_batch)

    def log_scale(self, name, theta_batch):
        """
        log10 of the analytic scale that decode needs for family `name` (lines in ratio form), None for the
        other families and for old artifacts with absolute line targets.
        """
        if self.prep.get(name, {}).get('ratio') is None:
            return None
        return line_log_scale(self.qh_nodes, self.input_keys, theta_batch)

    def predict_batch(self, theta_batch, with_sigma=True):
        """
        Predicts values for a batch of input theta and produces predictions for all families as a
        numpy array on the CPU.

        Parameters:
        theta_batch (array-like): Batch of input theta values to be processed and predicted.
        with_sigma (bool): False returns only the means (sigma None), without the predictive variance.

        Returns:
        dict: A dictionary containing predictions for all families in the dataset. Each key
              corresponds to a specific family, where the predictions may include means,
              standard deviations, or scalar values as described below:
              - For 'dust_mass_cloud': Returns a tuple where each element is a scalar (mean and std).
              - For 'T_fuv_hii' and 'N_fuv_hii': Returns a scalar without an associated error.
              - 'energy_fractions' (artifacts with the energy family): (n, 2) absorbed fraction a and continuum
                energy fraction c of the incident energy, with the transmission and the continuum rescaled to them.
        """
        x = self.standardize(theta_batch)
        dec = {}
        with torch.no_grad():                                                               # eval mode on families
            for name in self.cache:
                dec[name] = decode(self.prep[name], *fast_predict(self.cache[name], x, with_var=with_sigma),
                                   log_scale=self.log_scale(name, theta_batch))
        apply_constraints(self.energy_tabs, self.qh_nodes, self.input_keys, theta_batch, dec)
        # continuum below 912 A at the training floor
        zero_lyc_floor(self.wave_grid, dec)
        out = {self.WIRING[name][1]: v for name, v in dec.items()}
        sq = lambda v: None if v is None else v[:, 0]
        for key in ('dust_mass_cloud',):                                # dust mass is a scalar, not vector. remove
            out[key] = (out[key][0][:,0], sq(out[key][1]))              # useless dimension (mean, std)
        for key in ('T_fuv_hii', 'N_fuv_hii'):                          # scalar adimensional with no error
            out[key] = out[key][0][:,0]
        # True where the emulated L / H-beta is at the ratio floor of the training (not reliable, flagged)
        out['lines_below_floor'] = lines_below_floor(self.prep['lines'], out['lines_emergent'][0])
        return out


    CONSTRAINED = ('trans', 'cont', 'grain', 'energy')
    def physical_constrained(self, fam, mu, var, theta_batch):
        """
        Calculates the median and sigma of the specified family in physical units from the latent mean and variance of all
        families, applying constraints that couple different families together.

        This method processes the family data by mapping the latent space variables, applying physical constraints
        via `apply_constraints`, and ensuring proper adjustments, such as setting the continuum below a specific
        threshold to zero. It relies on specific family constraints and energy tabs where applicable.

        Parameters:
        fam : str
            The family identifier for which the calculations are performed.
        mu : dict
            Dictionary where keys are family identifiers and values are tensors
            representing the latent mean of the corresponding family.
        var : dict
            Dictionary where keys are family identifiers and values are tensors
            representing the latent variance of the corresponding family.
        theta_batch : torch.Tensor
            A batch tensor representing the input parameters for processing.

        Returns:
        torch.Tensor
            The decoded tensor in physical units for the specified family after applying the necessary constraints.
        """
        need = ([f for f in self.CONSTRAINED if f in mu] if fam in self.CONSTRAINED and self.energy_tabs is not None
                else [fam])
        dec = {f: decode(self.prep[f], torch.as_tensor(mu[f].T), torch.as_tensor(var[f].T),
                         log_scale=self.log_scale(f, theta_batch)) for f in need}
        apply_constraints(self.energy_tabs, self.qh_nodes, self.input_keys, theta_batch, dec)
        # continuum below 912 A at the training floor set to zero
        zero_lyc_floor(self.wave_grid, dec)
        return dec[fam]
