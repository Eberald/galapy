# Author: Enrico Veraldi
# the implementation of the emulator for the HII regions (inference class)

import numpy as np
import torch

from galapy.spectroscopy.svgp import (BatchedSVGPS, load_artifacts, predictive_cache, fast_predict, decode)

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
              'fuv_n': ('fuv_n_gp', 'N_fuv_hii')}

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
            sd = fam[name]['model']                                 # state dict of SVGP (name -> tensor)
            Z = sd['variational_strategy.inducing_points']          # (d_out, M, ard)
            gp = BatchedSVGPS(torch.zeros_like(Z), Z.shape[0], ard) # empty SVGP with the correct structure
            gp.load_state_dict(sd)                                  # load it with emulator data
            setattr(self, attr, gp.eval())                          # model in inference mode
            # cache the predictions of the emulator
            self.cache[name] = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k,v in predictive_cache(gp).items()}
            self.prep[name] = ex[f'{name}_prep']

        # GP trained on (X - x_mean) / x_std, so needed for the correct rescale
        self.x_mean = torch.as_tensor(ex['x_mean'], dtype=torch.float32, device=self.device)
        self.x_std = torch.as_tensor(ex['x_std'], dtype=torch.float32, device=self.device)

        # common informations
        self.input_keys = list(ex['input:keys'])
        self.qh_nodes = ex['qh_unit_nodes']
        self.line_names = ex.get('line_names')
        self.line_wavelengths_rest = ex.get('lines_vacuum')
        self.wave_grid = ex.get('wave_grid')
        return self

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
        th = np.asarray(theta_batch, dtype=float)                       # loading input parameters
        q = self.qh_nodes
        lt = th[:, self.input_keys.index('tau_SSP')]                    # take column of tau_SSP
        it = np.abs(lt[:, None] - q['log_tau'][None, :]).argmin(1)      # take nearest node
        if np.any(np.abs(q['log_tau'][it] - lt) > 1e-4):
            raise ValueError(f"[hii-emulator] log10 tau_SSP {lt} out of nodes range {q['log_tau']}")
        Zn = q['Z']
        Z = np.clip(10.0 ** th[:, self.input_keys.index('Z_star')], Zn[0], Zn[-1])              # clip Zstar to valid range
        hi = np.clip(np.searchsorted(Zn,Z), 1, len(Zn)-1)                                # two nodes Zstar used for interpolation
        w = (Z - Zn[hi - 1]) / (Zn[hi] - Zn[hi - 1])                                            # weight of interpolation
        w = np.where(np.isclose(w, 0.0), 0.0, np.where(np.isclose(w, 1.0), 1.0, w))      # attach the weight
        lo_q, hi_q = q['Qh'][it, hi - 1], q['Qh'][it, hi]                                       # two values of Qh used for interpolation
        out = np.where(w == 0.0, lo_q, np.where(w == 1.0, hi_q, (1.0 - w) * lo_q + w * hi_q))   # interpolation
        if not np.all(np.isfinite(out)):
            raise ValueError("[hii-emulator] Q_H, unit absent for the requested node: corpus does not cover the 48 nodes"
                             "(tau_SSP, Z_star)")

        return out

    def predict_batch(self, theta_batch):
        """
        Predicts values for a batch of input theta and produces predictions for all families as a
        numpy array on the CPU.

        Parameters:
        theta_batch (array-like): Batch of input theta values to be processed and predicted.

        Returns:
        dict: A dictionary containing predictions for all families in the dataset. Each key
              corresponds to a specific family, where the predictions may include means,
              standard deviations, or scalar values as described below:
              - For 'dust_mass_cloud': Returns a tuple where each element is a scalar (mean and std).
              - For 'T_fuv_hii' and 'N_fuv_hii': Returns a scalar without an associated error.
        """
        x = torch.as_tensor(theta_batch, dtype=torch.float32).to(self.device)               # to tensor
        x = (x - self.x_mean) / self.x_std                                                  # normalization
        out = {}
        with torch.no_grad():                                                               # eval mode on families
            for name, (_attr, key) in self.WIRING.items():
                out[key] = decode(self.prep[name], *fast_predict(self.cache[name], x))

        for key in ('dust_mass_cloud',):                                # dust mass is a scalar, not vector. remove
            out[key] = (out[key][0][:,0], out[key][1][:,0])             # useless dimension (mean, std)
        for key in ('T_fuv_hii', 'N_fuv_hii'):                          # scalar adimensional with no error
            out[key] = out[key][0][:,0]

        return out