# Author: Enrico Veraldi
# SVGP (Sparse Variational Gaussian Process) main module shared between HII and PDR sectors
# Also opearation of load artifacts and decode the GP output in the form of physical units

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (CholeskyVariationalDistribution, VariationalStrategy)

################# SVGP CLASS #################
class BatchedSVGPS(ApproximateGP):
    """
    D SVGPs independent (One SVGP = One output), with the same input theta.
    D = n_lines or n_components_PCA (for continuum)
    """

    def __init__(self, inducing_points, d_out, ard_dims=None):
        """
        The ard_dims (Automatic Relevance Determination dimensions)
        is deduced from the number of inducing points

        This is done for the kernel Matérn-5/2. Kernel measures how much
        close are two points; with ard_dims there is a lenghtscale for each dimension.
        In this way the GP can learn how much is relevant (in terms of correlation) each
        single input parameter.
        """

        if ard_dims is None:
            ard_dims = inducing_points.size(-1)
        # inducing_points shape: (d_out, M, ard_dims)

        bs = torch.Size([d_out]) # build of d_out independent SVGP in a single object
        # variational mean m and cholensku variational covariance L
        # we build the variational distribution on inducing points , q(u) = N(m,S), with S from Cholesky decomposition
        var_dist = CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=bs)
        # variational strategy compute q(f(x)) for any input x, and compute KL(q(f(x)) || p(f(x))) (divergence)
        # for the ELBO. With learn_inducing_locations=True, the inducing points are optimized during training
        var_strat = VariationalStrategy(self, inducing_points, var_dist, learn_inducing_locations=True)
        super().__init__(var_strat) # need to create the SVGP object
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=bs) # mean of the prior (trainable) one per output
        # Covariance between points (Kernel):
        # 1) Computation of weighted distance using ARD lengthscales: r = sqrt(sum((\theta_i - \theta'_i)^2 / l_i^2))
        # 2) Covariance used in the kernel (MaternKernel): k(r) = (1 + sqrt(5)*r + 5*r^2/3) * exp(-sqrt(5)*r)
        #
        # nu=2.5 sets the kernel type. With nu=2.5 (Matérn 5/2), sampled functions are twice differentiable,
        # representing a good compromise between an RBF kernel (infinitely smooth) and Matérn-1/2 (rough functions).
        # scale kernel k_tot = sigma^2 * k(r) (scaled M atern Kernel)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims,
                                                        batch_shape=bs), batch_shape=bs)

    def forward(self,x):
        """
        return the predictive distribution p(f(x)) for any input x, using the mean and covariance modules.
        """
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )

###############################################


def load_artifacts(paths):
    """
    Read trained emulator artifacts from disk and load into RAM.

    This function processes a list of file paths, loads serialized PyTorch objects from
    the specified files, and organizes the contents into two dictionaries:
    - `fam`: Contains trained Gaussian Processes (GPs) representing various families such as
      lines, continuum, etc.
    - `ex`: Contains additional extras, such as hyperparameters or other metadata.

    Parameters:
    paths (list[str]): A list of file paths to .pt files containing the serialized
        emulator data.

    Returns:
    tuple[dict, dict]: A tuple containing two dictionaries:
        - The first dictionary (`fam`) comprises the trained GPs.
        - The second dictionary (`ex`) encompasses extra data.
    """
    fam, ex = {}, {}
    for f in paths:
        blob = torch.load(f, map_location='cpu', weights_only=False)
        fam.update(blob['families'])
        ex.update(blob['extras'])

    return fam, ex

@torch.no_grad()
def predictive_cache(gp):
    """
    Computes pre-computed quantities for variational strategy prediction in a GP model
    under a whitened approximation. These quantities are independent of the input data x
    and are useful for efficiently making predictions for new input data.

    VariationalStrategy (whitened) predict, with L = chol(K_ZZ + jitter):
        mean(x) = k_xZ L^-T m + c
        var(x)  = k_xx + jitter + k_xZ L^-T (S - I) L^-1 k_Zx

    Summary:
        - Computes the Cholesky decomposition of the covariance matrix of inducing points.
        - Computes key intermediary results using the variational distribution's mean and
          covariance parameters.
        - Returns these pre-computed quantities in a dictionary for subsequent prediction steps.

    Args:
        gp (Module): A Gaussian Process model containing the variational strategy, mean
                     module, covariance module, and related parameters.

    Returns:
        Dict[str, Union[Tensor, float]]: A dictionary containing pre-computed quantities:
            - Z (Tensor): The inducing points' positions.
            - A (Tensor): L^-T m, where L is the Cholesky factor of the covariance matrix
                          of inducing points and m is the variational mean.
            - Q (Tensor): L^-T (S - I) L^-1, where S is the variational covariance and I
                          is the identity matrix.
            - jitter (float): Jitter used for numerical stability in Cholesky decomposition.
            - c (Tensor): Constant mean of the GP model.
            - outputscale (Tensor): Output scale of the kernel.
            - lenghtscale (Tensor): Lengthscale of the kernel.
    """
    vs = gp.variational_strategy                                                # import the choosen variational strategy
    Z = vs.inducing_points.detach()                                             # (d, M, D)
    eye = torch.eye(Z.shape[-2], dtype=torch.float64)                           # identity matrix MxM
    Kzz = gp.covar_module(Z).to_dense().double()                                # K_ZZ, covariance prior btw inducing points
    L = torch.linalg.cholesky(Kzz + vs.jitter_val * eye)                        # factorize K_ZZ = L L^T, jitter for numerical stability
    Linv = torch.linalg.solve_triangular(L, eye.expand_as(L), upper=False)      # compute L_inv resolving LX=I
    q = vs._variational_distribution
    Ls = q.chol_variational_covar.double().tril()                               # L_S

    #pre-computed quantities
    A = (Linv.transpose(-1,-2) @ q.variational_mean.double().unsqueeze(-1)).squeeze(-1) # L^-T m
    Q = Linv.transpose(-1, -2) @ (Ls @ Ls.transpose(-1,-2) - eye) @ Linv                # L^-T (S - I) L^-1
    k = gp.covar_module                                                                 # kernel

    return {'Z': Z, 'A': A.float(), 'Q' : Q.float(), 'jitter' : float(vs.jitter_val),
            'c' : gp.mean_module.constant.detach().reshape(-1),
            'outputscale' : k.outputscale.detach().reshape(-1),
            'lengthscale' : k.base_kernel.lengthscale.detach()}

def fast_predict(cache, x):
    """
    compute the meand and variance of GP in new point x using quantities prepared from predictive cache
    (same as gp(x) of pytorch, but without passing from it so that we can be more fast putting quantities in
    cache)

    """
    ls = cache['lengthscale'].to(x.device)                                                  # put lengthscale in device
    s = 5.0 ** 0.5 * torch.cdist(x.unsqueeze(0) / ls, cache['Z'].to(x.device) / ls)         # distance ARD
    K = cache['outputscale'][:, None, None] * (1.0 + s + s * s / 3.0) * torch.exp(-s)       # kernel between x and Z
    mean = torch.einsum('dnm,dm->dn', K, cache['A']) + cache['c'][:, None]
    var = (cache['outputscale'][:, None] + cache['jitter'] + torch.einsum('dnm,dmk,dnk->dn', K, cache['Q'], K))
    return mean, var.clamp_min(0.0)

def pca_decode(coeffs, components, pca_mean, mu, sd):
    """
    Decode PCA coefficients into the reconstructed physics field.

    This function takes the PCA coefficients produced by a Gaussian Process (GP),
    applies the inverse PCA transformation, and reconstructs the corresponding
    physics field (e.g., continuum field) by applying scaling and exponential
    transformation.

    Parameters
    ----------
    coeffs : list or numpy.ndarray
        The PCA coefficients produced by the GP. Expected as a 1D or 2D array.
    components : numpy.ndarray
        The PCA components matrix used during PCA decomposition. Should be 2D.
    pca_mean : numpy.ndarray
        The mean vector of the data used during PCA decomposition. Expected as
        a 1D array.
    mu : numpy.ndarray
        The mean of the field used to normalize the data. Expected as a 1D array.
    sd : numpy.ndarray
        The standard deviation of the field used to normalize the data. Expected
        as a 1D array.

    Returns
    -------
    numpy.ndarray
        The reconstructed physics field after applying the inverse PCA and
        data normalization. Output is a 2D array.

    Notes
    -----
    - The function assumes that all input arrays are compatible in terms of
      dimensions for matrix multiplication and addition.
    - The function performs the inverse PCA computation followed by unnormalization
      and exponential transformation to reconstruct the field.
    """
    z = np.asarray(coeffs).T @ np.asarray(components) + np.asarray(pca_mean)  # inverse PCA, (B, n_wl)
    log_field = z * np.asarray(sd) + np.asarray(mu)                           # de-normalization
    return np.power(10.0, log_field)

def lines_to_ratio(ytil, ref, log_scale, floor=0.0):
    """
    Line targets in ratio form, from the log10 of the line fluxes.

    The reference line (H-beta) becomes log10(L_ref) - log_scale, with log_scale = log10[Q_H (1 - f_esc)]
    known analytically from the inputs: this removes the orders of magnitude due to the ionizing photons.
    Every other line becomes log10(L_i / L_ref), floored at log10(floor) when floor > 0, so the GP does not
    spend its capacity on lines far below the reference. Inverted by ratio_to_lines.

    Parameters:
        ytil (np.ndarray): log10 of the line fluxes, (N, n_lines).
        ref (int): index of the reference line.
        log_scale (np.ndarray): log10 of the analytic scale of the reference line, (N,).
        floor (float): minimum of L_i / L_ref (0 = none).

    Returns:
        np.ndarray: the targets, (N, n_lines).
    """
    ytil = np.asarray(ytil, dtype=np.float64)
    t = ytil - ytil[:, [ref]]
    if floor > 0:
        t = np.maximum(t, np.log10(floor))
    t[:, ref] = ytil[:, ref] - np.asarray(log_scale, dtype=np.float64)
    return t


def ratio_to_lines(t, var_t, ref, log_scale):
    """
    Inverse of lines_to_ratio: mean and variance of log10 of the line fluxes from those of the targets.
    The GPs of the lines are independent, so the variance of a line is its own plus that of the reference.
    """
    y_ref = t[:, ref] + np.asarray(log_scale, dtype=np.float64)
    y, v = t + y_ref[:, None], var_t + var_t[:, [ref]]
    y[:, ref], v[:, ref] = y_ref, var_t[:, ref]
    return y, v


def decode(prep, mean, var, log_scale=None):
    """
    Decode the Gaussian Process (GP) prediction from the transformed space of training
    (normalized, log10, PCA optionally) into mean and variance in physical units.

    This function handles the transformation of the GP-predicted mean and variance back
    into the physical units based on the preprocessing configuration. If the preprocessing
    involves PCA (Principal Component Analysis), the function applies the inverse of the
    PCA transformation; otherwise, it performs a direct transformation from the log10
    normalized space. The resulting mean and variance are adjusted accordingly.

    Parameters:
        prep (dict): A dictionary containing preprocessing configuration. Expected keys:
            - 'pca' (dict or None): Components and mean from PCA preprocessing, if used.
            - 'mu' (array-like): Training dataset's mean in log10 normalized space.
            - 'sd' (array-like): Training dataset's standard deviation in log10 normalized space.
            - 'offset' (float, optional): Offset value computed during training, if available.
        mean (torch.Tensor): Predicted mean by the GP in the transformed space.
        var (torch.Tensor): Predicted variance by the GP in the transformed space.
        log_scale (array-like, optional): log10 of the analytic scale of the reference line, one per
            input point. Required when prep['ratio'] is set (lines in ratio form, see lines_to_ratio).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - y (np.ndarray): Decoded mean in physical units.
            - sigma (np.ndarray): Decoded standard deviation in physical units.
    """
    # from fast predict. in cpu
    mu_z = mean.detach().cpu().numpy()      # (d, N)
    var_z = var.detach().cpu().numpy().T    # (N, d)
    pca = prep['pca']
    #case of pca or not, different decode
    if pca is not None:
        C = np.asarray(pca['components'])                               # (n_pc, n_wl)
        if 'score_sd' in pca:                                           # undo the whitening of the scores
            s = np.asarray(pca['score_sd'])
            mu_z, var_z = mu_z * s[:, None], var_z * s ** 2
        y = pca_decode(mu_z, C, pca['mean'], prep['mu'], prep['sd'])    # (N, n_wl)
        var_z = var_z @ C ** 2                                          # (N, n_wl)
    elif prep.get('ratio') is not None:
        if log_scale is None:
            raise ValueError("[svgp] lines in ratio form: decode needs log_scale (log10 Q_H (1 - f_esc))")
        sd = np.asarray(prep['sd'])
        ylog, vlog = ratio_to_lines(mu_z.T * sd + np.asarray(prep['mu']), var_z * sd ** 2,
                                    prep['ratio']['ref'], log_scale)
        y = np.power(10.0, ylog)
        off = float(prep.get('offset', 0.0))
        return y - off, np.log(10.0) * y * np.sqrt(vlog)
    else:
        y = np.power(10.0, mu_z.T * np.asarray(prep['sd']) + np.asarray(prep['mu']))    # (N, d)

    sig_log = np.sqrt(var_z) * np.asarray(prep['sd'])   # std deviation, multiply by sd away the training division
    off = float(prep.get('offset', 0.0))                # offset computed by training (if present)
    return y - off, np.log(10.0) * y * sig_log          # return mean and sigma

