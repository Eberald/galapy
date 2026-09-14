# Author: Enrico Veraldi
# LHS core shared between hii and pdr

import numpy as np
import yaml
from scipy.stats import qmc

def load_charter(path):
    with open(path) as f:
        ch = yaml.safe_load(f)
    for key in ('sector', 'axes', 'lhs', 'nodes'):
        if key not in ch:
            raise ValueError(f"{path}: charter missing a crucial key: {key!r}")
    return ch

def sample_lhs(charter):
    """
    Generate samples using Latin Hypercube Sampling (LHS) for a given set of axes.

    This function leverages the LatinHypercube sampling method with optional
    optimization to minimize the discrepancy between the sampling distribution
    and uniformity. The generated samples are scaled according to the axis bounds,
    and logarithmic scales are applied where specified.

    Parameters:
        charter (dict): A dictionary containing the configuration for the LHS.

    Returns:
        tuple: A tuple containing:
            - ndarray: A 2D numpy array of shape (n_samples, d) containing the scaled LHS samples.
            - list: A list of axis names corresponding to the sampled dimensions.

    Raises:
        AssertionError: If LHS optimization does not reduce the discrepancy compared
                        to a plain LHS or if the absolute discrepancy exceeds 1e-3.

    Note:
        - The scaled values are adjusted to logarithmic space where specified.
        - The function ensures the sampled distribution maintains low discrepancy
          compared to a uniform distribution.
    """
    axes = charter['axes']
    names = [a['name'] for a in axes]
    d = len(axes)
    lo = np.array([float(a['min']) for a in axes])
    hi = np.array([float(a['max']) for a in axes])
    is_log = np.array([str(a.get('prior', 'uniform')).lower().startswith('log') for a in axes])
    lo_s = np.where(is_log, np.log10(np.where(is_log, lo, 1.0)), lo)
    hi_s = np.where(is_log, np.log10(np.where(is_log, hi, 1.0)), hi)

    n = int(charter['lhs']['n_samples'])
    seed = int(charter['lhs']['seed'])
    #random-cd random permutations between dimension coordinates for minimizing
    #the discrepancy (center discrepancy) from full uniform distribution
    opt = charter['lhs'].get('optimization', 'random-cd')
    unit = qmc.LatinHypercube(d=d, seed=seed, optimization=opt).random(n=n)
    # compute discrepancy (low value=near uniform) Centered $L_2$-Discrepancy Hickernell, F. J. (1998)
    cd_opt = qmc.discrepancy(unit)
    cd_plain = qmc.discrepancy(qmc.LatinHypercube(d=d, seed=seed).random(n=n))
    assert cd_opt < cd_plain, (
        f"LHS optimization ineffective: CD_opt={cd_opt:.3e} >= CD_plain={cd_plain:.3e} "
        f"(d={d}, n={n})")
    if  cd_opt < 1e-3:
        print(f"[WARNING] LHS absolute discrepancy CD_opt={cd_opt:.3e} > 1e-3 with d={d}, n={n}: "
              f"increase the n_samples for reducing the discrepancy")

    #rescale hypercube [0,1] to actual dimension of axes
    scaled = qmc.scale(unit, lo_s, hi_s)
    scaled[:, is_log] = 10.0 ** scaled[:, is_log]
    return scaled, names

def read_ssp_meta(path):
    """
    Reads and parses SSP metadata from a JSON file. (see physics/spectra.py)

    Arguments:
    path: str
        The file path to the JSON file containing SSP metadata.

    Returns:
    dict
        A dictionary representation of the parsed SSP metadata.
    """
    import json
    with open(path) as f:
        return json.load(f)


def write_grid_spec(path, jobs):
    """
    This function writes a grid specification to an HDF5 file. It processes a list of job dictionaries,
    extracting values for each key and storing them in datasets within the HDF5 file. String values are
    stored using a special string data type, while other data types are stored without conversion.

    Arguments:
        path (str): The file path where the HDF5 file will be created.
        jobs (list[dict]): A list of dictionaries representing job specifications (set free param).
            Each dictionary must have the same set of keys, and the data type of values for a given
            key must be consistent across all dictionaries.
    """
    import h5py
    with h5py.File(path, 'w') as f:
        for k in jobs[0]:
            vals = [j[k] for j in jobs]
            if isinstance(vals[0], str):
                f.create_dataset(k, data=vals, dtype=h5py.string_dtype())
            else:
                f.create_dataset(k, data=vals)