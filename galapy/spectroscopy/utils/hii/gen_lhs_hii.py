# Author: Enrico Veraldi
# Generation of LHS grid for HII regions
# NOTE: implement also the python script for creating a grid via CLI command

import argparse
import pathlib
import numpy as np


import galapy.internal.constants as CONST
import galapy.internal.globs as GLOBS
from galapy.internal.data import DataFile
import  galapy.spectroscopy.utils.lhs_core as CORE
from galapy.spectroscopy.utils.physics.abundances import (Chemistry, grain_scale_from_xi_d)

_CHEM = None # shared chemical module

def _chem():
    """
    class for load chemistry, done for reducing the overhead
    """
    global _CHEM
    if _CHEM is None:
        _CHEM = Chemistry()
    return _CHEM

def stromgren_stop_logN(logU, f_esc):
    """
    Calculates column density corresponding to escape fraction in the Stromgren sphere model.

    N_S = U c / alpha_B is the hydrogen column of the Stromgren layer [cm^-2] (c in cm s^-1,
    alpha_B in cm^3 s^-1), and N_stop = max(1 - f_esc, 1e-3) N_S the column at which CLOUDY
    stops ('stop column density').

    Parameters:
        logU : float
            log10 of the dimensionless ionization parameter U = Phi_H / (n_H c), the ratio
            of the densities of ionizing photons and of hydrogen at the illuminated face.

        f_esc : float
            The escape fraction, representing the fraction of ionizing photons that
            escape the sphere, ranging from 0.0 (no escape) to 1.0 (full escape).

    Returns:
        float
            log10 of the stopping hydrogen column density N_stop [cm^-2].
    """
    N_S = (10.0 ** logU) * CONST.clight['cm/s'] / CONST.alphaB_1e4K
    N_stop = max(1.0 - f_esc, 1e-3) * N_S # N_stop <= N_S always
    return float(np.log10(N_stop))

def build_jobs(sample, names, ssp_meta):
    """
    Builds a list of job configurations from sample data and specified stellar population metadata
    for the HII regions

    Parameters:
    sample: list
        A list of sample rows, where each row contains parameter data for each job.
    names: list
        A list of names corresponding to the columns in `sample`.
    ssp_meta: list
        A list of metadata dictionaries for stellar population models. Each dictionary
        contains key-value pairs representing model-specific parameters.

    Returns:
    list
        A list of job dictionaries, where each dictionary contains the configuration
        for a specific job. Each job contains calculated parameters and metadata.
    """
    idx = {n: i for i, n in enumerate(names)}
    jobs = []
    for j, row in enumerate(sample):
        logU = float(row[idx['logU']])
        lognH = float(row[idx['lognH_HII']])
        log_zeta_O = float(row[idx['log_zeta_O']])
        z_cmb = float(row[idx['z_CMB']])
        xi_d = float(row[idx['xi_d']])
        f_esc_target = float(row[idx['f_esc_target']])
        F_star = float(row[idx['F_star']])

        for ssp in ssp_meta:
            jobs.append({
                'job_id' : f"{j:05d}_{ssp['it']:03d}_{ssp['iz']:02d}",
                'logU' : logU,
                'lognH_HII' : lognH,
                'z_CMB' : z_cmb,
                'log_zeta_O' : log_zeta_O,
                'xi_d' : xi_d,
                'f_esc_target' : f_esc_target,
                'log_N_stop' : stromgren_stop_logN(logU, f_esc_target),
                'tau_SSP' : ssp['tau_SSP'],
                'Z_star' : ssp['Z_star'],
                'sed_file' : ssp['sed_file'],
                'F_star' : F_star,
                'element_scale_block' : "\n".join(_chem().element_scale_lines(log_zeta_O)),
                'grain_scale' : (_chem().metallicity_from_zeta(log_zeta_O) / _chem().Z_GC)
                                * grain_scale_from_xi_d(xi_d),
            })
    return jobs

def main():
    default_config = DataFile(CONST.CONFIG_HII_FILE, GLOBS.CONFIG_NEB).get_file()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--config', required=None, default=default_config, help='charter grid_hii.yaml, default galapy-dataset')
    ap.add_argument('-s', '--ssp-meta', required=None, default="data/cloudy_seds/ssp_metadata.json",
                    help='ssp_metadata.json ssp, default: data/cloudy_seds/ssp_metadata.json (galapy-sed-cloudy-extract)' )
    ap.add_argument('-o', '--output', required=None, default="data/grids/hii",
                    help='output directory, default: data/grids/hii')
    args = ap.parse_args()

    charter = CORE.load_charter(args.config)
    if charter['sector'] != 'hii' :
        raise ValueError('[lhs_hii] charter not right sector, expected hii')

    sample, names = CORE.sample_lhs(charter)
    ssp_meta = CORE.read_ssp_meta(args.ssp_meta)
    jobs = build_jobs(sample, names, ssp_meta)
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / 'lhs_hii.npy', sample)
    CORE.write_grid_spec(out / 'hii_grid_spec.h5', jobs)
    print(f"[lhs_hii] lhs_hii.npy and hii_grid_spec.h5 files created in {out}")

if __name__ == '__main__':
    main()
