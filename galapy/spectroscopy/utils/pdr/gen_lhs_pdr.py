# Author: Enrico Veraldi
# Generation of LHS grid for PDRs

import argparse
import pathlib
import numpy as np

import galapy.internal.constants as CONST
import galapy.internal.cloudy as CLOUDY
import galapy.internal.globs as GLOBS
from galapy.internal.data import DataFile

import galapy.spectroscopy.utils.lhs_core as CORE
from galapy.spectroscopy.utils.physics.abundances import Chemistry


_CHEM = None # shared chemical module between HII and PDR

def _chem():
    """
    class for load chemistry, done for reducing the overhead
    """
    global _CHEM
    if _CHEM is None:
        _CHEM = Chemistry()
    return _CHEM

def log_intensity_fuv(logG0):
    """
    Computes the logarithmic far-ultraviolet flux intensity

    The calculation applies the Habing one-sided flux normalization in base-ten
    logarithmic space and produces a scalar value suitable for downstream numerical
    processing.
    int_{6-13.6 eV} F_nu dnu = G0 x c u_Habing / 2 [erg cm^-2 s^-1]

    Parameters:
        logG0 (float): Base-ten logarithm of the FUV radiation field strength
            relative to the Habing field.

    Returns:
        float: Base-ten logarithm of the one-sided FUV flux intensity.

    Raises:
        TypeError: If logG0 cannot be used in numeric addition with a floating-point
            logarithmic normalization value.
        ValueError: If the Habing flux normalization cannot be converted to a valid
            base-ten logarithm.
    """
    return float(logG0 + np.log10(CONST.HABING_FLUX_ONE_SIDED))

def build_jobs(sample, names, ssp_meta):
    idx = {n: i for i, n in enumerate(names)}
    jobs = []

    for j, row in enumerate(sample):
        lognH = float(row[idx['lognH_PDR']])
        logG0 = float(row[idx['logG0']])
        log_zeta_O = float(row[idx['log_zeta_O']])
        A_V = float(row[idx['A_V_PDR']])
        log_chi_CR = float(row[idx['log_chi_CR']])
        xi_d = float(row[idx['xi_d']])
        z_cmb = float(row[idx['z_CMB']])
        F_star = float(row[idx['F_star']])
        # FUV tilt, approximated to the nearest multiple of BETA_STEP. so max 48xlevels(11) = 528
        beta = round(float(row[idx['beta']]) / CLOUDY.BETA_STEP) * CLOUDY.BETA_STEP

        for ssp in ssp_meta:
            jobs.append({
                'job_id' : f"{j:05d}_{ssp['it']:03d}_{ssp['iz']:02d}",
                'lognH_PDR' : lognH,
                'logG0' : logG0,
                'log_I_FUV' : log_intensity_fuv(logG0),
                'z_CMB' : z_cmb,
                'log_zeta_O' : log_zeta_O,
                'xi_d' : xi_d,
                'A_V_PDR' : A_V,
                'log_chi_CR' : log_chi_CR,
                'tau_SSP' : ssp['tau_SSP'],
                'Z_star' : ssp['Z_star'],
                'sed_base' : ssp['sed_file'],
                'sed_file' : ssp['sed_file'].replace('.sed', f'_b{abs(beta):.2f}.sed'),
                'beta' : beta,
                'F_star' : F_star,
                'element_scale_block' : "\n".join(_chem().element_scale_lines(log_zeta_O)),
                # dust mass per H = xi_d x metal mass per H (D/G = xi_d Z) in units of 'grains ISM'
                'grain_scale': xi_d * _chem().metal_mass_per_H(log_zeta_O) / CLOUDY.GRAIN_MASS_ISM_PER_H,
                # PAH scale
                'pah_scale': CLOUDY.PAH_SCALE_MW * xi_d * _chem().metal_mass_per_H(log_zeta_O) / CLOUDY.GRAIN_MASS_ISM_PER_H,
            })

    return jobs


def main():
    default_config = DataFile(CLOUDY.CONFIG_PDR_FILE, GLOBS.CONFIG_NEB).get_file()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--config', required=None, default=default_config,
                    help='charter grid_pdr.yaml, default galapy-dataset')
    ap.add_argument('-s', '--ssp-meta', required=None, default="data/cloudy_seds_trunc/ssp_metadata.json",
                    help='ssp_metadata.json ssp, default: data/cloudy_seds_trunc/ssp_metadata.json (galapy-sed-cloudy-extract --pdr)')
    ap.add_argument('-o', '--output', required=None, default="data/grids/pdr", help="output directory,"
                                                                                    "default: data/grids/pdr")
    args = ap.parse_args()

    charter = CORE.load_charter(args.config)
    if charter['sector'] != 'pdr' :
        raise ValueError("[lhs_pdr] charter not right sector, expected pdr")

    sample, names = CORE.sample_lhs(charter)
    ssp_meta = CORE.read_ssp_meta(args.ssp_meta)
    jobs = build_jobs(sample, names, ssp_meta)
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "lhs_pdr.npy", sample)
    CORE.write_grid_spec(out / "pdr_grid_spec.h5", jobs)
    print(f"[lhs_pdr] lhs_pdr.npy and pdr_grid_spec.h5 files created in {out}")


if __name__ == '__main__':
    main()
























