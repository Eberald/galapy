# Author: Enrico Veraldi
# script extracting table sed for Cloudy (CLOUDIA module)

import argparse
import os
import json
import numpy as np
from galapy.CompositeStellarPopulation import CSP

taus = np.array([1e6,2e6,5e6,1e7,2e7,5e7,7e7,1e8])
Z = np.array([0.0001,0.005,0.0010,0.0040, 0.0080, 0.0200])

def extract_ssp_seds(outdir, target_taus = taus,
                     target_Z  = Z,
                     ssp_lib = "parsec22.NT", truncate_lyman=None):
    """
    Extracts single stellar population spectral energy distributions (SSP SEDs)
    and saves them into files, while optionally truncating the Lyman continuum.

    This function utilizes a stellar population synthesis library to compute
    the SEDs for specified SSP ages (`target_taus`) and metallicities (`target_Z`),
    and then writes the SEDs into individual files in the specified output directory.
    Additionally, metadata about the generated SEDs is saved in a JSON file
    when the Lyman continuum is not truncated.

    Parameters:
        outdir: str
            Path to the output directory where the SED files and metadata will be stored.
        target_taus: list of float, optional
            List of SSP ages (in yr) for which SEDs will be extracted. Default is global `taus`.
        target_Z: list of float, optional
            List of SSP metallicities to target (absolute metallicity). Default is global `Z`.
        ssp_lib: str
            The name of the stellar population synthesis library to use (ones galapy implement)
            Default is "parsec22.NT".
        truncate_lyman: float or None, optional
            Minimum wavelength (in Angstroms) to process, effectively with key goal of
            truncating the Lyman continuum if specified. Default is None.

    Returns:
        list of dict
            Metadata for the extracted SEDs are produced when the Lyman continuum is not truncated.
            Each dictionary includes the following keys:
            - tau_SSP (float): The SSP age in yr.
            - Z_star (float): The SSP metallicity.
            - sed_file (str): The name of the file containing the generated SED.
            - it (int): Index of the selected age in the SSP library.
            - iz (int): Index of the selected metallicity in the SSP library.
            - Qh_unit (float): Unit-integrated ionizing photon rate associated
              with the SED file.
    """
    csp = CSP(ssp_lib=ssp_lib)
    t, Z = csp.t, csp.Z
    selected_it = [int(np.argmin(np.abs(t-tau))) for tau in target_taus]
    selected_iz = [int(np.argmin(np.abs(Z - Zstar))) for Zstar in target_Z]


    os.makedirs(outdir, exist_ok=True)
    metadata = []
    for it in selected_it:
        for iz in selected_iz:
            file_name = f"ssp_tau{t[it]:.3e}_Z{Z[iz]:.4f}.sed"
            path = os.path.join(outdir, file_name)
            csp.cloudy_sed_extract(path, it=it, iz=iz, lambda_min_A=truncate_lyman)
            if truncate_lyman is None:
                metadata.append(
                    {
                        'tau_SSP' : float(t[it]),
                        'Z_star' : float(Z[iz]),
                        'sed_file' : file_name,
                        'it' : it,
                        'iz' : iz,
                        'Qh_unit' : csp.cloudy_sed_QH(path),
                    }
                )

    if truncate_lyman is None:
        with open(os.path.join(outdir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    return metadata


def main():
    """
    Entry of the script

    This function parses command-line arguments, extracts SSP (Simple Stellar Population) 
    SEDs (Spectral Energy Distributions) based on the provided metadata and settings, 
    and outputs the results to a specified directory. Additional options allow truncation 
    of wavelengths and inclusion of metadata.

    Arguments:
        --ssp-lib (str): The library identifier for SSP extraction (galapy doc). Default is "parsec22.NT".
        --out (str): The output directory for the generated .sed files. Required.
        --truncate-lyman (Optional[float]): Truncates the wavelengths to lambda >= the provided 
                                            value in Angstroms. If not specified, wavelengths 
                                            are untruncated. Default is None.
        --tau (Optional[float], nargs='+'): SSP ages in yr to extract (e.g., --tau 1e6 2e6 5e6).
                                            If not specified, uses default ages: [1e6, 2e6, 5e6, 1e7, 
                                            2e7, 5e7, 7e7, 1e8].
        --Z (Optional[float], nargs='+'): SSP metallicities to extract (e.g., --Z 0.0001 0.005 0.0010). 
                                          If not specified, uses default metallicities: [0.0001, 0.005, 
                                          0.0010, 0.0040, 0.0080, 0.0200].
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-s','--ssp-lib', default='parsec22.NT', help='library SSP')
    ap.add_argument('-o','--out', required=True, help='output directory .sed')
    ap.add_argument('-l','--truncate-lyman', type=float, default=None,
                    help='cut lambda')
    ap.add_argument('-t','--tau', type=float, nargs='+', default=None,
                    help='SSP ages in years (e.g., --tau 1e6 2e6 5e6)')
    ap.add_argument('--Z', type=float, nargs='+', default=None,
                    help='SSP metallicities (e.g., --Z 0.0001 0.005 0.0010)')
    args = ap.parse_args()
    
    target_taus = np.array(args.tau) if args.tau is not None else taus
    target_Z = np.array(args.Z) if args.Z is not None else Z
    n = extract_ssp_seds(args.out, target_taus=target_taus, target_Z=target_Z,
                         ssp_lib=args.ssp_lib, truncate_lyman=args.truncate_lyman)

    tag = 'truncated' if args.truncate_lyman is not None else 'complete'
    print(f"[extract_ssp] {len(n)} nodes {tag} -> {args.out}, the SED is {tag}")


if __name__ == '__main__':
    main()