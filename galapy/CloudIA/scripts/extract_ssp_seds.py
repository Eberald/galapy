# script extracting table sed for Cloudy (CLOUDIA module)
# Author: Enrico Veraldi

import os, json
import numpy as np
from galapy.CompositeStellarPopulation import CSP

taus = np.array([1e6,2e6,5e6,1e7,2e7,5e7,7e7,1e8])
Z = np.array([0.0001,0.005,0.0010,0.0040, 0.0080, 0.0200])

def extract_ssp_seds(outdir, target_taus = taus, target_Z  = Z, ssp_lib = "parsec22.NT"):
    """
    Extracts SSP (Single Stellar Population) SEDs (Spectral Energy Distributions) for given ages
    and metallicities and saves the resulting data and metadata to the specified location.

    The function uses an external CSP (Composite Stellar Population) library to extract the SEDs
    corresponding to the specified parameters. It saves the extracted SED files in the provided
    output directory and generates a metadata file summarizing the characteristics of the extracted SEDs.

    Parameters:
        outdir (str): The output directory where the SED files and metadata file will be saved.
        target_taus (list[float], optional): A list of target ages for which SEDs should be extracted.
                                             Defaults to the global variable 'taus'.
        target_Z (list[float], optional): A list of target metallicities for which SEDs should be
                                          extracted. Defaults to the global variable 'Z'.
        ssp_lib (str, optional): The name of the SSP library to use for the SED extraction.
                                 Defaults to "parsec22.NT".

    Returns:
        list[dict]: A list of dictionaries containing metadata for each extracted SED file. Fields in the
                    metadata include:
                    - 'tau_SSP' (float): The selected age for the SED.
                    - 'Z_star' (float): The metallicity of the stellar population.
                    - 'sed_file' (str): The name of the SED file.
                    - 'it' (int): The index corresponding to the selected age in the grid.
                    - 'iz' (int): The index corresponding to the selected metallicity in the grid.
                    - 'Qh_unit' (float): The ionizing photon rate derived from the SED.
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
            csp.cloudy_sed_extract(path, it=it, iz=iz)
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

    with open(os.path.join(outdir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata