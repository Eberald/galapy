# script extracting table sed for Cloudy (CLOUDIA module)
# Author: Enrico Veraldi

import os, json
import numpy as np
from galapy.CompositeStellarPopulation import CSP

taus = np.array([1e6,2e6,5e6,1e7,2e7,5e7,7e7,1e8])

def extract_ssp_seds(outdir, target_taus = taus, ssp_lib = "parsec22.NT"):
    """
    Extract SSP SEDs for specified parameters and save metadata.

    This function generates Simple Stellar Population (SSP) Spectral Energy
    Distribution (SED) files for specified characteristic times (target_taus)
    and metallicity values, using a given SSP library. The output files are
    saved in the specified directory along with a metadata file in JSON format
    containing details about the generated SEDs.

    Parameters:
        outdir (str): Path to the output directory where SED files and metadata
                      will be saved.
        target_taus (list[float], optional): List of characteristic time
                      constants (tau) in years for which SSP SEDs will be
                      generated. Defaults to the global variable `taus`.
        ssp_lib (str): Name of the SSP library to be used for generating SEDs.
                      Defaults to "parsec22.NT".

    Returns:
        list[dict]: A list containing metadata dictionaries, each describing
                    one generated SSP SED file. Each dictionary contains the
                    following keys:
                        - tau_SSP (float): Characteristic time constant (in years)
                                          used to compute the SED.
                        - Z_star (float): Metallicity value associated with the SED.
                        - sed_file (str): Name of the generated SED file.
                        - it (int): Index of the time constant in the SSP library.
                        - iz (int): Index of the metallicity value in the SSP library.
                        - Qh_unit (float): Ionizing photon flux (Q_H) computed for the
                                          corresponding SSP.

    Raises:
        None
    """
    csp = CSP(ssp_lib=ssp_lib)
    t, Z = csp.ssp_t, csp.ssp_Z
    selected_it = [int(np.argmin(np.abs(t-tau))) for tau in target_taus]

    os.makedirs(outdir, exist_ok=True)
    metadata = []
    for it in selected_it:
        for iz in range(Z.size):
            file_name = f"ssp_tau{t[it]:.3e}_Z{Z[iz]:.4f}.sed"
            path = os.path.join(outdir, file_name)
            csp.to_cloudy_sed(path, it=it, iz=iz)
            metadata.append(
                {
                    'tau_SSP' : float(t[it]),
                    'Z_star' : float(Z[iz]),
                    'sed_file' : file_name,
                    'it' : it,
                    'iz' : iz,
                    'Qh_unit' : csp.cloudy_sed_QH(outdir),
                }
            )

    with open(os.path.join(outdir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata