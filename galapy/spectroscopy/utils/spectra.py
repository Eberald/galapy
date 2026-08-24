# Author: Enrico Veraldi
# extracting table sed for Cloudy (CLOUDIA module)
# this module is used for extraction of SSP or CSP from galapy in a table SED format
# (CLOUDY format) ready to be used in a CLOUDY input file
#
# the module works also as a script, used in this way
# $

import argparse
import os
import json
import numpy as np
from galapy.CompositeStellarPopulation import (CSP, load_SSP_table, reshape_SSP_table)
import galapy.internal.constants as CONST

#=============== DEFAULTS TAUS AND METALLICITIES ===============
taus = np.array([1e6,2e6,5e6,1e7,2e7,5e7,7e7,1e8])
Z = np.array([0.0001,0.0005,0.0010,0.0040, 0.0080, 0.0200])

#=============== LOADING FUNCTIONS FOR SEDS ===============

def load_ssp_cube(ssp_lib='parsec22.NT') -> tuple:
    """
    Loads the SSP cube from the specified spectral library.

    This function retrieves the data for wavelengths, time steps, metallicities,
    and the corresponding luminosity values by loading and reshaping the SSP table.

    Parameters:
        ssp_lib (str, optional): The name of the spectral library to load the SSP table from.
            Defaults to 'parsec22.NT'.

    Returns:
        tuple: A tuple containing:
            - l (numpy.ndarray): Array of wavelength values.
            - t (numpy.ndarray): Array of time step values.
            - Z (numpy.ndarray): Array of metallicity values.
            - L (numpy.ndarray): Reshaped 3D array of luminosity values with dimensions
              corresponding to (wavelengths, time steps, metallicities).
    """
    l, t, Z, L_flat = load_SSP_table(ssp_lib)
    L = reshape_SSP_table(L_flat, shape=(l.size, t.size, Z.size))
    return l, t, Z, L

def sed_from_ssp_cube_node(cube, it, iz) -> tuple:
    """
    Extracts a SED from an SSP cube node (tau_SSP, Z_star), 1Msun format.

    The function retrieves a specific spectral energy distribution (SED) from a
    given Simple Stellar Population (SSP) data cube based on the provided time
    and metallicity indices.

    Parameters:
        cube (tuple): A tuple containing the wavelength array, time grid, metallicity grid, and
            SED data cube. The expected structure is (l, _t, _Z, L), where:
            - l: Wavelength array.
            - _t: Time grid (ignored in this function).
            - _Z: Metallicity grid (ignored in this function).
            - L: Multidimensional array representing the SED cube.
        it (int): The index corresponding to the time grid in the SED cube.
        iz (int): The index corresponding to the metallicity grid in the SED cube.

    Returns:
        tuple: A tuple containing the wavelength array and the corresponding SED
            extracted from the input SSP cube at the specified indices.
    """
    l, _t, _Z, L = cube
    return l, L[:, it, iz]

def sed_from_csp(csp, age, sfh) -> tuple:
    """
    Generate the SED from a composite stellar population (CSP) model.

    This function sets the parameters for a given CSP model using the provided
    age and star formation history (SFH), and calculates the SED by computing
    the model's emission spectrum. This can be used in order to run a CLOUDY model
    with a specific CSP.

    Parameters:
        csp (CSP): The composite stellar population model object. It is expected to have the
            methods `set_parameters` and `core.emission`, and attributes `l` and `t`.
        age (float): The age (in appropriate time units) to set for the CSP model.
        sfh (array-like): The star formation history (SFH) to set for the CSP model. It should be
            compatible with the model's requirements.

    Returns:
        tuple: A tuple containing:
            - csp.l (numpy.ndarray): Wavelength grid for the SED (provided by the CSP model).
            - array (numpy.ndarray): Emission spectrum computed by the CSP model, reshaped to match the
              wavelength grid and provided SFH time structure.
    """
    csp.set_parameters(age, sfh)
    il = np.arange(len(csp.l), dtype=np.uint64)
    ftau = np.ones((len(il), csp.t.size))
    return csp.l, csp.core.emission(il, np.ascontiguousarray(ftau.ravel()))


# =============== WRITER ===============
def write_cloudy_sed(wavelength_A,
                     L_lambda,
                     outpath,
                     extrapolate=False,
                     lambda_min_A=None,
                     lambda_max_A=None) -> str:
    """
    Converts a SED to a CLOUDY table SED format.

    This function takes in wavelength and luminosity data, manipulates it to ensure appropriate formatting,
    and writes the output to a specified file in the format required by CLOUDY table SED. It allows for optional
    extrapolation and wavelength filtering.

    Parameters:
        wavelength_A (array-like): Array of wavelengths in Angstroms. Must contain at least two data points.
        L_lambda (array-like): Luminosity array corresponding to the wavelengths provided.
        outpath (str): File path where the CLOUDY table SED will be saved.
        extrapolate (bool, optional): If True, CLOUDY extrapolates the SED to the low-energy limit of the code.
            Defaults to False.
        lambda_min_A (float, optional): Minimum wavelength cutoff value in Angstroms. If provided, wavelengths below
            this value will be ignored. Defaults to None.
        lambda_max_A (float, optional): Maximum wavlenght cutoff value in Angstroms. If provided, wavelengths above
            this value will be ignored. Defaults to None.

    Raises:
        ValueError: Raised if the input wavelength array contains fewer than two data points or if the
            wavelength filtering leaves no data points.

    Returns:
        str: The file path where the CLOUDY table SED was written.
    """
    wavelength_A = np.asarray(wavelength_A, dtype=float)
    if wavelength_A.size < 2:
        raise ValueError('converting a SED to CLOUDY table sed format require at least 2 data')

    nuFnu = np.asarray(L_lambda, dtype=float) * wavelength_A * CONST.Lsun

    mask = np.ones(wavelength_A.shape, dtype=bool)
    if lambda_min_A is not None:
        mask &= wavelength_A >= float(lambda_min_A)
    if lambda_max_A is not None:
        mask &= wavelength_A <= float(lambda_max_A)
    if not np.all(mask):
        if not np.any(mask):
            raise ValueError(f'lambda_min_A={lambda_min_A} is too big for the SED (no points left after cut)')
        wavelength_A, nuFnu = wavelength_A[mask], nuFnu[mask]

    nuFnu = np.maximum(nuFnu, 1e-300)  # cut null fluxes
    order = np.argsort(wavelength_A)  # monotonic sort
    wavelength_A, nuFnu = wavelength_A[order], nuFnu[order]
    order = np.concatenate(([True], np.diff(wavelength_A) > 0.0))  # strictly monotonic sort
    wavelength_A, nuFnu = wavelength_A[order], nuFnu[order]

    # prefix units for table SED, in case of extrapolate, SED will be extrapolated to
    # the low-energy limit of the code
    prefix = "nuFnu units Angstroms" + (" extrapolate" if extrapolate else "")

    with open(outpath, 'w') as f:
        f.write('# CLOUDY table SED | col1 = lambda[Angstrom], col2 = nu*Fnu linear (erg/s per 1 Msun SSP)\n')
        for i, (w, n) in enumerate(zip(wavelength_A, nuFnu)):
            f.write(f'{w:.6e} {n:.6e} {prefix if i == 0 else ""}\n')

    f.close()
    return outpath

# =============== EXPORT ===============

def to_cloudy_sed(outpath,
                  cube = None,
                  csp = None,
                  age = None,
                  sfh = None,
                  it = None,
                  iz = None,
                  extrapolate = False,
                  lambda_min_A = None,
                  lambda_max_A = None) -> str:
    """
    Converts spectral data to a Cloudy-compatible SED file format.

    This function processes spectral data provided either as a preprocessed
    SSP cube or a CSP model. The data is then written to a file formatted for use
    with Cloudy.

    Parameters:
        outpath (str): The file path where the resulting SED data should be saved.
        cube (tuple, optional): Cube containing SSPs, typically structured spectral
            data with temporal and metallicity dimensions.
        csp (CSP, optional): Composite stellar population model representing
            star formation and age distributions.
        age (float, optional): Age associated with the stellar population. Typically required
            when using a CSP model.
        sfh (array-like, optional): Star formation history model. Required if using a CSP model.
        it (int, optional): Temporal index used to select an SSP node from the cube.
        iz (int, optional): Metallicity index to select an SSP node from the cube.
        extrapolate (bool, optional): Specifies whether to extrapolate data
            beyond the provided spectral range. Default is False.
        lambda_min_A (float, optional): Minimum wavelength (in Angstroms) to use
            for filtering or processing data.
        lambda_max_A (float, optional): Maximum wavelength (in Angstroms) to use
            for filtering or processing data.

    Raises:
        ValueError: Raised when neither (cube, it, iz) nor (csp, age, sfh)
            is provided as input.

    Returns:
        str: The output file path where the Cloudy-compatible SED file
            has been saved.
    """

    if cube is not None and it is not None and iz is not None:
        wave, spectrum = sed_from_ssp_cube_node(cube, it, iz)
    elif csp is not None and age is not None and sfh is not None:
        wave, spectrum = sed_from_csp(csp, age, sfh)
    else:
        raise ValueError('you must provide either (cube,it,iz) or (csp,age,sfh)')

    return write_cloudy_sed(wave, spectrum, outpath, extrapolate=extrapolate, lambda_min_A=lambda_min_A,
                            lambda_max_A=lambda_max_A)


# =============== QH ===============

def cloudy_sed_QH(sed_path, lyman_A=CONST.LymanA) -> float:
    """
    Calculate the hydrogen-ionizing photon rate from SED file.

    This function reads a spectral energy distribution (SED) from a file
    and calculates the hydrogen-ionizing photon rate using the continuum
    flux below the Lyman-alpha wavelength limit. The photon rate is calculated
    by integrating the flux over these wavelengths using the trapezoidal rule.

    Parameters:
        sed_path (str): Path to the SED file. The file must contain two columns:
            wavelength in Angstroms and flux in units of nu*F(nu).
        lyman_A (float, optional): The Lyman-alpha wavelength limit in Angstroms.
            Defaults to the constant CONST.LymanA

    Returns:
        float: The hydrogen-ionizing photon rate in photons per second.
    """
    wavelength_A, nuFnu = np.loadtxt(sed_path, comments='#', usecols=(0, 1), unpack=True)
    euv = wavelength_A < lyman_A
    if not np.any(euv):
        return 0.0
    wavelength_cm = wavelength_A[euv] * 1e-8
    order = np.argsort(wavelength_cm)
    return float(np.trapezoid(nuFnu[euv][order], wavelength_cm[order]) /
                 (CONST.hP["erg*s"] * CONST.clight["cm/s"]))


def cloudy_sed_QH_reference(cube, it, iz, lyman_A=CONST.LymanA) -> float:
    """
    Calculate Q_H (hydrogen-ionizing photon rate) from a raw SSP node in GalaPy format (1 Msun).

    This function computes the hydrogen-ionizing photon rate directly from the unwritten SSP table
    (raw data cube) for a specific age and metallicity node. It serves as a reference term.
    The calculation follows Eq. 42 from Ronconi+24 and integrates the spectral energy distribution
    below the Lyman limit to obtain the total ionizing photon rate per solar mass of stellar population
    formed.

    Parameters:
        cube (tuple): SSP data cube containing (wavelength, time, metallicity, luminosity arrays).
        it (int): Time index in the SSP cube corresponding to the desired stellar age.
        iz (int): Metallicity index in the SSP cube corresponding to the desired stellar metallicity.
        lyman_A (float, optional): Lyman limit wavelength in Angstroms. Defaults to CONST.LymanA.

    Returns:
        float: Hydrogen-ionizing photon rate in photons per second per solar mass.
    """
    l, _t, _Z, L = cube
    wavelength_A = np.asarray(l, dtype=float)
    L_lambda = np.asarray(L[:, it, iz], dtype=float)
    euv = wavelength_A < lyman_A
    if not np.any(euv):
        return 0.0
    wavelength_cm = wavelength_A[euv] * 1e-8
    nuFnu = L_lambda[euv] * wavelength_A[euv] * CONST.Lsun
    order = np.argsort(wavelength_cm)
    return float(np.trapezoid(nuFnu[order], wavelength_cm[order]) /
                 (CONST.hP["erg*s"] * CONST.clight["cm/s"]))


# =============== EXTRACTION SCRIPTS ===============

def extract_ssp_seds(outdir, target_taus = taus,
                     target_Z  = Z,
                     ssp_lib = "parsec22.NT", truncate_lyman=None, truncate_red=CONST.SED_cut,
                     extrapolate=False) -> tuple:
    """
    Extracts SSP (Simple Stellar Population) SEDs (Spectral Energy Distributions) for given target
    values of stellar ages and metallicities from a specified SSP library. The extracted data is saved
    to files in the output directory in table sed CLOUDY format.
    Metadata for the extracted SEDs is optionally created in JSON format.

    Parameters:
        outdir (str): Path to the output directory where the SED files and metadata will be saved.
        target_taus (list[float], optional): List of target stellar age values (in years) for which SEDs should be extracted.
            Default: taus.
        target_Z (list[float], optional): List of target stellar metallicities for which SEDs should be extracted.
            Default: Z.
        ssp_lib (str, optional): Name of the SSP library to load the cube from.
            Default: "parsec22.NT".
        truncate_lyman (float, optional): Minimum wavelength (in Angstroms) to include in the SED.
            If None, the SED is not truncated. Default: None.
        truncate_red (float, optional): Maximum wavelength (in Angstroms) to include in the SED.
            If None, the SED is not truncated on the red side. Default: CONST.SED_cut.
        extrapolate (bool, optional): If True, CLOUDY extrapolates the SED to the low-energy limit of the code.
            Default: False.

    Returns:
        tuple: A tuple containing:
            - int: Number of SED files generated.
            - list[dict]: Metadata for each extracted SED if `truncate_lyman` is None. The metadata includes:
                - tau_SSP (float): Stellar age value in years.
                - Z_star (float): Stellar metallicity.
                - sed_file (str): File name of the corresponding SED.
                - it (int): Index of the stellar age in the SSP cube.
                - iz (int): Index of the metallicity in the SSP cube.
                - Qh_unit (float): Hydrogen ionizing photon rate in units computed from the SED.
    """
    cube = load_ssp_cube(ssp_lib)
    _l, t, Z, _L = cube
    selected_it = [int(np.argmin(np.abs(t-tau))) for tau in target_taus]
    selected_iz = [int(np.argmin(np.abs(Z - Zstar))) for Zstar in target_Z]


    os.makedirs(outdir, exist_ok=True)
    metadata, n = [], 0
    for it in selected_it:
        for iz in selected_iz:
            file_name = f"ssp_tau{t[it]:.3e}_Z{Z[iz]:.4f}.sed"
            path = os.path.join(outdir, file_name)
            to_cloudy_sed(path, cube = cube, it=it, iz=iz, lambda_min_A=truncate_lyman,
                          lambda_max_A=truncate_red,extrapolate = extrapolate)
            n += 1
            if truncate_lyman is None:
                metadata.append(
                    {
                        'tau_SSP' : float(t[it]),
                        'Z_star' : float(Z[iz]),
                        'sed_file' : file_name,
                        'it' : it,
                        'iz' : iz,
                        'Qh_unit' : cloudy_sed_QH(path),
                    }
                )

    if truncate_lyman is None:
        with open(os.path.join(outdir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    return n, metadata


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
        --truncate-red (Optional[float]):  Truncates the wavelengths to lambda <= the provided
                                            value in Angstroms. If not specified, wavelengths
                                            are truncated at 1e6 A. Default is 1e6 A.
        --tau (Optional[float], nargs='+'): SSP ages in yr to extract (e.g., --tau 1e6 2e6 5e6).
                                            If not specified, uses default ages: [1e6, 2e6, 5e6, 1e7, 
                                            2e7, 5e7, 7e7, 1e8].
        --Z (Optional[float], nargs='+'): SSP metallicities to extract (e.g., --Z 0.0001 0.005 0.0010). 
                                          If not specified, uses default metallicities: [0.0001, 0.005, 
                                          0.0010, 0.0040, 0.0080, 0.0200].
    """
    ap = argparse.ArgumentParser(prog = "galapy-sed-cloudy-extract",description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-s','--ssp-lib', default='parsec22.NT', help='library SSP')
    ap.add_argument('-o','--out', required=True, help='output directory .sed')
    ap.add_argument('-l','--truncate-lyman', type=float, default=None,
                    help='cut lambda')
    ap.add_argument('-r','--truncate-red', type=float, default=CONST.SED_cut,
                    help=f'cut synchrotron: lambda <= value (A). Default {CONST.SED_cut:.3e}')
    ap.add_argument('-t','--tau', type=float, nargs='+', default=None,
                    help='SSP ages in years (e.g., --tau 1e6 2e6 5e6)')
    ap.add_argument('--Z', type=float, nargs='+', default=None,
                    help='SSP metallicities (e.g., --Z 0.0001 0.005 0.0010)')
    ap.add_argument('-e','--extrapolate', type=bool, nargs='+', default=None,
                    help='put extrapolate flag for make CLOUDY extrapolate low energy regime')
    args = ap.parse_args()
    
    target_taus = np.array(args.tau) if args.tau is not None else taus
    target_Z = np.array(args.Z) if args.Z is not None else Z
    n, _meta = extract_ssp_seds(args.out, target_taus=target_taus, target_Z=target_Z,
                                ssp_lib=args.ssp_lib, truncate_lyman=args.truncate_lyman,
                                truncate_red=args.truncate_red,extrapolate=args.extrapolate)

    tag = 'truncated' if args.truncate_lyman is not None else 'complete'
    print(f"[extract_ssp] {n} nodes {tag} -> {args.out}, the SED is {tag}")


if __name__ == '__main__':
    main()