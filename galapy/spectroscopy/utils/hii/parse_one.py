# Author: Enrico Veraldi
# Parser sector HII to hdf5 files. create <job_id>.h5 (one per model)

import re
import functools
import pathlib
import numpy as np

from galapy.internal.constants import M_Sun_G, LyLimit, LymanA, FUV_Lo_A, FUV_Hi_A, NH_to_AV_MW, clight

def _load_columns(path, ncol_expected):
    """
    Loads and validates columns from a specified output file path.

    Parameters:
    path : str or pathlib.Path
        The file path to the data to be loaded.
    ncol_expected : int
        The expected number of columns in the data.

    Returns:
    numpy.ndarray
        A 2D NumPy array containing the loaded numerical data.

    Raises:
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If the data is not a 2D array or if the number of columns in the
        data does not match the expected value.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing CLOUDY output: {p}")
    arr = np.loadtxt(p, comments='#')
    if arr.ndim != 2:
        raise ValueError(f"{p}: expected 2D table, instead dimensions {arr.shape} (check .out)")
    if arr.shape[1] != ncol_expected:
        raise ValueError(f"{p}: {arr.shape[1]} columns, expected columns: {ncol_expected}. ")
    return arr


def parse_cloudy_con(path, cols=(1, 2, 3, 4, 9)):
    """Parse the -save last continuum ... units angstrom no hash- output from
    CLOUDY simulations.

    This function extracts specified columns from the continuum output
    of the CLOUDY run.

    The default column indices (1-based) extracted are:
        - 1: Wavelengths [Angstrom].
        - 2: Incident flux.
        - 3: Pure attenuated incident flux (stellar transmitted)
        - 4: Diffuse flux emitted by the cloud (gas + grains, including lines).
        - 9: Flux from emission lines only.

    Columns related to flux are expressed in units of nuFnu
    [erg cm^-2 s^-1].

    Raises:
        ValueError: If the wavelengths in the first column are not strictly
        monotonic (either strictly increasing or strictly decreasing),
        a ValueError is raised.

    Args:
        path (str): The path to the file containing the CLOUDY output.
        cols (tuple of int): The 1-based column indices to extract from the
            file. Defaults to (1, 2, 3, 4, 9).

    Returns:
        tuple of numpy.ndarray: A tuple of NumPy arrays representing the
        extracted data columns in the order specified by cols.
    """
    arr = _load_columns(path, 9)
    out = [arr[:, c - 1] for c in cols]
    wave = out[0]
    # striclty monotonic check on wavelengths
    d = np.diff(wave)
    if not (np.all(d < 0) or np.all(d > 0)):
        raise ValueError(f"{path}: wavelength column (1) not strictly monotonic ")
    return tuple(out)


# diagnostic file extraction
def parse_cloudy_cong(path):
    """Parses the CONG file to extract the total contribution of dust components.

    This function extracts the last column that represents the total contribution from
    dust grains (graphite, silicates, and overall total) in an optically-thin limit.

    Args:
        path (str): The file path of the CONG diagnostic file.

    Returns:
        numpy.ndarray: A 1D array containing the total contribution values extracted
        from the file.
    """
    arr = _load_columns(path, 4)   # lambda | graphite | silicates | total
    return arr[:, -1]


def parse_cloudy_linelist(path):
    """
    Parses the CLOUDY emission line list output and extracts labels, wavelengths, and fluxes.

    This function processes a text file containing a list of emission lines saved
    in the format specific to the CLOUDY photoionization code. Each emission line is
    described by a label, a wavelength, and a flux intensity.

    Parameters:
        path (str): The path to the text file containing the Cloudy emission line list.

    Returns:
        dict: A dictionary with the following keys:
            - 'names' (numpy.ndarray): Array of extracted line labels as strings.
            - 'wavelengths' (numpy.ndarray): Array of wavelengths in Angstroms.
            - 'fluxes' (numpy.ndarray): Array of flux intensities.

    Raises:
        ValueError: If the file contains unparsable lines, invalid wavelength formats,
                    or lacks any valid data entries.
    """
    names, waves, fluxes = [], [], []
    unit_re = re.compile(r'^([0-9.eE+-]+)\s*([AaMmCc]?)$')   # es. 6562.80A, 157.6m, 3727
    scale = {'': 1.0, 'A': 1.0, 'M': 1e4, 'C': 1e8}           # conersion of Angstroms, Micron and Cm to Angstroms
    for raw in pathlib.Path(path).read_text().splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"{path}: parse not possible for line: {raw!r}")
        flux = float(parts[-1])
        m = unit_re.match(parts[-2]) #wavelength part
        if not m:
            raise ValueError(f"{path}: lambda with not recognize unit in: {raw!r} ")
        wl = float(m.group(1)) * scale[m.group(2).upper()]   # put unit in Maiusc, scale wl
        label = ' '.join(parts[:-2]) # re-compose token before the -2 in a single one ("H" "1" become "H 1")
        names.append(label)
        waves.append(wl)
        fluxes.append(flux)
    if not names:
        raise ValueError(f"{path}: no lines extracted")
    return {'names': np.array(names), 'wavelengths': np.array(waves),
            'fluxes': np.array(fluxes)}


def integrate_grain_abundance(path):
    """
    Integrates grain mass surface density from a file with grain abundance data.

    This function calculates the dust surface density, Sigma_d [g cm^-2],
    from a file produced with the 'save last grain abundance' command. The file
    contains zone-by-zone data where the grain mass density is reported, allowing
    for the integration of density over depth.

    file save last grain abundance:
        - Column 1 (depth_mid_zone) represents the depth in cm.
        - The last column ('total') is the summed mass density of dust per bin [g cm^-3].

    The surface density is computed using:
        Sigma_d = integral of (rho_d(depth) * d(depth)),
    where rho_d is the dust mass density [g cm^-3], and depth is given in cm,
    resulting in Sigma_d [g cm^-2].

    Notes:
    - This implementation assumes the 'last' keyword is used when generating
      the input file to select only the final converged iteration of the simulation.
      Without the 'last' keyword, files may concatenate profiles across iterations,
      causing overestimation of integrals.
    - The header format '#Depth<TAB>...<TAB>total' is strictly verified. Deviations
      may indicate changes in file format across versions or incorrect simulation runs.
    - Depth values in the file must be strictly monotonically increasing; otherwise,
      the file may contain concatenated iterations.

    Parameters:
        path (str): The file path to the grain abundance data.

    Returns:
        float: The integrated dust surface density, Sigma_d, in [g cm^-2].

    Raises:
        ValueError: If the file header is missing, malformed, or inconsistent with
                    expected format.
        ValueError: If the file contains no valid data rows.
        ValueError: If depth values are not strictly increasing, indicating a
                    problem with file content or simulation setup.
        ValueError: If the file contains data for fewer than two zones, which
                    renders column integrals undefined.
    """
    p = pathlib.Path(path)
    rows, header = [], None
    for raw in p.read_text().splitlines():
        if raw.startswith('#'):
            header = raw
            continue
        if raw.strip():
            rows.append([float(x) for x in raw.split('\t') if x.strip()])
    if header is None or not header.lower().startswith('#depth'):
        raise ValueError(f"{p}: expected header '#Depth<TAB>...<TAB>total', find {header!r} (format changed?)")
    if not header.rstrip().lower().endswith('total'):
        raise ValueError(f"{p}: last column is not'total': {header!r}")
    if not rows:
        raise ValueError(f"{p}: no zone found, aborted ")

    a = np.array(rows)
    depth, rho_d = a[:, 0], a[:, -1]      # depth [cm] , dust density [g cm^-3]

    # with 'last' the depth is striclty increasing, if not error (more iterations saved not the last)
    if np.any(np.diff(depth) <= 0):
        raise ValueError(f"{p}: depth not strictly increasing — concatenation of more then one iteration.")
    if len(depth) < 2:
        raise ValueError(f"{p}: only one zone in the file, column integral undefined.")

    return float(np.trapezoid(rho_d, depth))    # dust surface density [g cm^-2]


def fesc_from_recipe_C(wave_A, col2_incident, col3_transmitted):
    """
    Calculate the escape fraction f_esc by determining the ratio of
    ionizing photon rates (E > 1 Ryd, λ < 911.6 Å) between the transmitted and
    incident components.

    This function computes f_esc as the ratio of the integrated flux for wavelengths
    less than 911.6 Å from the transmitted component to the integrated flux
    from the incident component. The transmitted component is the pure stellar
    transmission without contamination from diffuse emission or lines, which aligns
    with the standard definition of Lyman Continuum (LyC) leakage.

    The calculation involves the numerical integration of both the transmitted and
    incident flux columns across the relevant wavelength range, given νFν (erg cm^-2 s^-1).
    The integration is performed in terms of wavelength (λ), eliminating the need for
    additional spectral weighting or conversions.

    Parameters:
    wave_A: ndarray
        Array of wavelengths in Ångströms.
    col2_incident: ndarray
        Array of incident flux values corresponding to wave_A.
    col3_transmitted: ndarray
        Array of transmitted flux values corresponding to wave_A.

    Returns:
    float
        The escape fraction f_esc, representing the ratio of transmitted to incident
        ionizing photon flux for wavelengths below 911.6 Å. Returns 0.0 if the
        wavelength range criterion is not met or the denominator is non-positive.
    """
    m = wave_A < LymanA
    if not np.any(m):
        return 0.0
    x = wave_A[m][::-1]
    num = np.trapezoid(col3_transmitted[m][::-1], x)
    den = np.trapezoid(col2_incident[m][::-1], x)
    return float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0

def fuv_transmittance_hii(wave_A, col2_incident, col3_transmitted, col4_own):
    """
    Computes the FUV transmittance (T_FUV) and nebular contribution (N_FUV) for the ionised part
    and its associated nebular continuum in the Habing band (6-13.6 eV).

    Summary:
    - T_FUV is the fraction of the initial FUV radiation transmitted through the ionized skin,
      defined as the ratio of integrated transmitted flux (col3) to the integrated incident flux
      (col2) over the 6-13.6 eV band.
    - N_FUV represents the nebular continuum flux emitted by the ionized skin in the FUV band,
      consisting of components like two-photon emission (1400 Å) and Ly-alpha (10.2 eV),
      normalized by the integrated incident flux (col2).

    This computation accounts for the influence of dust attenuation, Stromgren column
    parameters, and the enhancements from the nebular continuum. The result is dimensionless
    and provides inputs for simulating the FUV impact on photodissociation regions
    (PDRs) and related computations such as G0.

    Parameters:
    wave_A: array_like
        Wavelength array in Ångströms, defining the spectral range of interest.
    col2_incident: array_like
        Flux density corresponding to the incident FUV radiation in units of erg cm^-2 s^-1.
    col3_transmitted: array_like
        Flux density corresponding to the transmitted FUV radiation in units of erg cm^-2 s^-1.
    col4_own: array_like
        Flux density corresponding to the nebular continuum contribution in units of
        erg cm^-2 s^-1.

    Returns:
    tuple(float, float)
        - T_FUV: Dimensionless FUV transmittance ratio (0.0 to 1.0).
        - N_FUV: Dimensionless FUV nebular continuum contribution (≥ 0.0).
    """
    m = (wave_A >= FUV_Lo_A) & (wave_A <= FUV_Hi_A)
    if not np.any(m):
        return 0.0, 0.0
    x = wave_A[m][::-1]
    den = np.trapezoid(col2_incident[m][::-1], x)
    if den <= 0:
        return 0.0, 0.0
    T_fuv = float(np.clip(np.trapezoid(col3_transmitted[m][::-1], x) / den, 0.0, 1.0))
    N_fuv = float(max(np.trapezoid(col4_own[m][::-1], x) / den, 0.0))
    return T_fuv, N_fuv


def support_safe_ratio(col2, col3):
    """
    Calculate the safe ratio of col3 to col2 with specific handling for zero-division cases.

    Parameters:
        col2 (array-like of float): The denominator input array. It is expected to have non-negative
                                        values, representing the incident light column. Any value <= 0
                                        results in a default ratio of 1.
         col3 (array-like of float): The numerator input array, representing the observed or attenuated
                                        light column.

    Returns:
        numpy.ndarray: The computed safe ratio array with the same shape as `col3`.
    """
    col2 = np.asarray(col2, dtype=float)
    col3 = np.asarray(col3, dtype=float)
    out = np.ones_like(col3)
    np.divide(col3, col2, out=out, where=col2 > 0)
    return out


def s_k_factor(Qh_unit_node, logU, lognH):
    """
    Geometric factor.

    This factor is calculated using the 'ionization parameter', and
    CLOUDY outputs are given per unit AREA at the illuminated surface.
    The ionizing flux is expressed as phi_H = U * n_H * c
    [photons cm^-2 s^-1]. For a 1 Msun cluster of SSP formed at node
    k=(tau_SSP, Z_star), with the ionizing rate Q_H,k(1 Msun)
    (extracted from the SED ), the
    effective area is computed as:
        s_k = Q_H,k(1 Msun) / (U * n_H * c)     [cm^2].

    This factor (s_k) converts all outputs provided per cm^2
    (such as absolute row values, col3/col4, CONG, Sigma_d) into
    luminosities/masses per 1 Msun of formed SSP.

    Args:
        Qh_unit_node: Ionizing photon rate from the SSP in photons/s.
        logU: Logarithm of the dimensionless ionization parameter.
        lognH: Logarithm of the hydrogen number density in cm^-3.

    Returns:
        float: Geometric factor that converts outputs from per cm^2
        to luminosity/mass per 1 Msun of formed SSP.
    """
    return Qh_unit_node / (10.0**logU * 10.0**lognH * clight['cm/s'])

@functools.lru_cache(maxsize=None)
def spec_index(spec_path):
    """
    Generates a mapping of job IDs to their respective row indices from the
    specified hii_grid_spec.h5 file. The function reads the file only once per
    process and employs caching to optimize performance.

    Parameters:
    spec_path: str
        The file path to the hii_grid_spec.h5 file to be processed.

    Returns:
    dict
        A dictionary where keys are job IDs (str) and values are their
        corresponding row indices (int) in the file.

    Raises:
    ValueError
        If duplicate job IDs are detected in the specified file.
    """
    import h5py
    with h5py.File(spec_path, 'r') as s:
        ids = [b.decode() if isinstance(b, bytes) else b for b in s['job_id'][:]]
    index = {j: i for i, j in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError(f"{spec_path}: duplicated job_id in hii_grid_spec.h5")
    return index

def main():
    import argparse
    import json
    import h5py

    ap = argparse.ArgumentParser(description="PARSE_ONE_HII: workdir -> fragment grid_point")
    ap.add_argument('--workdir', default="data/output_hii", help='path to workdir (should be data/output_hii)')
    ap.add_argument('--spec',  default="data/grids/hii/hii_grid_spec.h5", help='path to hii_grid_spec.h5 (should be data/grids/hii/hii_grid_spec.h5')
    ap.add_argument('--ssp-meta', default="data/cloudy_seds/ssp_metadata.json", help='path to metadata.json of spectra (should be data/cloudy_seds/metadata.json)')
    ap.add_argument('--out', default="data/work", help='path to output file (should be data/output_hii)')
    args = ap.parse_args()

    wd = pathlib.Path(args.workdir)
    job_id = wd.name
    pre = wd / f"hii_{job_id}"

    with h5py.File(args.spec, 'r') as s:
        ids = [b.decode() if isinstance(b, bytes) else b for b in s['job_id'][:]]
        i = ids.index(job_id)
        p = {k: s[k][i] for k in ('logU', 'lognH_HII', 'z_CMB', 'log_zeta_O',
                                  'xi_d', 'f_esc', 'F_star', 'tau_SSP', 'Z_star')}
    meta = json.load(open(args.ssp_meta))
    Qh = {(m['tau_SSP'], m['Z_star']): m['Qh_unit'] for m in meta}
    s_k = s_k_factor(Qh[(float(p['tau_SSP']), float(p['Z_star']))],
                     float(p['logU']), float(p['lognH_HII']))

    wave, col2, col3, col4, col9 = parse_cloudy_con(f"{pre}.con", cols=(1, 2, 3, 4, 9))
    cong = parse_cloudy_cong(f"{pre}.con_grain")
    nebular = col4 - col9
    transmission = support_safe_ratio(col2, col3)
    lam_support = wave[col2 > 0]
    sed_support = (float(lam_support.min()), float(lam_support.max()))
    line_data = parse_cloudy_linelist(f"{pre}.lines")
    Sigma_d = integrate_grain_abundance(f"{pre}.dusa")

    with h5py.File(args.out, 'w') as f:
        f.create_dataset('continuum/wave_grid', data=wave.astype('f4'))
        f.create_dataset('line_names', data=line_data['names'].astype('S'))
        f.create_dataset('lines_emergent/wavelengths_rest',
                         data=line_data['wavelengths'].astype('f4'))
        g = f.create_group(f"grid_point_{job_id}")
        g.attrs.update({k: float(p[k]) for k in
                        ('logU', 'lognH_HII', 'z_CMB', 'log_zeta_O',
                         'xi_d', 'f_esc', 'F_star', 'tau_SSP', 'Z_star')})
        flag = wd / 'converged.flag'
        g.attrs['not_converged'] = bool(flag.exists() and flag.read_text().strip() == '0')
        I2 = np.trapezoid(col2[::-1], wave[::-1])
        I3 = np.trapezoid(col3[::-1], wave[::-1])
        I4 = np.trapezoid(col4[::-1], wave[::-1])
        g.attrs['energy_balance_rel'] = float(abs(I2 - (I3 + I4)) / I2)
        f_esc_meas = fesc_from_recipe_C(wave, col2, col3)
        g.attrs['f_esc_meas'] = float(f_esc_meas)
        g.create_dataset('f_esc_meas', data=f_esc_meas)
        g.create_dataset('continuum/nebular_emission_per_Msun', data=(nebular * s_k).astype('f4'))
        g.create_dataset('continuum/grain_diag_per_Msun', data=(cong * s_k).astype('f4'))
        g.create_dataset('continuum/transmission', data=transmission.astype('f4'))
        g.attrs['sed_support_A'] = sed_support
        g.create_dataset('lines_emergent/fluxes', data=(line_data['fluxes'] * s_k).astype('f4'))
        g.create_dataset('dust_mass_per_Msun', data=Sigma_d * s_k / M_Sun_G)
        g.attrs['dust_mass_units'] = 'Msun per Msun SSP formed'
    print(f"[parse_one/hii] {job_id} -> {args.out}")


if __name__ == '__main__':
    main()