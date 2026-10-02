# Author: Enrico Veraldi
# Parser sector HII to hdf5 files. create <job_id>.h5 (one per model)

import contextlib
import functools
import os
import pathlib
import re
import sys
import numpy as np

from galapy.internal.constants import M_Sun_G, LymanA, FUV_Lo_A, FUV_Hi_A, clight, Stellar_Max_A as STELLAR_MAX_A


#============================== UNITS
# Units of everything the parser writes, stored in the fragment itself
UNITS_SCHEMA = 'cloudia.hii.v2'

# root units datasets
ROOT_UNITS = {
    'continuum/wave_grid': (
        'Angstrom', 'wavelength of the CLOUDY continuum mesh, decreasing (increasing energy)',
        {'wavelength_medium': 'vacuum'}),
    'line_names': (
        None, 'CLOUDY labels of the lines (internal spaces collapsed), in the order of hii_lines.dat',
        {}),
    'lines_emergent/wavelengths_rest': (
        'Angstrom', 'rest wavelengths of the lines as printed by CLOUDY in their labels',
        {'wavelength_medium': 'air above 2000 A, vacuum below (CLOUDY print convention)'}),
    'lines_emergent/wavelengths_rest_vacuum': (
        'Angstrom', 'rest wavelengths of the lines in vacuum (air_to_vacuum_A)',
        {'wavelength_medium': 'vacuum'}),
    'incident/wave_grid': (
        'Angstrom', 'wavelength of the incident stellar SED, the table SED file read by CLOUDY '
                    '(galapy-sed-cloudy-extract), increasing',
        {'wavelength_medium': 'vacuum'}),
}

# members of grid_point_<job_id>
POINT_UNITS = {
    # grid axes and SSP node, copied from the spec
    'logU': ('dex', 'log10 of the ionization parameter U = Phi_H / (n_H c) at the illuminated face'),
    'lognH_HII': ('dex(cm-3)', 'log10 of the hydrogen density n_H (CLOUDY hden)'),
    'z_CMB': ('', 'redshift of the CMB of the run, T_CMB = 2.725 (1 + z_CMB) K'),
    'log_zeta_O': ('dex', 'log10 zeta_O, zeta_O = (O/H) / (O/H)_GC (Nicholls+17)'),
    'xi_d': ('', 'dust-to-metal mass ratio'),
    'f_esc_target': ('', 'target escape fraction of the H-ionizing photons (sets the stopping column)'),
    'F_star': ('', 'depletion strength F* of Jenkins (2009)'),
    'tau_SSP': ('yr', 'age of the SSP node'),
    'Z_star': ('', 'metallicity (mass fraction) of the SSP node'),
    # flags and diagnostics
    'not_converged': ('', 'flag: CLOUDY did not converge'),
    'cloudy_warnings': ('', 'flag: CLOUDY ended with warnings'),
    'energy_balance_rel': ('', '|E2 - (E3 + E4)| / E2(lambda < 10 um), E_n = int col_n dln(lambda): '
                               'energy imbalance over the STELLAR incident energy'),
    'cmb_incident_ratio': ('', 'E2(lambda >= 10 um) / E2(lambda < 10 um): incident energy of the CMB '
                               'over the stellar one'),
    'f_esc_meas': ('', 'escape fraction of the H-ionizing PHOTONS: int col3 dlambda / int col2 dlambda, '
                       'lambda < 911.6 A'),
    'sed_support_A': ('Angstrom', '(min, max) wavelength where the incident field (SED and CMB) is > 0: '
                                  'transmission measured inside, 1 outside'),
    # datasets
    'continuum/nebular_emission_per_Msun': (
        'erg s-1 Msun-1', 'nu L_nu of the outward own emission without lines, (col4 - col9) s_k, per Msun '
                          'of SSP formed; includes the grains heated by the CMB'),
    'continuum/grain_diag_per_Msun': (
        'erg s-1 Msun-1', 'nu L_nu of the grain emission (save continuum grain, optically thin) x s_k, '
                          'per Msun of SSP formed; diagnostic, never added to the SED'),
    'continuum/transmission': ('', 'col3 / col2 of the incident field (SED and CMB), 1 where col2 = 0'),
    'incident/sed_per_Msun': (
        'erg s-1 Msun-1', 'nu L_nu of the incident stellar SED per Msun of SSP formed, as written in the '
                          'table SED file (on incident/wave_grid): no CMB, not scaled by s_k'),
    'lines_emergent/fluxes': (
        'erg s-1 Msun-1', 'emergent line LUMINOSITIES per Msun of SSP formed: CLOUDY absolute intensities '
                          '(erg cm-2 s-1, into 4 pi) x s_k'),
    'T_fuv_hii': ('', 'Habing band (6-13.6 eV) ENERGY transmittance: int col3 dln(lambda) / int col2 dln(lambda)'),
    'N_fuv_hii': ('', 'Habing band energy of the own emission (lines included) over the incident one: '
                      'int col4 dln(lambda) / int col2 dln(lambda)'),
    'dust_mass_per_Msun': ('Msun Msun-1', 'dust mass of the cloud per Msun of SSP formed: Sigma_d s_k / M_sun'),
    's_k': ('cm2 Msun-1', 'area of the illuminated face per Msun of SSP formed: Q_H(1 Msun) / (U n_H c)'),
}


def write_root_dataset(f, name, data, table=ROOT_UNITS):
    """
    Creates the root dataset `name` and writes its attributes 'units' (when it has a unit),
    'description' and the extra ones of `table` (e.g. 'wavelength_medium').
    """
    unit, description, extra = table[name]
    ds = f.create_dataset(name, data=data)
    if unit is not None:
        ds.attrs['units'] = unit
    ds.attrs['description'] = description
    ds.attrs.update(extra)
    return ds


def write_units_legend(f, schema=UNITS_SCHEMA, table=POINT_UNITS):
    """
    Writes the units of the grid_point members once per file: the root attribute
    'units_schema' and the groups 'units' (name -> unit) and 'descriptions' (name -> text).
    """
    f.attrs['units_schema'] = schema
    units, descriptions = f.create_group('units'), f.create_group('descriptions')
    for name, (unit, description) in table.items():
        if unit is not None:
            units.attrs[name] = unit
        descriptions.attrs[name] = description


def point_members(group):
    """
    Names of the attributes and paths (relative to the group) of the datasets of a grid_point group.
    """
    import h5py
    members = set(group.attrs)
    group.visititems(lambda name, obj: members.add(name) if isinstance(obj, h5py.Dataset) else None)
    return members


def check_point_units(group, table=POINT_UNITS):
    """
    Every member of the grid_point group has its units in `table`, and every entry of `table` is
    written: a quantity added to the fragment without its units fails here, not downstream.
    """
    members = point_members(group)
    missing, stale = sorted(members - set(table)), sorted(set(table) - members)
    if missing or stale:
        raise ValueError(f"[parse_one/hii] {group.name}: units table out of date, members without "
                         f"units {missing}, units of absent members {stale}")


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
        raise FileNotFoundError(f"[parse_one/hii] missing CLOUDY output: {p}")
    arr = np.loadtxt(p, comments='#')
    if arr.ndim != 2:
        raise ValueError(f"[parse_one/hii] {p}: expected 2D table, instead dimensions {arr.shape} (check .out)")
    if arr.shape[1] != ncol_expected:
        raise ValueError(f"[parse_one/hii] {p}: {arr.shape[1]} columns, expected columns: {ncol_expected}. ")
    return arr


def parse_cloudy_con(path, cols=(1, 2, 3, 4, 9)):
    """Parse the -save last continuum ... units angstrom no hash- output from
    CLOUDY simulations.

    This function extracts specified columns from the continuum output
    of the CLOUDY run.

    The default column indices (1-based) extracted are:
        - 1: Wavelengths [Angstrom], in vacuum (Energy::angstromVac of CLOUDY).
        - 2: Incident flux (SED and, with the 'CMB' command, 4 pi nu B_nu(T_CMB)).
        - 3: Pure attenuated incident flux (stellar transmitted)
        - 4: Diffuse flux emitted by the cloud (gas + grains, including lines).
        - 9: Flux from emission lines only.

    Columns related to flux are expressed in units of nuFnu [erg cm^-2 s^-1], per cm^2
    of the illuminated face (intensity case: flxCell of save_do.cpp, zone_startend.cpp).

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
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[parse_one/hii] missing CLOUDY output: {p}")

    use_idx = [c - 1 for c in cols]
    arr = np.loadtxt(p, comments='#', usecols=use_idx)

    if arr.ndim != 2:
        raise ValueError(f"[parse_one/hii] {p}: expected 2D table, instead dimensions {arr.shape} (check .out)")

    wave = arr[:, 0]
    d = np.diff(wave)
    if not (np.all(d < 0) or np.all(d > 0)):
        raise ValueError(f"[parse_one/hii] {p}: wavelength column (1) not strictly monotonic")

    return tuple(arr[:, i] for i in range(len(cols)))


# diagnostic file extraction
def parse_cloudy_cong(path, wave_ref=None):
    """Parses the CONG file to extract the total contribution of dust components.

    This function extracts the last column that represents the total contribution from
    dust grains (graphite, silicates, and overall total) in an optically-thin limit, in
    nuFnu [erg cm^-2 s^-1] per cm^2 of the illuminated face, like the continuum.

    Args:
        path (str): The file path of the CONG diagnostic file.
        wave_ref (numpy.ndarray, optional): The wavelength column of the continuum ('.con').
            If given, the first column of the file must coincide with it: the grain emission
            is then on the same mesh, in the same order and units, as the continuum it is
            stored with.

    Returns:
        numpy.ndarray: A 1D array containing the total contribution values extracted
        from the file.

    Raises:
        ValueError: If `wave_ref` is given and the wavelength column differs from it.
    """
    arr = _load_columns(path, 4)   # lambda | graphite | silicates | total
    if wave_ref is not None:
        wave_ref = np.asarray(wave_ref, dtype=float)
        if arr.shape[0] != wave_ref.size or not np.allclose(arr[:, 0], wave_ref, rtol=1e-6, atol=0.0):
            raise ValueError(f"[parse_one/hii] {path}: wavelength column differs from the continuum mesh ")
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
            - 'wavelengths' (numpy.ndarray): Array of wavelengths in Angstroms, as printed by
              CLOUDY: air above 2000 A, vacuum below (t_wavl::sprt_wl; see air_to_vacuum_A).
            - 'fluxes' (numpy.ndarray): Array of line intensities: with 'absolute', linear,
              in erg cm^-2 s^-1 into 4 pi per cm^2 of the illuminated face (cdLine_ip of
              cddrive.cpp, Conv2PrtInten = 1 in the intensity case).

    Raises:
        ValueError: If the file contains unparsable lines, invalid wavelength formats,
                    or lacks any valid data entries.
    """
    names, waves, fluxes = [], [], []
    unit_re = re.compile(r'^([0-9.eE+-]+)\s*([AaMmCc]?)$')  # es. 6562.80A, 157.6m, 3727
    scale = {'': 1.0, 'A': 1.0, 'M': 1e4, 'C': 1e8}  # conversion to Angstroms

    for raw in pathlib.Path(path).read_text().splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith('#'):
            continue

        if line.lower().startswith('iteration'):
            names, waves, fluxes = [], [], []
            continue

        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"[parse_one/hii] {path}: parse not possible for line: {raw!r}")

        flux = float(parts[-1])
        m = unit_re.match(parts[-2])  # wavelength part
        if not m:
            raise ValueError(f"[parse_one/hii] {path}: lambda with not recognized unit in: {raw!r}")

        wl = float(m.group(1)) * scale[m.group(2).upper()]
        label = ' '.join(parts[:-2])  # reconstruct label

        names.append(label)
        waves.append(wl)
        fluxes.append(flux)

    if not names:
        raise ValueError(f"[parse_one/hii] {path}: no lines extracted")

    return {
        'names': np.array(names),
        'wavelengths': np.array(waves),
        'fluxes': np.array(fluxes)
    }


def air_to_vacuum_A(wl_A):
    """
    Vacuum wavelengths of the lines printed by CLOUDY.

    CLOUDY keeps vacuum wavelengths internally and prints them in air above 2000 A
    while the continuum mesh is printed in vacuum. This is the inverse CLOUDY uses
    itself: index of refraction of air of Peck & Reeder
    (1972), applied only above 2000 A, iterated twice because it depends on the vacuum
    wavenumber. The coefficients belong to that formula, not to galapy.internal.constants.

    Parameters:
        wl_A (array-like): Wavelengths in Angstrom, as printed by CLOUDY.

    Returns:
        numpy.ndarray: The vacuum wavelengths in Angstrom (unchanged at and below 2000 A).
    """
    wl_air = np.asarray(wl_A, dtype=float)
    wl_vac = wl_air.copy()
    m = wl_air > 2000.0
    for _ in range(2):
        sigma2 = (1.0e4 / wl_vac[m]) ** 2      # vacuum wavenumber squared [um^-2]
        n_air = 1.0 + 1.0e-8 * (8060.51 + 2480990.0 / (132.274 - sigma2) + 17455.7 / (39.32957 - sigma2))
        wl_vac[m] = wl_air[m] * n_air
    return wl_vac


def staged_sed(wd, job_id):
    """
    Path of the table SED file read by CLOUDY: the one named by the deck hii_<job_id>.in of the
    workdir, staged by run_core.stage_job into <workdir>/SED/.

    Raises:
        FileNotFoundError: If the deck is missing from the workdir.
        ValueError: If the deck has no 'table SED' command.
    """
    from galapy.spectroscopy.utils.run_core import sed_referenced_by
    deck = pathlib.Path(wd) / f"hii_{job_id}.in"
    if not deck.is_file():
        raise FileNotFoundError(f"[parse_one/hii] missing CLOUDY deck: {deck}")
    name = sed_referenced_by(deck.read_text())
    if name is None:
        raise ValueError(f"[parse_one/hii] {deck}: no 'table SED' command in the deck")
    return pathlib.Path(wd) / 'SED' / name


def parse_table_sed(path):
    """
    Parses the table SED file read by CLOUDY, as written by galapy-sed-cloudy-extract
    (extract_spectra.write_cloudy_sed).

    The file holds lambda [Angstrom] and nu*L_nu [erg s^-1] per 1 Msun of SSP formed, with the
    keywords 'nuFnu units Angstroms' (and optionally 'extrapolate') on the first data row.
    CLOUDY reads the wavelengths in vacuum (Energy::set, RYDLAM / lambda) and interpolates the
    SED in log-log, 0 outside the file: with 'extrapolate' it extends the reddest segment as a
    power law to the low-energy limit of the code, which is not part of the returned arrays.

    The columns are read as extract_spectra.cloudy_sed_QH reads them, i.e. as Qh_unit (and
    then s_k) was computed.

    Args:
        path (str): The path to the table SED file.

    Returns:
        tuple of numpy.ndarray: The wavelengths [Angstrom], strictly increasing, and nu*L_nu
        [erg s^-1 Msun^-1].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not in 'nuFnu units Angstroms', is not normalized per 1 Msun
            of SSP, has fewer than two points, or its wavelengths are not strictly increasing.
    """
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"[parse_one/hii] missing table SED: {p}")
    rows = [l for l in p.read_text().splitlines() if l.strip()]
    header = ' '.join(l for l in rows if l.lstrip().startswith('#')).lower()
    first = next((l for l in rows if not l.lstrip().startswith('#')), '').lower()
    if 'nufnu' not in first or 'angstrom' not in first:
        raise ValueError(f"[parse_one/hii] {p}: table SED not in 'nuFnu units Angstroms' "
                         f"(not written by galapy-sed-cloudy-extract?)")
    if 'per 1 msun' not in header:
        raise ValueError(f"[parse_one/hii] {p}: table SED not normalized per 1 Msun of SSP formed")
    wave, nuLnu = np.loadtxt(p, comments='#', usecols=(0, 1), unpack=True, ndmin=2)
    if wave.size < 2:
        raise ValueError(f"[parse_one/hii] {p}: less than two points in the table SED")
    if np.any(np.diff(wave) <= 0):
        raise ValueError(f"[parse_one/hii] {p}: wavelengths of the table SED not strictly increasing")
    return wave, nuLnu


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
        raise ValueError(f"[parse_one/hii] {p}: expected header '#Depth<TAB>...<TAB>total', find {header!r} (format changed?)")
    if not header.rstrip().lower().endswith('total'):
        raise ValueError(f"[parse_one/hii] {p}: last column is not'total': {header!r}")
    if not rows:
        raise ValueError(f"[parse_one/hii] {p}: no zone found, aborted ")

    a = np.array(rows)
    depth, rho_d = a[:, 0], a[:, -1]      # depth [cm] , dust density [g cm^-3]

    # with 'last' the depth is striclty increasing, if not error (more iterations saved not the last)
    if np.any(np.diff(depth) <= 0):
        raise ValueError(f"[parse_one/hii] {p}: depth not strictly increasing — "
                         f"concatenation of more then one iteration.")
    if len(depth) < 2:
        raise ValueError(f"[parse_one/hii] {p}: only one zone in the file, column integral undefined.")

    return float(np.trapezoid(rho_d, depth))    # dust surface density [g cm^-2]


def fesc(wave_A, col2_incident, col3_transmitted):
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

    Both are ratios of ENERGY fluxes: the nuFnu columns are integrated in d(ln lambda), which
    gives int F_lambda dlambda over the band. They correct G0, an energy flux in the Habing band
    (the PDR deck imposes int F_nu dnu over 6-13.6 eV), and the PDR must receive the energy the
    ionized skin transmits. An integral in d(lambda) would give photon rates instead (as fesc).

    This computation accounts for the influence of dust attenuation, Stromgren column
    parameters, and the enhancements from the nebular continuum. The result is dimensionless
    and provides inputs for simulating the FUV impact on photodissociation regions
    (PDRs) and related computations such as G0.

    Parameters:
    wave_A: array_like
        Wavelength array in Ångströms, DECREASING (the native CLOUDY order).
    col2_incident: array_like
        Incident nuFnu [erg cm^-2 s^-1].
    col3_transmitted: array_like
        Transmitted nuFnu [erg cm^-2 s^-1].
    col4_own: array_like
        Own diffuse emission nuFnu (lines included) [erg cm^-2 s^-1].

    Returns:
    tuple(float, float)
        - T_FUV: Dimensionless FUV transmittance ratio (0.0 to 1.0).
        - N_FUV: Dimensionless FUV nebular continuum contribution (≥ 0.0).
    """
    m = (wave_A >= FUV_Lo_A) & (wave_A <= FUV_Hi_A)
    if not np.any(m):
        return 0.0, 0.0
    x = np.log(wave_A[m][::-1])
    den = np.trapezoid(col2_incident[m][::-1], x)
    if den <= 0:
        return 0.0, 0.0
    T_fuv = float(np.clip(np.trapezoid(col3_transmitted[m][::-1], x) / den, 0.0, 1.0))
    N_fuv = float(max(np.trapezoid(col4_own[m][::-1], x) / den, 0.0))
    return T_fuv, N_fuv


def _band_energy(wave_A, col, mask):
    """
    Energy flux int col d(ln lambda) of a nuFnu column over the bins in `mask`, independent of
    the order of wave_A. 0 with less than two bins.
    """
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(abs(np.trapezoid(np.asarray(col, dtype=float)[mask], np.log(wave_A[mask]))))


def energy_balance(wave_A, col2_incident, col3_transmitted, col4_own, split_A=None):
    """
    Relative energy balance of the cloud: incident = transmitted + diffuse outward emission.

    The columns are nuFnu = lambda F_lambda, so the energy flux is the integral in d(ln lambda):
        I_n = int nuFnu d(ln lambda) = int F_lambda d(lambda)     [erg cm^-2 s^-1].
    The integral in d(lambda) would instead be proportional to the PHOTON flux (as in fesc),
    which is not conserved: the dust re-emits each absorbed UV photon as many IR photons.

    With the 'CMB' command col2 also holds 4 pi nu B_nu(T_CMB), which in many models carries
    more energy than the SED. The CMB balances itself (what the cloud absorbs, it re-emits),
    so the imbalance is referred to the STELLAR incident energy, int col2 below `split_A`
    (STELLAR_MAX_A in the parser): otherwise it would measure the CMB and stay blind to the
    stellar budget, which is what a double-counted component would break.

    Parameters:
    wave_A: ndarray
        Array of wavelengths in Ångströms, strictly monotonic (either order).
    col2_incident: ndarray
        Incident flux (nuFnu) corresponding to wave_A.
    col3_transmitted: ndarray
        Transmitted flux (nuFnu) corresponding to wave_A.
    col4_own: ndarray
        Diffuse outward emission (nuFnu, lines included) corresponding to wave_A.
    split_A: float, optional
        Wavelength [Angstrom] below which col2 is the stellar incident field. None: the whole
        col2 is the reference.

    Returns:
    float
        |I2 - (I3 + I4)| / E_ref, independent of the order of wave_A, with E_ref = I2, or the
        col2 energy below `split_A`. NaN if the reference energy is zero.
    """
    lnl = np.log(wave_A)
    I2 = np.trapezoid(col2_incident, lnl)
    I3 = np.trapezoid(col3_transmitted, lnl)
    I4 = np.trapezoid(col4_own, lnl)
    E_ref = I2 if split_A is None else _band_energy(wave_A, col2_incident, wave_A < split_A)
    if E_ref == 0:
        return float('nan')
    # the abs() at the denominator makes the ratio independent of the wavelength order
    return float(abs(I2 - (I3 + I4)) / abs(E_ref))


def cmb_incident_ratio(wave_A, col2_incident, split_A=STELLAR_MAX_A):
    """
    Incident energy above `split_A` over the incident energy below it: with the 'CMB' command,
    the CMB over the stellar incident field (the SED adds at most ~1% above 10 um). Large
    values flag the models whose continua are dominated by the CMB: the imbalance referred to
    the stellar energy is (1 + ratio) times the one referred to the total incident energy.

    Parameters:
    wave_A: ndarray
        Array of wavelengths in Ångströms, strictly monotonic (either order).
    col2_incident: ndarray
        Incident flux (nuFnu) corresponding to wave_A.
    split_A: float
        Wavelength [Angstrom] separating the stellar and the CMB incident field.

    Returns:
    float
        The dimensionless ratio, NaN if there is no stellar incident energy.
    """
    stellar = _band_energy(wave_A, col2_incident, wave_A < split_A)
    if stellar == 0:
        return float('nan')
    return _band_energy(wave_A, col2_incident, wave_A >= split_A) / stellar


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
        s_k = Q_H,k(1 Msun) / (U * n_H * c)     [cm^2 Msun^-1].

    This factor (s_k) converts all outputs provided per cm^2
    (such as absolute row values, col3/col4, CONG, Sigma_d) into
    luminosities/masses per 1 Msun of formed SSP.

    Args:
        Qh_unit_node: Ionizing photon rate of the SSP node in photons s^-1 Msun^-1.
        logU: Logarithm of the dimensionless ionization parameter.
        lognH: Logarithm of the hydrogen number density in cm^-3.

    Returns:
        float: Geometric factor [cm^2 Msun^-1] that converts outputs from per cm^2
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
        raise ValueError(f"[parse_one/hii] {spec_path}: duplicated job_id in hii_grid_spec.h5")
    return index


@contextlib.contextmanager
def atomic_h5(path):
    """
    Writes an HDF5 file atomically
    """
    import h5py
    path = pathlib.Path(path)
    part = path.with_name(path.name + '.part')
    try:
        with h5py.File(str(part), 'w') as f:
            yield f
        os.replace(part, path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def require_successful_run(wd):
    """
    Only a succesfull run of run_core is parsed. Returns not_converged (bool).
    """
    flag = wd / 'converged.flag'
    if not flag.is_file():
        raise FileNotFoundError(f"[parse_one/hii] {flag}: absent -- convergence unknown (DD-4). "
                                f"Run the job with run_core first.")
    failed = wd / 'RUN_FAILED'
    if failed.is_file():
        raise RuntimeError(f"[parse_one/hii] {wd}: RUN_FAILED ({failed.read_text().strip()}) -- "
                           f"the run did not succeed, it is not parsed")
    return flag.read_text().strip() == '0'


def resolve_out(out, job_id):
    """
    --out is a DIRECTORY (an existing one, or a path without extension, like the default
    data/hii/parsed) -> <dir>/<job_id>.h5; otherwise it is the FILE to write (<id>.h5, or
    <id>.h5.part as the Slurm script of v70 passes before its own `mv`). Parents are created.
    """
    out = pathlib.Path(out)
    if out.is_dir() or out.suffix == '':
        out = out / f'{job_id}.h5'
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def build_parser():
    """
    parser
    """
    import argparse
    ap = argparse.ArgumentParser(prog='python -m galapy.spectroscopy.utils.hii.parse_one_hii',
                                 description="PARSE_ONE_HII: workdir of one job -> one <job_id>.h5")
    ap.add_argument('workdir', help="work directory of single job (es: data/hii/work/<job_id>)")
    ap.add_argument('--spec', default="data/grids/hii/hii_grid_spec.h5",
                    help='path to hii_grid_spec.h5 (default: data/grids/hii/hii_grid_spec.h5)')
    ap.add_argument('--ssp-meta', default="data/cloudy_seds/ssp_metadata.json",
                    help='path to ssp_metadata.json (default: data/cloudy_seds/ssp_metadata.json)')
    ap.add_argument('--out', default="data/hii/parsed",
                    help="output: a file (e.g. <job_id>.h5), or a directory (existing, or a path "
                         "without extension) -> <dir>/<job_id>.h5 (default: data/hii/parsed)")
    return ap


def main(argv=None):
    """
    workdir of one job -> one fragment <job_id>.h5.
    """
    import json
    import h5py

    args = build_parser().parse_args(argv)

    wd = pathlib.Path(args.workdir)
    job_id = wd.name
    pre = wd / f"hii_{job_id}"
    # BEFORE reading a single number: only a successful run is parsed
    not_converged = require_successful_run(wd)
    # a run with CLOUDY warnings (exit code 2, marker of run_core) is parsed and flagged
    cloudy_warnings = (wd / 'WARNINGS').is_file()
    out = resolve_out(args.out, job_id)

    i = spec_index(args.spec).get(job_id)
    if i is None:
        raise KeyError(f"[parse_one/hii] {job_id}: absent in {args.spec} (job not part of grid)")
    with h5py.File(args.spec, 'r') as s:
        p = {k: s[k][i] for k in ('logU', 'lognH_HII', 'z_CMB', 'log_zeta_O',
                                  'xi_d', 'f_esc_target', 'F_star', 'tau_SSP', 'Z_star')}
    with open(args.ssp_meta) as fh:
        meta = json.load(fh)
    node = {(m['tau_SSP'], m['Z_star']): m for m in meta}[(float(p['tau_SSP']), float(p['Z_star']))]
    s_k = s_k_factor(node['Qh_unit'], float(p['logU']), float(p['lognH_HII']))

    # incident stellar SED: the table SED file CLOUDY read, the same of Qh_unit (hence of s_k)
    sed_path = staged_sed(wd, job_id)
    if node.get('sed_file', sed_path.name) != sed_path.name:
        raise ValueError(f"[parse_one/hii] {job_id}: the deck reads {sed_path.name}, {args.ssp_meta} "
                         f"gives {node['sed_file']} for the SSP node")
    sed_wave, sed_nuLnu = parse_table_sed(sed_path)

    # wavelength and continuum and lines
    wave, col2, col3, col4, col9 = parse_cloudy_con(f"{pre}.con", cols=(1, 2, 3, 4, 9))
    # grains, on the mesh of the continuum
    cong = parse_cloudy_cong(f"{pre}.con_grain", wave_ref=wave)
    # emission CLOUD with no emission lines
    nebular = col4 - col9
    # Transmission
    transmission = support_safe_ratio(col2, col3)
    lam_support = wave[col2 > 0]
    sed_support = (float(lam_support.min()), float(lam_support.max()))
    line_data = parse_cloudy_linelist(f"{pre}.lines")
    Sigma_d = integrate_grain_abundance(f"{pre}.dusa")

    with atomic_h5(out) as f:
        # units: root attribute + legend of the grid_point members, attributes of the root datasets
        write_units_legend(f)
        write_root_dataset(f, 'continuum/wave_grid', wave.astype('f4'))
        write_root_dataset(f, 'line_names', line_data['names'].astype('S'))
        write_root_dataset(f, 'lines_emergent/wavelengths_rest', line_data['wavelengths'].astype('f4'))
        write_root_dataset(f, 'lines_emergent/wavelengths_rest_vacuum',
                           air_to_vacuum_A(line_data['wavelengths']).astype('f4'))
        write_root_dataset(f, 'incident/wave_grid', sed_wave.astype('f4'))
        g = f.create_group(f"grid_point_{job_id}")
        g.attrs.update({k: float(p[k]) for k in
                        ('logU', 'lognH_HII', 'z_CMB', 'log_zeta_O',
                         'xi_d', 'f_esc_target', 'F_star', 'tau_SSP', 'Z_star')})
        g.attrs['not_converged'] = bool(not_converged)  # DD-4
        g.attrs['cloudy_warnings'] = bool(cloudy_warnings)
        g.attrs['energy_balance_rel'] = energy_balance(wave, col2, col3, col4, split_A=STELLAR_MAX_A)
        g.attrs['cmb_incident_ratio'] = cmb_incident_ratio(wave, col2)
        f_esc_meas = fesc(wave, col2, col3)
        g.attrs['f_esc_meas'] = float(f_esc_meas)
        g.create_dataset('f_esc_meas', data=f_esc_meas)
        T_fuv, N_fuv = fuv_transmittance_hii(wave, col2, col3, col4)
        g.create_dataset('T_fuv_hii', data=T_fuv)
        g.create_dataset('N_fuv_hii', data=N_fuv)
        g.create_dataset('continuum/nebular_emission_per_Msun', data=(nebular * s_k).astype('f4'))
        g.create_dataset('continuum/grain_diag_per_Msun', data=(cong * s_k).astype('f4'))
        g.create_dataset('continuum/transmission', data=transmission.astype('f4'))
        g.create_dataset('incident/sed_per_Msun', data=sed_nuLnu.astype('f4'))
        g.attrs['sed_support_A'] = sed_support
        g.create_dataset('lines_emergent/fluxes', data=(line_data['fluxes'] * s_k).astype('f4'))
        g.create_dataset('dust_mass_per_Msun', data=Sigma_d * s_k / M_Sun_G)
        g.create_dataset('s_k', data=s_k)
        # inside atomic_h5: a member written without its units leaves no fragment
        check_point_units(g)
    print(f"[parse_one/hii] {job_id} -> {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())