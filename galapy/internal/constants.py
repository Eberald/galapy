""" Definitions of physical and mathematical constants
"""

# Solar luminosity [ erg / s ]
Lsun = 3.828e+33
sunL = 1. / Lsun

# Speed of light in vacuum
clight = {
    'cm/s' : 2.99792458e+10,
    'm/s'  : 2.99792458e+8,
    'A/s'  : 2.99792458e+18,
}

# Mpc to cm
Mpc_to_cm = 3.086e+24

# Planck constant
hP = {
    'eV/Hz' : 4.1357e-15,
    'erg*s' : 6.6262e-27,
}

# Angstrom to keV
def Ang_to_keV ( wavelength ) :
    return 1.e-3 * hP['eV/Hz'] * clight['A/s'] / wavelength

#================== CLOUDIA =========================
#General
alphaB_1e4K = 2.59e-13     # cm^3/s  case B recombination at T_e = 1e4 K
                           # (Osterbrock & Ferland 2006, Tab. 2.1).
M_Sun_G   = 1.989e33       # g  (IAU 2015 B3)
NH_to_AV_MW = 1.87e21      # cm^-2 mag^-1  Bohlin, Savage & Drake 1978 (R_V=3.1)

## SPECTRA
LyLimit=911.76
LymanA=911.6
SED_cut=1.0e6
FUV_Lo_A=911.76 # this two are Habing band (6-13.6 eV) on Angstroms
FUV_Hi_A=2066.0 # //
Stellar_Max_A = 1.0e5  #Split between the stellar and the CMB part of the incident field [Angstrom].

## ABUNDANCES
#  Standard Atomic Weight (IUPAC, https://iupac.qmul.ac.uk/AtWt/)
# Need for Z_gas translation from abundances
ATOMIC_WEIGHT = {
    'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974,
    'S': 32.06, 'Cl': 35.45, 'Ar': 39.95, 'K': 39.098, 'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
    'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
}

# Cloudy Names for the elements
CLOUDY_NAME = {
    'H': 'HYDROGEN', 'He': 'HELIUM', 'Li': 'LITHIUM', 'Be': 'BERYLLIUM',
    'B': 'BORON', 'C': 'CARBON', 'N': 'NITROGEN', 'O': 'OXYGEN',
    'F': 'FLUORINE', 'Ne': 'NEON', 'Na': 'SODIUM', 'Mg': 'MAGNESIUM',
    'Al': 'ALUMINIUM', 'Si': 'SILICON', 'P': 'PHOSPHORUS', 'S': 'SULPHUR',
    'Cl': 'CHLORINE', 'Ar': 'ARGON', 'K': 'POTASSIUM', 'Ca': 'CALCIUM',
    'Sc': 'SCANDIUM', 'Ti': 'TITANIUM', 'V': 'VANADIUM', 'Cr': 'CHROMIUM',
    'Mn': 'MANGANESE', 'Fe': 'IRON', 'Co': 'COBALT', 'Ni': 'NICKEL',
    'Cu': 'COPPER', 'Zn': 'ZINC',
}

#files from dataset (galapy-dataset/Nebular/abundances)
# - abundances GC
# - scaling for metallicity of abundances
# - conversion from Nichols factor zeta to metallicity
ABN_FILE   = 'GC.abn'
SCALE_FILE = 'scaling_abd_Z.yaml'
ZMAP_FILE  = 'zeta_Z.npz'

# dust to metal milky way
# xi_d,MW = (D/G)_MW / Z_gas^MW,
# GC is metallicity of MW (0.014254, Nicholls+17), D/G_MW (1/162,Zubko+04)
XI_D_MW = 0.43

# solar references for absolute metallicities
SOLAR_REFERENCES = {
    'GASS10':   0.013370,   # Grevesse+10
    'Caffau11': 0.015300,   # used in Ronconi+24
    'PARSEC':   0.015240,   # PARSEC ssp scales
}

## HII and PDR CONFIGS
CONFIG_HII_FILE = 'grid_hii.yaml'
TEMPLATE_KEYS_HII = ('job_id', 'lognH_HII', 'logU', 'z_CMB', 'F_star', 'log_zeta_O', 'element_scale_block',
                     'grain_scale', 'sed_file', 'log_N_stop',)
EXP_FIELDS = ('grain_scale',)

## LINES
HII_LINES_FILE = 'hii_full.yaml'
PDR_LINES_FILE = 'pdr_full.yaml'
HII_LINES_OFF = 'LineList_HII.dat'
PDR_LINES_OFF = 'LineList_PDR.dat'

#================================= UNITS =================================
Units_Legend = ('units', 'descriptions')

Root_Units_HII = {
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
Points_Units_HII = {
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

Root_Shared_HII = ('continuum/wave_grid', 'line_names', 'lines_emergent/wavelengths_rest',
               'lines_emergent/wavelengths_rest_vacuum', 'incident/wave_grid')

