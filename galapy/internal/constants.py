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

#================== PHYSICS USED BY CLOUDIA =========================
alphaB_1e4K = 2.59e-13     # cm^3/s  case B recombination at T_e = 1e4 K
                           # (Osterbrock & Ferland 2006, Tab. 2.1).
M_Sun_G   = 1.989e33       # g  (IAU 2015 B3)
NH_to_AV_MW = 1.87e21      # cm^-2 mag^-1  Bohlin, Savage & Drake 1978 (R_V=3.1)
# atomic mass unit [g] (CODATA 2018)
AMU_G = 1.66053906660e-24

## SPECTRA
LyLimit=911.76  # Lyman limit (vacuum, Angstrom), 13.6 eV
LymanA=911.6
FUV_Lo_A=911.76 # this two are Habing band (6-13.6 eV) on Angstroms
FUV_Hi_A=2066.0 # //

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
