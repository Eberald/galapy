# Author: ENrico Veraldi
""" Definitions tied to CLOUDY and to the CloudIA grids built with it: element names, data and
configuration files, input templates, default SSP nodes, and the units of the grid products.
"""

import numpy as np

#================================= SPECTRA =================================
SED_cut = 1.0e6          # red end of the table SED written for CLOUDY [Angstrom] (cuts the synchrotron tail)
Stellar_Max_A = 1.0e5    # split between the stellar and the CMB part of the incident field [Angstrom]

#========================= DEFAULT SSP NODES (galapy-sed-cloudy-extract) =========================
# ages [yr] and metallicities (mass fraction) of the SSPs extracted as table SEDs for the grid
SSP_TAUS = np.array([1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 7e7, 1e8])
SSP_Z = np.array([0.0001, 0.0005, 0.0010, 0.0040, 0.0080, 0.0200])

#================================ ABUNDANCES ================================
# CLOUDY names of the elements
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

# files from dataset (galapy-dataset/Nebular/abundances)
# - abundances GC
# - scaling for metallicity of abundances
# - conversion from Nichols factor zeta to metallicity
ABN_FILE   = 'GC.abn'
SCALE_FILE = 'scaling_abd_Z.yaml'
ZMAP_FILE  = 'zeta_Z.npz'

# grain mass per H atom of CLOUDY 'grains ISM' at scale 1 [g]
# (C25: Dust to gas ratio of the .out x gas mass per H / grain_scale, constant within 1e-4)
GRAIN_MASS_ISM_PER_H = 1.4934e-26

#============================ HII and PDR CONFIGS ============================
CONFIG_HII_FILE = 'grid_hii.yaml'
CONFIG_PDR_FILE = 'grid_pdr.yaml'
TEMPLATE_KEYS_HII = ('job_id', 'lognH_HII', 'logU', 'z_CMB', 'F_star', 'log_zeta_O', 'element_scale_block',
                     'grain_scale', 'sed_file', 'log_N_stop',)
TEMPLATE_KEYS_PDR = ('job_id', 'lognH_PDR', 'log_I_FUV', 'z_CMB', 'F_star', 'log_zeta_0', 'element_scale_block',
                     'grain_scale', 'pah_scale', 'log_chi_CR', 'A_V_PDR', 'sed_file', 'sed_base', 'beta',)
EXP_FIELDS_HII = ('grain_scale',)
EXP_FIELDS_PDR = ('grain_scale', 'pah_scale',)

#================================= PDR CONST =================================

PAH_SCALE_MW = 10.0         # factor cloudy for PAH MW abundance (10 = 10^-6.52 of Tielens (2008))
                            # 'pah_scale':   PAH_SCALE_MW * xi_d * m_Z(zeta_O) / GRAIN_MASS_ISM_PER_H
# PAH bands windows od own continuum,  PAHFIT (Smith et al. 2007) and CLOUDY (Hazy 1)
PAH_WINDOWS_UM = {'PAH_3.3': (3.30, 0.048), 'PAH_6.2': (6.18, 0.204), 'PAH_7.7': (7.94, 0.48),
                  'PAH_8.6': (8.60, 0.336), 'PAH_11.3': (11.30, 0.172)}

# FUV tilt of the PDR field beta slop minimum and step
BETA_MIN, BETA_STEP = -2.5, 0.25

#================================== LINES ==================================
HII_LINES_FILE = 'hii_full.yaml'
PDR_LINES_FILE = 'pdr_full.yaml'
HII_LINES_OFF = 'LineList_HII.dat'
PDR_LINES_OFF = 'LineList_PDR.dat'

#================================= UNITS =====================================
Units_Legend = ('units', 'descriptions')

#================================= UNITS-HII =================================
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

# members of grid_point_<job_id> (HII)
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
    'pumping_removed': ('', 'flag: continuum pumping removed from the pumped lines (.pump read)'),
    'energy_balance_rel': ('', '|E2 - (E3 + E4)| / E2(lambda < 10 um), E_n = int col_n dln(lambda): '
                               'energy imbalance over the STELLAR incident energy'),
    'cmb_incident_ratio': ('', 'E2(lambda >= 10 um) / E2(lambda < 10 um): incident energy of the CMB '
                               'over the stellar one'),
    'reprocessing_excess': ('', '(E4 - E_abs) / E_abs, E_abs = E2 - E3, E_n = int col_n dln(lambda): emitted over '
                                'absorbed energy, minus 1 (signed)'),
    'absorbed_fraction': ('', 'E_abs / E2: share of the incident energy (SED and CMB) absorbed by the cloud'),
    'lyc_emit_fraction': ('', 'E4(lambda < 911.76 A) / E_abs: diffuse Lyman continuum emitted over the absorbed '
                              'energy, before its rescaling (with sphere CLOUDY counts its inward half twice)'),
    'f_esc_meas': ('', 'escape fraction of the H-ionizing PHOTONS: int col3 dlambda / int col2 dlambda, '
                       'lambda < 911.6 A'),
    'sed_support_A': ('Angstrom', '(min, max) wavelength where the incident field (SED and CMB) is > 0: '
                                  'transmission measured inside, 1 outside'),
    # datasets
    'continuum/nebular_emission_per_Msun': (
        'erg s-1 Msun-1', 'nu L_nu of the outward own emission without lines, (col4 - col9) s_k, per Msun '
                          'of SSP formed; includes the grains heated by the CMB (emission, CMB not subtracted); '
                          'cells below the print precision interpolated; lambda < 911.76 A scaled to the energy '
                          'budget'),
    'continuum/grain_diag_per_Msun': (
        'erg s-1 Msun-1', 'nu L_nu of the grain emission: emissivity of the LAST zone in the optically thin '
                          'limit (save continuum grain, erg cm-3 s-1) x thickness of the cloud x s_k, per Msun '
                          'of SSP formed; exact for uniform grain temperature; diagnostic, never added to the SED'),
    'continuum/transmission': ('', 'col3 / col2 of the incident field (SED and CMB), 1 where col2 = 0; absorption '
                                   'only (with sphere scattering does not attenuate the incident field)'),
    'incident/sed_per_Msun': (
        'erg s-1 Msun-1', 'nu L_nu of the incident stellar SED per Msun of SSP formed, as written in the '
                          'table SED file (on incident/wave_grid): no CMB, not scaled by s_k'),
    'lines_emergent/fluxes': (
        'erg s-1 Msun-1', 'emergent line LUMINOSITIES per Msun of SSP formed: CLOUDY absolute intensities '
                         '(erg cm-2 s-1, into 4 pi) x s_k; contrast against the CMB (CLOUDY default); lines of '
                          'the pump list (below 10 um) without continuum pumping'),
    'T_fuv_hii': ('', 'Habing band (6-13.6 eV) ENERGY transmittance: int col3 dln(lambda) / int col2 dln(lambda)'),
    'N_fuv_hii': ('', 'Habing band energy of the own emission (lines included) over the incident one: '
                      'int col4 dln(lambda) / int col2 dln(lambda)'),
    'dust_mass_per_Msun': ('Msun Msun-1', 'dust mass of the cloud per Msun of SSP formed: Sigma_d s_k / M_sun'),
    's_k': ('cm2 Msun-1', 'area of the illuminated face per Msun of SSP formed: Q_H(1 Msun) / (U n_H c)'),
}

Root_Shared_HII = ('continuum/wave_grid', 'line_names', 'lines_emergent/wavelengths_rest',
                   'lines_emergent/wavelengths_rest_vacuum', 'incident/wave_grid')

#================================= UNITS-PDR =================================
Root_Units_PDR = dict(Root_Units_HII)

# different from Root_Units_HII
Root_Units_PDR['line_names'] = ( None, 'CLOUDY labels of the lines (internal spaces collapsed), '
                                       'in the order of pdr_lines.dat', {})
Root_Units_PDR['incident/wave_grid'] = ( 'Angstrom', 'wavelength of the incident stellar SED, the table SED file '
                                                     'read by CLOUDY, truncated the Lyman limit '
                                                     '(galapy-sed-cloudy-extract --pdr), increasing',
                                         {'wavelength_medium': 'vacuum'})
Root_Units_PDR['continuum/pah_wave_grid'] = ( 'Angstrom', 'wavelengths of the continuum mesh inside the PAH '
                                                          'windows (PAH_WINDOWS_UM), decreasing',
                                              {'wavelength_medium': 'vacuum'})

# members of grid_point_<job_id> (PDR)
Points_Units_PDR = {
    # grid axes and SSP node, copied from the spec
    'lognH_PDR': ('dex(cm-3)', 'log10 of the hydrogen density n_H of the slab (CLOUDY hden)'),
    'logG0': ('dex', 'log10 G0, FUV field in Habing units (6-13.6 eV, one-sided) imposed by the deck'),
    'log_zeta_O': ('dex', 'log10 zeta_O, zeta_O = (O/H) / (O/H)_GC (Nicholls+17)'),
    'A_V_PDR': ('mag', 'visual extinction at which CLOUDY stops (stop AV, point source)'),
    'log_chi_CR': ('dex', 'log10 of the cosmic-ray ionization rate over the galactic background'),
    'xi_d': ('', 'dust-to-metal mass ratio'),
    'z_CMB': ('', 'redshift of the CMB of the run, T_CMB = 2.725 (1 + z_CMB) K'),
    'F_star': ('', 'depletion strength F* of Jenkins (2009)'),
    'beta': ('', 'FUV tilt of the incident SED, nuFnu x (nu/nu0)^beta inside the Habing band (snapped to BETA_STEP)'),
    'tau_SSP': ('yr', 'age of the SSP node'),
    'Z_star': ('', 'metallicity (mass fraction) of the SSP node'),
    # flags and diagnostics
    'not_converged': ('', 'flag: CLOUDY did not converge'),
    'cloudy_warnings': ('', 'flag: CLOUDY ended with warnings'),
    'pumping_removed': ('', 'flag: continuum pumping removed from the pumped lines (.pump read)'),
    'energy_balance_rel': ('', '|E2 - (E3 + E4)| / E2(lambda < 10 um), E_n = int col_n dln(lambda): '
                            'energy imbalance over the STELLAR incident energy'),
    'cmb_incident_ratio': ('', 'E2(lambda >= 10 um) / E2(lambda < 10 um): incident energy of the CMB '
                            'over the stellar one'),
    'reprocessing_excess': ('', '(E4 - E_abs) / E_abs, E_abs = E2 - E3, E_n = int col_n dln(lambda):'
                                'emitted over absorbed energy, minus 1 (signed)'),
    'absorbed_fraction': ('', 'E_abs / E2: share of the incident energy (SED and CMB) absorbed by the slab'),
    'lyc_emit_fraction': ('', 'E4(lambda < 911.76 A) / E_abs: diffuse Lyman continuum emitted over the absorbed '
                            'energy (expected ~0: no ionizing photon reaches the slab)'),
    'logG0_meas': ('dex', 'log10 of int col2 dln(lambda) over the Habing band divided by HABING_FLUX_ONE_SIDED: '
                    'G0 that CLOUDY applied'),
    'sed_support_A': ('Angstrom', '(min, max) wavelength where the incident field (SED and CMB) is > 0:'
                        'transmission measured inside, 1 outside'),
    # datasets
    'continuum/nebular_emission_per_area': ('erg cm-2 s-1', 'nu F_nu of the outward own emission without lines, '
                                                             'col4 - col9, per cm2 of the illuminated face; includes'
                                                             ' the grains heated by the CMB; cells below the print precision'
                                                             'interpolated'),
    'continuum/pah_emission_per_area': ('erg cm-2 s-1', 'continuum/nebular_emission_per_area on continuum/pah_wave_grid '
                                                        '(PAH windows, underlying continuum not subtracted)'),
    'continuum/grain_diag_per_area': ('erg cm-2 s-1', 'nu F_nu of the grain emission: emissivity of the LAST zone in '
                                                      'the optically thin limit (save continuum grain, erg cm-3 s-1) '
                                                      'x thickness of the slab; diagnostic, never added'),
    'continuum/transmission': ('', 'col3 / col2 of the incident field (SED and CMB), 1 where col2 = 0'),
    'extinction/A_lambda': ('mag', '-2.5 log10(continuum/transmission): extinction curve of the slab, 0 where col2 = 0'),
    'incident/sed_per_Msun': ('erg s-1 Msun-1', 'nu L_nu of the truncated and tilted (beta) table SED file per Msun of SSP'
                                                'formed (on incident/wave_grid): shape only, CLOUDY renormalizes it to G0'),
    'lines_emergent/fluxes': ('erg cm-2 s-1', 'emergent line intensities per cm2 of the illuminated face (CLOUDY absolute,'
                                                ' into 4 pi); lines of the pump list without continuum pumping'),
    'dust_mass_per_area': ('Msun cm-2', 'dust mass column of the slab: Sigma_d / M_sun'),
}

Root_Shared_PDR = Root_Shared_HII + ('continuum/pah_wave_grid',)