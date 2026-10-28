# Author: Enrico Veraldi
""" Constants of the CloudIA emulators: inputs and output families, preprocessing of the targets,
calibration of the predictive sigma, and the thresholds and line tables of their validation.
"""

import math

#================================ HII: INPUTS ================================
# input free parameters of the training on HII
INPUT_KEYS_HII = ('logU', 'lognH_HII', 'log_zeta_O', 'z_CMB', 'xi_d', 'f_esc_meas', 'F_star', 'tau_SSP', 'Z_star')
LOG10_INPUTS_HII = ('tau_SSP', 'Z_star')
# axes that are dominant in the training, for PCA more impact (stratification of holdout and calibration clouds)
AXES_DOMINANT_KEYS_HII = ('logU', 'lognH_HII', 'log_zeta_O')
AXES_DOMINANT_HII = tuple(INPUT_KEYS_HII.index(k) for k in AXES_DOMINANT_KEYS_HII)

#=========================== HII: OUTPUT FAMILIES ===========================
FUV_N_OFFSET = 1e-6  # N_FUV >= 0, can be zero and in this case we add a small offset to avoid log(0)
# (name, compressed with PCA, offset added before the log10) of every family trained on HII
FAMILIES_HII = (('lines', False, 0.0), ('cont', True, 0.0), ('grain', True, 0.0), ('trans', True, 0.0),
                ('dust', False, 0.0), ('fuv_t', False, 0.0), ('fuv_n', False, FUV_N_OFFSET))
WAVE_FAMS = ('cont', 'grain', 'trans')              # families defined on the continuum wavelength grid
VECTOR_FAMS = ('lines', 'cont', 'grain', 'trans')   # families that get the relative floor

#========================= PREPROCESSING OF THE TARGETS =========================
LOG_FLOOR = 1e-35   # avoid -inf on null fluxes
# Default floor of each vector family before the log10, relative to the peak of each model (row): values below
# rel_floor * max(row) are set to it. The floor fixes the depth where a spectrum goes to ~0: in log
# space a transition transparent -> opaque is a jump of -log10(rel_floor) dex, which a stationary GP kernel smooths
# over, spilling error onto the cells near the jump. So the floor should be as HIGH as the physics allows:
#   - trans: T = transmitted/incident <= 1, so T < 1e-5 is an opaque cloud, indistinguishable from 0. Floor 1e-5
#     vs 1e-10 (holdout, 40 epochs): error below 912 A p50 0.20 -> 0.12 dex, p90 0.76 -> 0.49 dex, cells with
#     |dT| > 0.1 35% -> 27%, nothing changes above 912 A.
#   - cont, grain: the spectrum spans ~10 dex (free-free radio tail far below the peak), a higher floor clips
#     cells that are needed (at 1e-5: 32% of the cells > 100 um, error p95 0.2 -> 2.3 dex), so they keep 1e-10
#     (except below 912 A, see REL_FLOOR_LYC).
#   - lines: with ratio targets the floor is ratio_floor (on L_i / H-beta), so no relative floor: it would be
#     relative to the brightest line (Ly-alpha ~ 20-40 H-beta) and clip at ~2e-9 H-beta, above ratio_floor.
#     With absolute targets they get REL_FLOOR_ABS_LINES.
REL_FLOOR = {'lines': 0.0, 'cont': 1e-10, 'grain': 1e-10, 'trans': 1e-5}
REL_FLOOR_ABS_LINES = 1e-10  # relative floor of the lines with --line-targets absolute
# Floor that REPLACES rel_floor at the ionizing wavelengths (lambda < CONST.LyLimit) of the families on the
# continuum wavelength grid. The Lyman break is where the transparent/opaque cliff lives, while the deep tails that
# need the low floor (free-free radio) are all at lambda > 912 A, so the two regimes get separate floors:
#   - cont, lyc 1e-5 + rest 1e-10 vs 1e-10 everywhere (holdout, 40 epochs): error below 912 A p50 0.39 -> 0.18 dex,
#     p90 2.1 -> 0.83 dex, |err|/peak p95 0.078 -> 0.046; above 912 A unchanged (radio tail p95 0.21 -> 0.20 dex).
#     Same gain below 912 A as a flat 1e-5, without its loss above (radio tail p95 2.3 dex).
#   - grain: no ionizing floor. Below 912 A the grain emission is already < 1e-10 of the peak in every model (a
#     constant column), so the ionizing floor changes nothing (identical holdout errors). A flat 1e-7 / 1e-5
#     gains <= 10% at 1-100 um but loses on the weak cells (1e-5: 100 um - 1 cm p90 0.18 -> 0.48 dex): 1e-10.
# Families not listed have no ionizing floor (their rel_floor holds at every wavelength).
REL_FLOOR_LYC = {'cont': 1e-5}
SD_FLOOR = 0.01     # minimum sd (dex) of the z-score: near-constant column otherwise blow numerical noise up to unit variance and inflate the PCA rank

#=============================== LINES: RATIO FORM ===============================
RATIO_REF = ('h 1', 4862.7)  # reference line of the ratio form (species, vacuum wavelength in Angstrom): H-beta
# Default floor of L_i / H-beta in the ratio form: lines below it are emulated at the floor. 1e-5 is below
# the faintest line ever measured in a nebular spectrum, ~2e-5 I(H-beta) (Orion nebula, VLT/UVES, 555 lines;
# Esteban et al. 2004, MNRAS, arXiv:astro-ph/0408249). Galaxy spectra are far shallower: the deepest composite
# (JWST/NIRSpec, ~600 object-hours, z ~ 2-3) reaches ~0.1% of H-alpha ~ 3e-3 H-beta (Strom et al. 2023, ApJL,
# arXiv:2308.13508), and ultra-deep spectra of single z = 6-8 galaxies (13-18 h NIRSpec G395M) detect lines at
# a few 1e-2 H-beta, e.g. He II 4686 / H-beta = 0.041 +- 0.008 at S/N ~ 5 (Rai et al. 2026, arXiv:2609.35526).
# A lower floor makes the GP spend capacity on unobservable ratios spanning many dex: on the HII holdout,
# 1e-10 instead of 1e-4 worsened the metal lines above 1e-3 H-beta by 10-25%.
RATIO_FLOOR = 1e-5
# floor of (1 - f_esc) in the analytic scale of the reference line (inference): f_esc -> 1 is not log(0)
ONE_MINUS_FESC_FLOOR = 1e-4

#========================== CALIBRATION OF THE SIGMA ==========================
CALIB_FRAC = 0.05   # default fraction of the CLOUDS kept out of training to estimate the scale s_k of sigma
CALIB_FAMS_HII = tuple(f for f, _, _ in FAMILIES_HII)  # default: every family calibrated
SIGMA_SCALE_MIN = 1.0  # lower clip of s_k: the calibration only widens sigma, never shrinks it
# Median of |r| for r ~ N(0, 1): Phi^-1(0.75). Half of a standard normal lies within +-0.6745.
MEDIAN_ABS_STD_NORMAL = 0.6744897501960817
# Sigma calibration and its check: s_k from the 95% quantile of |r| on the
# calibration clouds; on the holdout every family must then hold 93-97% of the points within 1.96 sigma
# (cov95) and at least 60% within 1 sigma (cov68: a core that is not overconfident). With heavy-tailed
# residuals a calibrated cov95 implies a conservative cov68, so cov68 has no upper bound.
CALIB_QUANTILE = 0.95
COV95_RANGE = (0.93, 0.97)
COV68_MIN = 0.60

#================================ VALIDATION ================================
FRAC_5PCT_DEX = math.log10(1.05)        # "5% in log-space": RMSE of log10(pred/true)
FRAC_10PCT_DEX = math.log10(1.10)
# some critical lines (species, wavelength as printed by CLOUDY)
CRITICAL_HII = {'H_alpha': ('H 1', 6562.8), 'H_beta': ('H 1', 4861.3),
                'OIII_5007': ('O 3', 5006.8), 'NII_6584': ('N 2', 6583.5),
                'OII_3726': ('O 2', 3726.0), 'OII_3729': ('O 2', 3728.8),
                'CIII_1909': ('Blnd', 1909.0)}
# lines scanned along the input axes (ARD and monotonicity check)
SCAN_HII = {'SII_6716': ('S 2', 6716.4), 'SII_6731': ('S 2', 6730.8),
            'SIII_9069': ('S 3', 9068.6), 'SIII_9531': ('S 3', 9530.6),
            'OI_6300': ('O 1', 6300.3), 'H_alpha': ('H 1', 6562.8),
            'OIII_5007': ('O 3', 5006.8), 'OII_3726': ('O 2', 3726.0), 'OII_3729': ('O 2', 3728.8)}
