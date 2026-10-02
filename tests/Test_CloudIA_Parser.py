# Author: Enrico Veraldi
# Tests on parsing functions

import numpy as np
import pytest

# HII parser
from galapy.spectroscopy.utils.hii.parse_one_hii import (fesc, fuv_transmittance_hii, s_k_factor, support_safe_ratio,
                                                         energy_balance, cmb_incident_ratio, air_to_vacuum_A,
                                                         parse_cloudy_cong, STELLAR_MAX_A, UNITS_SCHEMA,
                                                         ROOT_UNITS, POINT_UNITS)
from galapy.internal.constants import LymanA, FUV_Lo_A, FUV_Hi_A, clight

C_CGS = clight['cm/s']


#================================================= test on fesc
@pytest.mark.unit
def test_fesc_full_transmission_below_threshold():
    wave = np.array([900.0, 700.0, 500.0])          # decreasing, < 911.6
    col = np.array([1.0, 3.0, 2.0])
    assert fesc(wave, col, col) == pytest.approx(1.0)

@pytest.mark.unit
def test_fesc_zero_transmission():
    wave = np.array([900.0, 700.0, 500.0])
    col2 = np.array([1.0, 1.0, 1.0])
    assert fesc(wave, col2, np.zeros_like(col2)) == pytest.approx(0.0)


@pytest.mark.unit
def test_fesc_half_transmission():
    wave = np.array([900.0, 800.0, 600.0, 400.0])
    col2 = np.array([1.0, 2.0, 1.5, 1.0])
    assert fesc(wave, col2, 0.5 * col2) == pytest.approx(0.5)


@pytest.mark.unit
def test_fesc_only_counts_below_lyman():
    wave = np.array([5000.0, 1000.0, 900.0, 500.0])
    col2 = np.array([1.0, 1.0, 1.0, 1.0])
    col3 = np.array([0.0, 0.0, 1.0, 1.0])
    assert fesc(wave, col2, col3) == pytest.approx(1.0)


@pytest.mark.unit
def test_fesc_clipped_to_unit_interval():
    wave = np.array([900.0, 700.0, 500.0])
    col2 = np.array([1.0, 1.0, 1.0])
    assert fesc(wave, col2, 2.0 * col2) == pytest.approx(1.0)


@pytest.mark.unit
def test_fesc_no_ionizing_incident_returns_zero():
    wave = np.array([3000.0, 2000.0])                # all wavelenghts > 911.6
    assert fesc(wave, np.ones(2), np.ones(2)) == 0.0


#================================================= test fuv_transmittance_hii
@pytest.mark.unit
def test_tfuv_full_transmission_in_band():
    wave = np.array([2000.0, 1500.0, 1000.0])
    col2 = np.array([1.0, 2.0, 1.0])
    T, N = fuv_transmittance_hii(wave, col2, col2, np.zeros_like(col2))
    assert T == pytest.approx(1.0)
    assert N == pytest.approx(0.0)


@pytest.mark.unit
def test_tfuv_half_and_nfuv_ratio():
    wave = np.array([2000.0, 1800.0, 1400.0, 1000.0])
    col2 = np.array([1.0, 2.0, 3.0, 1.0])
    T, N = fuv_transmittance_hii(wave, col2, 0.5 * col2, 0.25 * col2)
    assert T == pytest.approx(0.5)
    assert N == pytest.approx(0.25)


@pytest.mark.unit
def test_fuv_band_restriction():
    wave = np.array([5000.0, 2000.0, 1000.0, 500.0])  # 5000 and 500 are ot of bands
    col2 = np.array([9.0, 1.0, 1.0, 9.0])
    col3 = np.array([0.0, 1.0, 1.0, 0.0])
    T, _ = fuv_transmittance_hii(wave, col2, col3, np.zeros_like(col2))
    assert T == pytest.approx(1.0)


@pytest.mark.unit
def test_tfuv_clipped_nfuv_nonnegative():
    wave = np.array([2000.0, 1500.0, 1000.0])
    col2 = np.ones(3)
    T, N = fuv_transmittance_hii(wave, col2, 3.0 * col2, -1.0 * col2)
    assert T == pytest.approx(1.0)
    assert N >= 0.0


@pytest.mark.unit
def test_fuv_no_incident_returns_zeros():
    wave = np.array([200.0, 100.0]) # put of bands
    assert fuv_transmittance_hii(wave, np.ones(2), np.ones(2), np.ones(2)) == (0.0, 0.0)


@pytest.mark.unit
def test_fuv_ratios_are_energy_weighted():
    # nuFnu integrated in dln(lambda) (energy): in dlambda (photon rates) it would be 0.25
    wave = np.array([2000.0, 1500.0, 1000.0])
    col = np.array([1.0, 0.0, 0.0])
    T, N = fuv_transmittance_hii(wave, np.ones(3), col, col)
    expected = np.log(2000.0 / 1500.0) / (2.0 * np.log(2.0))
    assert T == pytest.approx(expected)
    assert N == pytest.approx(expected)


#================================================= test energy balance and CMB
def _star_and_cmb():
    wave = np.logspace(7, 2, 400)                 # decreasing, as the CLOUDY mesh
    star = np.where(wave < 3.0e4, 1.0, 0.0)       # stellar incident field
    cmb = np.where(wave > 2.0e5, 50.0, 0.0)       # isotropic CMB of the run
    return wave, star, cmb


@pytest.mark.unit
def test_energy_balance_is_referred_to_the_stellar_incident_energy():
    wave, star, cmb = _star_and_cmb()
    col2 = star + cmb
    col3 = 0.5 * col2
    col4 = 0.5 * col2 + 0.01 * star               # 1% of the stellar energy counted twice
    stellar = energy_balance(wave, col2, col3, col4, split_A=STELLAR_MAX_A)
    assert stellar == pytest.approx(0.01, rel=1e-9)
    assert energy_balance(wave, col2, col3, col4) < stellar    # diluted by the CMB energy
    assert energy_balance(wave[::-1], col2[::-1], col3[::-1], col4[::-1],
                          split_A=STELLAR_MAX_A) == pytest.approx(stellar)


@pytest.mark.unit
def test_cmb_incident_ratio_compares_the_two_sides_of_the_split():
    wave, star, cmb = _star_and_cmb()
    lnw = np.log(wave[::-1])
    expected = np.trapezoid(cmb[::-1], lnw) / np.trapezoid(star[::-1], lnw)
    assert cmb_incident_ratio(wave, star + cmb) == pytest.approx(expected, rel=1e-9)
    assert cmb_incident_ratio(wave, star) == 0.0
    assert np.isnan(cmb_incident_ratio(wave, cmb))


#================================================= test air -> vacuum wavelengths
def _cloudy_print_air(wl_vac):
    """The print rule of CLOUDY (t_wavl::sprt_wl): vacuum -> air above 2000 A."""
    wl_vac = np.asarray(wl_vac, dtype=float)
    sigma2 = (1.0e4 / wl_vac) ** 2
    n_air = 1.0 + 1.0e-8 * (8060.51 + 2480990.0 / (132.274 - sigma2) + 17455.7 / (39.32957 - sigma2))
    return np.where(wl_vac > 2000.0, wl_vac / n_air, wl_vac)


@pytest.mark.unit
def test_air_to_vacuum_halpha():
    assert air_to_vacuum_A([6562.80])[0] == pytest.approx(6564.613, abs=2e-3)


@pytest.mark.unit
def test_air_to_vacuum_is_identity_up_to_2000A():
    wl = np.array([1215.67, 1906.68, 2000.0])
    np.testing.assert_array_equal(air_to_vacuum_A(wl), wl)


@pytest.mark.unit
def test_air_to_vacuum_inverts_the_cloudy_print_rule():
    wl_vac = np.array([2500.0, 4862.68, 6564.61, 1.0e4, 1.5768e6, 2.0e8])   # optical to radio
    np.testing.assert_allclose(air_to_vacuum_A(_cloudy_print_air(wl_vac)), wl_vac, rtol=1e-7)


#================================================= test grain continuum and units
@pytest.mark.unit
def test_cong_must_share_the_continuum_mesh(tmp_path):
    wave = np.logspace(6, 2, 50)
    path = tmp_path / 'model.con_grain'
    np.savetxt(path, np.column_stack([wave, wave, wave, wave]))
    assert parse_cloudy_cong(path, wave_ref=wave).shape == wave.shape
    with pytest.raises(ValueError, match='continuum mesh'):
        parse_cloudy_cong(path, wave_ref=wave * (1.0 + 1e-3))
    with pytest.raises(ValueError, match='continuum mesh'):
        parse_cloudy_cong(path, wave_ref=wave[1:])


@pytest.mark.unit
def test_units_tables_are_well_formed():
    assert UNITS_SCHEMA.startswith('cloudia.hii.')
    for name, (unit, description) in POINT_UNITS.items():
        assert isinstance(unit, str) and description.strip(), name
    for name, (unit, description, extra) in ROOT_UNITS.items():
        assert (unit is None or isinstance(unit, str)) and description.strip(), name
        assert isinstance(extra, dict)
    assert ROOT_UNITS['continuum/wave_grid'][2]['wavelength_medium'] == 'vacuum'
    assert POINT_UNITS['continuum/nebular_emission_per_Msun'][0] == 'erg s-1 Msun-1'
    assert POINT_UNITS['s_k'][0] == 'cm2 Msun-1'


#================================================= test wavelength order
@pytest.mark.unit
def test_wavelength_order_convention_matters():
    wave_desc = np.array([2000.0, 1500.0, 1000.0])
    col = np.array([1.0, 2.0, 1.0])
    T_desc, _ = fuv_transmittance_hii(wave_desc, col, col, np.zeros_like(col))
    T_asc, _ = fuv_transmittance_hii(wave_desc[::-1], col[::-1], col[::-1],
                                     np.zeros_like(col))
    assert T_desc == pytest.approx(1.0)
    assert T_asc == 0.0                              # wrong order gives a slinet zero


#================================================= test s_k geometric factor
@pytest.mark.unit
def test_s_k_formula():
    Qh, logU, lognH = 1e52, -2.0, 3.0
    expected = Qh / (10**logU * 10**lognH * C_CGS)
    assert s_k_factor(Qh, logU, lognH) == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_s_k_linear_in_Q():
    assert (s_k_factor(2e52, -2.0, 3.0) ==
            pytest.approx(2 * s_k_factor(1e52, -2.0, 3.0)))


@pytest.mark.unit
def test_s_k_inverse_in_U_and_nH():
    base = s_k_factor(1e52, -2.0, 3.0)
    assert s_k_factor(1e52, -1.0, 3.0) == pytest.approx(base / 10.0)
    assert s_k_factor(1e52, -2.0, 4.0) == pytest.approx(base / 10.0)


@pytest.mark.unit
def test_band_constants_sane():
    assert LymanA < FUV_Lo_A < FUV_Hi_A


#==================================================== test support_safe_ratio
#

def _cols_with_gap(n=64, gap=slice(40, 56)):
    wave = np.linspace(2.0e6, 1.0e2, n)
    col2 = np.ones(n)
    col3 = 0.25 * np.ones(n)
    col2[gap] = 0.0   #after lambda_cut no incident
    col3[gap] = 0.0
    return wave, col2, col3, gap


@pytest.mark.unit
def test_transmission_is_unity_outside_support():
    _w, col2, col3, gap = _cols_with_gap()
    t = support_safe_ratio(col2, col3)
    assert np.all(t[gap] == 1.0)


@pytest.mark.unit
def test_transmission_is_the_ratio_inside_support():
    _w, col2, col3, gap = _cols_with_gap()
    t = support_safe_ratio(col2, col3)
    inside = np.ones(col2.size, dtype=bool)
    inside[gap] = False
    assert t[inside] == pytest.approx(0.25)


@pytest.mark.unit
def test_a_lambda_is_zero_outside_support():
    _w, col2, col3, gap = _cols_with_gap()
    A = -2.5 * np.log10(support_safe_ratio(col2, col3))
    assert np.all(A[gap] == 0.0)
    assert not np.any(np.isclose(A, 75.0, atol=1.0))


@pytest.mark.unit
def test_contract_never_produces_nan_or_inf():
    _w, col2, col3, _gap = _cols_with_gap()
    t = support_safe_ratio(col2, col3)
    assert np.all(np.isfinite(t))
    assert np.all(np.isfinite(-2.5 * np.log10(np.clip(t, 1e-300, None))))


@pytest.mark.unit
def test_zero_incident_everywhere_is_fully_transparent():
    t = support_safe_ratio(np.zeros(16), np.zeros(16))
    assert np.all(t == 1.0)


@pytest.mark.unit
def test_negative_incident_is_treated_as_outside_support():
    col2 = np.array([1.0, -1.0, 1.0])
    col3 = np.array([0.5, 0.5, 0.5])
    t = support_safe_ratio(col2, col3)
    assert t[1] == 1.0 and np.all(t >= 0.0)


@pytest.mark.unit
def test_transmission_dtype_and_shape_are_preserved():
    _w, col2, col3, _gap = _cols_with_gap(n=97)
    t = support_safe_ratio(col2, col3)
    assert t.shape == col3.shape and t.dtype == np.dtype(float)