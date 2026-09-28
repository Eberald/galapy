# Author: Enrico Veraldi
# Tests on parsing functions

import numpy as np
import pytest

# HII parser
from galapy.spectroscopy.utils.hii.parse_one_hii import (fesc, fuv_transmittance_hii, s_k_factor, support_safe_ratio)
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