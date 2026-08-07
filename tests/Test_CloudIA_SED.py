# Author: Enrico Veraldi
# check the sed extraction in CLOUDY format
# galapy/spectroscopy/utils/spectra.py

import numpy as np
import pytest

import galapy.internal.constants as CONST
import galapy.spectroscopy.utils.spectra as spc


@pytest.fixture
def spectrum():
    """
    Synthetic broken power-law spectrum used as input for the writer.

    The blue side follows a Rayleigh-Jeans-like lambda^-4 law and steepens to
    lambda^-1.2 above 1.5e6 Angstrom, so that both the Lyman cut and the red
    cut of `write_cloudy_sed` act on a non-trivial spectral shape.

    Returns:
        tuple: (lam, L_lambda) with lam in Angstrom, sampled logarithmically
            from 10 to 1e10 Angstrom.
    """
    lam = np.logspace(1, 10, 3000)
    y = 1e-5 * (lam / 1e4) ** -4.0
    tail = lam > 1.5e6
    y[tail] = y[tail][0] * (lam[tail] / lam[tail][0]) ** -1.2
    return lam, y


@pytest.fixture
def cube():
    """
    Minimal in-memory SSP cube with the layout returned by `load_ssp_cube`.

    Built without touching the GalaPy database, so that the cube-based entry
    points can be exercised offline. Each (it, iz) node is a distinct rescaling
    of the same power law, which lets the tests check that the right slice is
    selected.

    Returns:
        tuple: (l, t, Z, L) with L of shape (l.size, t.size, Z.size).
    """
    l = np.logspace(1, 7, 500)
    t = np.array([1e6, 1e7])
    Z = np.array([0.0004, 0.0040, 0.0200])
    base = 1e-5 * (l / 1e4) ** -4.0
    L = np.empty((l.size, t.size, Z.size))
    for it in range(t.size):
        for iz in range(Z.size):
            L[:, it, iz] = base * (it + 1) * (iz + 1)
    return l, t, Z, L


def _read(path):
    """
    Reads back a CLOUDY table SED file, ignoring comments and the unit prefix.

    Parameters:
        path (str): Path to the file written by `write_cloudy_sed`.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Wavelengths in Angstrom and the
            corresponding nu*Fnu values.
    """
    lam, val = [], []
    for raw in open(path):
        raw = raw.strip()
        if not raw or raw.startswith('#'):
            continue
        tok = raw.split()
        lam.append(float(tok[0])); val.append(float(tok[1]))
    return np.array(lam), np.array(val)


# =============== WRITER: FORMAT ===============

@pytest.mark.unit
def test_sed_format_six_facts(tmp_path, spectrum):
    """
    Checks the six properties CLOUDY requires from a `table SED` file: the unit
    prefix sits on the first data row only, `extrapolate` is absent unless
    requested, fluxes are strictly positive and wavelengths strictly increasing.
    """
    lam, y = spectrum
    p = spc.write_cloudy_sed(lam, y, tmp_path / 'a.sed')
    lines = [l for l in open(p) if not l.startswith('#')]
    assert 'nuFnu units Angstroms' in lines[0]
    assert 'extrapolate' not in lines[0]
    assert all('units' not in l for l in lines[1:])
    w, v = _read(p)
    assert np.all(v > 0)
    assert np.all(np.diff(w) > 0)
    assert w.size >= 2


@pytest.mark.unit
def test_extrapolate_keyword_is_written(tmp_path, spectrum):
    """
    Verifies that `extrapolate=True` appends the CLOUDY `extrapolate` keyword to
    the unit prefix, and that it stays confined to the first data row.
    """
    lam, y = spectrum
    p = spc.write_cloudy_sed(lam, y, tmp_path / 'e.sed', extrapolate=True)
    lines = [l for l in open(p) if not l.startswith('#')]
    assert lines[0].rstrip().endswith('nuFnu units Angstroms extrapolate')
    assert all('extrapolate' not in l for l in lines[1:])


@pytest.mark.unit
def test_duplicate_wavelengths_are_deduplicated(tmp_path):
    """
    CLOUDY rejects repeated wavelengths, so the writer must collapse duplicates
    and leave a strictly increasing grid.
    """
    lam = np.array([1e3, 1e3, 2e3, 3e3])
    p = spc.write_cloudy_sed(lam, np.ones(4), tmp_path / 'd.sed')
    w, _ = _read(p)
    assert w.size == 3 and np.all(np.diff(w) > 0)


@pytest.mark.unit
def test_unsorted_input_is_sorted(tmp_path):
    """
    The writer must not assume a monotonic input grid: a shuffled SED has to be
    written out in increasing wavelength order, keeping each flux with its own
    wavelength.
    """
    lam = np.array([3e3, 1e3, 2e3])
    L_lambda = np.array([3.0, 1.0, 2.0])
    p = spc.write_cloudy_sed(lam, L_lambda, tmp_path / 's.sed')
    w, v = _read(p)
    assert np.all(np.diff(w) > 0)
    expected = np.sort(lam) * np.array([1.0, 2.0, 3.0]) * CONST.Lsun
    assert np.allclose(v, expected, rtol=1e-6)


@pytest.mark.unit
def test_nonpositive_flux_is_floored_not_dropped(tmp_path):
    """
    Null or negative fluxes are floored to a tiny positive value rather than
    removed, so the wavelength grid keeps its original sampling.
    """
    lam = np.array([1e3, 2e3, 3e3])
    p = spc.write_cloudy_sed(lam, np.array([1.0, 0.0, 1.0]), tmp_path / 'f.sed')
    w, v = _read(p)
    assert w.size == 3 and np.all(v > 0)


@pytest.mark.unit
def test_two_points_are_required(tmp_path):
    """
    A single-point SED cannot be interpolated by CLOUDY and must be rejected,
    whatever array-like the wavelengths come in as: the length check must run
    after the input has been coerced to an ndarray.
    """
    with pytest.raises(ValueError):
        spc.write_cloudy_sed(np.array([1e3]), np.array([1.0]), tmp_path / 'x.sed')
    with pytest.raises(ValueError):
        spc.write_cloudy_sed([1e3], [1.0], tmp_path / 'x.sed')


@pytest.mark.unit
def test_list_input_is_accepted(tmp_path):
    """
    Plain lists are a valid input and must be coerced, not rejected with an
    AttributeError from an ndarray-only attribute access.
    """
    p = spc.write_cloudy_sed([1e3, 2e3, 3e3], [1.0, 2.0, 3.0], tmp_path / 'l.sed')
    w, v = _read(p)
    assert np.allclose(w, [1e3, 2e3, 3e3], rtol=1e-6)
    assert np.allclose(v, np.array([1e3, 4e3, 9e3]) * CONST.Lsun, rtol=1e-6)


@pytest.mark.unit
def test_cut_leaving_no_points_raises(tmp_path, spectrum):
    """
    A wavelength cut that discards the whole SED must raise instead of writing
    an empty file that CLOUDY would fail on much later.
    """
    lam, y = spectrum
    with pytest.raises(ValueError):
        spc.write_cloudy_sed(lam, y, tmp_path / 'x.sed', lambda_min_A=1.0e12)


# =============== WRITER: TRUNCATIONS ===============

@pytest.mark.unit
def test_blue_truncation_applied(tmp_path, spectrum):
    """
    `lambda_min_A` removes the ionizing side of the SED.
    """
    lam, y = spectrum
    w, _ = _read(spc.write_cloudy_sed(lam, y, tmp_path / 'b.sed',
                                      lambda_min_A=CONST.LyLimit))
    assert w.min() >= CONST.LyLimit


@pytest.mark.unit
def test_red_truncation_applied(tmp_path, spectrum):
    """
    `lambda_max_A` removes the far-IR/radio tail, where the SSP tables are not
    trustworthy as a CLOUDY input.
    """
    lam, y = spectrum
    w, _ = _read(spc.write_cloudy_sed(lam, y, tmp_path / 'r.sed',
                                      lambda_max_A=CONST.SED_cut))
    assert w.max() <= CONST.SED_cut


@pytest.mark.unit
def test_both_truncations_compose(tmp_path, spectrum):
    """
    The two cuts are independent and can be applied together, still leaving a
    strictly increasing grid.
    """
    lam, y = spectrum
    w, _ = _read(spc.write_cloudy_sed(lam, y, tmp_path / 'c.sed',
                                      lambda_min_A=CONST.LyLimit,
                                      lambda_max_A=CONST.SED_cut))
    assert w.min() >= CONST.LyLimit
    assert w.max() <= CONST.SED_cut
    assert np.all(np.diff(w) > 0)


@pytest.mark.unit
def test_blue_cut_leaves_the_red_end_untouched(tmp_path, spectrum):
    """
    Adding `lambda_min_A` on top of a red cut must only move the blue end: the
    red edge has to stay exactly where the red cut alone put it.
    """
    lam, y = spectrum
    wh, _ = _read(spc.write_cloudy_sed(lam, y, tmp_path / 'h.sed',
                                       lambda_max_A=CONST.SED_cut))
    wp, _ = _read(spc.write_cloudy_sed(lam, y, tmp_path / 'p.sed',
                                       lambda_min_A=CONST.LyLimit,
                                       lambda_max_A=CONST.SED_cut))
    assert wh.max() == wp.max()
    assert wp.min() > wh.min()


# =============== QH ===============

@pytest.mark.unit
def test_QH_roundtrip(tmp_path, spectrum):
    """
    Q_H recomputed from the written file must match the value integrated
    directly from the input arrays. The tolerance probes the precision of the
    `%.6e` formatting used by the writer, not the physics.
    """
    lam, y = spectrum
    HC = CONST.hP["erg*s"] * CONST.clight["cm/s"]
    m = lam < CONST.LymanA
    order = np.argsort(lam[m])
    ref = float(np.trapezoid((y[m] * lam[m] * CONST.Lsun)[order],
                             lam[m][order] * 1e-8) / HC)
    got = spc.cloudy_sed_QH(spc.write_cloudy_sed(lam, y, tmp_path / 'rt.sed'))
    assert abs(got - ref) / ref < 1e-6


@pytest.mark.unit
def test_QH_invariant_under_red_cut(tmp_path, spectrum):
    """
    The red cut sits far above the Lyman limit, so it cannot change the ionizing
    photon budget at all.
    """
    lam, y = spectrum
    q_full = spc.cloudy_sed_QH(spc.write_cloudy_sed(lam, y, tmp_path / 'q1.sed'))
    q_cut = spc.cloudy_sed_QH(spc.write_cloudy_sed(lam, y, tmp_path / 'q2.sed',
                                                   lambda_max_A=CONST.SED_cut))
    assert q_full == q_cut


@pytest.mark.unit
def test_QH_is_zero_on_lyman_truncated_file(tmp_path, spectrum):
    """
    A file truncated at the Lyman limit carries no ionizing photons. Note this
    relies on CONST.LyLimit (911.76 A) sitting just above the integration bound
    CONST.LymanA (911.6 A), so no point survives the cut below it.
    """
    lam, y = spectrum
    p = spc.write_cloudy_sed(lam, y, tmp_path / 'z.sed', lambda_min_A=CONST.LyLimit)
    assert spc.cloudy_sed_QH(p) == 0.0


@pytest.mark.unit
def test_QH_reference_matches_file_QH(tmp_path, cube):
    """
    Cross-checks the two independent Q_H implementations: `cloudy_sed_QH_reference`
    integrates the raw SSP node, `cloudy_sed_QH` re-reads the written file. They
    must agree to within the writer's formatting precision.
    """
    it, iz = 1, 2
    ref = spc.cloudy_sed_QH_reference(cube, it, iz)
    got = spc.cloudy_sed_QH(spc.to_cloudy_sed(tmp_path / 'ref.sed',
                                              cube=cube, it=it, iz=iz))
    assert ref > 0.0
    assert abs(got - ref) / ref < 1e-6


# =============== EXPORT ===============

@pytest.mark.unit
def test_sed_from_ssp_cube_node_selects_the_right_slice(cube):
    """
    `sed_from_ssp_cube_node` must return the wavelength grid untouched and the
    (it, iz) slice of the luminosity cube.
    """
    l, _t, _Z, L = cube
    w, y = spc.sed_from_ssp_cube_node(cube, 1, 2)
    assert np.array_equal(w, l)
    assert np.array_equal(y, L[:, 1, 2])


@pytest.mark.unit
def test_to_cloudy_sed_from_cube_node(tmp_path, cube):
    """
    The cube entry point writes a file whose fluxes are the node luminosities
    converted to nu*Fnu, with the cuts forwarded to the writer.
    """
    p = spc.to_cloudy_sed(tmp_path / 'n.sed', cube=cube, it=0, iz=0,
                          lambda_min_A=CONST.LyLimit)
    w, v = _read(p)
    l, _t, _Z, L = cube
    m = l >= CONST.LyLimit
    assert np.allclose(w, l[m], rtol=1e-6)
    assert np.allclose(v, (L[:, 0, 0] * l * CONST.Lsun)[m], rtol=1e-6)


@pytest.mark.unit
def test_to_cloudy_sed_requires_a_source(tmp_path):
    """
    Neither (cube, it, iz) nor (csp, age, sfh) given: the call must fail loudly
    rather than write nothing.
    """
    with pytest.raises(ValueError):
        spc.to_cloudy_sed(tmp_path / 'none.sed')
    with pytest.raises(ValueError):
        spc.to_cloudy_sed(tmp_path / 'none.sed', it=0, iz=0)
