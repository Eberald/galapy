# Author: Enrico Veraldi
# Tests related to emulators (SVGPs)
import sys

import h5py
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('gpytorch')

TAUS = (1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 7e7, 1e8)
ZS = (1e-4, 5e-4, 1e-3, 4e-3, 8e-3, 2e-2)
CLIGHT = 2.99792458e10


def qh_true(tau, Z):
    return 1e47 * (tau / 1e6) ** -2.0 * (1.0 + 10.0 * Z)

def write_hii_corpus(path, pumped=True, drop_flag=False, reps=1, opaque_frac=0.0):
    import galapy.internal.cloudy as CLOUDY
    rng = np.random.default_rng(0)
    wave = np.geomspace(100.0, 1e6, 60)
    with h5py.File(path, 'w') as f:
        u = f.create_group('units')
        for k, (unit, _d) in CLOUDY.Points_Units_HII.items():
            if unit is not None:
                u.attrs[k] = unit
        f.create_dataset('line_names', data=np.array([b'H  1', b'H  1', b'O  3', b'N  2']))
        f.create_dataset('lines_emergent/wavelengths_rest_vacuum', data=[6564.6, 4862.7, 5008.2, 6585.3])
        f.create_dataset('continuum/wave_grid', data=wave)
        j = 0
        for tau, Z in [(t, z) for _ in range(reps) for t in TAUS for z in ZS]:
            if True:
                logU, lognH = rng.uniform(-4, -1), rng.uniform(0, 4)
                g = f.create_group(f'grid_point_{j:05d}')
                g.attrs.update({'logU': logU, 'lognH_HII': lognH, 'log_zeta_O': rng.uniform(-2, 0.5),
                                'z_CMB': rng.uniform(0, 12), 'xi_d': rng.uniform(0.1, 1),
                                'f_esc_meas': (rng.uniform(0, 0.05) if opaque_frac and rng.uniform() < opaque_frac
                                               else rng.uniform(0, 0.9)), 'F_star': rng.uniform(0, 1),
                                'tau_SSP': tau, 'Z_star': Z, 'not_converged': False})
                if not drop_flag:
                    g.attrs['pumping_removed'] = pumped if j else True
                a = 10.0 ** (logU + 40.0) * (tau / 1e6) ** -1
                g['lines_emergent/fluxes'] = a * np.array([2.86, 1.0, 2.0 * (1 + logU / 10), 0.3])
                g['continuum/nebular_emission_per_Msun'] = a * (wave / 1e3) ** -0.5 * (1 + logU / 10)
                g['continuum/grain_diag_per_Msun'] = a * np.exp(-((np.log10(wave) - 5) ** 2))
                g['continuum/transmission'] = np.clip(0.5 + 0.1 * logU * (wave / 1e6), 1e-3, 1)
                g['dust_mass_per_Msun'] = 1e-4 * 10 ** lognH
                g['T_fuv_hii'] = 0.5 + 0.1 * logU
                g['N_fuv_hii'] = 1e-3 * (1 + Z)
                g['s_k'] = qh_true(tau, Z) / (10.0 ** (logU + lognH) * CLIGHT)
                j += 1
    return path


@pytest.fixture(scope='module')
def hii_artifact(tmp_path_factory):
    from galapy.spectroscopy.utils.hii import train_emulator_hii as T
    d = tmp_path_factory.mktemp('hii')
    corpus = write_hii_corpus(d / 'hii_grid.h5')
    argv = ['train', '--corpus', str(corpus), '--out', str(d / 'emu.pt'), '--meta', str(d / 'emu.json'),
            '--inducing', '8', '--inducing-lines', '8', '--epochs', '2', '--batch', '48',
            '--device', 'cpu']
    old, sys.argv = sys.argv, argv
    try:
        T.main()
    finally:
        sys.argv = old
    return d / 'emu', corpus


@pytest.mark.unit
def test_training_reads_the_nodes_in_log10(tmp_path):
    from galapy.spectroscopy.utils.hii.train_emulator_hii import load_corpus
    import galapy.internal.emulators as EMUL
    X, _Y, meta = load_corpus(write_hii_corpus(tmp_path / 'c.h5'))
    it, iz = EMUL.INPUT_KEYS_HII.index('tau_SSP'), EMUL.INPUT_KEYS_HII.index('Z_star')
    assert np.allclose(sorted(set(X[:, it])), np.log10(TAUS))
    assert np.allclose(sorted(set(X[:, iz])), np.log10(ZS))
    assert meta['pumping_removed'] is True


@pytest.mark.unit
@pytest.mark.parametrize('case', ['missing', 'mixed'])
def test_corpus_must_declare_the_pumping(tmp_path, case):
    from galapy.spectroscopy.utils.hii.train_emulator_hii import load_corpus
    path = write_hii_corpus(tmp_path / 'c.h5', pumped=False, drop_flag=(case == 'missing'))
    with pytest.raises(ValueError, match='pumping removed'):
        load_corpus(path)


@pytest.mark.unit
def test_artifact_is_cpu_and_predict_contract(hii_artifact):
    from galapy.spectroscopy.svgp import load_artifacts
    from galapy.spectroscopy.hii_emulator import HIIEmulator
    stem, corpus = hii_artifact
    fam, _ex = load_artifacts([f'{stem}_{p}.pt' for p in ('lines', 'cont', 'grain', 'trans', 'scalars')])
    assert all(v.device.type == 'cpu' for f in fam.values() for v in f['model'].values())
    e = HIIEmulator().load(stem, device='cpu')
    theta = np.array([[-2.0, 2.0, -0.5, 1.0, 0.4, 0.3, 0.5, np.log10(t), np.log10(0.004)] for t in TAUS])
    out = e.predict_batch(theta)
    for key, n in (('lines_emergent', 4), ('continuum_nebular', 60), ('transmission', 60)):
        mean, sigma = out[key]
        assert isinstance(mean, np.ndarray) and mean.shape == sigma.shape == (8, n)
        assert np.all(mean > 0) and np.all(sigma >= 0)
    assert out['dust_mass_cloud'][0].shape == (8,) and out['T_fuv_hii'].shape == (8,)


@pytest.mark.unit
def test_predict_standardises_the_input_and_matches_gpytorch(hii_artifact):
    import gpytorch
    from galapy.spectroscopy.hii_emulator import HIIEmulator
    from galapy.spectroscopy.svgp import fast_predict
    e = HIIEmulator().load(hii_artifact[0])
    theta = torch.tensor([[-2.0, 2.0, -0.5, 1.0, 0.4, 0.3, 0.5, 6.5, -2.4]])
    x = (theta - e.x_mean) / e.x_std
    with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
        ref = e.line_gp(x)
        mean, var = fast_predict(e.cache['lines'], x)
    np.testing.assert_allclose(mean.numpy(), ref.mean.numpy(), rtol=1e-4, atol=1e-6)
    # fast_predict returns the variance of y = latent + likelihood noise, times the calibration scale s_k^2
    c = e.cache['lines']
    ref_var = ref.variance + (c['noise'][:, None] if 'noise' in c else 0.0)
    ref_var = ref_var * (c['var_scale'][:, None] if 'var_scale' in c else 1.0)
    np.testing.assert_allclose(var.numpy(), ref_var.numpy(), rtol=2e-4)

@pytest.mark.unit
def test_qh_unit_is_read_from_the_corpus_and_linear_in_Z(hii_artifact):
    from galapy.spectroscopy.hii_emulator import HIIEmulator
    e = HIIEmulator().load(hii_artifact[0])
    row = lambda t, z: [-2.0, 2.0, -0.5, 1.0, 0.4, 0.3, 0.5, np.log10(t), np.log10(z)]
    q = e.Qh_unit_lookup(np.array([row(5e6, 1e-3), row(5e6, 2.5e-3)]))
    assert q[0] == pytest.approx(qh_true(5e6, 1e-3), rel=1e-6)
    assert q[1] == pytest.approx(0.5 * (qh_true(5e6, 1e-3) + qh_true(5e6, 4e-3)), rel=1e-6)
    with pytest.raises(ValueError, match='nodes'):
        e.Qh_unit_lookup(np.array([row(3e6, 1e-3)]))


@pytest.mark.unit
def test_holdout_split_is_by_cloud_stratified_and_deterministic():
    from galapy.spectroscopy.utils.validate_emulator import holdout_split, cloud_id
    n_c, n_n = 200, 6
    keys = [f'grid_point_{c:05d}_{i:03d}_{i % 3:02d}' for c in range(n_c) for i in range(n_n)]
    X = np.repeat(np.random.default_rng(3).uniform(size=(n_c, 3)), n_n, axis=0)
    m = holdout_split(keys, X, (0, 1, 2), 0.1, 7)
    side = {}
    for k, t in zip(keys, m):
        side.setdefault(cloud_id(k), set()).add(bool(t))
    assert all(len(s) == 1 for s in side.values())
    assert 0.07 * n_c <= sum(True in s for s in side.values()) <= 0.13 * n_c
    assert np.array_equal(m, holdout_split(keys, X, (0, 1, 2), 0.1, 7))
    assert not np.array_equal(m, holdout_split(keys, X, (0, 1, 2), 0.1, 8))


@pytest.mark.unit
def test_training_holds_out_clouds_and_records_them(hii_artifact):
    import json
    from galapy.spectroscopy.utils.hii.train_emulator_hii import load_corpus
    from galapy.spectroscopy.hii_emulator import HIIEmulator
    stem, corpus = hii_artifact
    hold = json.loads(open(f'{stem}_holdout.json').read())
    meta = json.loads((stem.parent / 'emu.json').read_text())
    X, _Y, m = load_corpus(corpus)
    test = np.isin(m['keys'], hold['keys'])
    assert 0 < test.sum() < len(X) and meta['holdout']['n_test'] == test.sum()
    e = HIIEmulator().load(stem)
    np.testing.assert_allclose(e.x_mean.numpy(), X[~test].mean(0), rtol=1e-5, atol=1e-5)
    assert {'epoch', 'rmse_z', 'cov68'} <= set(meta['holdout']['final']['lines'])
    z = np.load(f'{stem}_gate_set_frozen.npz')
    assert len(z['X']) == test.sum() and list(z['families']) == sorted(e.cache)


@pytest.mark.unit
def test_validate_emulator_runs_every_hii_check(hii_artifact, tmp_path):
    import json
    from galapy.spectroscopy.utils import validate_emulator as V
    stem, corpus = hii_artifact
    rep = tmp_path / 'r.json'
    V.main(['--sector', 'hii', '--emulator', str(stem), '--meta', str(stem.parent / 'emu.json'),
            '--corpus', str(corpus), '--report', str(rep),
            '--tp-threads', '1', '--tp-batches', '8', '--tp-min-time', '0.05'])   # small throughput grid
    checks = json.loads(rep.read_text())['gates']
    assert {c for c, v in V.CHECKS.items() if v[1] == 'hii'} == set(checks)
    assert all(v['status'] in ('PASS', 'FAIL', 'SKIP') for v in checks.values()), \
        {k: v['detail'] for k, v in checks.items() if v['status'] == 'ERROR'}
    skipped = ['Midnodes'] + ([] if torch.cuda.is_available() else ['GPU'])   # need --midnode-corpus / a GPU
    for c in skipped:
        assert checks[c]['status'] == 'SKIP' and checks[c]['detail']['reason']
    assert checks['Calibr.']['blocking'] and rep.with_suffix('.md').is_file()


@pytest.mark.unit
def test_acceptance_gates_read_the_frozen_holdout(hii_artifact):
    from galapy.spectroscopy.hii_emulator import HIIEmulator
    from galapy.spectroscopy.utils.validate_emulator import acceptance_gates
    stem, _corpus = hii_artifact
    e = HIIEmulator().load(stem)
    rep = acceptance_gates(e, f'{stem}_gate_set_frozen.npz', base=e)
    assert set(rep['checks']) == {'check calibration', 'check no_degradation'}
    assert rep['checks']['check no_degradation']['pass'] and rep['n_points'] > 0

@pytest.mark.unit
def test_tau_transform_round_trip_keeps_T_in_0_1():
    import torch
    from galapy.spectroscopy.svgp import target_transform, decode
    tr = {'kind': 'tau', 'tau_min': 1e-3, 'tau_max': np.log(1e5)}
    T = np.array([[0.0, 1e-7, 0.3, 0.999, 1.0, 1.0006]])
    tau = target_transform(tr, T)
    assert np.all(tau >= 1e-3) and np.all(tau <= np.log(1e5))
    prep = {'pca': None, 'ratio': None, 'mu': np.zeros(6), 'sd': np.ones(6), 'transform': tr}
    y, s = decode(prep, torch.as_tensor(np.log10(tau).T), torch.full((6, 1), 1e-4))
    np.testing.assert_allclose(y, np.exp(-tau), rtol=1e-12)
    assert np.all(y > 0) and np.all(y <= 1) and np.all(s >= 0)


@pytest.mark.unit
def test_constraints_close_the_energy_budget_and_bound_T_and_grain(monkeypatch):
    import galapy.internal.emulators as EMUL
    monkeypatch.setattr(EMUL, 'ENERGY_RESCALE', True)
    from galapy.spectroscopy.hii_emulator import incident_tables, apply_constraints
    rng = np.random.default_rng(1)
    wave = np.geomspace(1.0, 1e7, 300)[rng.permutation(300)]             # unsorted, as the CLOUDY grid
    wi = np.geomspace(10.0, 1e6, 120)
    taus, Zs = np.log10([1e6, 5e6]), np.array([1e-3, 4e-3, 2e-2])
    sed = np.exp(-((np.log(wi)[None, None, :] - np.log(3e3)) ** 2)) * (1 + np.arange(6).reshape(2, 3, 1))
    qn = {'log_tau': taus, 'Z': Zs, 'sed_inc': sed, 'wave_inc': wi}
    tabs = incident_tables(qn, wave)
    n = 40
    keys = ['tau_SSP', 'Z_star']
    th = np.column_stack([rng.choice(taus, n), np.log10(rng.uniform(1e-3, 2e-2, n))])
    T = np.exp(-rng.uniform(0.2, 2.0, (n, 1)) * (1.0 + (wave[None, :] < 912)))   # absorbs where the SED is
    C = rng.uniform(0.5, 2.0, (n, 300)) * 1e-2
    G = 1.5 * C * (wave[None, :] > 1e4)                                     # above cont on purpose
    a, q = rng.uniform(0.01, 0.6, n), rng.uniform(0.3, 1.2, n)
    out = {'trans': (T, 0.01 * T), 'cont': (C, 0.1 * C), 'grain': (G, 0.1 * G),
           'energy': (np.column_stack([a / (1 - a), q]), np.full((n, 2), 0.01))}
    apply_constraints(tabs, qn, keys, th, out)
    Tn, Cn, Gn = out['trans'][0], out['cont'][0], out['grain'][0]
    assert np.all(Tn > 0) and np.all(Tn <= 1) and np.all(Gn <= Cn)
    # closure of the Energy check (validate_emulator.energy) with the interpolated SED of each row
    o = np.argsort(wave)
    lw, lwi = np.log(wave[o]), np.log(wi)
    it = np.abs(th[:, 0][:, None] - taus[None]).argmin(1)
    Z = 10 ** th[:, 1]
    hi = np.clip(np.searchsorted(Zs, Z), 1, 2)
    w = (Z - Zs[hi - 1]) / (Zs[hi] - Zs[hi - 1])
    L = (1 - w)[:, None] * sed[it, hi - 1] + w[:, None] * sed[it, hi]
    trap = getattr(np, 'trapezoid', None) or np.trapz
    E = trap(L, lwi, axis=1)
    Ti = np.array([np.interp(lwi, lw, t[o]) for t in Tn])
    Ci = np.array([np.interp(lwi, lw, c[o], left=0, right=0) for c in Cn])
    np.testing.assert_allclose(1 - trap(Ti * L, lwi, axis=1) / E, a, rtol=1e-8)
    np.testing.assert_allclose(trap(Ci, lwi, axis=1) / E, q * a, rtol=1e-8)
    np.testing.assert_allclose(out['energy'][0], np.column_stack([a, q * a]), rtol=1e-12)
    out2 = {'trans': (T, None)}
    assert apply_constraints(None, qn, keys, th, out2)['trans'][0] is T
    monkeypatch.setattr(EMUL, 'ENERGY_RESCALE', False)
    out3 = {'trans': (T, 0.01 * T), 'cont': (C, 0.1 * C), 'grain': (G, 0.1 * G),
            'energy': (np.column_stack([a / (1 - a), q]), np.full((n, 2), 0.01))}
    apply_constraints(tabs, qn, keys, th, out3)
    np.testing.assert_array_equal(out3['trans'][0], T)
    np.testing.assert_array_equal(out3['cont'][0], C)
    np.testing.assert_array_equal(out3['grain'][0], np.minimum(G, C))
    np.testing.assert_allclose(out3['energy'][0], np.column_stack([a, q * a]), rtol=1e-12)

@pytest.mark.unit
def test_lyc_floor_cells_go_to_zero_and_grain_stays_below_cont():
    import galapy.internal.emulators as EMUL
    from galapy.spectroscopy.hii_emulator import zero_lyc_floor
    wave = np.array([100.0, 500.0, 900.0, 1000.0, 5000.0, 1e5])
    fl = EMUL.REL_FLOOR_LYC['cont']
    peak = 1e30
    C = np.array([[1.0 * fl, 3.0 * fl, 1.2 * fl, 0.5 * fl, 1.0, 1e-3]]) * peak   # peak at 5000 A
    G = np.array([[1e-3, 1e-3, 1e-3, 1e-3, 0.5, 1e-3]]) * peak
    out = {'cont': (C, 0.1 * C), 'grain': (G, 0.1 * G)}
    zero_lyc_floor(wave, out)
    Cn, Gn = out['cont'][0], out['grain'][0]
    # below 912 A: floor-level cells -> 0, the cell at 3x floor kept; above 912 A nothing changes (even below floor)
    np.testing.assert_array_equal(Cn[0], [0.0, 3.0 * fl * peak, 0.0, 0.5 * fl * peak, peak, 1e-3 * peak])
    assert np.all(Gn <= Cn) and Gn[0, 4] == 0.5 * peak
    np.testing.assert_array_equal(out['cont'][1], 0.1 * C)

@pytest.mark.unit
def test_lines_at_the_ratio_floor_are_flagged():
    import galapy.internal.emulators as EMUL
    from galapy.spectroscopy.hii_emulator import lines_below_floor
    prep = {'ratio': {'ref': 1, 'floor': 1e-5}}
    hb = 1e30
    L = np.array([[2.0, 1.0, 1.0e-5, 1.4e-5, 2.0e-5, 1e-3],
                  [0.5, 1.0, 1.0e-3, 0.9e-5, 5.0e-5, 1e-6]]) * hb
    f = lines_below_floor(prep, L)
    np.testing.assert_array_equal(f, [[False, False, True, True, False, False],
                                      [False, False, False, True, False, True]])
    assert not lines_below_floor({'ratio': None}, L).any()
    assert EMUL.LINE_FLOOR_FLAG_FACTOR == 1.5

@pytest.mark.unit
def test_pca_on_a_subset_keeps_the_statistics_of_every_row(monkeypatch):
    import galapy.internal.emulators as EMUL
    from galapy.spectroscopy.utils.hii import train_emulator_hii as T
    monkeypatch.setattr(EMUL, 'PCA_CHUNK', 7)          # several chunks, the last one partial
    rng = np.random.default_rng(1)
    Y = (10.0 ** (rng.normal(size=(60, 3)) @ rng.normal(size=(3, 12)))).astype(np.float32)
    ytil = np.log10(np.asarray(Y, np.float64))
    fit, mu, sd, k, n_fit = T.pca_rank(Y, fit_max=0)
    np.testing.assert_allclose(mu, ytil.mean(0), rtol=1e-12)
    np.testing.assert_allclose(sd, np.maximum(ytil.std(0), EMUL.SD_FLOOR), rtol=1e-12)
    assert n_fit == 60 and k <= 4
    # all rows: the scores are those of the PCA of the whole z-scored matrix, whitened
    Yz, prep = T.preprocess_targets(Y, True, k_granted=3, fit_max=0)
    z = (ytil - mu) / sd
    s = (z - z.mean(0)) @ np.asarray(prep['pca']['components']).T
    np.testing.assert_allclose(np.abs(Yz), np.abs(s / s.std(0)), atol=1e-9)
    # subset: mean/sd still on every row, the PCA on fit_max rows only, recorded
    _, mu2, sd2, _, n2 = T.pca_rank(Y, fit_max=25, seed=3)
    np.testing.assert_allclose(mu2, mu, rtol=1e-12)
    assert n2 == 25
    Yz2, prep2 = T.preprocess_targets(Y, True, k_granted=3, fit_max=25, seed=3)
    assert Yz2.shape == (60, 3) and prep2['pca']['n_fit'] == 25
    np.testing.assert_allclose(Yz2.std(0), 1.0, rtol=1e-9)


@pytest.mark.unit
def test_training_stops_when_the_estimated_memory_does_not_fit(tmp_path, monkeypatch, capsys):
    from galapy.spectroscopy.utils.hii import train_emulator_hii as T
    corpus = write_hii_corpus(tmp_path / 'c.h5')
    need = T.estimate_train_memory(1000, 9244, 850, 100, 100)
    assert need < T.estimate_train_memory(1000, 9244, 850, 100, 0) < T.estimate_train_memory(2000, 9244, 1700, 200, 0)
    avail = T.available_memory()
    assert avail is None or avail > 0
    monkeypatch.setattr(T, 'available_memory', lambda: 2 ** 20)            # 1 MiB
    monkeypatch.setattr(sys, 'argv', ['train', '--corpus', str(corpus), '--out', str(tmp_path / 'e.pt'),
                                      '--meta', str(tmp_path / 'e.json'), '--epochs', '1', '--device', 'cpu'])
    with pytest.raises(SystemExit, match='estimated peak memory'):
        T.main()
    assert not (tmp_path / 'e_cont.pt').exists()


@pytest.mark.unit
def test_sigma_x_fit_recovers_an_input_dependent_noise():
    from galapy.spectroscopy.utils.validate_emulator import fit_sigma_x, sigma_x_factor
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.5, 1.5, (20000, 3))
    true_sd = np.exp(0.8 * X[:, :1] - 0.5 * X[:, 1:2] ** 2) * np.ones((1, 2))   # (n, 2), up to x10 in sigma
    r = rng.standard_normal((20000, 2)) * true_sd                                 # residuals over sigma_GP = 1
    sx, s = fit_sigma_x(X[:10000], r[:10000], s_min=0.1)
    sd = s * np.sqrt(sigma_x_factor(sx, X[10000:]))
    rt = np.abs(r[10000:]) / sd
    assert abs(np.mean(rt <= 1) - 0.68) < 0.03 and abs(np.mean(rt <= 1.96) - 0.95) < 0.015
    assert np.median(np.abs(np.log(sd / true_sd[10000:]))) < 0.15                 # sigma within ~15%
    rc = np.abs(r[10000:]) / (np.quantile(np.abs(r[:10000]), 0.95, axis=0) / 1.96)  # constant sigma
    assert np.mean(rc <= 1) > 0.75                                                # the core is too wide


@pytest.mark.unit
def test_fast_predict_applies_the_sigma_x_factor_only_to_the_variance(hii_artifact):
    from galapy.spectroscopy.hii_emulator import HIIEmulator
    from galapy.spectroscopy.svgp import fast_predict, sigma_x_features
    e = HIIEmulator().load(hii_artifact[0])
    c = dict(e.cache['lines'])
    for k in ('var_x_coef', 'var_x_lo', 'var_x_hi', 'var_x_clip'):
        c.pop(k, None)
    x = torch.randn(5, e.x_mean.shape[0])
    m0, v0 = fast_predict(c, x)
    D = x.shape[1]
    P = sigma_x_features(x, -torch.ones(D), torch.ones(D)).shape[1]
    assert P == 1 + D + D * (D + 1) // 2
    coef = torch.zeros(v0.shape[0], P)
    coef[:, 0], coef[:, 1] = 0.5, 1.0                                             # g = 0.5 + x_1 (clipped to +-1)
    c.update({'var_x_coef': coef, 'var_x_lo': -torch.ones(D), 'var_x_hi': torch.ones(D), 'var_x_clip': (-4.6, 4.6)})
    m1, v1 = fast_predict(c, x)
    g = 0.5 + x[:, 0].clamp(-1, 1)
    torch.testing.assert_close(m1, m0)
    torch.testing.assert_close(v1, v0 * torch.exp(g)[None, :], rtol=1e-5, atol=0)

@pytest.mark.unit
def test_sigma_x_cross_validation_over_clouds_keeps_cov95_on_new_clouds():
    from galapy.spectroscopy.utils.validate_emulator import fit_sigma_x, sigma_x_factor
    rng = np.random.default_rng(1)
    D, d = 12, 4

    def clouds(nc, per):                                                          # clusters in x + cloud effect
        X = np.repeat(rng.normal(size=(nc, D)), per, 0) + 0.3 * rng.normal(size=(nc * per, D))
        lv = 0.8 * X[:, :1] + np.repeat(rng.normal(0, 1.0, (nc, d)), per, 0)
        return X, np.exp(0.5 * lv) * rng.normal(size=(nc * per, d)), np.repeat(np.arange(nc), per)

    Xc, rc, gc = clouds(40, 27)                                                   # 1080 models, 40 clouds
    Xh, rh, _ = clouds(300, 15)
    cov95 = {}
    for name, gr in (('fixed', None), ('cv', gc)):
        sx, s = fit_sigma_x(Xc, rc, quantile=0.95, s_min=0.0, groups=gr)
        cov95[name] = np.mean(np.abs(rh / (s * np.sqrt(sigma_x_factor(sx, Xh)))) <= 1.96)
    assert cov95['fixed'] < 0.88                                                  # the failure of the fixed ridge
    assert abs(cov95['cv'] - 0.95) < 0.025
    assert sx['ridge_over_n'].shape == (d,)

@pytest.mark.unit
def test_regime_blend_weights_mixture_sigma_and_lyc_splice():
    from galapy.spectroscopy.hii_emulator import regime_weight, blend_regimes, regime_of
    import galapy.internal.constants as CONST
    w = regime_weight(np.array([0.0, 0.02, 0.1, 0.5, np.sqrt(0.02 * 0.1)]), (0.02, 0.1))[:, 0]
    assert np.allclose(w[:4], [1, 1, 0, 0]) and abs(w[4] - 0.5) < 1e-12           # smoothstep, half at the log middle
    assert regime_of('trans_thin') == ('trans', 'thin') and regime_of('fuv_t') == ('fuv_t', None)
    wave = np.array([500.0, 800.0, 1200.0, 5000.0])
    lyc = wave < CONST.LyLimit
    W = np.array([[1.0], [0.0], [0.5]])
    mO, sO, mT, sT = np.full((3, 4), 0.2), np.full((3, 4), 0.01), np.full((3, 4), 0.8), np.full((3, 4), 0.02)
    dec = {'trans_opaque': (mO, sO), 'trans_thin': (mT, sT), 'cont': (np.ones((3, 4)), np.ones((3, 4))),
           'lyc_opaque': (np.full((3, 2), 3.0), np.full((3, 2), 0.1)),
           'lyc_thin': (np.full((3, 2), 3.0), np.full((3, 2), 0.1))}
    blend_regimes(dec, W, wave)
    assert set(dec) == {'trans', 'cont'}
    m, s = dec['trans']
    assert np.allclose(m[0], 0.2) and np.allclose(m[1], 0.8) and np.allclose(m[2], np.sqrt(0.2 * 0.8))
    assert np.allclose(s[2], np.sqrt(0.5 * 0.01 ** 2 + 0.5 * 0.02 ** 2 + 0.25 * 0.6 ** 2))   # includes the disagreement
    assert np.allclose(dec['cont'][0][:, lyc], 3.0) and np.allclose(dec['cont'][0][:, ~lyc], 1.0)
    assert np.allclose(dec['cont'][1][:, lyc], 0.1)
    one = {'trans_thin': (mT, sT)}                                                  # opaque skipped on the batch
    blend_regimes(one, W, wave)
    assert one['trans'][0] is mT


@pytest.fixture(scope='module')
def hii_artifact_regimes(tmp_path_factory):
    from galapy.spectroscopy.utils.hii import train_emulator_hii as T
    import galapy.internal.emulators as EMUL
    d = tmp_path_factory.mktemp('hii_reg')
    corpus = write_hii_corpus(d / 'hii_grid.h5', reps=3, opaque_frac=0.4)
    argv = ['train', '--corpus', str(corpus), '--out', str(d / 'emu.pt'), '--meta', str(d / 'emu.json'),
            '--inducing', '8', '--inducing-lines', '8', '--epochs', '2', '--batch', '48', '--device', 'cpu',
            '--checkpoint-dir', str(d / 'ckpt')]
    old, sys.argv, old_min = sys.argv, argv, EMUL.REGIME_MIN_MODELS
    EMUL.REGIME_MIN_MODELS = 10
    try:
        T.main()
    finally:
        sys.argv, EMUL.REGIME_MIN_MODELS = old, old_min
    return d / 'emu', corpus


@pytest.mark.unit
def test_regime_experts_train_save_and_predict(hii_artifact_regimes):
    import json
    import galapy.internal.constants as CONST
    from galapy.spectroscopy.hii_emulator import HIIEmulator, regime_weight
    from galapy.spectroscopy.svgp import fast_predict, decode
    stem, corpus = hii_artifact_regimes
    files = {p: (stem.parent / f'emu_{p}.pt').is_file() for p in ('trans', 'trans_opaque', 'trans_thin', 'lyc')}
    assert files == {'trans': False, 'trans_opaque': True, 'trans_thin': True, 'lyc': True}
    assert (stem.parent / 'ckpt').is_dir() and not list((stem.parent / 'ckpt').iterdir())
    meta = json.loads((stem.parent / 'emu.json').read_text())
    assert meta['regimes']['blend'] == [0.02, 0.1]
    assert meta['epochs_per_family']['trans_opaque'] > meta['epochs_per_family']['trans_thin']   # equal steps
    e = HIIEmulator().load(stem)
    assert {'trans_opaque', 'trans_thin', 'lyc_opaque', 'lyc_thin'} <= set(e.cache) and 'trans' not in e.cache
    th = np.tile(np.array([-2.5, 2.0, -0.5, 3.0, 0.5, 0.0, 0.5, np.log10(2e6), np.log10(4e-3)]), (3, 1))
    th[:, 5] = [0.01, 0.045, 0.6]                                                  # opaque, band, thin
    out = e.predict_batch(th)
    T, sT = out['transmission']
    assert T.shape == (3, 60) and np.all((T > 0) & (T <= 1)) and np.all(sT >= 0)
    # the blend of the experts decoded one by one
    x = e.standardize(th)
    with torch.no_grad():
        dO = decode(e.prep['trans_opaque'], *fast_predict(e.cache['trans_opaque'], x))[0]
        dT = decode(e.prep['trans_thin'], *fast_predict(e.cache['trans_thin'], x))[0]
        dC = decode(e.prep['cont'], *fast_predict(e.cache['cont'], x))[0]
    w = regime_weight(th[:, 5], (0.02, 0.1))
    assert np.allclose(T[0], dO[0], rtol=1e-6) and np.allclose(T[2], dT[2], rtol=1e-6)
    assert np.allclose(T[1], np.exp(w[1] * np.log(dO[1]) + (1 - w[1]) * np.log(dT[1])), rtol=1e-6)
    # continuum above 912 A from the single cont family (grain <= cont does not touch it)
    lyc = np.asarray(e.wave_grid) < CONST.LyLimit
    assert np.allclose(out['continuum_nebular'][0][:, ~lyc], dC[:, ~lyc], rtol=1e-6)
    # a batch fully thin does not evaluate the opaque experts and gives the same thin rows
    out_t = e.predict_batch(th[2:])
    assert np.allclose(out_t['transmission'][0], T[2:], rtol=1e-6)
    assert list(e._expert_rows('trans_thin', w)) == [1, 2] and e._expert_rows('cont', w) is None
    e._expert_rows = lambda name, w: None
    out_all = e.predict_batch(th)
    for k in ('transmission', 'continuum_nebular'):
        for u, v in zip(out[k], out_all[k]):
            assert np.allclose(u, v, rtol=1e-5)


@pytest.mark.unit
def test_regime_experts_validate_context(hii_artifact_regimes):
    import argparse
    from galapy.spectroscopy.utils import validate_emulator as V
    stem, corpus = hii_artifact_regimes
    hold = V.read_holdout(f'{stem}_holdout.json')
    ns = argparse.Namespace(sector='hii', emulator=str(stem), meta=f'{stem.parent}/emu.json',
                            holdout=f'{stem}_holdout.json', corpus=str(corpus), max_points=0, seed=0, n_ks=200,
                            noise_diagnostic=False, midnode_corpus=None)
    ctx = V.Context(ns)
    assert len(ctx.keys) == len(hold['keys'])
    T, sT = ctx.phys('trans')
    C, _ = ctx.phys('cont')
    assert T.shape == ctx.Y['trans'].shape and C.shape == ctx.Y['cont'].shape
    for f in ('trans_opaque', 'trans_thin', 'lyc_opaque', 'lyc_thin'):                # targets of the base family
        assert ctx.Yz[f].shape == ctx.mu[f].shape
    st, _ = V.calibration(ctx)                                                      # rows of each regime only
    assert st in (V.PASS, V.FAIL, V.SKIP)

@pytest.mark.unit
def test_train_family_checkpoint_resume(tmp_path):
    from galapy.spectroscopy.utils.hii.train_emulator_hii import train_family
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    Y = np.stack([np.sin(X[:, 0]), X[:, 1] * X[:, 2]], 1).astype(np.float32)
    kw = dict(M=16, epochs=4, batch=50, lr=1e-2, device='cpu', seed=1, eval_every=2, select=(X[:40], Y[:40]))
    ref = train_family(X, Y, **kw)[0].state_dict()

    class Kill:                      # the job dies at the walltime during epoch 3 (after 2 checkpoints)
        n = 0
        def update(self, c):
            self.n += 1
            if self.n == 3:
                raise KeyboardInterrupt
    ck = str(tmp_path / 'lines.pt')
    with pytest.raises(KeyboardInterrupt):
        train_family(X, Y, progress=Kill(), ckpt=ck, **kw)
    assert torch.load(ck, weights_only=False)['epoch'] == 2
    m, lik, hist, ep_t, te_hist, best = train_family(X, Y, ckpt=ck, **kw)
    assert len(hist) == 4 and torch.load(ck, weights_only=False)['epoch'] == 4
    for k, v in ref.items():
        assert torch.allclose(v, m.state_dict()[k], rtol=1e-5, atol=1e-6), k
    # finished: nothing trained again (no progress update), same model
    class NoUpdate:
        def update(self, c):
            assert c == 0 or c == 4.0 * 1.0, c
    m2 = train_family(X, Y, ckpt=ck, progress=NoUpdate(), **kw)[0]
    assert all(torch.equal(v, m2.state_dict()[k]) for k, v in m.state_dict().items())
    # other targets: checkpoint ignored, trained from the start
    m3, _, hist3, *_ = train_family(X, 2 * Y, ckpt=ck, **kw)
    assert len(hist3) == 4
