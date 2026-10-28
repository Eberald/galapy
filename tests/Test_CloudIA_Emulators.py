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


def write_hii_corpus(path, pumped=True, drop_flag=False):
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
        for tau in TAUS:
            for Z in ZS:
                logU, lognH = rng.uniform(-4, -1), rng.uniform(0, 4)
                g = f.create_group(f'grid_point_{j:05d}')
                g.attrs.update({'logU': logU, 'lognH_HII': lognH, 'log_zeta_O': rng.uniform(-2, 0.5),
                                'z_CMB': rng.uniform(0, 12), 'xi_d': rng.uniform(0.1, 1),
                                'f_esc_meas': rng.uniform(0, 0.9), 'F_star': rng.uniform(0, 1),
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
    np.testing.assert_allclose(var.numpy(), ref.variance.numpy(), rtol=2e-4)


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