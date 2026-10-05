# Author: Enrico Veraldi
# test the entire process that brings a run from the CLOUDY deck to the fragment
# so test the run_core.py, paerse_one_*.py and merge_grid.p

import json
import os
import pathlib
import shutil
import stat
import sys
import h5py
import numpy as np
import pytest

from galapy.spectroscopy.utils import merge_grid
from galapy.spectroscopy.utils import run_core as rc
from galapy.spectroscopy.utils.extract_spectra import write_cloudy_sed
from galapy.spectroscopy.utils.hii import parse_one_hii as phii
from galapy.spectroscopy.utils.hii.run_hii import main as run_hii_main
from galapy.internal.constants import (Units_Legend,Root_Units_HII, Points_Units_HII)

#=============================== Sectors registers
# the free parameters per sector and some defaults to test purposes
SPEC_KEYS = {'hii': ('logU', 'lognH_HII', 'z_CMB', 'log_zeta_O', 'xi_d', 'f_esc_target', 'F_star')}
# synthetic SSP node, written as galapy-sed-cloudy-extract writes it (the parser stores it)
SED_LAMBDA = np.logspace(1.5, 6, 40)
SED_WRITERS = {'hii': lambda p: write_cloudy_sed(SED_LAMBDA, 1e-3 * (SED_LAMBDA / 1e3) ** -2.2, p)}
NODE = {'tau_SSP': 6.5, 'Z_star': 0.02}

#=============================== CLOUDY TEST (fake CLOUDY for test purposes)
CLOUDY_TEST = r'''#!@PYTHON@
import os, sys, time
import numpy as np
BANNER = os.environ.get('TEST_CLOUDY_BANNER', 'Cloudy 25.00')
if len(sys.argv) == 1:
    sys.stdout.write("                       %s\n" % BANNER)
    sys.exit(0)
prefix = sys.argv[sys.argv.index('-r') + 1]
sector, job_id = prefix.split('_', 1)
if os.environ.get('TEST_CLOUDY_LOG'):
    open(os.environ['TEST_CLOUDY_LOG'], 'a').write(prefix + '\n')
time.sleep(float(os.environ.get('TEST_CLOUDY_SLEEP', '0')))
n = 300
w = np.logspace(6, 1.5, n)
if job_id in os.environ.get('TEST_CLOUDY_BROKEN', '').split(','):
    w = w.copy(); w[[3, 7]] = w[[7, 3]]
col2 = np.where((w > 912) & (w < 5e5), 1e-3 * (w / 1e3) ** -1.2, 0.0)
col3, col4, z = 0.6 * col2, 1e-5 * (w / 1e3) ** -0.5, np.zeros(n)
np.savetxt(prefix + '.con', np.column_stack([w, col2, col3, col4, z, z, z, z, 0.1 * col4]),
           header='Cont nu\tincident\ttrans\tDiffOut\tnet trans\treflc\ttotal\treflin\toutlin')
g = 1e-7 * np.exp(-(np.log10(w) - 5.0) ** 2)
np.savetxt(prefix + '.con_grain', np.column_stack([w, 0.4 * g, 0.6 * g, g]),
           header='lambda\tgra\tsil\ttotal')
open(prefix + '.lines', 'w').write(
    'H  1  6562.80A  1.0e-3\nO  3  5006.84A  2.0e-3\nC  2  157.636m  3.0e-4\n')
rows = ''.join('%.5e\t1.000e-26\t2.000e-26\t3.000e-26\n' % d for d in np.linspace(1e15, 1e18, 12))
open(prefix + '.dusa', 'w').write('#Depth\tgra01\tsil01\ttotal\n' + rows)
for ext in {'hii': ('.grain_temp',), 'pdr': ('.pdr', '.heat', '.cool')}[sector]:
    open(prefix + ext, 'w').write('# synthetic save: parse_one does not read it\n')
if os.environ.get('TEST_CLOUDY_MISSING'):
    os.remove(prefix + os.environ['TEST_CLOUDY_MISSING'])
out = ["                       %s\n" % BANNER]
if os.environ.get('TEST_CLOUDY_CAUTION'):
    out.append(" C-Iterate to convergence did not converge in 10 iterations.\n")
rc = '0'
if job_id in os.environ.get('TEST_CLOUDY_WARNINGS', '').split(','):
    out += [" W-Radiation pressure was large, the model may be unstable.\n",
            "  C-Grain temperature above sublimation, W- not at line start.\n",
            " W-Negative optical depth in the Lya line.\n",
            " W-Radiation pressure was large, the model may be unstable.\n"]
    rc = '2'
open(prefix + '.out', 'w').writelines(out)
sys.exit(int(os.environ.get('TEST_CLOUDY_RC', rc)))
'''

# the W- lines written by CLOUDY_TEST, as warnings_from_out must return them
TEST_WARNINGS = ['W-Radiation pressure was large, the model may be unstable.',
                 'W-Negative optical depth in the Lya line.']

TEST_DECK = ('title cloudia test\n'
        'table SED "ssp_003_02.sed"\n'
        'save last line list ".lines" "{ll}" emergent absolute column\n')


@pytest.fixture
def fake_cloudy(tmp_path, monkeypatch):
    exe = tmp_path / 'cloudy'
    exe.write_text(CLOUDY_TEST.replace('@PYTHON@', sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv('CLOUDY_EXE', str(exe))
    for k in ('TEST_CLOUDY_CAUTION', 'TEST_CLOUDY_SLEEP', 'TEST_CLOUDY_BANNER',
              'TEST_CLOUDY_RC', 'TEST_CLOUDY_BROKEN', 'TEST_CLOUDY_WARNINGS',
              'TEST_CLOUDY_MISSING'):
        monkeypatch.delenv(k, raising=False)
    log = tmp_path / 'cloudy_calls.log'
    monkeypatch.setenv('TEST_CLOUDY_LOG', str(log))
    return log


def make_grid(root, sector, n=4):
    """gen_lhs and gen_input generations"""
    ids = [f'{j:05d}_003_02' for j in range(n)]
    root = pathlib.Path(root)
    runs = root / sector
    (runs / 'SED').mkdir(parents=True)
    SED_WRITERS[sector](runs / 'SED' / 'ssp_003_02.sed')
    ll_name = rc.SECTORS[sector].linelist
    for j in ids:
        (runs / f'{sector}_{j}.in').write_text(TEST_DECK.format(ll=ll_name))
    (runs / f'jobs_{sector}.txt').write_text('\n'.join(ids) + '\n')
    spec = runs / f'{sector}_grid_spec.h5'
    with h5py.File(spec, 'w') as f:
        f.create_dataset('job_id', data=np.array([i.encode() for i in ids]))
        for k in SPEC_KEYS[sector]:
            f.create_dataset(k, data=np.linspace(-3.0, -1.0, n))
        for k, v in NODE.items():
            f.create_dataset(k, data=np.full(n, v))
    ll = root / 'lines'
    ll.mkdir(exist_ok=True)
    (ll / ll_name).write_text('H  1 6562.80A\n')
    argv = ['--runs', str(runs), '--linelists', str(ll), '--spec', str(spec)]
    if sector == 'hii':
        meta = runs / 'ssp_metadata.json'
        meta.write_text(json.dumps([{**NODE, 'Qh_unit': 1e40}]))
        argv += ['--ssp-meta', str(meta)]
    return runs, ids, argv


@pytest.fixture
def hii(tmp_path):
    """sector hii"""
    return make_grid(tmp_path, 'hii')


def _manifest(path):
    doc = json.loads(pathlib.Path(path).read_text())
    return doc, {r['job_id']: r for r in doc['jobs']}


def _calls(log):
    return len(log.read_text().split()) if log.exists() else 0


def _content(path):
    """complete comtemt of a fragment (dataset + attrs), for comparison"""
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj[()]
        out[name + '@attrs'] = {k: obj.attrs[k] for k in obj.attrs}

    with h5py.File(path, 'r') as f:
        f.visititems(visit)
    return out


# ============================================================ deck->fragment
@pytest.mark.unit
def test_fragment_is_written_by_the_run_worker(hii,
                                               fake_cloudy):
    runs, ids, argv = hii
    assert run_hii_main(argv + ['--jobs', '2']) == 0
    for j in ids:
        frag, flag = runs / 'parsed' / f'{j}.h5', runs / 'work' / j / 'converged.flag'
        assert frag.is_file() and frag.stat().st_mtime >= flag.stat().st_mtime
    assert not list((runs / 'parsed').glob('*.part'))
    doc, _ = _manifest(runs / 'work' / 'run_manifest.json')
    assert doc['mode'] == 'run+parse' and doc['n_ok'] == doc['n_parse_ok'] == len(ids)
    assert doc['parse_module'] == 'galapy.spectroscopy.utils.hii.parse_one_hii'


@pytest.mark.unit
def test_non_converged_is_parsed_and_marked(hii,
                                            fake_cloudy, monkeypatch):
    monkeypatch.setenv('TEST_CLOUDY_CAUTION', '1')
    runs, ids, argv = hii
    assert run_hii_main(argv + ['--job-id', ids[0]]) == 0
    assert (runs / 'work' / ids[0] / 'NOT_CONVERGED').is_file()
    with h5py.File(runs / 'parsed' / f'{ids[0]}.h5', 'r') as f:
        assert bool(f[f'grid_point_{ids[0]}'].attrs['not_converged'])
    doc, _ = _manifest(runs / 'work' / 'run_manifest.json')
    assert doc['n_not_converged'] == 1 and doc['n_parse_ok'] == 1


@pytest.mark.unit
def test_failed_run_is_never_parsed(hii, fake_cloudy, monkeypatch):
    monkeypatch.setenv('TEST_CLOUDY_RC', '1')
    runs, ids, argv = hii
    assert run_hii_main(argv + ['--job-id', ids[0]]) == 1
    wd = runs / 'work' / ids[0]
    assert all((wd / f'hii_{ids[0]}{e}').is_file() for e in rc.SECTORS['hii'].saves)
    assert (wd / 'RUN_FAILED').is_file()
    assert not list((runs / 'parsed').glob(f'{ids[0]}*'))
    _, st = _manifest(runs / 'work' / 'run_manifest.json')
    assert st[ids[0]]['status'] == 'failed' and st[ids[0]]['parse'] is None
    with pytest.raises(RuntimeError, match='RUN_FAILED'):
        phii.main([str(wd), *argv[argv.index('--spec'):], '--out', str(runs / 'manual')])


@pytest.mark.unit
def test_parse_failure_is_isolated_and_leaves_no_file(hii, fake_cloudy,
                                                      monkeypatch, capsys):
    runs, ids, argv = hii
    monkeypatch.setenv('TEST_CLOUDY_BROKEN', ids[1])
    assert run_hii_main(argv + ['--jobs', '2']) == 1
    _, st = _manifest(runs / 'work' / 'run_manifest.json')
    assert st[ids[1]]['status'] == 'ok' and st[ids[1]]['parse']['status'] == 'failed'
    assert 'parse_cloudy_con' in st[ids[1]]['parse']['error']
    assert sorted(p.name for p in (runs / 'parsed').iterdir()) == \
        [f'{j}.h5' for j in ids if j != ids[1]]
    assert f'traceback of the first parse failure ({ids[1]})' in capsys.readouterr().err


@pytest.mark.unit
def test_resume_does_the_minimum(tmp_path, fake_cloudy):
    runs, ids, argv = make_grid(tmp_path, 'hii', n=5)
    assert run_hii_main(argv) == 0 and _calls(fake_cloudy) == 5
    frags, work = runs / 'parsed', runs / 'work'
    (frags / f'{ids[1]}.h5').unlink()                                 # lost
    t = (work / ids[2] / 'converged.flag').stat().st_mtime            # fragment old then run
    os.utime(frags / f'{ids[2]}.h5', (t - 60, t - 60))
    (work / ids[3] / 'converged.flag').unlink()                       # run interrupted
    (work / ids[3] / 'TIMEOUT').write_text('timeout after 1800 s\n')
    shutil.rmtree(work / ids[4])                                      # raw deleted
    assert run_hii_main(argv + ['--resume']) == 0
    assert _calls(fake_cloudy) == 6                                   # only ids[3]
    _, st = _manifest(work / 'run_manifest.json')
    assert {j: (r['status'], r['parse']['status']) for j, r in st.items()} == {
        ids[0]: ('skipped', 'skipped'), ids[1]: ('skipped', 'ok'),
        ids[2]: ('skipped', 'ok'), ids[3]: ('ok', 'ok'), ids[4]: ('skipped', 'skipped')}
    assert not (work / ids[3] / 'TIMEOUT').exists()


@pytest.mark.unit
def test_parse_only_writes_fragments_without_cloudy(hii, fake_cloudy,
                                                    monkeypatch):
    runs, ids, argv = hii
    assert run_hii_main(argv + ['--no-parse']) == 0 and _calls(fake_cloudy) == len(ids)
    assert not (runs / 'parsed').exists()
    monkeypatch.delenv('CLOUDY_EXE')
    monkeypatch.setenv('PATH', str(runs / 'empty'))
    assert run_hii_main(argv + ['--parse-only', '--jobs', '2']) == 0
    assert sorted(p.name for p in (runs / 'parsed').glob('*.h5')) == [f'{j}.h5' for j in ids]
    doc, _ = _manifest(runs / 'parsed' / 'parse_manifest.json')
    assert doc['mode'] == 'parse-only' and doc['n_parse_ok'] == len(ids)
    assert doc['cloudy_banner'] is None
    run_doc, _ = _manifest(runs / 'work' / 'run_manifest.json')
    assert run_doc['mode'] == 'run' and '25.00' in run_doc['cloudy_banner']


@pytest.mark.unit
def test_serial_and_pool_write_identical_fragments(hii, fake_cloudy,
                                                   tmp_path):
    runs, ids, argv = hii
    f1, f2 = tmp_path / 'f1', tmp_path / 'f2'
    assert run_hii_main(argv + ['--frags', str(f1)]) == 0
    assert run_hii_main(argv + ['--frags', str(f2), '--jobs', '2']) == 0
    _, st = _manifest(runs / 'work' / 'run_manifest.json')
    assert list(st) == ids
    for j in ids:
        a, b = _content(f1 / f'{j}.h5'), _content(f2 / f'{j}.h5')
        assert a.keys() == b.keys()
        for k, v in a.items():
            if k.endswith('@attrs'):
                assert v.keys() == b[k].keys()
                assert all(np.array_equal(v[q], b[k][q]) for q in v)
            else:
                assert np.array_equal(v, b[k])


@pytest.mark.unit
def test_fragment_carries_fuv_scalars(hii, fake_cloudy):
    runs, ids, argv = hii
    assert run_hii_main(argv + ['--job-id', ids[0]]) == 0
    w, c2, c3, c4, _ = phii.parse_cloudy_con(runs / 'work' / ids[0] / f'hii_{ids[0]}.con')
    T_ref, N_ref = phii.fuv_transmittance_hii(w, c2, c3, c4)
    with h5py.File(runs / 'parsed' / f'{ids[0]}.h5', 'r') as f:
        g = f[f'grid_point_{ids[0]}']
        assert g['T_fuv_hii'][()] == pytest.approx(T_ref)
        assert g['N_fuv_hii'][()] == pytest.approx(N_ref)
        assert 0.0 < g['T_fuv_hii'][()] <= 1.0 and g['N_fuv_hii'][()] >= 0.0


#================================================== WARNING
def _warning_flag(frag, job_id):
    with h5py.File(frag, 'r') as f:
        return bool(f[f'grid_point_{job_id}'].attrs['cloudy_warnings'])


@pytest.mark.unit
def test_warnings_from_out_keeps_W_lines_once_in_order():
    out = ('                       Cloudy 25.00\n'
           ' W-first warning.\n'
           ' C-Iterate to convergence did not converge in 10 iterations.\n'
           '  C-a caution with W- inside.\n'
           '   W-second warning.\n'
           ' W-first warning.\n')
    assert rc.warnings_from_out(out) == ['W-first warning.', 'W-second warning.']
    assert rc.warnings_from_out('                       Cloudy 25.00\n') == []


@pytest.mark.unit
def test_warning_run_is_parsed_and_flagged(hii, fake_cloudy,
                                           monkeypatch, capsys):
    runs, ids, argv = hii
    monkeypatch.setenv('TEST_CLOUDY_WARNINGS', ids[1])
    assert run_hii_main(argv + ['--jobs', '2']) == 0
    work = runs / 'work'
    doc, st = _manifest(work / 'run_manifest.json')
    assert doc['n_warning'] == 1 and doc['n_ok'] == len(ids) - 1
    assert doc['n_failed'] == doc['n_parse_failed'] == 0 and doc['n_parse_ok'] == len(ids)
    r = st[ids[1]]
    assert r['status'] == 'warning' and r['returncode'] == rc.ES_WARNINGS
    assert r['converged'] is True and r['warnings'] == TEST_WARNINGS
    assert (work / ids[1] / 'WARNINGS').read_text().splitlines() == TEST_WARNINGS
    assert (work / ids[1] / 'converged.flag').read_text() == '1\n'
    assert not (work / ids[1] / 'RUN_FAILED').exists()
    for j in ids:
        assert (work / j / 'WARNINGS').exists() == (j == ids[1])
        assert ('warnings' in st[j]) == (j == ids[1])
        assert _warning_flag(runs / 'parsed' / f'{j}.h5', j) == (j == ids[1])
    err = capsys.readouterr().err
    assert f'[W] {ids[1]}' in err and 'warning=1' in err


@pytest.mark.unit
def test_exit_code_2_without_W_lines_is_still_a_warning(hii, fake_cloudy,
                                                        monkeypatch):
    runs, ids, argv = hii
    monkeypatch.setenv('TEST_CLOUDY_RC', '2')
    assert run_hii_main(argv + ['--job-id', ids[0]]) == 0
    _, st = _manifest(runs / 'work' / 'run_manifest.json')
    assert st[ids[0]]['status'] == 'warning' and st[ids[0]]['warnings'] == []
    assert (runs / 'work' / ids[0] / 'WARNINGS').read_text() == ''
    assert _warning_flag(runs / 'parsed' / f'{ids[0]}.h5', ids[0])


@pytest.mark.unit
def test_exit_code_2_with_missing_saves_is_a_failure(hii, fake_cloudy,
                                                     monkeypatch):
    runs, ids, argv = hii
    monkeypatch.setenv('TEST_CLOUDY_WARNINGS', ids[0])
    monkeypatch.setenv('TEST_CLOUDY_MISSING', '.dusa')
    assert run_hii_main(argv + ['--job-id', ids[0]]) == 1
    wd = runs / 'work' / ids[0]
    assert (wd / 'RUN_FAILED').is_file() and not (wd / 'WARNINGS').exists()
    _, st = _manifest(runs / 'work' / 'run_manifest.json')
    assert st[ids[0]]['status'] == 'failed' and st[ids[0]]['missing'] == ['.dusa']
    assert 'warnings' not in st[ids[0]] and st[ids[0]]['parse'] is None
    assert not list((runs / 'parsed').glob(f'{ids[0]}*'))


@pytest.mark.unit
def test_warning_and_non_convergence_are_independent(hii, fake_cloudy,
                                                     monkeypatch):
    runs, ids, argv = hii
    monkeypatch.setenv('TEST_CLOUDY_WARNINGS', ids[0])
    monkeypatch.setenv('TEST_CLOUDY_CAUTION', '1')
    assert run_hii_main(argv + ['--job-id', ids[0]]) == 0
    wd = runs / 'work' / ids[0]
    assert (wd / 'NOT_CONVERGED').is_file() and (wd / 'WARNINGS').is_file()
    assert (wd / 'converged.flag').read_text() == '0\n'
    doc, st = _manifest(runs / 'work' / 'run_manifest.json')
    assert st[ids[0]]['status'] == 'warning' and st[ids[0]]['converged'] is False
    assert doc['n_warning'] == doc['n_not_converged'] == doc['n_parse_ok'] == 1
    with h5py.File(runs / 'parsed' / f'{ids[0]}.h5', 'r') as f:
        a = f[f'grid_point_{ids[0]}'].attrs
        assert bool(a['not_converged']) and bool(a['cloudy_warnings'])


@pytest.mark.unit
def test_rerun_clears_stale_warnings(hii, fake_cloudy,
                                     monkeypatch):
    runs, ids, argv = hii
    monkeypatch.setenv('TEST_CLOUDY_WARNINGS', ids[0])
    assert run_hii_main(argv + ['--job-id', ids[0]]) == 0
    assert (runs / 'work' / ids[0] / 'WARNINGS').is_file()
    monkeypatch.delenv('TEST_CLOUDY_WARNINGS')
    assert run_hii_main(argv + ['--job-id', ids[0]]) == 0
    assert not (runs / 'work' / ids[0] / 'WARNINGS').exists()
    _, st = _manifest(runs / 'work' / 'run_manifest.json')
    assert st[ids[0]]['status'] == 'ok' and 'warnings' not in st[ids[0]]
    assert not _warning_flag(runs / 'parsed' / f'{ids[0]}.h5', ids[0])


@pytest.mark.unit
def test_parse_only_and_resume_keep_the_warning_flag(hii, fake_cloudy,
                                                     monkeypatch):
    runs, ids, argv = hii
    monkeypatch.setenv('TEST_CLOUDY_WARNINGS', ids[0])
    assert run_hii_main(argv + ['--job-id', ids[0], '--no-parse']) == 0
    assert _calls(fake_cloudy) == 1
    frag = runs / 'parsed' / f'{ids[0]}.h5'
    assert run_hii_main(argv + ['--job-id', ids[0], '--parse-only']) == 0
    assert _warning_flag(frag, ids[0])
    frag.unlink()
    assert run_hii_main(argv + ['--job-id', ids[0], '--resume']) == 0
    assert _calls(fake_cloudy) == 1 and _warning_flag(frag, ids[0])


@pytest.mark.unit
def test_merge_keeps_the_warning_flag(hii, fake_cloudy,
                                      monkeypatch, tmp_path):
    runs, ids, argv = hii
    monkeypatch.setenv('TEST_CLOUDY_WARNINGS', ids[2])
    assert run_hii_main(argv) == 0
    out = tmp_path / 'hii_grid.h5'
    assert merge_grid.main(['--frags', str(runs / 'parsed'), '--out', str(out)]) == 0
    with h5py.File(out, 'r') as f:
        assert {j: bool(f[f'grid_point_{j}'].attrs['cloudy_warnings']) for j in ids} == \
            {j: j == ids[2] for j in ids}


#================================================================ merge h5
@pytest.mark.unit
def test_merge_builds_corpus_from_run_fragments(hii, fake_cloudy,
                                                tmp_path):
    runs, ids, argv = hii
    assert run_hii_main(argv) == 0
    (runs / 'parsed' / 'residual.h5.part').write_bytes(b'truncated')
    out = tmp_path / 'hii_grid.h5'
    assert merge_grid.main(['--frags', str(runs / 'parsed'), '--out', str(out)]) == 0
    assert not out.with_name(out.name + '.part').exists()
    with h5py.File(out, 'r') as f:
        assert f.attrs['n_grid_points'] == len(ids)
        assert sorted(k for k in f if k.startswith('grid_point_')) == \
            [f'grid_point_{j}' for j in ids]
        assert all(d in f for d in Root_Units_HII)


@pytest.mark.unit
def test_merge_rejects_fragments_with_different_line_list(hii,
                                                          fake_cloudy, tmp_path):
    runs, ids, argv = hii
    assert run_hii_main(argv) == 0
    (runs / 'work' / ids[1] / f'hii_{ids[1]}.lines').write_text(
        'H  1  6562.80A  1.0e-3\nO  3  5006.84A  2.0e-3\nN  2  6583.45A  3.0e-4\n')
    assert run_hii_main(argv + ['--parse-only', '--job-id', ids[1]]) == 0
    out = tmp_path / 'hii_grid.h5'
    with pytest.raises(ValueError, match='line_names'):
        merge_grid.merge(runs / 'parsed', out)
    assert not out.exists() and not out.with_name(out.name + '.part').exists()


#================================================================ units
@pytest.mark.unit
def test_fragment_declares_the_units_of_every_member(hii, fake_cloudy):
    runs, ids, argv = hii
    assert run_hii_main(argv + ['--job-id', ids[0]]) == 0
    with h5py.File(runs / 'parsed' / f'{ids[0]}.h5', 'r') as f:
        g = f[f'grid_point_{ids[0]}']
        members = phii.point_members(g)
        assert set(f['units'].attrs) == members == set(f['descriptions'].attrs)
        assert {k: f['units'].attrs[k] for k in members} == {k: phii.Points_Units_HII[k][0] for k in members}
        for name, (unit, _description, extra) in phii.Root_Units_HII.items():
            attrs = dict(f[name].attrs)
            assert attrs.get('units') == unit and attrs['description']
            assert all(attrs[k] == v for k, v in extra.items())
        assert 'dust_mass_units' not in g.attrs


@pytest.mark.unit
def test_fragment_carries_vacuum_wavelengths(hii, fake_cloudy):
    runs, ids, argv = hii
    assert run_hii_main(argv + ['--job-id', ids[0]]) == 0
    with h5py.File(runs / 'parsed' / f'{ids[0]}.h5', 'r') as f:
        native = f['lines_emergent/wavelengths_rest'][:].astype(float)
        vacuum = f['lines_emergent/wavelengths_rest_vacuum'][:]
    np.testing.assert_allclose(vacuum, phii.air_to_vacuum_A(native), rtol=1e-6)
    assert vacuum[0] == pytest.approx(6564.61, abs=0.01)        # H-alpha, 6562.80 A in air


@pytest.mark.unit
def test_merge_carries_the_units(hii, fake_cloudy, tmp_path):
    runs, ids, argv = hii
    assert run_hii_main(argv) == 0
    out = tmp_path / 'hii_grid.h5'
    assert merge_grid.main(['--frags', str(runs / 'parsed'), '--out', str(out)]) == 0
    with h5py.File(runs / 'parsed' / f'{ids[0]}.h5', 'r') as fr, h5py.File(out, 'r') as f:
        for name in merge_grid.Units_Legend:
            assert dict(f[name].attrs) == dict(fr[name].attrs)
        for d in Root_Units_HII:
            assert dict(f[d].attrs) == dict(fr[d].attrs)


@pytest.mark.unit
@pytest.mark.parametrize('which', [0, 1])
def test_merge_rejects_fragments_without_the_incident_grid(hii, fake_cloudy, tmp_path, which):
    runs, ids, argv = hii
    assert run_hii_main(argv) == 0
    with h5py.File(runs / 'parsed' / f'{ids[which]}.h5', 'a') as f:
        del f['incident/wave_grid']             # a fragment of an older parser
    out = tmp_path / 'hii_grid.h5'
    with pytest.raises(ValueError, match='older parser'):
        merge_grid.merge(runs / 'parsed', out)
    assert not out.exists() and not out.with_name(out.name + '.part').exists()


@pytest.mark.unit
@pytest.mark.parametrize('which, alter', [(0, 'drop'), (1, 'drop'), (1, 'change')])
def test_merge_rejects_fragments_with_different_units(hii, fake_cloudy, tmp_path, which, alter):
    runs, ids, argv = hii
    assert run_hii_main(argv) == 0
    with h5py.File(runs / 'parsed' / f'{ids[which]}.h5', 'a') as f:
            f['units'].attrs['dust_mass_per_Msun'] = 'g g-1'
    out = tmp_path / 'hii_grid.h5'
    with pytest.raises(ValueError, match='units'):
        merge_grid.merge(runs / 'parsed', out)
    assert not out.exists() and not out.with_name(out.name + '.part').exists()


@pytest.mark.unit
def test_validate_grid_reads_the_units_of_the_corpus(hii, fake_cloudy, tmp_path):
    from galapy.spectroscopy.utils import validate_grid
    runs, ids, argv = hii
    assert run_hii_main(argv) == 0
    out = tmp_path / 'hii_grid.h5'
    assert merge_grid.main(['--frags', str(runs / 'parsed'), '--out', str(out)]) == 0
    report = tmp_path / 'report.json'
    validate_grid.main(['--sector', 'hii', '--corpus', str(out), '--report', str(report),
                        '--markdown', str(tmp_path / 'report.md')])
    checks = {c['name']: c for c in json.loads(report.read_text())['checks']}
    assert checks['units_legend']['status'] == 'PASS', checks['units_legend']['detail']


#======================================================== parse done manually
@pytest.mark.unit
def test_parse_one_refuses_unsuccessful_workdirs(hii, fake_cloudy):
    runs, ids, argv = hii
    assert run_hii_main(argv + ['--job-id', ids[0], '--no-parse']) == 0
    wd, out = runs / 'work' / ids[0], runs / 'manual'
    inputs = argv[argv.index('--spec'):]
    (wd / 'converged.flag').unlink()
    with pytest.raises(FileNotFoundError, match='DD-4'):
        phii.main([str(wd), *inputs, '--out', str(out)])
    (wd / 'converged.flag').write_text('1\n')
    (wd / 'RUN_FAILED').write_text('returncode=2 missing=[]\n')
    with pytest.raises(RuntimeError, match='RUN_FAILED'):
        phii.main([str(wd), *inputs, '--out', str(out)])
    assert not out.exists()


@pytest.mark.unit
def test_parse_one_out_is_a_file_or_a_directory(hii, fake_cloudy):
    runs, ids, argv = hii
    assert run_hii_main(argv + ['--job-id', ids[0], '--no-parse']) == 0
    wd, inputs = str(runs / 'work' / ids[0]), argv[argv.index('--spec'):]
    assert phii.main([wd, *inputs, '--out', str(runs / 'manual')]) == 0
    assert (runs / 'manual' / f'{ids[0]}.h5').is_file()
    assert phii.main([wd, *inputs, '--out', str(runs / 'x' / 'name.h5')]) == 0
    assert (runs / 'x' / 'name.h5').is_file()


@pytest.mark.unit
def test_parse_one_write_is_atomic(hii, fake_cloudy,
                                   monkeypatch):
    runs, ids, argv = hii
    assert run_hii_main(argv + ['--job-id', ids[0], '--no-parse']) == 0

    def boom(*a, **k):
        raise RuntimeError('rise exception ad half of\' writing')

    monkeypatch.setattr(phii, 'fuv_transmittance_hii', boom)
    out = runs / 'manual'
    with pytest.raises(RuntimeError, match='writing'):
        phii.main([str(runs / 'work' / ids[0]), *argv[argv.index('--spec'):], '--out', str(out)])
    assert list(out.iterdir()) == []

#======================================================== incident SED
@pytest.mark.unit
def test_fragment_carries_the_incident_sed_of_the_file(hii, fake_cloudy):
    runs, ids, argv = hii
    assert run_hii_main(argv + ['--job-id', ids[0]]) == 0
    wave, nuLnu = np.loadtxt(runs / 'work' / ids[0] / 'SED' / 'ssp_003_02.sed', comments='#',
                             usecols=(0, 1), unpack=True)
    with h5py.File(runs / 'parsed' / f'{ids[0]}.h5', 'r') as f:
        np.testing.assert_allclose(f['incident/wave_grid'][:], wave, rtol=1e-6)
        np.testing.assert_allclose(f[f'grid_point_{ids[0]}/incident/sed_per_Msun'][:], nuLnu, rtol=1e-6)


@pytest.mark.unit
def test_parse_one_needs_the_staged_sed(hii, fake_cloudy):
    runs, ids, argv = hii
    assert run_hii_main(argv + ['--job-id', ids[0], '--no-parse']) == 0
    wd, out = runs / 'work' / ids[0], runs / 'manual'
    (wd / 'SED' / 'ssp_003_02.sed').unlink()
    with pytest.raises(FileNotFoundError, match='table SED'):
        phii.main([str(wd), *argv[argv.index('--spec'):], '--out', str(out)])
    assert list(out.iterdir()) == []


@pytest.mark.unit
def test_parse_one_rejects_the_sed_of_another_node(hii, fake_cloudy):
    runs, ids, argv = hii
    assert run_hii_main(argv + ['--job-id', ids[0], '--no-parse']) == 0
    wd, inputs = str(runs / 'work' / ids[0]), argv[argv.index('--spec'):]
    meta = runs / 'ssp_metadata.json'
    meta.write_text(json.dumps([{**NODE, 'Qh_unit': 1e40, 'sed_file': 'ssp_other.sed'}]))
    with pytest.raises(ValueError, match='ssp_other.sed'):
        phii.main([wd, *inputs, '--out', str(runs / 'manual')])
    meta.write_text(json.dumps([{**NODE, 'Qh_unit': 1e40, 'sed_file': 'ssp_003_02.sed'}]))
    assert phii.main([wd, *inputs, '--out', str(runs / 'manual')]) == 0
