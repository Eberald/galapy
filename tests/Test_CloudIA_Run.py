# Author: Enrico Veraldi
# check the run of CLOUDY. so file run_*.py in HII and PDR and the run_core.py

import json
import os
import pathlib
import stat
import subprocess
import sys
import pytest

from galapy.spectroscopy.utils import CloudyNotFound
from galapy.spectroscopy.utils import run_core as rc
from galapy.spectroscopy.utils.run_core import (SECTORS, CONVERGENCE_CAUTION, converged_from_out, is_complete,
                                                parse_index_selector, plan_jobs, read_manifest, run_job,
                                                sed_referenced_by, stage_job)

# sectors under analysis
HII = SECTORS['hii']

@pytest.fixture(autouse=True)
def _cloudy_env_isolato(monkeypatch):
    """
    Fixture to automatically isolate the environment by removing specific environment variables.
    """
    for var in ('CLOUDY_EXE', 'CLOUDY_DATA_PATH', 'CLOUDIA_LINELISTS'):
        monkeypatch.delenv(var, raising=False)

# Drivers, i.e. the runner under analysis
DRIVERS = [
    'galapy.spectroscopy.utils.run_core',
    'galapy.spectroscopy.utils.hii.run_hii',
]

# a python script that fake a CLOUDY run reproducing the stuff that the run should observe
CLOUDY_TEST = r'''#!/usr/bin/env python3
import os, sys, time
BANNER = os.environ.get('TEST_CLOUDY_BANNER', 'Cloudy 25.00')           #banner get and down write
if len(sys.argv) == 1:
    sys.stdout.write("                       %s\n" % BANNER)
    sys.exit(0)
prefix = sys.argv[sys.argv.index('-p') + 1] #take element after -p
time.sleep(float(os.environ.get('TEST_CLOUDY_SLEEP', '0')))             #fake time of execution cloudy
drop = os.environ.get('TEST_CLOUDY_DROP', '')
for ext in ('.con', '.con_grain', '.lines', '.dusa', '.grain_temp'):
    if ext == drop:
        continue
    open(prefix + ext, 'w').write('# fake %s\n' % ext)                  #creation fake output, in case not dropp
out = ["                       %s\n" % BANNER]
if os.environ.get('TEST_CLOUDY_CAUTION'):
    out.append(" @@CAUTION@@\n")                                        #generation list out with also possible no conv test
open(prefix + '.out', 'w').writelines(out)
sys.exit(int(os.environ.get('TEST_CLOUDY_RC', '0')))                    #test exit code
'''
CLOUDY_TEST = CLOUDY_TEST.replace('@@CAUTION@@', CONVERGENCE_CAUTION)

# test input cloudy
TEST_INPUT = ('title cloudia test\n'
        'table SED "ssp_003_02.sed"\n'
        'ionization parameter -2.0\n'
        'save last line list ".lines" "hii_lines.dat" emergent absolute column\n')


@pytest.fixture
def test_cloudy(tmp_path, monkeypatch):
    """
    Create a temporary test CLOUDY executable for testing purposes and modify
    the environment variables relevant to the test executable.

    Parameters:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest,
            used to create the test executable.
        monkeypatch (pytest.MonkeyPatch): A pytest utility to safely patch
            and test environment-related behaviors.

    Returns:
        pathlib.Path: The path to the temporary 'cloudy' executable.
    """
    exe = tmp_path / 'cloudy'
    exe.write_text(CLOUDY_TEST)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv('CLOUDY_EXE', str(exe))
    monkeypatch.delenv('TEST_CLOUDY_CAUTION', raising=False)
    monkeypatch.delenv('TEST_CLOUDY_SLEEP', raising=False)
    monkeypatch.delenv('TEST_CLOUDY_DROP', raising=False)
    monkeypatch.delenv('TEST_CLOUDY_BANNER', raising=False)
    return exe


@pytest.fixture
def runs(tmp_path):
    """
    Set up a directory tree for testing, with input files,
    directories, and other related data.

    Returns:
        tuple: A tuple consisting of:
            - Path to the root of the directory structure created by this
              fixture.
            - List of job identifiers created.
            - Path to the directory containing line lists.

    Args:
        tmp_path (Path): Temporary directory path provided by pytest.
    """
    d = tmp_path / 'runs'
    (d / 'SED').mkdir(parents=True)
    ids = [f'j{k:03d}' for k in range(4)]
    for j in ids:
        (d / f'{HII.name}_{j}.in').write_text(TEST_INPUT)
    (d / 'SED' / 'ssp_003_02.sed').write_text('# test sed\n1.0 1.0\n2.0 2.0\n')
    (d / f'jobs_{HII.name}.txt').write_text('\n'.join(ids) + '\n')
    ll = tmp_path / 'linelists'
    ll.mkdir()
    (ll / HII.linelist).write_text('H  1 6562.80A\n')
    return d, ids, ll

# JOB SELECTION ========================================================================

@pytest.mark.unit
def test_parse_index_selector_forms():
    """
    Test function for validating the behavior of the `parse_index_selector` function.

    Parameters:
        None

    Raises:
        AssertionError: If any of the test assertions fail.

    Returns:
        None
    """
    assert parse_index_selector('0-9,20,30-32') == list(range(10)) + [20, 30, 31, 32]
    assert parse_index_selector('7') == [7]
    assert parse_index_selector(' 3 , 1 ,3 ') == [1, 3]


@pytest.mark.unit
def test_parse_index_selector_rejects_bad_ranges():
    """
    Tests the `parse_index_selector` function to ensure it rejects invalid range
    specifications.

    Raises:
        ValueError: If the range specified is reversed (e.g., '9-0') or
        otherwise invalid (e.g., '-3').
    """
    with pytest.raises(ValueError):
        parse_index_selector('9-0')
    with pytest.raises(ValueError):
        parse_index_selector('-3')


@pytest.mark.unit
def test_read_manifest_rejects_duplicates_and_empty(tmp_path):
    """
    Test the behavior of the `read_manifest` function when encountering duplicate lines and empty input.

    Parameters:
        tmp_path (Path): A temporary file path provided by the `pytest` fixture.

    Raises:
        ValueError: Raised when the manifest file contains duplicate or empty lines.
    """
    p = tmp_path / f'jobs_{HII.name}.txt'
    p.write_text('a\nb\na\n')
    with pytest.raises(ValueError):
        read_manifest(p)
    p.write_text('\n\n')
    with pytest.raises(ValueError):
        read_manifest(p)


@pytest.mark.unit
def test_plan_jobs_preserves_manifest_order():
    """
    Unit test for the `plan_jobs` function, ensuring that it preserves the order
    of the manifest while applying specified selections or filters.

    Test cases include:
    1. Selecting jobs based on index positions.
    2. Selecting jobs based on specific job IDs.
    3. Verifying that no selection returns the full manifest in order.

    Parameters:
        None

    Raises:
        None

    Returns:
        None
    """
    ids = ['a', 'b', 'c', 'd']
    assert plan_jobs(ids, index_sel='2,0') == ['a', 'c']
    assert plan_jobs(ids, job_ids=['d', 'b']) == ['b', 'd']
    assert plan_jobs(ids) == ids


@pytest.mark.unit
def test_plan_jobs_rejects_incoherent_selection():
    """
    Tests the plan_jobs function for various incoherent selection inputs.
    """
    ids = ['a', 'b']
    with pytest.raises(ValueError):
        plan_jobs(ids, index_sel='0', job_ids=['a'])
    with pytest.raises(IndexError):
        plan_jobs(ids, index_sel='0-5')
    with pytest.raises(KeyError):
        plan_jobs(ids, job_ids=['zz'])


# .sed FROM CLOUDY INPUT =================================================
@pytest.mark.unit
def test_sed_referenced_by_reads_the_deck():
    """
    Test on sed reading
    """
    assert sed_referenced_by(TEST_INPUT) == 'ssp_003_02.sed'
    assert sed_referenced_by('TABLE SED "sub/dir/x.sed"\n') == 'x.sed'
    assert sed_referenced_by('ionization parameter -2\n') is None


# STAGING ================================================================
@pytest.mark.unit
def test_stage_job_mounts_deck_sed_and_linelist(tmp_path, runs):
    """
    Test function that verifies the correct mounting of the deck, SED, and linelist files
    for a staged job. This test ensures that the deck file, the SED, and the linelist file are
    correctly located and created in the specified working directory during staging.

    Parameters:
    tmp_path (pathlib.Path): A temporary directory provided by pytest for test file operations.
    runs (tuple): A tuple containing a directory structure, ID list, and line list directory
        required for the test setup
    """
    d, ids, ll = runs
    wd, deck = stage_job(d, tmp_path / 'work', HII, ids[0], linelist_dir=ll)
    assert deck.is_file()
    assert (wd / 'SED' / 'ssp_003_02.sed').is_file()
    assert (wd / HII.linelist).is_file()


@pytest.mark.unit
def test_stage_job_fails_loudly_on_missing_sed(tmp_path, runs):
    """
    Marks a unit test that validates the behavior of the `stage_job` function when an expected SED file is missing.

    Parameters:
    tmp_path (pathlib.Path): A temporary directory provided by pytest for test file operations.
    runs (tuple): A tuple containing a directory structure, ID list, and line list directory
        required for the test setup.

    Raises:
    FileNotFoundError: If the required SED file is missing from the directory.
    """
    d, ids, ll = runs
    (d / 'SED' / 'ssp_003_02.sed').unlink()
    with pytest.raises(FileNotFoundError, match='SED'):
        stage_job(d, tmp_path / 'work', HII, ids[0], linelist_dir=ll)


@pytest.mark.unit
def test_stage_job_fails_loudly_on_missing_linelist(tmp_path, runs):
    """
    Marks a unit test that validates the behavior of the `stage_job` function when an expected line list file is missing.

    Parameters:
    tmp_path (pathlib.Path): A temporary directory provided by pytest for test file operations.
    runs (tuple): A tuple containing a directory structure, ID list, and line list directory
        required for the test setup.

    Raises:
    FileNotFoundError: If the required line list file is missing from the directory.
    """
    d, ids, ll = runs
    (ll / HII.linelist).unlink()
    with pytest.raises(FileNotFoundError, match='line list'):
        stage_job(d, tmp_path / 'work', HII, ids[0], linelist_dir=ll)


# CONVERGENCE / RESUME =========================================================
@pytest.mark.unit
def test_converged_from_out_detects_the_caution():
    """
    Test if the 'converged_from_out' function detects the convergence caution flag correctly.
    """
    assert converged_from_out('tutto bene\n')
    assert not converged_from_out(' ' + CONVERGENCE_CAUTION + '\n')


@pytest.mark.unit
def test_is_complete_requires_flag_and_all_saves(tmp_path):
    """
    Test whether the `is_complete` function correctly identifies completion state based
    on the existence of required save files and the presence of a "converged.flag" file.

    Args:
        tmp_path (Path): A temporary directory path provided by pytest.

    Raises:
        AssertionError: If the `is_complete` function does not correctly identify the
        completion state in cases where required files exist, or when files are deleted.

    """
    wd = tmp_path / 'j000'
    wd.mkdir()
    for ext in HII.saves:
        (wd / f'{HII.name}_j000{ext}').write_text('x')
    assert not is_complete(wd, HII, 'j000')
    (wd / 'converged.flag').write_text('1\n')
    assert is_complete(wd, HII, 'j000')
    (wd / f'{HII.name}_j000.dusa').unlink()
    assert not is_complete(wd, HII, 'j000')


# END-TO-END ==========================================================================
@pytest.mark.unit
def test_run_job_end_to_end(tmp_path, runs, test_cloudy):
    """
    Tests the `run_job` function using a simulated end-to-end scenario.

    This test verifies the proper functionality of the `run_job` function by
    simulating a realistic job run.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest.
        runs (tuple): Contains mock data, including directories, IDs, and other
            test-specific details required for job execution.
        test_cloudy (str): A mock representation of the cloudy simulator script.

    Raises:
        AssertionError: If any conditions verifying the job's success or output
            consistency are not met.
    """
    d, ids, ll = runs
    r = run_job(ids[0], d, tmp_path / 'work', 'hii', str(test_cloudy), 60,
                linelist_dir=ll)
    assert r['status'] == 'ok' and r['converged'] is True
    wd = pathlib.Path(r['workdir'])
    assert (wd / 'converged.flag').read_text().strip() == '1'
    assert not (wd / 'NOT_CONVERGED').exists()
    assert all((wd / f'{HII.name}_{ids[0]}{e}').is_file() for e in HII.saves)


@pytest.mark.unit
def test_non_convergence_is_marked_not_failed(tmp_path, runs, test_cloudy, monkeypatch):
    """T-1.6 tollera una frazione di non convergenti: si MARCA, non si scarta --
    e soprattutto non si spaccia per riuscito."""
    monkeypatch.setenv('TEST_CLOUDY_CAUTION', '1')
    d, ids, ll = runs
    r = run_job(ids[0], d, tmp_path / 'work', 'hii', str(test_cloudy), 60,
                linelist_dir=ll)
    wd = pathlib.Path(r['workdir'])
    assert r['status'] == 'ok'  # NON e' un fallimento
    assert r['converged'] is False
    assert (wd / 'NOT_CONVERGED').is_file()
    assert (wd / 'converged.flag').read_text().strip() == '0'


@pytest.mark.unit
def test_missing_save_is_a_failure(tmp_path, runs, test_cloudy, monkeypatch):
    """
    Test case for verifying that a missing required save file results in a failure status.

    Parameters:
    tmp_path (pathlib.Path): A temporary directory path provided by pytest.
    runs (Tuple): A tuple containing the test setup data, including input files,
        job IDs, and other configuration-specific data.
    test_cloudy (str): The test configuration or executable path for the "Cloudy"
        software simulation.
    monkeypatch (pytest.MonkeyPatch): Pytest utility for modifying and mocking
        environment variables and attributes during the test.

    Raises:
    AssertionError: If the job's status is not 'failed' or if the missing file
        list does not contain the expected entry.
    """
    monkeypatch.setenv('TEST_CLOUDY_DROP', '.dusa')
    d, ids, ll = runs
    r = run_job(ids[0], d, tmp_path / 'work', 'hii', str(test_cloudy), 60,
                linelist_dir=ll)
    assert r['status'] == 'failed' and r['missing'] == ['.dusa']


@pytest.mark.unit
def test_resume_skips_completed_jobs(tmp_path, runs, test_cloudy):
    """
    This function is a unit test designed to ensure that completed jobs are skipped when the resume flag
    is used in subsequent calls to the run_job function.

    Parameters:
    tmp_path (pathlib.Path): A temporary directory path used for the test.
    runs (tuple): A tuple containing test data, identifiers, and additional parameters used for the configurations.
    test_cloudy (Any): A test-specific parameter representing a configuration file or state for cloudy simulation.

    Raises:
    AssertionError: Raised if the job is not skipped properly or its runtime is not reported as 0.0 when resumed.
    """
    d, ids, ll = runs
    work = tmp_path / 'work'
    run_job(ids[0], d, work, 'hii', str(test_cloudy), 60, linelist_dir=ll)
    again = run_job(ids[0], d, work, 'hii', str(test_cloudy), 60,
                    linelist_dir=ll, resume=True)
    assert again['status'] == 'skipped' and again['seconds'] == 0.0


@pytest.mark.unit
def test_timeout_is_marked(tmp_path, runs, test_cloudy, monkeypatch):
    """
    Marks a test case to validate job handling under timeout conditions, ensuring that a
    job is identified with a 'timeout' status and a specific timeout marker file is
    created in the designated work directory.

    Parameters:
        tmp_path (Path): A temporary directory path provided by pytest.
        runs (tuple): A tuple containing a working directory, job identifiers,
            and the linelist directory.
        test_cloudy (Path): Path to the test_cloudy executable or script.
        monkeypatch (pytest.MonkeyPatch): Utility to modify environments or
            functions during testing.

    Raises:
        AssertionError: If the job's status is not labeled as 'timeout' in the
            results dictionary, or if the corresponding 'TIMEOUT' marker file
            is not created in the job's work directory.
    """
    monkeypatch.setenv('TEST_CLOUDY_SLEEP', '2')
    d, ids, ll = runs
    r = run_job(ids[0], d, tmp_path / 'work', 'hii', str(test_cloudy), 1,
                linelist_dir=ll)
    assert r['status'] == 'timeout'
    assert (pathlib.Path(r['workdir']) / 'TIMEOUT').is_file()


@pytest.mark.unit
def test_pool_preserves_result_order(tmp_path, runs, test_cloudy):
    """
    Unit test to verify that the pool preserves the result order after processing jobs.
    This ensures that the sequence of results matches the order of the submitted job IDs.

    Parameters:
    tmp_path (Path): Temporary path fixture for creating isolated directories.
    runs (tuple): Contains a dictionary of input data, a list of expected job IDs,
        and a line list directory for the run.
    test_cloudy (str): Cloudy executable path used for the test.

    Raises:
    AssertionError: If the order of job IDs in the results does not match
        the expected job ID order.
    """
    d, ids, ll = runs
    res = rc.run_all(ids, d, tmp_path / 'work', 'hii', str(test_cloudy), 60,
                     nproc=2, linelist_dir=ll)
    assert [r['job_id'] for r in res] == ids


# REQUIRE CLOUDY CONTRACT ==================================================================
@pytest.mark.unit
def test_dry_run_does_not_require_cloudy(tmp_path, runs, monkeypatch, capsys):
    """
    Test that the dry-run functionality does not require the CLOUDY executable.

    Parameters:
    tmp_path: pytest fixture
        Represents a temporary directory that is used to modify the PATH for
        the test.
    runs: tuple
        Contains the run directory, list of identifiers, and associated data
        used for testing.
    monkeypatch: pytest fixture
        Used to modify or delete environment variables during the test.
    capsys: pytest fixture
        Captures output written to stdout and stderr for assertion purposes.
    """
    monkeypatch.delenv('CLOUDY_EXE', raising=False)
    monkeypatch.setenv('PATH', str(tmp_path / 'vuoto'))
    d, ids, ll = runs
    rcode = rc.main(['--runs', str(d), '--index', '0-1', '--dry-run'], 'hii')
    assert rcode == 0
    assert not (d / 'work').exists()
    assert ids[1] in capsys.readouterr().out


@pytest.mark.unit
def test_missing_cloudy_aborts_before_any_work(tmp_path, runs, monkeypatch, capsys):
    """
    Test function to verify that the program aborts early if the `CLOUDY_EXE` environment
    variable is missing and does not perform any tasks.

    Parameters:
    tmp_path : pathlib.Path
        Temporary directory created for the test, used to simulate file paths.
    runs : tuple
        Fixture providing preconfigured test data, including paths and identifiers for runs.
    monkeypatch : pytest.MonkeyPatch
        Utility to modify environment variables or other attributes during testing.
    capsys : pytest.CaptureFixture
        Fixture to capture output (stdout and stderr) during test execution.

    Raises:
    AssertError
        If the program does not abort as expected, fails to include specific messages in the
        error output, or if any work directory is created.
    """
    monkeypatch.delenv('CLOUDY_EXE', raising=False)
    monkeypatch.setenv('PATH', str(tmp_path / 'vuoto'))
    d, ids, ll = runs
    rcode = rc.main(['--runs', str(d), '--index', '0'], 'hii')
    err = capsys.readouterr().err
    assert rcode == 1
    assert not (d / 'work' / ids[0]).exists()


@pytest.mark.unit
def test_wrong_version_aborts(tmp_path, runs, test_cloudy, monkeypatch, capsys):
    """
    Unit test for ensuring version mismatch results in program end.

    Args:
        tmp_path (pathlib.Path): Fixture for creating temporary directory during tests.
        runs (tuple): Simulated runs data including directory, IDs, and line lists.
        test_cloudy (pytest fixture): Mocked environment simulating the "Cloudy" software.
        monkeypatch (pytest.MonkeyPatch): Fixture for dynamically modifying and controlling the
            environment during testing.
        capsys (pytest.CaptureFixture): Fixture for capturing printed output during testing.
    """
    monkeypatch.setenv('TEST_CLOUDY_BANNER', 'Cloudy 23.01')
    d, ids, ll = runs
    rcode = rc.main(['--runs', str(d), '--index', '0', '--linelists', str(ll)], 'hii')
    err = capsys.readouterr().err.lower()
    assert rcode == 1
    assert 'version' in err or 'versione' in err


@pytest.mark.unit
def test_main_writes_run_manifest_with_banner(tmp_path, runs, test_cloudy):
    """
    Test function to verify the main functionality of writing a run manifest with
    a banner using the `rc.main` function.

    Parameters:
    tmp_path (Path): Path object for a temporary directory used for the
                     working directory.
    runs (tuple): A tuple containing the directory with mock runs, list of run
                  IDs, and location of line lists.
    test_cloudy (Mock): A mock object representing the fake cloudy instance.

    Raises:
    AssertionError: When any of the assertions fail, including incorrect return
                    code, improper run manifest content, or invalid banner
                    details in the manifest.
    """
    d, ids, ll = runs
    work = tmp_path / 'work'
    rcode = rc.main(['--runs', str(d), '--work', str(work),
                     '--linelists', str(ll), '--jobs', '2'], 'hii')
    assert rcode == 0
    doc = json.loads((work / 'run_manifest.json').read_text())
    assert doc['n_ok'] == len(ids) and doc['n_failed'] == 0
    assert '25.00' in doc['cloudy_banner']
    assert doc['cloudy_required'] == 'C25.00'


@pytest.mark.unit
def test_sector_drivers_are_import_safe():
    """
    Verifies that the sector drivers listed in the DRIVERS collection are import-safe
    and contain a 'main' attribute.

    Raises
    ------
    AssertionError
        If any module in DRIVERS fails to be imported or does not contain a 'main'
        attribute.
    """
    import importlib
    for mod in DRIVERS:
        m = importlib.import_module(mod)
        assert hasattr(m, 'main')