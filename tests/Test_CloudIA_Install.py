# Author: Enrico Veraldi
# End-to-end test of the CLOUDY installation routine in
# galapy/spectroscopy/utils/__init__.py
# SLOW TEST, ONLY WHEN ENV VARIABLE CLOUDIA_SLOW=1

import os

import pytest

from galapy.spectroscopy.utils import (
    ensure_cloudy, detect_cloudy, cloudy_banner, require_cloudy, CLOUDY_REQUIRED,
)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('CLOUDIA_SLOW') != '1',
                    reason="downloads ~1 GB and compiles CLOUDY from source (~10 min);"
                           " runs only with CLOUDIA_SLOW=1")
def test_ensure_cloudy_real_download_and_compile(tmp_path, monkeypatch):
    """
    Tests the process of downloading, compiling, and validating the Cloudy software using the
    `ensure_cloudy` utility. This test ensures that the Cloudy binary and data paths are created,
    work correctly, and meet the required version specifications.

    The test only runs when the `CLOUDIA_SLOW` environment variable is explicitly set to `1`, as
    it involves downloading approximately 1 GB of data and compiling the software from source.
    It is the only test able to catch failures of CLOUDY's own build system (for instance a
    poisoned `tmp_cloudyconfig.*` left over in a non-pristine source tree), a class of error
    that is by construction invisible to any mocked unit test.

    Isolation is enforced on all three detection routes implemented by `detect_cloudy()`:
    `$CLOUDY_EXE` and `$CLOUDY_DATA_PATH` are unset for the duration of the test, and the
    presence of a `cloudy` executable on `$PATH` causes an explicit skip. Without this,
    `ensure_cloudy()` would return on its very first line and every assertion below would
    pass against a pre-existing installation, certifying nothing.

    The environment variables are restored in a `finally` block: `ensure_cloudy()` writes them
    through `os.environ.setdefault()`, a direct mutation that `monkeypatch` does not track and
    would therefore not undo on teardown.

    Attributes:
        tmp_path (Path): Temporary directory used as the installation prefix, so that the
            test never touches the real one (`~/.galapy/cloudy`).
        monkeypatch (pytest.MonkeyPatch): Fixture used to auto-confirm the interactive prompt.

    Raises:
        AssertionError: If any of the checks, such as compilation success, executable
            accessibility, binary banner correctness, or version compliance, fails.
    """
    prefix = tmp_path / 'cloudy'
    monkeypatch.setattr('builtins.input', lambda *_: 'y')

    saved_exe = os.environ.pop('CLOUDY_EXE', None)
    saved_data = os.environ.pop('CLOUDY_DATA_PATH', None)
    try:
        if detect_cloudy().found:
            pytest.skip("a 'cloudy' executable is reachable via PATH: ensure_cloudy() would "
                        "short-circuit on its first line and the test would pass without "
                        "downloading or compiling anything")

        det = ensure_cloudy(prefix=str(prefix), interactive=True)

        assert det.found is True
        assert str(prefix) in det.exe, (
            f"executable {det.exe!r} lies outside the test prefix {str(prefix)!r}: "
            "a pre-existing installation was returned instead of a fresh build")

        assert os.path.isfile(det.exe)
        assert os.access(det.exe, os.X_OK)
        assert os.path.isdir(det.data_path)

        banner = cloudy_banner(det.exe)
        assert banner is not None, "the binary did not produce a readable banner"
        assert CLOUDY_REQUIRED.lstrip('C') in banner, (
            f"unexpected version: banner={banner!r}, requested {CLOUDY_REQUIRED}")

        req = require_cloudy(check_version=True)
        assert req.found is True
        assert req.version == banner

        assert os.environ.get('CLOUDY_EXE') == det.exe
        assert os.environ.get('CLOUDY_DATA_PATH') == det.data_path
        det2 = ensure_cloudy(prefix=str(prefix), interactive=True)
        assert det2.exe == det.exe

    finally:
        if saved_exe is None:
            os.environ.pop('CLOUDY_EXE', None)
        else:
            os.environ['CLOUDY_EXE'] = saved_exe
        if saved_data is None:
            os.environ.pop('CLOUDY_DATA_PATH', None)
        else:
            os.environ['CLOUDY_DATA_PATH'] = saved_data