# Author: Enrico Veraldi
# declaration of package and check of CLOUDY installation

import os
import shutil
from collections import namedtuple

__all__ = ['CLOUDY', 'CloudyInstall', 'CloudyNotFound', 'detect_cloudy', 'cloudy_banner',
           'require_cloudy', 'ensure_cloudy']

# CLOUDY version used for production
CLOUDY_REQUIRED = 'C25.00'
CLOUDY_URL_TAR = 'https://data.nublado.org/cloudy_releases/c25/c25.00.tar.gz'
CLOUDY_SHA256 = '12a4fac7a29f888f56b37885df0067c92eae47a53c78060743e3c92140add5f3'

# installation
CloudyInstall = namedtuple('CloudyInstall', 'exe data_path version found')

class CloudyNotFound(RuntimeError) :
    """
    CLOUDY code not found and is necessary for the execution of the code
    """

def detect_cloudy():
    """
    Detect the Cloudy executable and associated data path.

    This function attempts to locate the Cloudy astrophysical simulation software
    by checking system environment variables and default installation paths. It
    returns a `CloudyInstall` object containing the executable path, data path,
    version, and a flag indicating whether the executable was found.

    Returns:
        CloudyInstall: An object that includes the executable path, data path,
        version, and a boolean value indicating whether the Cloudy executable
        was successfully located.
    """
    exe = os.environ.get('CLOUDY_EXE') or shutil.which('cloudy')
    data = os.environ.get('CLOUDY_DATA_PATH')
    
    if exe is None and data:
        candidate = os.path.join(os.path.dirname(data.rstrip(os.sep)), 'source', 'cloudy.exe')
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            exe = candidate
    
    return CloudyInstall(exe=exe, data_path=data, version=None, found=exe is not None)

# THE ONLY EXECUTION OF THE IMPORT
CLOUDY = detect_cloudy()

#=========================================================

def cloudy_banner(exe=None, timeout=300):
    """
    Runs the Cloudy application and retrieves the banner line of its output.

    This function attempts to execute the Cloudy software and examines its output
    to extract and return the first line containing the word "Cloudy". If the
    application is unavailable, times out, or encounters an error during execution,
    it returns None.

    Parameters:
    exe: str, optional
        The path to the Cloudy executable. If not provided, the default executable
        path specified in the CLOUDY global configuration is used.
    timeout: int, optional
        The maximum time in seconds to wait for the Cloudy executable to complete.
        Defaults to 300 seconds.

    Returns:
    str or None
        The banner line containing the word "Cloudy" from the executable's output,
        or None if the executable was unavailable, an error occurred, or no
        matching output was found.
    """
    import subprocess
    exe = exe or CLOUDY.exe
    if exe is None:
        return None
    
    try:
        out = subprocess.run([exe], input='', capture_output=True, text=True,
                             timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    
    for line in out.splitlines():
        if line.strip().startswith('Cloudy'):
            return line.strip()

    return None

def require_cloudy(check_version=True):
    """
    Checks for the presence of Cloudy software and optionally verifies its version.

    This function determines if the Cloudy executable is available in the system and, if
    `check_version` is enabled, verifies that the detected version meets the required version
    specified by `CLOUDY_REQUIRED`. If Cloudy is not found or the version check fails, an
    exception will be raised.

    Parameters:
        check_version (bool): Indicates whether to validate the detected version of Cloudy.
            Defaults to True.

    Returns:
        det: An object containing details about the Cloudy executable, including its path
        and version if available.

    Raises:
        CloudyNotFound: Raised if the Cloudy executable is not found or if the version
        check fails when `check_version` is True.
    """
    det = detect_cloudy()
    if not det.found:
        raise CloudyNotFound(f'Cloudy not found')
    if check_version:
        banner = cloudy_banner(det.exe)
        if banner and CLOUDY_REQUIRED.lstrip('C') not in banner:
            raise CloudyNotFound(f'Cloudy version {CLOUDY_REQUIRED} is requested')

        det = det._replace(version=banner)

    return det

def _reporthook(block_num, block_size, total_size):
    """
    Reports the progress of a download.

    This function is a callback typically used with data downloading functions to display the progress
    of the download to the console. It computes the percentage of the file downloaded and prints it
    along with the downloaded size. If the total size is unknown, it only displays the downloaded size.

    Args:
        block_num (int): The current block number being processed.
        block_size (int): The size of each block in bytes.
        total_size (int): The total size of the file in bytes. If the total size is unknown, this
        value may be -1.
    """
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        print(f"\r  downloaded {downloaded/1e6:7.1f} / {total_size/1e6:7.1f} MB "
              , end="", flush=True)
    else:
        print(f"\r  downloaded {downloaded/1e6:7.1f} MB ",
              end="", flush=True)


def ensure_cloudy(prefix=None, interactive=None, jobs=None, url=None, sha256=None):
    """
    Installs and configures the Cloudy software if it is not already detected on the system.

    This function ensures that the Cloudy astrophysical simulation tool is installed and available
    for use. If Cloudy is not detected, the function prompts the user for permission to download,
    install, and configure it. Once the installation is complete, environment variables required for
    using Cloudy are printed, and the Cloudy detection function is invoked again to verify the
    installation.

    Parameters:
        prefix (Optional[str]): Path to the directory where Cloudy should be installed. Defaults to
            a subdirectory in the user's home directory.
        interactive (Optional[bool]): Whether to prompt the user interactively during installation.
            Defaults to True in a TTY environment unless overridden by a CI environment.
        jobs (Optional[int]): Number of parallel jobs to use during the compilation process. Defaults
            to the number of CPU cores or 2 if the CPU count is unavailable.
        url (Optional[str]): URL to download the Cloudy source code tarball. If not provided, a
            default URL is used.
        sha256 (Optional[str]): Expected SHA256 hash of the downloaded tarball. If not provided,
            integrity verification using SHA256 is skipped.

    Raises:
        CloudyNotFound: If Cloudy is not found, the installation is aborted by the user, or the
            SHA256 integrity verification fails.

    Returns:
        CloudyDetection: A detection result object indicating the outcome of the Cloudy installation
            and configuration process.
    """
    import hashlib as _hl
    import platform
    import subprocess
    import sys
    import tarfile
    import tempfile
    import urllib.request

    det = detect_cloudy()
    if det.found:
        return det

    if interactive is None:
        interactive = sys.stdin.isatty() and not os.environ.get('CI')
    if not interactive:
       raise CloudyNotFound(f'Cloudy not found and not interactive, manual installation required: '
                            f'python -m galapy.spectroscopy.utils --install-cloudy')

    system = platform.system()
    machine = platform.machine()

    # --- make -------------------------------------------------------------
    make_bin = shutil.which('make')
    if make_bin is None:
        if system == 'Darwin':
            hint = (
                "Install the Xcode Command Line Tools with:\n"
                "    xcode-select --install\n"
                "or via Homebrew:\n"
                "    brew install make"
            )
        elif system == 'Windows':
            hint = (
                "Native Windows (MSVC/nmake) is not supported by this build path: Cloudy's "
                "Makefile assumes a GCC/Clang-compatible, POSIX-style toolchain. Use one of:\n"
                "    MSYS2  : install from https://www.msys2.org/, then from the MSYS2 shell:\n"
                "             pacman -S make mingw-w64-x86_64-gcc\n"
                "             and ensure that MSYS2/MinGW's bin directory is on PATH\n"
                "    WSL    : run this installer from inside a WSL Linux distribution instead "
                "(it will then be detected as system == 'Linux')"
            )
        else:
            hint = (
                "Install it via your system package manager, e.g.:\n"
                "    Debian/Ubuntu : sudo apt install build-essential\n"
                "    RHEL/CentOS   : sudo yum groupinstall 'Development Tools'\n"
                "    conda         : conda install -c conda-forge make"
            )
        raise CloudyNotFound(
            f"'make' not found in PATH. Cloudy requires GNU Make to be compiled from source.\n"
            f"{hint}\n"
            f"Then re-run: python -m galapy.spectroscopy.utils --install-cloudy"
        )

    # --- C++ compiler -------------------------------------------------------
    cxx_bin = shutil.which('g++') or shutil.which('c++') or shutil.which('clang++')
    if cxx_bin is None:
        if system == 'Darwin':
            hint = (
                "Install the Xcode Command Line Tools with:\n"
                "    xcode-select --install\n"
                "This provides Apple Clang (clang++), used as the default C++ compiler on macOS."
            )
        elif system == 'Windows':
            hint = (
                "No GCC/Clang-compatible C++ compiler found. MSVC (cl.exe) is not supported by "
                "this build path (incompatible flag syntax and build system). Install a "
                "GCC toolchain via MSYS2:\n"
                "    pacman -S mingw-w64-x86_64-gcc\n"
                "and ensure it is on PATH, or run this installer from inside WSL instead."
            )
        else:
            hint = (
                "    Debian/Ubuntu : sudo apt install g++\n"
                "    conda         : conda install -c conda-forge gxx_linux-64"
            )
        raise CloudyNotFound(
            f"No C++ compiler (g++/c++/clang++) found in PATH. Cloudy is written in C++ and "
            f"requires a C++11-compatible, GCC/Clang-compatible compiler to be built.\n{hint}\n"
            f"Then re-run: python -m galapy.spectroscopy.utils --install-cloudy"
        )

    if system == 'Windows':
        print(
            "NOTE: building on Windows via a detected GCC/Clang-compatible toolchain "
            "(MSYS2/MinGW or similar) found in PATH. This path has not been independently "
            "verified against Cloudy's official build instructions for Windows -- proceed "
            "with awareness that this is not a confirmed-supported platform for this "
            "installer."
        )
    if system == 'Darwin':
        print(
            "NOTE: on macOS, 'c++'/'g++' typically resolve to Apple Clang rather than GCC. "
            "The OPT flags below are GCC-style; Apple Clang generally accepts the same "
            "syntax, but this has not been independently verified against the "
            "Huang-CL/cloudy mirror build instructions for macOS."
        )

    prefix = os.path.abspath(prefix or os.path.expanduser('~/.galapy/cloudy'))
    url = url or CLOUDY_URL_TAR
    sha256 = CLOUDY_SHA256 if sha256 is None else sha256
    jobs = jobs or (os.cpu_count() or 2)
    print(f"Cloudy not found, starting installation:\n"
          f"    source   = {url}\n"
          f"    prefix   = {prefix}\n"
          f"    platform = {system} / {machine}\n"
          f"    make     = {make_bin}\n"
          f"    c++      = {cxx_bin}\n"
          f"    compile  = make -j {jobs}\n")
    
    if input("Starting installation? [y/N] ").strip().lower() not in ('y', 'yes', 's', 'si'):
        raise CloudyNotFound("installation aborted by user")

    os.makedirs(prefix, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tar = os.path.join(tmpdir, 'cloudy.tar.gz')
        urllib.request.urlretrieve(url, tar,reporthook=_reporthook)
        print()

        with open(tar, 'rb') as fh:
            digest = _hl.sha256(fh.read()).hexdigest()
        if sha256 and digest != sha256:
            raise CloudyNotFound(f"SHA256 mismatch: expected {sha256}, got {digest}")
        if not sha256:
            print(f"WARNING! SHA256 hash not provided, proceeding with installation")
        with tarfile.open(tar) as tf:
            tf.extractall(prefix)

    opt_flags = (
        'OPT=-O3 -ftrapping-math -fno-math-errno '
        '-fasynchronous-unwind-tables -Wno-deprecated-declarations'
    )
    root = next(os.path.join(prefix, d) for d in sorted(os.listdir(prefix))
                if os.path.isdir(os.path.join(prefix, d, 'source')))
    source_dir = os.path.join(root, 'source')

    subprocess.check_call([make_bin, 'clean'], cwd=source_dir)

    try:
        subprocess.check_call(
            # Explicit 'cloudy.exe' target, matching the already-ratified
            # build recipe in the CloudIA spec (v58.md, Dockerfile Stage 2):
            # bare 'make' would also build 'data' and 'vh128sum.exe', which
            # are out of scope for this installer and not needed by GalaPy.
            [make_bin, 'cloudy.exe', f'-j{jobs}', opt_flags],
            cwd=source_dir
        )
    except subprocess.CalledProcessError as e:
        raise CloudyNotFound(
            f"Cloudy compilation failed (make exit code {e.returncode}). "
            f"Sources extracted at: {source_dir}. "
            f"Inspect the make output above for the specific cause."
        ) from e

    exe = os.path.join(source_dir, 'cloudy.exe')
    data = os.path.join(root, 'data')

    exe = os.path.join(root, 'source', 'cloudy.exe')
    data = os.path.join(root, 'data')
    print("\n Done, add to the environment the following env variables:\n"
          f"export CLOUDY_EXE={exe}\n"
          f"export CLOUDY_DATA_PATH={data}")

    os.environ.setdefault('CLOUDY_EXE', exe)
    os.environ.setdefault('CLOUDY_DATA_PATH', data)
    return detect_cloudy()

    




