# CLOUDY Installation and Setup

CloudIA requires the CLOUDY astrophysical simulation software (specifically version `C25.00`) to model nebular emission. 
To facilitate this, the utility implements the module `utils/install_cloudy.py` and `utils/__init__.py`, exposing both 
a Command Line Interface (CLI) and a Python API to automatically check, download, compile, and configure the CLOUDY 
environment.

## Command Line Interface (CLI)

The module exposes the CLI entrypoint `galapy-install-cloudy` to check the installation status or build CLOUDY from source.

### Usage

```text
usage: galapy-install-cloudy [-h] [--check] [--prefix DIR] [--jobs N]
```

### Options

| Flag | Argument | Description                                                                       |
| :--- | :--- |:----------------------------------------------------------------------------------|
| `-h`, `--help` | None | Show help message and exit.                                                       |
| `--check`, `-c` | None | Search for an existing CLOUDY installation and report it. Exit status `0` if CLOUDY is found and its banner reports `C25.00`, `1` otherwise. |
| `--prefix`, `-p` | `DIR` | Directory prefix where CLOUDY will be compiled. <br> Default: `~/.galapy/cloudy`. |
| `--jobs`, `-j` | `N` | Parallel compilation threads (`make -jN`). <br> Default: `os.cpu_count()`.        |

### System Prerequisites

To compile CLOUDY from source, your system must have the following tools available on your system `PATH`:

- **GNU Make** (`make`)
- **C++ Compiler** (`g++`, `c++`, or `clang++` supporting C++11)

---

## Environment Configuration

GalaPy/CloudIA communicates with CLOUDY via two environment variables:

1. **`CLOUDY_EXE`**: The absolute path to the compiled `cloudy.exe` binary.
2. **`CLOUDY_DATA_PATH`**: The search path of CLOUDY: the current directory `.` followed by the CLOUDY `data`
   directory containing atomic data and opacities.

The executable is located by `detect_cloudy`, which tries in order:

1. `$CLOUDY_EXE`;
2. a `cloudy` executable on `$PATH`;
3. `$CLOUDY_DATA_PATH/../source/cloudy.exe`, i.e. the binary next to the last `data` directory of the search path.

When set, `CLOUDY_DATA_PATH` **replaces** the default search path of CLOUDY (`+`, i.e. `.` followed by the
`data` directory of the build), so the current directory is searched only if it is listed explicitly. The
installer therefore writes the variable through `cloudy_search_path`, which puts `.` in front of the `data`
directory unless `.` or `+` is already there. The runner stages each job's SED and abundance file in its own
working directory, and CLOUDY finds them only through this entry (see [Running CLOUDY](run_hii.md#2-staging)).
If you set the variable by hand, keep it.

### Shell Persistence

During an interactive compilation, the installer will prompt to make these variables permanent. 
If approved, it appends a configuration block to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
# >>> CloudIA CLOUDY environment >>>
export CLOUDY_EXE="/home/user/.galapy/cloudy/c25.00/source/cloudy.exe"
export CLOUDY_DATA_PATH=".:/home/user/.galapy/cloudy/c25.00/data"
# <<< CloudIA CLOUDY environment <<<
```

## Python API

::: galapy.spectroscopy.utils.cloudy_search_path
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.detect_cloudy
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.cloudy_banner
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.require_cloudy
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.ensure_cloudy
    options:
      show_root_heading: true
      show_source: false

### Example Code

```python
from galapy.spectroscopy.utils import require_cloudy, CloudyNotFound

try:
    # Verify if CLOUDY is correctly installed and configured
    install_details = require_cloudy(check_version=True)
    print(f"CLOUDY is available at: {install_details.exe}")
except CloudyNotFound:
    print("CLOUDY is not installed. Run 'galapy-install-cloudy' from your terminal.")
```
