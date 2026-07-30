# Extraction of Single Stellar Populations in CLOUDY format

CloudIA implements the module `utils/spectra.py` to manage the extraction of Simple Stellar Populations (SSP) and 
Composite Stellar Populations (CSP) from GalaPy's spectral libraries. These stellar spectra are converted into the 
standard `table SED` format required by CLOUDY simulations to model the incident radiation field.

The extraction tools are accessible both as a Command Line Interface (CLI) and programmatically through a Python API.

## Command Line Interface (CLI)

The module exposes a CLI entrypoint `galapy-sed-cloudy-extract` to directly generate the necessary `.sed` files for a 
set of stellar ages ($\tau$) and metallicities ($Z$).

### Usage

```text
usage: galapy-sed-cloudy-extract [-h] [-s SSP_LIB] -o OUT [-l TRUNCATE_LYMAN]
                                 [-t TAU [TAU ...]] [--Z Z [Z ...]]
                                 [--e E [E ...]]
```

### Options

| Flag | Argument | Description                                                                                                                                                |
| :--- | :--- |:-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-h`, `--help` | None | Show help message and exit.                                                                                                                                |
| `-s`, `--ssp-lib` | `SSP_LIB` | Name of the SSP spectral library. Default: `parsec22.NT`.                                                                                                  |
| `-o`, `--out` | `OUT` | Directory path where output files and metadata will be saved. **(Required)**                                                                               |
| `-l`, `--truncate-lyman` | `TRUNCATE_LYMAN` | Minimum wavelength threshold in Angstroms. If provided, wavelengths below this value are truncated.                                                        |
| `-t`, `--tau` | `TAU [TAU ...]` | List of target SSP ages in years (e.g., `--tau 1e6 2e6 5e6`). <br> Default: `[1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 7e7, 1e8]`.                                    |
| `--Z` | `Z [Z ...]` | List of target SSP metallicities (e.g., `--Z 0.0001 0.0005 0.0010`). <br> Default: `[0.0001, 0.0005, 0.0010, 0.0040, 0.0080, 0.0200]`.                     |
| `--e`, `--extrapolate` | `E [E ...]` | Flag/value to enable extrapolation.  |

### Example

To extract complete SED files for younger stars at low metallicity with extrapolation enabled:

```bash
galapy-sed-cloudy-extract -o ./ssp_seds -t 1e6 5e6 --Z 0.0001 0.0010 --e True
```

---

## Output Structure and File Formats

When running the batch extraction, the output directory will contain:

- An individual `.sed` file for each stellar age and metallicity node.
- A `metadata.json` summary catalog (only generated if `truncate_lyman` is `None`).

### 1. The CLOUDY `.sed` File Format

The resulting file is formatted to match the CLOUDY `table SED` specification:

- A comment line starting with `#` describing the columns.
- Two space-separated columns:
    - **Wavelength** in Angstroms ($\text{A}$).
    - **$\nu F_\nu$ linear** ($\lambda L_\lambda$ or $\nu L_\nu$) scaled to units of $\text{erg/s}$ per $1\,M_\odot$ SSP.
- Wavelengths are filtered to be strictly monotonically increasing.
- The first data row contains the unit and configuration prefix: `nuFnu units Angstroms` (and optionally `extrapolate` if requested).

Example file header (with extrapolation):
```text
# CLOUDY table SED | col1 = lambda[Angstrom], col2 = nu*Fnu linear (erg/s per 1 Msun SSP)
9.100000e+01 1.000000e-300 nuFnu units Angstroms extrapolate
9.110000e+01 3.245890e+30 
9.120000e+01 5.892011e+32 
...
```

### 2. The `metadata.json` Catalog

This file lists all exported nodes in a JSON array of objects.

Example:
```json
[
  {
    "tau_SSP": 1000000.0,
    "Z_star": 0.0001,
    "sed_file": "ssp_tau1.000e+06_Z0.0001.sed",
    "it": 0,
    "iz": 0,
    "Qh_unit": 1.341258e+47
  }
]
```

# Python API

::: galapy.spectroscopy.utils.spectra.load_ssp_cube
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.spectra.sed_from_ssp_cube_node
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.spectra.sed_from_csp
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.spectra.write_cloudy_sed
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.spectra.to_cloudy_sed
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.spectra.cloudy_sed_QH
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.spectra.cloudy_sed_QH_reference
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.spectra.extract_ssp_seds
    options:
      show_root_heading: true
      show_source: false

### Example Code

```python
from galapy.spectroscopy.utils.spectra import extract_ssp_seds

# Batch extract SSPs with extrapolation enabled
n_files, metadata = extract_ssp_seds(
    outdir="./ssp_seds",
    target_taus=[1e6, 2e6],
    target_Z=[0.0001, 0.0200],
    extrapolate=True
)
print(f"Generated {n_files} SED files.")
```


