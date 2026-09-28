# Merging the grid fragments

CloudIA implements the module `utils/merge_grid.py` to merge the fragments `<job_id>.h5` of a sector into a
single **corpus file**. The parser writes one fragment per CLOUDY model (see [Parse HII](parse_hii.md)). The
corpus, `data/products/hii_grid.h5` or `data/products/pdr_grid.h5`, is what the downstream stages read: the
validation of the grid and the training of the emulators.

The merge is the last step of the production chain of a sector (see [Running CLOUDY](run_hii.md)):

| Step | Command | Produces |
| :--- | :--- | :--- |
| 5 | `galapy-run-cloudy-hii` | `data/hii/parsed/<job_id>.h5`, one fragment per model |
| 6 | `galapy-h5 --hii` | `data/products/hii_grid.h5`, the corpus |

The fragments are only read, never modified or deleted, so the corpus can be regenerated from them at any time,
e.g. after new jobs have been run.

!!! warning
    The merge takes **every** `*.h5` file of the fragment directory. Keep one grid per directory: fragments of
    two grids that share the same CLOUDY version and line list pass the checks and end up in the same corpus,
    unless two of their `job_id`s happen to coincide.

---

## Command Line Interface (CLI)

The module exposes the CLI entrypoint `galapy-h5`.

### Usage

```text
usage: galapy-h5 [-h] [--hii | --pdr] [--frags FRAGS] [--out OUT] [--cloudy-version CLOUDY_VERSION]
                 [--ssp-lib SSP_LIB]
```

### Options

| Flag | Argument | Description |
| :--- | :--- | :--- |
| `-h`, `--help` | None | Show help message and exit. |
| `--hii` | None | HII sector: default fragments `data/hii/parsed`, default output `data/products/hii_grid.h5`. Mutually exclusive with `--pdr`. |
| `--pdr` | None | PDR sector: default fragments `data/pdr/parsed`, default output `data/products/pdr_grid.h5`. Mutually exclusive with `--hii`. |
| `--frags` | `FRAGS` | Directory of the fragments `<job_id>.h5`. It is the `--frags` of the runner. Overrides the sector default. |
| `--out` | `OUT` | Path of the corpus file. Missing parent directories are created. Overrides the sector default. |
| `--cloudy-version` | `CLOUDY_VERSION` | CLOUDY version recorded in the corpus metadata. <br> Default: `C25.00`. |
| `--ssp-lib` | `SSP_LIB` | SSP library recorded in the corpus metadata. <br> Default: `parsec22.NT`. |

A sector flag is optional: without it, both `--frags` and `--out` must be given explicitly. With it, either path
can still be overridden.

The HII defaults chain with those of the runner: `galapy-run-cloudy-hii` writes its fragments to
`<runs>/parsed`, i.e. `data/hii/parsed` with the default `--runs`.

### Examples

Merge the HII grid produced with the default paths:

```bash
galapy-h5 --hii
```

Merge a grid staged elsewhere:

```bash
galapy-h5 --hii --frags /scratch/hii/parsed --out /scratch/products/hii_grid.h5
```

Build a corpus from a test subset, without touching the production one:

```bash
galapy-h5 --frags data/hii_test/parsed --out data/products/hii_grid_test.h5
```

A successful run reports the number of fragments and of grid points:

```text
[merge] 9000 fragments (jobs), 9000 grid_point, file generated in data/products/hii_grid.h5
```

---

## How the merge works

The function `merge` proceeds in four steps.

**1. Fragment discovery.** It takes all the `*.h5` files of `--frags`, sorted by name. Fragments still being
written by the parser (`<job_id>.h5.part`) do not match the pattern and are ignored. The output file is excluded
too, so the corpus can be written inside the fragment directory itself. An empty directory is an error.

**2. Shared datasets.** The three root datasets of the fragments depend only on the CLOUDY version, the deck
template and the line list, so they are the same for every model of a grid:

```python
ROOT_SHARED = ('continuum/wave_grid', 'line_names', 'lines_emergent/wavelengths_rest')
```

They are copied **once**, from the first fragment. Every other fragment is compared against that copy
(`np.array_equal`), and any difference aborts the merge, naming the fragment and the dataset. This keeps a corpus
from mixing grids with different wavelength meshes or line lists, whose spectra would not be comparable element
by element.

**3. Grid points.** Every group `grid_point_<job_id>` of every fragment is copied as it is, with all its datasets
and attributes. The group name comes from the `job_id`, so two fragments of the same grid can never collide. The
same job found in two files (e.g. a renamed copy of a fragment) aborts the merge.

**4. Metadata.** The root attributes `cloudy_version`, `ssp_lib` and `n_grid_points` are written, the last one
counting the copied groups.

The corpus is written to `<out>.part` and renamed to `<out>` only once complete. On any error the partial file
is deleted, and a previous corpus at `<out>` is left untouched.

### What the merge does not do

- **No filtering.** Non-converged models and models with CLOUDY warnings are merged like the others. They are
  recognised by their attributes `not_converged` and `cloudy_warnings`, and filtered by the consumers of the
  corpus.
- **No check of the metadata.** `--cloudy-version` and `--ssp-lib` are *declared*, not verified, since the
  fragments do not carry them. They must match the `cloudy_version` and `ssp_library` of the grid charter
  (`grid_hii.yaml`) and the `cloudy_banner` recorded in the run report of the runner.

---

## Output Structure and File Formats

### 1. The corpus file

```text
hii_grid.h5
├── attrs: cloudy_version, ssp_lib, n_grid_points
├── continuum/wave_grid                    # shared, one copy
├── line_names                             # shared, one copy
├── lines_emergent/wavelengths_rest        # shared, one copy
├── grid_point_00000_003_02/               # one group per model, as in its fragment
├── grid_point_00000_003_03/
└── ...
```

The root attributes are:

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `cloudy_version` | `str` | The `--cloudy-version` of the merge, e.g. `C25.00`. |
| `ssp_lib` | `str` | The `--ssp-lib` of the merge, e.g. `parsec22.NT`. |
| `n_grid_points` | `int` | Number of `grid_point_*` groups in the file. |

The shared datasets and the content of each `grid_point_<job_id>` group — sampled parameters, continua, line
luminosities, diagnostics and quality flags — are described in [Parse HII](parse_hii.md#1-the-fragment-job_idh5).

### 2. Failure modes

| Error | Cause |
| :--- | :--- |
| `ValueError: [merge] <frags>: no fragments available` | No `*.h5` file in `--frags`: wrong directory, or nothing parsed yet. |
| `ValueError: [merge] <fragment>: '<dataset>' different from first fragment` | Fragments of different grids (CLOUDY version or line list) in the same directory. |
| `RuntimeError: Unable to synchronously copy object (destination object already exists)` | The same `grid_point_<job_id>` found in two files. |

These errors abort the merge with exit status `1`, and leave no output file. A missing sector together with a
missing `--frags` or `--out` is a usage error, with exit status `2`.

---

# Python API

::: galapy.spectroscopy.utils.merge_grid.merge
    options:
      show_root_heading: true
      show_source: false

### Example Code

```python
import h5py
import numpy as np

from galapy.spectroscopy.utils.merge_grid import merge

n = merge('data/hii/parsed', 'data/products/hii_grid.h5')
print(f"{n} grid points merged")

# read the corpus, keeping the converged models without warnings
with h5py.File('data/products/hii_grid.h5', 'r') as f:
    print(f.attrs['cloudy_version'], f.attrs['ssp_lib'], f.attrs['n_grid_points'])
    wave = f['continuum/wave_grid'][:]
    names = f['line_names'][:].astype(str)
    good = [k for k in f if k.startswith('grid_point_')
            and not f[k].attrs['not_converged']
            and not f[k].attrs.get('cloudy_warnings', False)]
    X = np.array([[f[k].attrs[a] for a in ('logU', 'lognH_HII', 'log_zeta_O')] for k in good])
    L_lines = np.stack([f[k]['lines_emergent/fluxes'][:] for k in good])
```
