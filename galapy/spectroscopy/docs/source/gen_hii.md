# HII region: sampling and input files generation

CloudIA implements the modules `utils/hii/gen_lhs_hii.py` and `utils/hii/gen_input_hii.py` to build the
grid of the HII regions. The two modules cover the two consecutive stages of the grid production:

1. **Sampling** — `galapy-gen-lhs-hii` draws the free parameters of the HII region from a *Latin Hypercube
   Sample* (LHS) over the axes declared in a charter file, crosses every sample with the SSP nodes extracted by
   [`galapy-sed-cloudy-extract`](ssp_extraction.md), and derives the quantities that cannot be sampled directly
   (stopping column density, chemistry block, grain scaling).
2. **Rendering** — `galapy-gen-hii` turns each job of that grid into a ready-to-run CLOUDY input deck (`.in`),
   by filling the Jinja2 template shipped with the dataset, and stages next to the decks the `.sed` files they
   reference.

The sampling core shared with the PDR sector lives in `utils/lhs_core.py`, so that both sectors
consume the same charter format and produce the same grid-spec layout.

!!! warning
    The charter (`grid_hii.yaml`) and the deck template (`hii.in.j2`) live in the galapy-dataset, not in the
    code.

---

## The HII grid charter

The charter is the single source of truth of the grid: it declares the sampled axes, the LHS settings and the
SSP node counts. It is read by `lhs_core.load_charter`, which rejects any file missing one of the four
mandatory keys `sector`, `axes`, `lhs`, `nodes`; `galapy-gen-lhs-hii` additionally refuses a charter whose
`sector` is not `hii`.

```yaml
sector: hii
cloudy_version: C25.00
ssp_library: parsec22.NT

lhs:
  n_samples: 9000            # LHS points, each one crossed with every SSP node
  seed: 42                   # reproducibility
  optimization: random-cd    # scipy.stats.qmc.LatinHypercube optimization

axes:                        # order matters
  - {name: logU,          min: -4.0,   max: -1.0, prior: uniform}
  - {name: lognH_HII,     min:  0.0,   max:  4.0, prior: uniform}
  - {name: log_zeta_O,    min: -2.3,   max:  0.5, prior: uniform}
  - {name: z_CMB,         min:  0.0,   max: 12.0, prior: uniform}
  - {name: xi_d,          min:  0.001, max:  0.5, prior: uniform}
  - {name: f_esc_target,  min:  0.0,   max:  0.9, prior: uniform}
  - {name: F_star,        min:  0.0,   max:  1.0, prior: uniform}

nodes:
  n_tau_ssp: 8               # 8 x 6 = 48 SSP nodes
  n_zstar: 6
```

### The sampled axes

| Axis | Symbol | Meaning |
| :--- | :--- | :--- |
| `logU` | $\log U$ | Ionization parameter at the illuminated face. Sets the CLOUDY `ionization parameter` command and the Strömgren column density. |
| `lognH_HII` | $\log n_{\rm H}$ | Hydrogen density of the cloud, in ${\rm cm}^{-3}$ (`hden`). |
| `log_zeta_O` | $\log\zeta_O$ | Metallicity axis, see [Abundances](abundances.md). Drives the whole chemistry block and the metallicity term of the grain scaling. |
| `z_CMB` | $z$ | Redshift of the CMB thermal floor (`CMB`). |
| `xi_d` | $\xi_d$ | Dust-to-metal ratio, gauged on $\xi_{d,\rm MW} = 0.43$. |
| `f_esc_target` | $f_{\rm esc}$ | Target escape fraction of ionizing photons; converted into the stopping column density. |
| `F_star` | $F_*$ | Jenkins (2009) depletion strength (`metals deplete jenkins 2009`). |

An axis whose `prior` starts with `log` is sampled uniformly in $\log_{10}$ space between `min` and `max` and
returned already converted back to linear units; every other axis is sampled uniformly between its bounds. In
the fiducial charter all axes are `uniform`, since the quantities that are logarithmic are already declared as
such by their name (`logU`, `lognH_HII`, `log_zeta_O`).

### Sampling quality

`lhs_core.sample_lhs` draws the hypercube with `scipy.stats.qmc.LatinHypercube` and compares the *centered
$L_2$-discrepancy* (Hickernell 1998) of the optimized sample against a plain LHS drawn with the same seed.
An optimization that does not lower the discrepancy aborts the run (`AssertionError`). The absolute discrepancy is
also checked against a $10^{-3}$ threshold, and a warning invites to raise `n_samples` when the sample is not
uniform enough.

### Derived quantities

Two job fields are *not* sampled but computed from the axes.

**Stopping column density.** The escape fraction is translated into a stopping criterion through the Strömgren
sphere model, by `stromgren_stop_logN`:

$$
N_S = \frac{10^{\log U}\, c}{\alpha_B},
\qquad
N_{\rm stop} = \max\left(1 - f_{\rm esc},\; 10^{-3}\right) \times N_S
$$

with $c$ in ${\rm cm\,s^{-1}}$ and $\alpha_B = 2.59\times10^{-13}\ {\rm cm^3\,s^{-1}}$ the case-B recombination
coefficient at $T_e = 10^4$ K (`CONST.alphaB_1e4K`). The floor on $1 - f_{\rm esc}$ guarantees
$N_{\rm stop} \le N_S$ and a finite $\log N_{\rm stop}$ even for $f_{\rm esc} \to 1$. The job stores
$\log_{10} N_{\rm stop}$.

**Grain scaling.** The argument of the CLOUDY `grains ISM` command factorises into a metallicity term and a
dust-to-metal term (see [Abundances](abundances.md)):

$$
{\rm grain\_scale} = \frac{Z(\zeta_O)}{Z_{\rm GC}} \times g(\xi_d)
$$

computed as `Chemistry.metallicity_from_zeta(log_zeta_O) / Chemistry.Z_GC * grain_scale_from_xi_d(xi_d)`. The
`Chemistry` instance is built once per process and cached, to avoid re-reading the chemistry files for each of
the hundreds of thousands of jobs.

---

## Sampling: `galapy-gen-lhs-hii`

The module `utils/hii/gen_lhs_hii.py` exposes the CLI entrypoint `galapy-gen-lhs-hii`, which builds the LHS
sample and the job specification of the whole grid.

### Usage

```text
usage: galapy-gen-lhs-hii [-h] [-c CONFIG] [-s SSP_META] [-o OUTPUT]
```

### Options

| Flag | Argument | Description                                                                                                            |
| :--- | :--- |:-----------------------------------------------------------------------------------------------------------------------|
| `-h`, `--help` | None | Show help message and exit.                                                                                            |
| `-c`, `--config` | `CONFIG` | Path to the grid charter. <br> Default: `grid_hii.yaml` resolved from the GalaPy database (`Nebular/Configs`).          |
| `-s`, `--ssp-meta` | `SSP_META` | Path to the `ssp_metadata.json` written by `galapy-sed-cloudy-extract`. <br> Default: `data/cloudy_seds/ssp_metadata.json`. |
| `-o`, `--output` | `OUTPUT` | Output directory of the grid. Created if missing. <br> Default: `data/grids/hii`.                                       |

### Example

Generate the grid from the dataset charter, using SEDs extracted in a custom directory:

```bash
galapy-gen-lhs-hii -s ./ssp_seds/ssp_metadata.json -o ./data/grids/hii
```

---

## Rendering: `galapy-gen-hii`

The module `utils/hii/gen_input_hii.py` exposes the CLI entrypoint `galapy-gen-hii`, which renders the job
specification into CLOUDY decks and stages the SEDs they need.

### Usage

```text
usage: galapy-gen-hii [-h] [-s SPEC] [-t TEMPLATE] [-d SED_DIR] [-o OUT]
                      [-l LIMIT]
```

### Options

| Flag | Argument | Description                                                                                                       |
| :--- | :--- |:------------------------------------------------------------------------------------------------------------------|
| `-h`, `--help` | None | Show help message and exit.                                                                                       |
| `-s`, `--spec` | `SPEC` | Job specification produced by `galapy-gen-lhs-hii`. <br> Default: `data/grids/hii/hii_grid_spec.h5`.               |
| `-t`, `--template` | `TEMPLATE` | Jinja2 deck template. <br> Default: `None`, i.e. `hii.in.j2` resolved from the GalaPy database (`Nebular/Templates`). |
| `-d`, `--sed-dir` | `SED_DIR` | Directory of the `.sed` files written by `galapy-sed-cloudy-extract`. <br> Default: `data/cloudy_seds`.            |
| `-o`, `--out` | `OUT` | Output directory of the `.in` decks. Created if missing. It is the `--runs` of `galapy-run-cloudy-hii`. <br> Default: `data/hii`. |
| `-l`, `--limit` | `LIMIT` | Render only `N` jobs, picked evenly spread over the grid (`np.linspace`), instead of the whole spec. <br> Default: `None`, i.e. all jobs. |

`--limit` is meant for smoke tests and pipeline dry-runs: since the subsample is taken on a regular stride
rather than from the head of the file, it still spans the full extent of every axis.

### Example

Render a 100-deck subsample of the grid, for a quick validation run:

```bash
galapy-gen-hii -s ./data/grids/hii/hii_grid_spec.h5 -d ./ssp_seds -o ./data/hii -l 100
```

---

## Output Structure and File Formats

### 1. The sampling stage

`galapy-gen-lhs-hii` writes two files in its output directory:

| File | Type | Description |
| :--- | :--- | :--- |
| `lhs_hii.npy` | `float64[n_samples, 7]` | The raw LHS sample, one row per point, columns ordered as the `axes` block of the charter. |
| `hii_grid_spec.h5` | HDF5 | The job specification, one dataset per field, all of length $N_{\rm jobs}$. |

The grid spec is written in *columnar* form by `lhs_core.write_grid_spec`: each job field becomes a dataset of
length $N_{\rm jobs}$, strings being stored with `h5py.string_dtype()`. Its datasets are:

| Dataset | Type | Description |
| :--- | :--- | :--- |
| `job_id` | `str` | Unique job identifier, `{sample}_{it}_{iz}`. |
| `logU`, `lognH_HII`, `log_zeta_O`, `z_CMB`, `xi_d`, `f_esc_target`, `F_star` | `float64` | The sampled axes, copied verbatim from the LHS row. |
| `log_N_stop` | `float64` | $\log_{10} N_{\rm stop}$ derived from `logU` and `f_esc_target`. |
| `grain_scale` | `float64` | Linear argument of `grains ISM`. |
| `element_scale_block` | `str` | The `element scale factor ...` lines for this $\log\zeta_O$, newline-joined. |
| `tau_SSP`, `Z_star` | `float64` | Age and metallicity of the SSP node. |
| `sed_file` | `str` | Name of the `.sed` file of the SSP node. |

Note that `xi_d`, `f_esc_target`, `tau_SSP` and `Z_star` are not consumed by the template: they are carried
along for provenance, so that the spec alone is enough to reconstruct how each deck was built.

### 2. The rendering stage

`galapy-gen-hii` populates its output directory with:

```text
data/hii/
├── hii_00000_003_02.in        # one deck per job
├── hii_00042_003_02.in
├── ...
├── jobs_hii.txt               # manifest, one job_id per line
└── SED/                       # staged SEDs, deduplicated
    ├── ssp_tau1.000e+06_Z0.0001.sed
    └── ...
```

SEDs are accumulated in a set while the decks are rendered, so each file is copied once no matter how many
jobs reference it.

Staging keeps the decks relocatable: since the template references the SED by bare filename
(`table SED "{{ sed_file }}"`), the output directory can be shipped to the compute node as a single unit. The two
other files the deck references, the abundance file `GC.abn` and the line list `hii_lines.dat`, are shared by all
the jobs, and are staged by the runner at run time (see [Running CLOUDY](run_hii.md#2-staging)).

### 3. The rendered deck

The `hii.in.j2` template maps the job fields onto the CLOUDY commands as follows:

| Job field | CLOUDY command |
| :--- | :--- |
| `job_id` | `title HII id=...`, and the `save last ...` output filenames |
| `lognH_HII` | `hden` |
| `logU` | `ionization parameter` |
| `z_CMB` | `CMB` |
| `F_star` | `metals deplete jenkins 2009` |
| `log_zeta_O` | `metals ... log` |
| `element_scale_block` | the `element scale factor ...` block |
| `grain_scale` | `grains ISM ... linear` |
| `sed_file` | `table SED "..."` |
| `log_N_stop` | `stop column density` |

The physical configuration fixed by the template — identical for every job of the grid — is a closed spherical
geometry (`sphere` plus `double optical depths`), stopping either at the column density set by $f_{\rm esc}$ or
at $T = 4000$ K, with `iterate to convergence`. The fiducial chemistry is read from `abundances "GC.abn"`, and
then rescaled by the `metals` and `element scale factor` commands. The template also sets `turbulence 5 km/s`
and the `cosmic rays background`. PAHs are absent by construction in the HII sector, which also allows
`no grain qheat` to save run time.

Each deck saves the emergent continuum, the grain continuum, the line list (against `hii_lines.dat`), the grain
abundance and the grain temperature, all prefixed with `hii_{job_id}`.

---

# Python API

::: galapy.spectroscopy.utils.lhs_core.load_charter
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.lhs_core.sample_lhs
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.lhs_core.read_ssp_meta
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.lhs_core.write_grid_spec
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.gen_lhs_hii.stromgren_stop_logN
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.gen_lhs_hii.build_jobs
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.gen_input_hii.load_jobs
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.gen_input_hii.render_one
    options:
      show_root_heading: true
      show_source: false

