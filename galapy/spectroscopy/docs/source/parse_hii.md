# HII region: parsing the CLOUDY models

CloudIA implements the module `utils/hii/parse_one_hii.py` to turn the working directory of **one** CLOUDY run
into **one** HDF5 fragment, `<job_id>.h5`. The parser reads the saves of the model and rescales them from the
per-unit-area outputs of CLOUDY to quantities per solar mass of SSP formed. Along the way it computes the
diagnostics of the run: measured escape fraction, FUV transmittance, energy budget and dust mass.

The parser is normally not called by hand. [`galapy-run-cloudy-hii`](run_hii.md) runs it in the same worker,
right after each successful CLOUDY run, and `--parse-only` re-runs it on existing working directories. It can
also be called standalone, to debug a single model or as a separate step of a pipeline (the Nextflow process
`PARSE_ONE_HII`).

The fragments of a grid are then merged into a single corpus file by `galapy-merge-h5 --hii`.

!!! warning
    Only **successful** runs are parsed: the working directory must hold `converged.flag` and no `RUN_FAILED`
    marker (see [Running CLOUDY](run_hii.md#3-running-cloudy)). Non-converged models and models with CLOUDY
    warnings are parsed and flagged.

---

## Inputs

The parser combines three sources, all resolved from the `job_id`, i.e. the **name of the working directory**:

| Source | Path | What is read |
| :--- | :--- | :--- |
| Working directory | `data/hii/work/<job_id>/` | Outcome markers and the CLOUDY saves `hii_<job_id>.*`. |
| Job specification | `--spec`, i.e. `hii_grid_spec.h5` | The row of the job: the 7 sampled axes plus `tau_SSP`, `Z_star`. |
| SSP metadata | `--ssp-meta`, i.e. `ssp_metadata.json` | $Q_H$ per $M_\odot$ (`Qh_unit`) of the SSP node `(tau_SSP, Z_star)`. |

The job is looked up in the spec through an index `job_id → row`, built once per process and cached
(`spec_index`). Under `galapy-run-cloudy-hii` a worker therefore reads the `job_id` column only once, however
many models it parses. A duplicated `job_id` in the spec raises `ValueError`. A job that is not part of the spec
raises `KeyError`.

The SSP node is matched on the exact `(tau_SSP, Z_star)` pair. The spec and the metadata must therefore come from
the same `galapy-sed-cloudy-extract` run.

Of the saves written by the deck, four are parsed:

| File | Content | Parsed by |
| :--- | :--- | :--- |
| `hii_<job_id>.con` | Continuum: $\lambda$ plus the components below, in $\nu F_\nu$ [${\rm erg\,cm^{-2}\,s^{-1}}$] | `parse_cloudy_con` |
| `hii_<job_id>.con_grain` | Grain emission: $\lambda$, graphite, silicates, total | `parse_cloudy_cong` |
| `hii_<job_id>.lines` | Emergent absolute line intensities [${\rm erg\,cm^{-2}\,s^{-1}}$] | `parse_cloudy_linelist` |
| `hii_<job_id>.dusa` | Dust mass density per zone | `integrate_grain_abundance` |

`hii_<job_id>.grain_temp` is saved for diagnostics only, and is not read.

!!! note "Units, verified on the CLOUDY C25.00 source"
    The deck sets the field with `ionization parameter` and no radius (*intensity case*), so every output is
    **per cm² of the illuminated face** (`zone_startend.cpp`: "intensity per unit inner cloud area").

    | Output | Units | Source |
    | :--- | :--- | :--- |
    | `save continuum`, columns 2–9 | $\nu F_\nu$ [${\rm erg\,cm^{-2}\,s^{-1}}$] | `save_do.cpp`, `flxCell` |
    | `save continuum ... units angstrom`, column 1 | Å, **vacuum** | `energy.cpp`, `angstromVac` |
    | `save continuum grain` | as the continuum, same mesh and order | `save_do.cpp`, branch `CONG` |
    | `save line list ... absolute` | ${\rm erg\,cm^{-2}\,s^{-1}}$ into $4\pi$, **linear** | `cddrive.cpp`, `cdLine_ip` |
    | wavelength in the line labels | Å/µm/cm, **air** above 2000 Å, vacuum below | `cddefines.cpp`, `t_wavl::sprt_wl` |
    | `save grain abundance` | ${\rm g\,cm^{-3}}$ per zone, depth in cm | `save_do.cpp`, branch `DUSA` |
    | `CMB z` | adds $4\pi\nu B_\nu(2.725\,(1+z)\,{\rm K})$ to the incident column | `parse_CMB.cpp` |

---

## Command Line Interface (CLI)

The module has no console script: it is run as a module.

### Usage

```text
usage: python -m galapy.spectroscopy.utils.hii.parse_one_hii [-h] [--spec SPEC] [--ssp-meta SSP_META]
                                                             [--out OUT]
                                                             workdir
```

### Options

| Flag | Argument | Description |
| :--- | :--- | :--- |
| `-h`, `--help` | None | Show help message and exit. |
| `workdir` | `WORKDIR` | Working directory of a single job, e.g. `data/hii/work/<job_id>`. Its name is the `job_id`. **(Required)** |
| `--spec` | `SPEC` | Job specification produced by `galapy-gen-lhs-hii`. <br> Default: `data/grids/hii/hii_grid_spec.h5`. |
| `--ssp-meta` | `SSP_META` | `ssp_metadata.json` written by `galapy-sed-cloudy-extract`. <br> Default: `data/cloudy_seds/ssp_metadata.json`. |
| `--out` | `OUT` | Output fragment, as a file or a directory, see below. <br> Default: `data/hii/parsed`. |

`--out` is interpreted by `resolve_out`:

- an **existing directory**, or a path **without extension** (like the default `data/hii/parsed`), is a
  directory, and the fragment is written to `<out>/<job_id>.h5`;
- anything else is the **file** to write, e.g. `<job_id>.h5`, or `<job_id>.h5.part` for a batch script that
  renames the file itself once done.

Missing parent directories are created.

### Example

Parse a single model into the default fragment directory:

```bash
python -m galapy.spectroscopy.utils.hii.parse_one_hii data/hii/work/00042_003_02
```

The same, for a grid staged elsewhere, writing to an explicit file:

```bash
python -m galapy.spectroscopy.utils.hii.parse_one_hii /scratch/hii/work/00042_003_02 \
    --spec /scratch/grids/hii_grid_spec.h5 --ssp-meta /scratch/seds/ssp_metadata.json \
    --out /scratch/hii/parsed/00042_003_02.h5
```

---

## From the CLOUDY outputs to the physical quantities

### 1. The run check

`require_successful_run` is called **before** any number is read:

- a missing `converged.flag` raises `FileNotFoundError`, since the convergence of the run is unknown;
- a `RUN_FAILED` marker raises `RuntimeError`, reporting the reason stored in the marker;
- otherwise the flag gives the convergence status (`0` means not converged), stored as the attribute
  `not_converged` of the fragment;
- the `WARNINGS` marker, written by the runner when CLOUDY ended with warnings, is stored as the attribute
  `cloudy_warnings`. The working directories are transient, so the fragment is where this information survives.

### 2. The continuum

`parse_cloudy_con` reads five columns of `save continuum` (1-based indices, CLOUDY header names in brackets):

| Column | Symbol | Content |
| :--- | :--- | :--- |
| 1 [`nu`] | $\lambda$ | Wavelength in Å, in vacuum (the deck saves with `units angstrom`). |
| 2 [`incident`] | ${\rm col}_2$ | Incident field: the SED and the CMB of the run. |
| 3 [`trans`] | ${\rm col}_3$ | Transmitted SED: the incident one attenuated by the cloud, with no diffuse emission. |
| 4 [`DiffOut`] | ${\rm col}_4$ | Diffuse outward emission of the cloud: gas and grains, lines included. |
| 9 [`outlin`] | ${\rm col}_9$ | Outward line emission only. |

The wavelength column must be strictly monotonic, otherwise the file is rejected. The integrals below are
computed with the trapezoidal rule, on the wavelength grid reversed into increasing $\lambda$.

### 3. The geometric factor $s_k$

The deck fixes the incident field through the `ionization parameter`, so CLOUDY returns every output **per unit
area** of the illuminated face. The flux of ionizing photons there is $\Phi_H = U\, n_{\rm H}\, c$. For the SSP
node $k = (\tau_{\rm SSP}, Z_*)$, of ionizing photon rate $Q_{H,k}$ per $1\,M_\odot$ formed, the area that
receives exactly that rate is

$$
s_k = \frac{Q_{H,k}(1\,M_\odot)}{U\, n_{\rm H}\, c} \qquad [{\rm cm^2}\,M_\odot^{-1}]
$$

computed by `s_k_factor` from `logU`, `lognH_HII` and `Qh_unit` [photons ${\rm s^{-1}}\,M_\odot^{-1}$]. Multiplying
any per-${\rm cm^2}$ output by $s_k$ turns it into a luminosity (or a mass) **per $1\,M_\odot$ of SSP formed**, the
normalization of the GalaPy SSP libraries. $s_k$ is stored in the fragment, so every per-$M_\odot$ quantity can be
brought back to the per-${\rm cm^2}$ units of CLOUDY.

### 4. Derived quantities

**Nebular continuum.** The diffuse emission without the lines,

$$
L^{\rm neb}_\lambda = \left({\rm col}_4 - {\rm col}_9\right) s_k
\qquad [{\rm erg\,s^{-1}}\ (\nu L_\nu)\ {\rm per}\ M_\odot]
$$

The lines are removed here because they are stored separately, as integrated fluxes.

**Transmission.** The fraction of the incident SED that crosses the cloud,

$$
T(\lambda) = \frac{{\rm col}_3}{{\rm col}_2}
$$

set to $1$ wherever ${\rm col}_2 = 0$ (`support_safe_ratio`). Since $T$ is meaningful only where there is an
incident field, the wavelength range with ${\rm col}_2 > 0$ is stored as `sed_support_A`.

**Measured escape fraction.** The ratio between transmitted and incident radiation below the Lyman limit,
$\lambda < 911.6$ Å (`CONST.LymanA`):

$$
f_{\rm esc}^{\rm meas} = \frac{\int_{\lambda < 911.6} {\rm col}_3\, d\lambda}{\int_{\lambda < 911.6} {\rm col}_2\, d\lambda}
$$

Since ${\rm col}_n$ is $\nu F_\nu = \lambda F_\lambda$, $\int \nu F_\nu\, d\lambda = hc \int (F_\lambda / h\nu)\, d\lambda$
is proportional to the photon rate. $f_{\rm esc}^{\rm meas}$ is therefore the escape fraction of ionizing
**photons**, with no further weighting. It is clipped to $[0, 1]$, and set to $0$ when the grid has no point
below the Lyman limit. Compared with `f_esc_target`, the value used by `galapy-gen-lhs-hii` to set the stopping
column density, it measures how well the Strömgren stopping criterion reproduces the target.

**FUV terms.** Over the Habing band, $6$–$13.6$ eV, i.e. $911.76 \le \lambda \le 2066$ Å
(`CONST.FUV_Lo_A`, `CONST.FUV_Hi_A`), `fuv_transmittance_hii` computes

$$
T_{\rm FUV} = \frac{\int_{\rm FUV} {\rm col}_3\, d\ln\lambda}{\int_{\rm FUV} {\rm col}_2\, d\ln\lambda},
\qquad
N_{\rm FUV} = \frac{\int_{\rm FUV} {\rm col}_4\, d\ln\lambda}{\int_{\rm FUV} {\rm col}_2\, d\ln\lambda}
$$

$T_{\rm FUV}$, clipped to $[0, 1]$, is the fraction of the stellar FUV field that crosses the ionized layer.
$N_{\rm FUV} \ge 0$ is the FUV emission of the ionized gas itself (two-photon continuum, Ly$\alpha$), in units of
the incident field. Both are set to $0$ when the band is empty or not illuminated. Together they describe the
FUV field reaching the photodissociation region behind the HII region.

Both are ratios of **energy** fluxes ($\int \nu F_\nu\, d\ln\lambda = \int F_\lambda\, d\lambda$), unlike
$f_{\rm esc}^{\rm meas}$: they correct $G_0$, an energy flux in the Habing band (the PDR deck imposes
$\int F_\nu\, d\nu$ over 6–13.6 eV), and the PDR must receive the energy that the ionized layer transmits.

**Energy budget.** The relative mismatch between the incident energy and the energy that leaves the cloud,
computed by `energy_balance`:

$$
\epsilon = \frac{\left| I_2 - (I_3 + I_4) \right|}{I_2^{\star}},
\qquad
I_n = \int {\rm col}_n\, d\ln\lambda = \int F_{\lambda,n}\, d\lambda,
\qquad
I_2^{\star} = \int_{\lambda < 10\,\mu{\rm m}} {\rm col}_2\, d\ln\lambda
$$

with $I_2$, $I_3$, $I_4$ integrated over the full wavelength grid, and stored as `energy_balance_rel`. The
integral is taken in $d\ln\lambda$ because $\int \nu F_\nu\, d\lambda$, the weighting of $f_{\rm esc}^{\rm meas}$,
counts **photons**. Photon number is not conserved: the dust re-emits every absorbed UV photon as many IR
photons, so a photon budget would never close. $\epsilon$ is `NaN` if the reference energy is zero.

The reference $I_2^{\star}$ is the **stellar** incident energy. With `CMB z_CMB` the incident column also holds
$4\pi\nu B_\nu(T_{\rm CMB})$, which in many models carries more energy than the SED: referred to the whole $I_2$,
the imbalance would measure the CMB, which balances itself, and stay blind to a double-counted stellar
component. The split at $10\,\mu$m (`STELLAR_MAX_A`) is exact enough for this: below it the CMB of every
$z_{\rm CMB} \le 12$ carries $\sim 3\times10^{-14}$ of its energy, above it the SSP nodes carry at most $1.3\%$ of
theirs. The weight of the CMB is stored as

$$
\texttt{cmb\_incident\_ratio} = \frac{\int_{\lambda \ge 10\,\mu{\rm m}} {\rm col}_2\, d\ln\lambda}{I_2^{\star}}
$$

In the models where it is large, the print precision of CLOUDY ($3$ decimals) sets the floor of $\epsilon$:
the grid validation grades the energy budget only below a threshold of this ratio.

!!! warning "The CMB in the per-$M_\odot$ outputs"
    The CMB is a field per unit area that does not scale with the stellar mass, while $s_k$ scales everything per
    $M_\odot$. The incident column is neither stored nor emulated: through it the CMB only reaches the
    diagnostics (`energy_balance_rel`, `cmb_incident_ratio`, `sed_support_A`). It cancels in
    `continuum/transmission` (the same transmission of the medium for SED and CMB, pure absorption), and it is
    negligible in $f_{\rm esc}^{\rm meas}$, $T_{\rm FUV}$ and $N_{\rm FUV}$ (below 2066 Å).

    The emulated quantities are touched only through the **own emission**: the grains re-emit the CMB they
    absorb, and CLOUDY includes it in ${\rm col}_4$. The observed continuum is the contrast against the
    background, $I_{\rm obs}(\lambda) = I_{\rm emitted}(\lambda) - (1 - T(\lambda))\, 4\pi\nu B_\nu(T_{\rm CMB})$.
    On the pilot grid this correction is below $10^{-3}$ of the own emission in most models, a few per cent at
    high $z$ and intermediate $U n_{\rm H}$, and 33–37% of the far-IR energy (above 50 µm) at $z \approx 9$–10
    with $n_{\rm H} \approx 1$–3 cm⁻³. How to subtract it (da Cunha et al. 2013) is an open modelling decision.

**Dust mass.** `integrate_grain_abundance` integrates the dust mass density $\rho_d$ (the `total` column of
`save grain abundance`, in ${\rm g\,cm^{-3}}$) over the depth $r$ of the zones:

$$
\Sigma_d = \int \rho_d\, dr \quad [{\rm g\,cm^{-2}}],
\qquad
M_d = \frac{\Sigma_d\, s_k}{M_\odot} \quad [M_\odot\ {\rm per}\ M_\odot\ {\rm of\ SSP\ formed}]
$$

The file is validated strictly: its header must start with `#Depth` and end with `total`, it must contain at
least two zones, and the depth must be strictly increasing. A non-increasing depth means that several iterations
were concatenated, i.e. the deck lost its `last` keyword, and the integral would be overestimated.

**Lines.** `parse_cloudy_linelist` reads the `column` format of `save line list`, one line per row: label,
wavelength with its unit, intensity. The label is rebuilt by joining its tokens with a single space, so the
CLOUDY label `H  1` is stored as `H 1`. The wavelength is converted to Å (`m` for µm: $\times 10^4$; `c` for cm:
$\times 10^8$; `A` or no unit: $\times 1$). A row starting with `iteration` restarts the list, so only the last
iteration is kept. The intensities, `absolute` and `emergent`, are multiplied by $s_k$. The lines follow the order
of `hii_lines.dat`, i.e. of the master list (see [Line list](line_list.md)).

CLOUDY prints the line wavelengths in **air** above 2000 Å (far-IR and radio lines included) and in vacuum
below, while the continuum mesh is in vacuum. The printed values are stored as they are, since they identify the
line (Decision 15), and `air_to_vacuum_A` converts them to vacuum with the same formula CLOUDY uses
(`t_wavl::p_wlAirVac`, Peck & Reeder 1972): H$\alpha$ $6562.80 \to 6564.61$ Å. The vacuum values are the ones to
place the lines on the continuum mesh, or to compare with an observed wavelength.

**Grain emission.** The `total` column of `save continuum grain`, multiplied by $s_k$, is kept as a diagnostic
of the dust emission. Its wavelength column must coincide with that of the continuum (`parse_cloudy_cong(...,
wave_ref=wave)`), so the stored array is on `continuum/wave_grid`.

---

## Output Structure and File Formats

### 1. The fragment `<job_id>.h5`

```text
00042_003_02.h5
├── attrs: units_schema                    # 'cloudia.hii.v1'
├── units/                                 # attrs: unit of every member of grid_point_*
├── descriptions/                          # attrs: description of every member of grid_point_*
├── continuum/wave_grid                    # shared
├── line_names                             # shared
├── lines_emergent/wavelengths_rest        # shared
├── lines_emergent/wavelengths_rest_vacuum # shared
└── grid_point_00042_003_02/
    ├── f_esc_meas
    ├── T_fuv_hii
    ├── N_fuv_hii
    ├── dust_mass_per_Msun
    ├── s_k
    ├── continuum/
    │   ├── nebular_emission_per_Msun
    │   ├── grain_diag_per_Msun
    │   └── transmission
    └── lines_emergent/
        └── fluxes
```

**Units.** The units are written in the fragment itself, from the tables `ROOT_UNITS` and `POINT_UNITS` of the
parser (unit strings in the astropy *generic* syntax: `Msun` is `solMass`, `''` dimensionless or flag, `dex(X)`
the log10 of a quantity in X):

- each **root** dataset carries the attributes `units`, `description` and, for the wavelengths,
  `wavelength_medium`;
- the members of every `grid_point_<job_id>` group, attributes and datasets, are described **once per file** by
  the root groups `units` and `descriptions`, keyed by attribute name or by dataset path relative to the group
  (e.g. `continuum/transmission`). Writing them once avoids repeating the same strings in every model of the
  corpus;
- the root attribute `units_schema` versions the tables. `check_point_units` refuses to write a fragment with a
  member missing from them, and `galapy-merge-h5` refuses to merge fragments with different units.

The four **root** datasets depend only on the CLOUDY version, the deck template and `hii_lines.dat`, so they are
identical in every fragment of a grid. `galapy-merge-h5` verifies this, and keeps a single copy of them in the corpus file:

| Dataset | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `continuum/wave_grid` | `float32[n_λ]` | Å, vacuum | Wavelength grid of the continuum, in the native CLOUDY order. |
| `line_names` | `bytes[n_lines]` | — | CLOUDY labels of the lines, e.g. `b'H 1'`. |
| `lines_emergent/wavelengths_rest` | `float32[n_lines]` | Å, air above 2000 Å | Rest wavelengths of the lines, as printed by CLOUDY. |
| `lines_emergent/wavelengths_rest_vacuum` | `float32[n_lines]` | Å, vacuum | The same, in vacuum (`air_to_vacuum_A`). |

The group `grid_point_<job_id>` holds the model itself. It is named after the job, so the fragments of a grid
can be merged without collisions. Its datasets are:

| Dataset | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `continuum/nebular_emission_per_Msun` | `float32[n_λ]` | `erg s-1 Msun-1` ($\nu L_\nu$) | $({\rm col}_4 - {\rm col}_9)\, s_k$ |
| `continuum/grain_diag_per_Msun` | `float32[n_λ]` | `erg s-1 Msun-1` ($\nu L_\nu$) | Grain emission $\times\, s_k$ (diagnostic) |
| `continuum/transmission` | `float32[n_λ]` | `''` | $T(\lambda) = {\rm col}_3 / {\rm col}_2$ |
| `lines_emergent/fluxes` | `float32[n_lines]` | `erg s-1 Msun-1` | Emergent line **luminosities** |
| `f_esc_meas` | scalar | `''` | Measured ionizing escape fraction (photons) |
| `T_fuv_hii` | scalar | `''` | FUV energy transmittance of the ionized layer |
| `N_fuv_hii` | scalar | `''` | FUV emission of the ionized layer, in units of the incident energy |
| `dust_mass_per_Msun` | scalar | `Msun Msun-1` | Dust mass of the cloud per $M_\odot$ of SSP formed |
| `s_k` | scalar | `cm2 Msun-1` | Area of the illuminated face per $M_\odot$ of SSP formed |

Every quantity "per Msun" is per $M_\odot$ of SSP **formed**.

and its attributes:

| Attribute | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `logU` | `float` | `dex` | $\log_{10} U$, copied from the spec. |
| `lognH_HII` | `float` | `dex(cm-3)` | $\log_{10} n_{\rm H}$, copied from the spec. |
| `log_zeta_O` | `float` | `dex` | $\log_{10}\zeta_O$, copied from the spec. |
| `z_CMB`, `xi_d`, `f_esc_target`, `F_star` | `float` | `''` | The other sampled axes, copied from the spec. |
| `tau_SSP` | `float` | `yr` | Age of the SSP node. |
| `Z_star` | `float` | `''` | Metallicity (mass fraction) of the SSP node. |
| `not_converged` | `bool` | `''` | `True` if CLOUDY did not converge (`converged.flag` = `0`). |
| `cloudy_warnings` | `bool` | `''` | `True` if CLOUDY ended with warnings (exit code `2`, marker `WARNINGS`). |
| `f_esc_meas` | `float` | `''` | Same value as the dataset, for quick filtering. |
| `energy_balance_rel` | `float` | `''` | The energy-budget mismatch $\epsilon$, over the stellar incident energy. |
| `cmb_incident_ratio` | `float` | `''` | Incident energy of the CMB over the stellar one. |
| `sed_support_A` | `float[2]` | `Angstrom` | $(\lambda_{\min}, \lambda_{\max})$ where the incident field (SED and CMB) is non-zero. |

### 2. Atomic writing

The fragment is first written to `<out>.part`, and renamed to its final name only once the file is complete
(`atomic_h5`). On any error the partial file is deleted. The runner also deletes the old fragment before each new
run of a job. Together, the two rules guarantee that **a fragment exists only if the last run of its job was
parsed successfully**. This is the invariant that `galapy-run-cloudy-hii --resume` relies on.

### 3. Failure modes

The messages of the parser checks are prefixed with `[parse_one/hii]`. No error leaves a partial fragment behind:

| Error | Cause |
| :--- | :--- |
| `FileNotFoundError: ... converged.flag: absent` | The run never reached its end (timeout, crash), or was never run. |
| `RuntimeError: ... RUN_FAILED (...)` | CLOUDY returned a non-zero code or missed a save: the run is not parsed. |
| `KeyError: ... absent in <spec>` | The working directory does not belong to the grid of `--spec`. |
| `KeyError: (tau_SSP, Z_star)` | The SSP node of the job is missing from `--ssp-meta`. |
| `FileNotFoundError` | A save is missing. The runner would have marked such a run `RUN_FAILED`. |
| `ValueError: ... not strictly monotonic` | Corrupted continuum. |
| `ValueError: ... columns, expected columns: 4` | The grain continuum does not have the expected layout. |
| `ValueError: ... wavelength column differs from the continuum mesh` | The grain continuum is not on the mesh of the continuum (e.g. a different `units` on its save). |
| `ValueError: ... units table out of date` | A quantity is written without its units in `POINT_UNITS` (a development error). |
| `ValueError: ... expected header '#Depth<TAB>...<TAB>total'` | The grain abundance format changed. |
| `ValueError: ... depth not strictly increasing` | Several iterations were saved: the `last` keyword is missing from the deck. |
| `ValueError: ... no lines extracted` | Empty line list. |

---

# Python API

::: galapy.spectroscopy.utils.hii.parse_one_hii.main
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.require_successful_run
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.resolve_out
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.spec_index
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.parse_cloudy_con
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.parse_cloudy_cong
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.parse_cloudy_linelist
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.integrate_grain_abundance
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.s_k_factor
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.fesc
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.fuv_transmittance_hii
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.energy_balance
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.cmb_incident_ratio
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.air_to_vacuum_A
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.support_safe_ratio
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.atomic_h5
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.write_units_legend
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.write_root_dataset
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.parse_one_hii.check_point_units
    options:
      show_root_heading: true
      show_source: false

### Example Code

```python
import h5py

# read back one fragment
with h5py.File('data/hii/parsed/00042_003_02.h5', 'r') as f:
    wave = f['continuum/wave_grid'][:]
    names = f['line_names'][:].astype(str)
    g = f['grid_point_00042_003_02']
    print(dict(g.attrs))
    L_neb = g['continuum/nebular_emission_per_Msun'][:]
    L_lines = g['lines_emergent/fluxes'][:]
    print(f"f_esc: target {g.attrs['f_esc_target']:.3f}, measured {g.attrs['f_esc_meas']:.3f}")

    # units: of the root datasets on the datasets, of the grid_point members in the root legend
    print(f.attrs['units_schema'], f['continuum/wave_grid'].attrs['units'],
          f['continuum/wave_grid'].attrs['wavelength_medium'])
    units = dict(f['units'].attrs)
    print(units['continuum/nebular_emission_per_Msun'], units['dust_mass_per_Msun'], units['tau_SSP'])
    print(f['descriptions'].attrs['energy_balance_rel'])
```
