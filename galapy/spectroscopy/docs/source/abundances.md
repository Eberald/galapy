# Chemical abundances and the $\zeta_O$–$Z$ map

CloudIA implements the module `utils/physics/abundances.py` to manage the gas-phase chemistry used by the
HII-region and PDR models. The module encodes the *Galactic Concordance* (GC) abundance set of
**Nicholls et al. (2017)** and its scaling with metallicity, translating a single scaling parameter,
$\zeta_O$, into: the full 30-element abundance pattern, the gas mass fraction $Z_{\rm gas}$, the grain scaling
factor, and the `element scale factor ...` directives injected into the CLOUDY input decks.

The module is accessible both as a Command Line Interface (CLI), to (re)build the $\zeta_O \leftrightarrow Z_{\rm gas}$
lookup table, and programmatically through the `Chemistry` Python API.

!!! warning
    The chemistry files live in the galapy-dataset, not in the code.

---

## The scaling parameter $\zeta_O$

$\zeta_O$ is the linear scaling of the oxygen abundance with respect to the Galactic Concordance fiducial point,
i.e. $\zeta_O = 1$ ($\log \zeta_O = 0$) corresponds to $12 + \log({\rm O/H}) = 8.760$ and $Z_{\rm GC} = 0.014254$.
Every method of the API takes and returns the **base-10 logarithm** $\log_{10}\zeta_O$, which is validated to lie
in the interval $[-3, +1]$.

The abundance of each element $X$ relative to hydrogen follows Eq. 7 of Nicholls+17:

$$
\left(\frac{X}{\rm H}\right) = \left(\frac{X}{\rm H}\right)_{0} \times
10^{\,\Delta_O(X) \,+\, \log\zeta_O}
$$

where $(X/{\rm H})_0$ is the fiducial GC value and $\Delta_O(X)$ is the depletion/enrichment offset in dex.
Hydrogen is fixed to 1 by construction and helium does **not** carry the uniform $\log\zeta_O$ term.

$\Delta_O(X)$ is piecewise-linear in $\log\zeta_O$, with slope set by the tabulated $\xi_O(X)$ (Table 2 of
Nicholls+17) and the two breakpoints $\chi_0 = -0.50$, $\chi_1 = +0.25$:

$$
\Delta_O(X) =
\begin{cases}
\xi_O(X), & \log\zeta_O < \chi_0 \\[4pt]
\dfrac{\xi_O(X)}{\chi_0}\,\log\zeta_O, & \chi_0 \le \log\zeta_O \le \chi_1 \\[8pt]
\dfrac{\xi_O(X)}{\chi_0}\,\chi_1, & \log\zeta_O > \chi_1
\end{cases}
$$

Two elements are treated separately (`special` block of the YAML):

| Element | Prescription | Formula |
| :--- | :--- | :--- |
| **He** | `he_enrichment` (Eq. 4) | ${\rm He/H} = A\,(1 + B\,\zeta_O)$, with $A = 8.3503\times10^{-2}$, $B = 0.17031$ |
| **N** | `primary_secondary` (Eq. 9) | $\Delta_O({\rm N}) = \log_{10}\!\left(10^{a} + 10^{\,\log\zeta_O + b}\right)$, with $a = -0.764$, $b = -0.082$ |

From the abundance pattern the code derives the **absolute** gas mass fraction

$$
Z_{\rm gas}(\zeta_O) = \frac{\sum_{X \neq {\rm H,He}} A_X\, n_X}{\sum_{X} A_X\, n_X}
$$

with $A_X$ the atomic weights tabulated in `galapy.internal.constants.ATOMIC_WEIGHT`. Note that $Z_{\rm gas}$ is
an *absolute* mass fraction: no solar scaling is involved. The conversion to solar units is an explicit,
opt-in step (`Chemistry.to_solar_units`).

---

## Command Line Interface (CLI)

The relation $Z_{\rm gas}(\log\zeta_O)$ has no closed-form inverse, so the inverse mapping
$\log\zeta_O(Z_{\rm gas})$ is tabulated once on a dense grid and interpolated at run time. The module exposes
the CLI entrypoint `galapy-build-zeta-map` to generate that table.

### Usage

```text
usage: galapy-build-zeta-map [-h] -o OUT [-a ABN] [-s SCALE]
                             [-l LZ_MIN] [-u LZ_MAX] [-n N]
```

### Options

| Flag | Argument | Description                                                                                                   |
| :--- | :--- |:--------------------------------------------------------------------------------------------------------------|
| `-h`, `--help` | None | Show help message and exit.                                                                                   |
| `-o`, `--out` | `OUT` | Output path of the `.npz` map. Should point to `zeta_Z.npz` inside the dataset directory. **(Required)**: there is no default, so that the database map is never overwritten by accident. |
| `-a`, `--abn` | `ABN` | Path to the fiducial abundance file. <br> Default: `None`, i.e. `GC.abn` resolved from the GalaPy database.    |
| `-s`, `--scale` | `SCALE` | Path to the scaling YAML file. <br> Default: `None`, i.e. `scaling_abd_Z.yaml` from the GalaPy database.      |
| `-l`, `--lz-min` | `LZ_MIN` | Lower edge of the $\log\zeta_O$ grid. <br> Default: `-3.0`.                                                   |
| `-u`, `--lz-max` | `LZ_MAX` | Upper edge of the $\log\zeta_O$ grid. <br> Default: `+1.0`.                                                   |
| `-n`, `--n` | `N` | Number of grid points. Must be $\ge 2$. <br> Default: `4001`.                                                  |

The defaults are collected in the module-level constant `ZMAP_DEFAULT_GRID = (-3.0, 1.0, 4001)`, which also
sets the range enforced by the internal $\log\zeta_O$ validation. Shrinking the grid below the default number
of points is allowed but triggers a warning, since the inversion accuracy degrades.

### Example

Rebuild the map in place, after having edited the chemistry files in the dataset:

```bash
galapy-build-zeta-map -o $HOME/.galapy/galapy_database/Nebular/Abundances/zeta_Z.npz
```

---

## Output Structure and File Formats

### 1. The `zeta_Z.npz` map

A compressed NumPy archive with four arrays:

| Key | Type | Description |
| :--- | :--- | :--- |
| `log_zeta_O` | `float64[n]` | Uniform grid of $\log_{10}\zeta_O$ between `lz_min` and `lz_max`. |
| `Z_gas` | `float64[n]` | Corresponding absolute gas mass fraction $Z_{\rm gas}$. |
| `sha_abn` | `str` | SHA-256 of the `.abn` file used to build the map. |
| `sha_scaling` | `str` | SHA-256 of the scaling YAML used to build the map. |

`build_zeta_map` verifies that `Z_gas` is **strictly increasing** before writing: a non-monotonic branch would
make the inversion ambiguous and raises `ValueError`. The file is written to a `.tmp` companion and then
atomically renamed, so a failed run never leaves a half-written map behind.

### 2. The integrity check

At construction time `Chemistry` re-computes the SHA-256 of the abundance and scaling files it just read and
compares them against the hashes stored in the map. On mismatch it raises `ChemistryHashMismatch`, the
module's dedicated `RuntimeError` subclass:

```text
ChemistryHashMismatch: zeta_Z.npz generated from chemistry files (.abn and .yaml) different from
the loaded ones. You can re-generate it via cli command `galapy-build-zeta-map`.
```

This guarantees that an edited chemistry can never be silently paired with a stale inverse map.

### 3. The fiducial `.abn` file

The parser of `GC.abn` is deliberately strict, because CLOUDY itself is:

- lines starting with `#` are comments, a line starting with `*` terminates the table;
- **empty lines are rejected** — CLOUDY aborts on them;
- the `GRAINS` keyword is rejected, since grains are already handled by the template (double counting);
- element names must belong to `galapy.internal.constants.CLOUDY_NAME`, and abundances must be positive;
- exactly **30 elements** are required; anything else means some element would be silently switched off.

Abundances are normalised to hydrogen on read, so `fiducial['H'] == 1.0` always.

### 4. CLOUDY chemistry directives

`Chemistry.element_scale_lines` emits the block injected into the CLOUDY deck. Only elements with a
non-negligible $\xi_O$ (plus the two special cases He and N) get a line: elements with $\xi_O = 0$ already
scale with $\zeta_O$ alone and need no explicit correction.

```text
element scale factor helium      +0.012473 log
element scale factor nitrogen    -0.681204 log
element scale factor carbon      -0.437000 log
element scale factor sodium      -0.300000 log
...
```

`Chemistry.write_abn_file` is the companion *validation* tool: instead of scale factors it writes a complete
`.abn` file with the abundances already scaled to a given $\log\zeta_O$, useful to cross-check the deck
directives against a directly-tabulated chemistry.

### 5. Grains

The full argument of the CLOUDY `grains ISM` command factorises as

$$
{\rm grain\_fac} = \frac{Z(\zeta_O)}{Z_{\rm GC}} \times g(\xi_d),
\qquad
g(\xi_d) = \frac{\xi_d}{\xi_{d,{\rm MW}}}
$$

with $\xi_{d,\rm MW} = 0.43$ (`CONST.XI_D_MW`), obtained from the Milky Way dust-to-gas ratio
$(D/G)_{\rm MW} = 1/162$ (Zubko+04) and $Z^{\rm MW}_{\rm GC} = 0.014254$ (Nicholls+17).
`grain_scale_from_xi_d` returns **only** the $\xi_d$-dependent factor $g(\xi_d)$; the metallicity term must be
multiplied in by the caller, via `Chemistry.metallicity_from_zeta`.

---

## Solar references

`Chemistry.to_solar_units` converts an absolute $Z_{\rm gas}$ into $\log_{10}(Z_{\rm gas}/Z_\odot)$ against one
of the following references:

| Reference | $Z_\odot$ | Note |
| :--- | :--- | :--- |
| `'GC'` (default) | `0.014254` | Derived at run time as `metallicity_from_zeta(0.0)`, i.e. self-consistent with the loaded chemistry. |
| `'GASS10'` | `0.013370` | Grevesse et al. 2010. |
| `'Caffau11'` | `0.015300` | Caffau et al. 2011, used in Ronconi+24. |
| `'PARSEC'` | `0.015240` | Scale of the PARSEC SSP libraries — use this one when comparing to GalaPy's stellar metallicities. |

---

# Python API

::: galapy.spectroscopy.utils.physics.abundances.ChemistryHashMismatch
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.physics.abundances.Chemistry
    options:
      show_root_heading: true
      show_source: false
      members:
        - delta_O
        - abundance_pattern
        - metallicity_from_zeta
        - zeta_from_metallicity
        - oh_from_zeta
        - zeta_from_oh
        - to_solar_units
        - element_scale_lines
        - write_abn_file

::: galapy.spectroscopy.utils.physics.abundances.build_zeta_map
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.physics.abundances.grain_scale_from_xi_d
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.physics.abundances.format_cloudy_float
    options:
      show_root_heading: true
      show_source: false
