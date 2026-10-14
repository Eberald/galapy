# Validating the grid

CloudIA implements the module `utils/validate_grid.py` to decide whether a grid can be handed to the emulators.
It reads the **corpus file** written by [`galapy-merge-h5`](merge_grid.md) and, when they are still on disk, the
working directories of the CLOUDY runs, and submits them to a battery of **checks**. Each check tests one
property that a correct grid must have: a conservation law, an atomic-physics invariant, a known observational
locus, the consistency between two outputs of the same run, or the coverage of the parameter space.

The validation is the step that follows the merge in the production chain of a sector (see
[Running CLOUDY](run_hii.md)):

| Step | Command | Produces |
| :--- | :--- | :--- |
| 6 | `galapy-merge-h5 --hii` | `data/products/hii_grid.h5`, the corpus |
| 7 | `galapy-validate-grid --sector hii` | `data/validation/report_hii.json` and `report_hii.md` |

The checks fall in two families:

- **checks on the corpus**, which only need the merged file (plus, for some of them, the charter, the LHS sample
  and the SSP metadata). They run on every grid;
- **checks on the raw outputs**, whose name starts with `rw|`. They read the files that CLOUDY wrote in the
  working directories (`.out`, `.con`, `.lines`, `.dusa`, the deck `.in` and the staged SED) and verify what the
  parser does not store in the corpus. They run only if the working directories have been kept.

Nothing is modified: the corpus and the working directories are only read.

!!! warning
    In the current version every check is registered for the **HII** sector only. With `--sector pdr` no check
    runs, and the report is an empty `PASS`.

---

## Inputs

Only the sector is required. Every other input has a default path, and a missing input does not stop the
validation: the checks that need it end in `SKIP`, and the reason is written in the report.

| Input | Flag | Default | Read by |
| :--- | :--- | :--- | :--- |
| Corpus | `--data` | `data/products/<sector>_grid.h5` | Every check on the corpus; the `rw|` checks that compare raw outputs with stored values. |
| Grid charter | `--config` | `data/grids/<sector>/grid_<sector>.yaml`, else `data/grids/<sector>/<sector>.yaml`, `data/<sector>/grid_<sector>.yaml`, `grid_<sector>.yaml`, else the charter of the GalaPy database (`Nebular/Configs`) | Number of planned runs, axis bounds, required CLOUDY version, SSP library. |
| LHS sample | `--lhs` | `data/grids/<sector>/lhs_<sector>.npy`, else `data/grids/<sector>/lhs.npy`, `data/<sector>/lhs_<sector>.npy` | Uniformity of the sampling design. |
| SSP metadata | `--ssp-meta` | `data/cloudy_seds/ssp_metadata.json`, else `data/grids/<sector>/ssp_metadata.json`, `data/<sector>/ssp_metadata.json` | $Q_H$ of the SSP nodes. |
| Working directories | `--raw-workdirs` | `data/<sector>/work`, if it exists | Every `rw|` check, and the second half of `convergence`. |

The **planned runs** are the product of the LHS samples by the SSP nodes of the charter,
$N_{\rm planned} = n_{\rm samples} \times n_{\tau} \times n_Z$ (e.g. $9000 \times 8 \times 6 = 432\,000$), or the
value of `--n-expected`. They are the denominator of the convergence fraction and the multiplier of the runtime
projection: without them, a run that crashed before writing its fragment would go unnoticed.

The checks that inspect full spectra (`grain_continuum_sanity`, `line_free_continua`,
`transparency_outside_sed_support`) read them for a random subsample of at most `--max-spectra` converged
models, drawn with `--seed`, so that the same command always grades the same models.

---

## Command Line Interface (CLI)

The module exposes the CLI entrypoint `galapy-validate-grid`.

### Usage

```text
usage: galapy-validate-grid [-h] --sector {hii,pdr} [--data DATA] [--report REPORT] [--markdown MARKDOWN]
                            [--config CONFIG] [--lhs LHS] [--ssp-meta SSP_META] [--raw-workdirs RAW_WORKDIRS]
                            [--n-expected N_EXPECTED] [--max-spectra MAX_SPECTRA] [--seed SEED] [--strict]
```

### Options

| Flag | Argument | Description |
| :--- | :--- | :--- |
| `-h`, `--help` | None | Show help message and exit. |
| `--sector` | `{hii,pdr}` | Sector of the grid. **(Required)** |
| `--data`, `--corpus` | `DATA` | Corpus file. <br> Default: `data/products/<sector>_grid.h5`. |
| `--report` | `REPORT` | JSON report. <br> Default: `data/validation/report_<sector>.json`. |
| `--markdown` | `MARKDOWN` | Markdown report. <br> Default: `data/validation/report_<sector>.md`. |
| `--config` | `CONFIG` | Grid charter. <br> Default: see [Inputs](#inputs). |
| `--lhs` | `LHS` | Continuous LHS sample, before the crossing with the SSP nodes. <br> Default: see [Inputs](#inputs). |
| `--ssp-meta` | `SSP_META` | `ssp_metadata.json` written by `galapy-sed-cloudy-extract`. <br> Default: see [Inputs](#inputs). |
| `--raw-workdirs` | `RAW_WORKDIRS` | Directory of the kept CLOUDY working directories. <br> Default: `data/<sector>/work`. |
| `--n-expected` | `N_EXPECTED` | Planned runs. Overrides the value derived from the charter. |
| `--max-spectra` | `MAX_SPECTRA` | Models subsampled by the spectral checks. <br> Default: `2000`. |
| `--seed` | `SEED` | Seed of the subsample. <br> Default: `0`. |
| `--strict` | None | A blocking check that ends in `SKIP` fails the validation. |

---

## Statuses, blocking checks and report

Every check ends in one of five statuses:

| Status | Meaning |
| :--- | :--- |
| `PASS` | The property holds. |
| `FAIL` | The property is violated; the detail names the worst models. |
| `SKIP` | An input is missing (the detail says which), or no model falls in the evaluation window. |
| `PARTIAL` | Every part of the check that ran passed, but some parts were skipped. |
| `ERROR` | The check raised an exception; the error and the tail of the traceback are reported. |

A check is **blocking** when its failure means that the grid is unusable (wrong units, energy not conserved,
wrong input SED, corrupted outputs). The validation fails, with exit status `1`, if any blocking check ends in
`FAIL` or `ERROR`, or in `SKIP` under `--strict`. The non-blocking checks grade the physical plausibility of the
grid: their failure must be understood before training, but it does not stop the pipeline. A corpus that cannot
be opened is a failure by itself (`data_load`).

The JSON report records the sector, the corpus path, the number of models and of converged models, the overall
status, the list of failed blocking checks, the inputs that could not be loaded, the full table of thresholds
used, and for every check its name, status, blocking flag and a detail with the numbers behind the verdict. The
Markdown report presents the same content as a summary table (`Check`, `Status`, `Blocking`, `Summary`)
followed by the details of every check.

---

## The checks

| Check | Family | Blocking | Physical property tested |
| :--- | :--- | :---: | :--- |
| `units_legend` | corpus | yes | The corpus is in the units every check assumes. |
| `convergence` | corpus + raw | yes | The radiative transfer of almost every planned run converged. |
| `ionizing_photon_budget` | SSP metadata | yes | The SED files carry the ionizing photons of the SSP library. |
| `energy_balance` | corpus | yes | No cloud emits more energy than it absorbs. |
| `balmer_decrement` | corpus | | H$\alpha$/H$\beta$ follows recombination Case B. |
| `bpt_star_forming_locus` | corpus | | The models lie in the star-forming region of the BPT diagram. |
| `sampling_design_uniformity` | LHS | | Every axis of the design is uniformly covered. |
| `grain_continuum_sanity` | corpus | | The dust emits as grains in radiative equilibrium. |
| `line_free_continua` | corpus | | No line is left in the stored continua. |
| `oiii_doublet_ratio` | corpus | | [O III] 4959/5007 equals its atomic value. |
| `dust_mass_scaling` | corpus | | The dust mass follows the Strömgren scaling. |
| `old_node_ionizing_output` | SSP metadata | | Old SSP nodes emit no ionizing photons. |
| `escape_fraction_coverage` | corpus | | The measured escape fraction covers its range and tracks the target. |
| `depletion(jk)_axis_coverage` | corpus | | The converged models still cover the depletion axis uniformly. |
| `fuv_terms` | corpus | | The FUV transmitted to the PDR is grey across the Habing band. |
| `transparency_outside_sed_support` | corpus | yes | The transmission is a physical transmission. |
| `rw|cloudy_version_in_outputs` | raw | yes | Every model comes from the same CLOUDY version. |
| `rw|abundance_echo` | raw | yes | CLOUDY used the intended gas-phase chemistry. |
| `rw|line_raster_closure` | raw | | The line-only continuum holds exactly the listed line fluxes. |
| `rw|grain_abundance_file` | raw | yes | The dust mass is integrated over one cloud, once. |
| `rw|incident_continuum_matches_sed` | raw | yes | CLOUDY was illuminated by the intended SED shape. |
| `rw|line_list_complete` | raw | yes | Every requested line comes out of CLOUDY. |
| `rw|runtime_per_model` | raw | yes | The full grid fits in the computing budget. |
| `rw|energy_balance_from_raw_output` | raw | yes | Energy is conserved, and the parser measured it correctly. |

The sections below group the checks by the physics they test. The symbols are those of
[Parse HII](parse_hii.md): ${\rm col}_2$ is the incident field, ${\rm col}_3$ the transmitted one, ${\rm col}_4$ the
diffuse outward emission and ${\rm col}_9$ its line part, all in $\nu F_\nu$; every energy is an integral over
$d\ln\lambda$, i.e. $\int \nu F_\nu\, d\ln\lambda = \int F_\lambda\, d\lambda$.

### 1. Integrity and provenance of the grid

These checks do not test physics directly: they guarantee that the numbers the physical checks read are what
they claim to be.

**`units_legend`** *(blocking)*. The corpus declares its own units: a root legend giving the unit of every member
of the `grid_point_*` groups, and unit and wavelength-medium attributes on the root datasets (see
[Parse HII](parse_hii.md#1-the-fragment-job_idh5)). The check compares them, one by one, with the tables of
`galapy.internal.constants` from which the parser writes them. Every threshold of this page is expressed in those
units, so a difference (e.g. luminosities per cm² instead of per $M_\odot$, or line wavelengths in vacuum where
air is expected) would make every following verdict meaningless. A corpus written by a parser older than the
units has no legend, and the check is skipped.

- *Pass*: the same unit for every member and every root dataset, the same wavelength media.

**`convergence`** *(blocking)*. The deck runs CLOUDY with `iterate to convergence`: the optical depths of the lines
and of the continuum in the outward direction are known only after a first pass through the cloud, and the
iterations go on until they stop changing. A non-converged model therefore has inconsistent escape probabilities,
hence wrong line ratios and wrong cooling. Non-converged models are flagged, not discarded, by the runner, and
this check measures how many there are:

$$
f_{\rm conv} = \frac{N_{\rm converged}}{N_{\rm planned}}
$$

The denominator is the number of planned runs, not of fragments: a run that crashed or timed out has no
fragment, and counts as a failure. When the planned runs are unknown the fragments are used instead, and the
report warns that crashes are invisible.

On the kept working directories the check also verifies the bookkeeping: the convergence read from the main
output (absence of the caution *Iterate to convergence did not converge*) must agree with the content of
`converged.flag`, with the presence of the `NOT_CONVERGED` marker and with the `not_converged` attribute of the
corpus. Outputs that do not end with *Cloudy exited OK* are listed for inspection.

- *Pass*: $f_{\rm conv} > 99\%$, and no flag disagrees with its output.

**`rw|cloudy_version_in_outputs`** *(blocking)*. Atomic data, grain opacities, line labels and default physics
change between CLOUDY releases, so a grid run with two versions is not a single physical model. The banner of
every kept main output must contain the required version: the `cloudy_version` of the charter, else the one
recorded in the corpus, else the version required by CloudIA. Several thresholds of this page (e.g. the
[O III] ratio) are computed from the atomic data of that version.

- *Pass*: every output from the required version.

**`rw|line_list_complete`** *(blocking)*. The emulator learns a line vector of fixed length and order, that of the
master list (see [Line list](line_list.md)). CLOUDY may fail to find a requested line and leave it out of the
output, which would shift every following line by one place. For every kept run the number of lines requested
in the list file named by the deck must equal the number returned in the line output, and every returned flux
must be finite and non-negative. The lines of the main output that mention a line not found or unknown are
collected for inspection.

- *Pass*: requested = returned, with valid fluxes, in every run.

**`rw|grain_abundance_file`** *(blocking)*. The dust mass of a model is the sum over the zones of the dust mass
density times the zone thickness (see [Parse HII](parse_hii.md#4-derived-quantities)). The deck saves the
density with the keyword `last`, i.e. for the last iteration only. Without it every iteration would be appended
to the same file, and the integral would count the cloud as many times as there were iterations. The check
verifies that each file holds exactly one block of zones: a single `#Depth … total` header, rows of constant
width, at least two zones, depth strictly increasing from the illuminated face. A second block shows up as a
depth that restarts from zero. The file must then be accepted by the integrator of the parser, with a finite
and positive column of dust.

- *Pass*: every file.

**`rw|runtime_per_model`** *(blocking)*. A resource check rather than a physical one: the mean execution time
of the kept runs, multiplied by the planned runs, gives the projected cost of the full grid. The budget is
5400 core hours, i.e. about 45 s per model for the $432\,000$ models of the fiducial HII grid. It is meant to be
run on a pilot subset, before launching the production grid. Without the planned runs the check is skipped.

- *Pass*: projected cost $\le 5400$ core hours.

### 2. The stellar input

**`ionizing_photon_budget`** *(blocking)*. Every model is normalised through the hydrogen-ionizing photon rate of
its SSP node, $Q_H$: it sets the area $s_k$ that turns the per-cm² outputs of CLOUDY into luminosities per
$M_\odot$ formed (see [Parse HII](parse_hii.md#3-the-geometric-factor-s_k)), so an error on $Q_H$ propagates
linearly into every luminosity and every mass of the corpus. The rate is

$$
Q_H = \frac{1}{hc}\int_{\lambda_{\min}}^{911.76\,{\rm Å}} \nu L_\nu\, d\lambda ,
$$

the photons with energy above 1 Ryd, counted as CLOUDY counts them (log-log interpolation of the table, up to
the H threshold). The check compares, node by node, the $Q_H$ of the SED file written by
[`galapy-sed-cloudy-extract`](ssp_extraction.md) (stored in the SSP metadata) with the $Q_H$ recomputed directly
from the SSP library at the same age and metallicity. The extraction resamples and cuts the spectrum, and adds
nodes at the Lyman edge: a loss of ionizing photons there would make every model of the node too faint, with no
other visible symptom. Nodes whose library $Q_H$ is zero (old populations) cannot be graded in relative terms and
are listed apart.

- *Pass*: the worst node within 5%.

**`old_node_ionizing_output`**. Hydrogen-ionizing photons are produced by O and early-B stars, whose lifetimes are
at most a few tens of Myr. For each stellar metallicity the check takes the nodes closest to 70 and 100 Myr
(within a factor 1.5 in age) and divides their $Q_H$ by that of the youngest node. Beyond this age only hot
evolved stars (post-AGB stars, white dwarfs) contribute, at a level orders of magnitude below a young
population: a larger ratio points to nodes mixed up in age or metallicity, or to a defect of the library. The
check also confirms that the old nodes describe, as intended, the limit in which the HII region fades away.

- *Pass*: both ratios $< 1\%$ for every metallicity.

**`rw|incident_continuum_matches_sed`** *(blocking)*. The deck sets the intensity of the field through the
ionization parameter, so the absolute level of the incident spectrum is fixed by CLOUDY: what the run inherits
from the SED file is only its **shape**. The check verifies that CLOUDY was illuminated by the file named in the
`table SED` command of the deck, read as intended. The file is interpolated in log-log onto the CLOUDY mesh and
compared with the incident column ${\rm col}_2$ through two numbers that do not depend on the normalisation:

$$
\eta_{\rm ion} = \frac{\int_{\lambda < 911.6\,{\rm Å}} \nu F_\nu\, d\lambda}{\int \nu F_\nu\, d\ln\lambda}
\;\propto\; \frac{\rm ionizing\ photons}{\rm total\ energy},
\qquad
\eta_{\rm FUV} = \frac{\int_{\rm Habing} \nu F_\nu\, d\ln\lambda}{\int \nu F_\nu\, d\ln\lambda}
$$

The first measures the hardness of the ionizing spectrum, which sets the energy deposited per photoionization,
hence the electron temperature and the ionization structure of the gas. The second is the share of the energy in
the Habing band ($6$–$13.6$ eV, $911.76$–$2066$ Å), which feeds the FUV field of the photodissociation region.
Both are computed below $10\,\mu$m, since above it the incident column also holds the CMB of the run, which is
not part of the file. A mismatch means a wrong file staged, wrong unit keywords, an unwanted extrapolation, or a
file misread by CLOUDY.

- *Pass*: both numbers within 5% in every run.

### 3. Energy conservation

A cloud in steady state, with no internal energy source other than the negligible heating by cosmic rays, must
return all the energy it receives: what it does not transmit, it absorbs and re-emits. Three checks test this
budget from different angles.

**`energy_balance`** *(blocking)*. The reprocessing budget stored by the parser compares the energy the cloud
emits with the energy it absorbs, $E_{\rm abs} = E_2 - E_3$. The check grades the **observable** excess, the
emission at $\lambda \ge 911.76$ Å in excess of the absorbed energy:

$$
\Delta_{\rm obs} = \frac{E_4(\lambda \ge 911.76\,{\rm Å}) - E_{\rm abs}}{E_{\rm abs}}
= \texttt{reprocessing\_excess} - \texttt{lyc\_emit\_fraction}
$$

The diffuse Lyman continuum is left out on purpose. In the closed `sphere` geometry CLOUDY counts its inward half
twice, a known excess that the parser corrects separately (see
[Parse HII](parse_hii.md#4-derived-quantities)); the test makes sure that nothing else creates energy, in the
part of the spectrum that the emulators actually learn. The test is one-sided: it guards against the spurious
creation of energy (e.g. a double-counted component), which would inflate every emulated luminosity.

Only converged models that absorb at least $2\times10^{-3}$ of their incident energy are graded. In thinner models
$E_{\rm abs}$ is the difference of two nearly equal numbers printed with four significant digits, and the ratio is
dominated by the print precision; the cut also leaves out the models whose incident field is dominated by the
CMB, which they absorb and re-emit in balance. The percentiles of the total excess and of the Lyman-continuum
fraction are reported along with the worst models.

- *Pass*: $\Delta_{\rm obs} \le 2\%$ for at least 95% of the graded models.

**`rw|energy_balance_from_raw_output`** *(blocking)*. The global budget, recomputed from scratch on the continuum
output of every kept run:

$$
\epsilon = \frac{\left| I_2 - (I_3 + I_4) \right|}{I_2^{\star}},
\qquad
I_n = \int {\rm col}_n\, d\ln\lambda,
\qquad
I_2^{\star} = \int_{\lambda < 10\,\mu{\rm m}} {\rm col}_2\, d\ln\lambda
$$

i.e. the incident energy must equal the transmitted plus the emitted one, referred to the **stellar** incident
energy (the CMB part of ${\rm col}_2$ balances itself, and referred to it the imbalance would be blind to an error
on the stellar component). The integral is taken over $d\ln\lambda$, as an energy, because photons are not
conserved: dust turns every absorbed UV photon into many IR photons. The check then compares the recomputed
$\epsilon$ with the `energy_balance_rel` attribute stored in the corpus, which guards against errors of the
parser (wrong column, reversed wavelength order, wrong quadrature). A run without stellar incident energy means
that the cloud was not illuminated at all.

- *Pass*: median $\epsilon < 1\%$, every stored attribute within $2\times10^{-3}$ of the recomputed value, no
  run without incident energy.

**`transparency_outside_sed_support`** *(blocking)*. The stored transmission $T(\lambda) = {\rm col}_3/{\rm col}_2$
is the fraction of the incident beam that crosses the cloud without being absorbed or scattered out. The
transmitted column holds no diffuse emission, so a passive medium can only attenuate it: $0 \le T \le 1$. A value
above 1 would mean that emission leaked into the transmitted column, or that the columns are misaligned. Where
the incident field is zero, outside the wavelength support of the SED (`sed_support_A`), $T$ carries no
information and is set by convention to 1, the neutral value: GalaPy multiplies $T$ onto SEDs whose support can
differ from that of the run, and any other value there would add or remove light that the cloud never saw. On
the subsample of converged models, the check verifies both conditions bin by bin.

- *Pass*: no bin off by more than $10^{-6}$, inside or outside the support.

### 4. Atomic physics and nebular diagnostics

**`balmer_decrement`**. In an HII region optically thick to the Lyman lines (*Case B* of recombination), every
Lyman photon is absorbed on the spot and eventually converted into a Balmer photon plus Ly$\alpha$; the ratio of
the Balmer lines then depends only weakly on the electron temperature and density. At $T_e = 10^4$ K and
$n_e = 10^2\,{\rm cm^{-3}}$,

$$
\left(\frac{{\rm H}\alpha}{{\rm H}\beta}\right)_{\rm B} = 2.86
$$

(Storey & Hummer 1995; Osterbrock & Ferland 2006), from 3.04 at 5000 K to 2.75 at 20 000 K. The check takes the
converged models around the typical conditions of Galactic HII regions, $|\log U + 3| \le 0.25$, and at
$z_{\rm CMB} \le 1$, where the CMB does not alter the thermal balance, and compares the **median** emergent ratio
with Case B. The window spans all the other axes (density, metallicity, dust), so the median averages over the
temperatures set by metallicity cooling and over the internal attenuation by dust, which raises the emergent
ratio. A larger deviation would point to a wrong temperature structure, to excessive dust attenuation, or to
broken radiative transfer of the hydrogen lines.

- *Pass*: $|{\rm median}/2.86 - 1| < 5\%$.

**`oiii_doublet_ratio`**. The [O III] $\lambda\lambda$4959, 5007 lines share the same upper level, $^1D_2$, and are
optically thin. Their ratio is therefore independent of temperature, density and abundance, and fixed by atomic
physics alone, by the ratio of the Einstein coefficients and of the photon energies:

$$
\frac{F_{4959}}{F_{5007}} = \frac{A_{4959}}{A_{5007}}\,\frac{\lambda_{5007}}{\lambda_{4959}} = 0.346
$$

with the atomic data of CLOUDY C25. Differential dust attenuation over 50 Å is negligible. Since nothing physical
can move this ratio, the check is graded on the **worst** converged model, not on a median: any deviation means
that a line was misidentified in the master list, mislabelled, blended with something else, or corrupted
between CLOUDY and the corpus.

- *Pass*: worst model within 1% of 0.346.

**`bpt_star_forming_locus`**. The Baldwin–Phillips–Terlevich diagram, $x = \log([{\rm N\,II}]\,6584/{\rm H}\alpha)$
versus $y = \log([{\rm O\,III}]\,5007/{\rm H}\beta)$, separates gas photoionized by stars from gas ionized by
harder sources. Two demarcations are used:

$$
y_{\rm K01} = \frac{0.61}{x - 0.47} + 1.19,
\qquad
y_{\rm K03} = \frac{0.61}{x - 0.05} + 1.30
$$

The Kewley et al. (2001) line is a **theoretical** upper envelope: the maximum excitation that photoionization by
stellar populations can reach. The Kauffmann et al. (2003) line is the **empirical** upper edge of the SDSS
star-forming galaxies. A model counts as below a line only on its left branch, $x < x_0$. Since every model of
the grid is ionized by a stellar population, it must lie below the Kewley line: a model above it would require
an ionizing spectrum harder than stars can produce, i.e. a wrong SED shape, wrong atomic data or a wrong
nitrogen abundance. The test is restricted to $\log U \le -2$, the range over which the Kewley line was derived.
The fraction below the Kauffmann line is reported but not graded: a grid that spans extreme metallicities,
densities and ionization parameters legitimately populates the region between the two lines.

- *Pass*: at least 95% of the converged models with $\log U \le -2$ below the Kewley et al. (2001) line.

### 5. The stored continua

**`grain_continuum_sanity`**. The diagnostic grain continuum is the emissivity of the last zone times the cloud
thickness (see [Parse HII](parse_hii.md#4-derived-quantities)). On the subsample of converged models it must
look like the emission of grains in radiative equilibrium:

- (a) **no negative value**: an emissivity cannot be negative;
- (b) **peak of $\nu F_\nu$ between 5 and 500 µm**. For a modified black body with emissivity index $\beta = 2$
  the peak falls at $\lambda_{\rm peak} \simeq 2400\,\mu{\rm m}\,/\,(T_d/{\rm K})$, so the window corresponds to
  $T_d \approx 5$–$500$ K, which contains every equilibrium temperature expected in and around an HII region
  (typically a few tens of K);
- (c) **no ultraviolet emission**: below 0.3 µm the grain emission must stay below $10^{-3}$ of the peak.
  Thermal emission there would require $T_d \gtrsim 10^4$ K, far above sublimation. The HII decks contain no PAHs
  and switch off the stochastic heating of grains (`no grain qheat`), so not even transient temperature spikes
  are possible: a UV signal means leaked stellar light or a column mix-up;
- (d) **far-IR scale**: above 20 µm the diffuse continuum of the cloud is dominated by dust, so the energy of the
  grain continuum there must be of the same order as that of the stored nebular continuum. The ratio is not
  expected to be one, since the last zone is not at the mean temperature of the cloud, but a ratio far from
  unity would reveal a wrong thickness or a wrong scaling of the diagnostic.

- *Pass*: (a)–(c) for at least 99% of the sample, and the median far-IR ratio in $[0.1, 10]$.

**`line_free_continua`**. The lines are stored separately, as integrated luminosities; the nebular continuum
(${\rm col}_4 - {\rm col}_9$, see [Parse HII](parse_hii.md#4-derived-quantities)) and the grain continuum must
therefore contain no line at all, or the line energy would be counted twice when the emulated SED is assembled.
Each line of the master list is placed on the continuum mesh at its **vacuum** wavelength (the mesh is in vacuum;
the air wavelength of H$\alpha$ is off by 1.8 Å, enough to miss its bin). For every model the peak over the line
bin and its two neighbours is compared with two local baselines, the medians of three bins on each side,
separated from the line by a gap of two bins. The line is flagged if the peak exceeds the **larger** baseline by
more than 10%. Taking the larger of the two makes the test insensitive to continuum edges: at a recombination
edge, such as the Balmer jump, the continuum is a step, and the peak does not rise above the higher side. The
grain continuum is pure dust emission, so in it any spike is a defect.

- *Pass*: at most 1% of the (model, line) pairs flagged, separately for the two continua.

**`rw|line_raster_closure`**. CLOUDY returns the lines in two forms: integrated, in the line list, and rasterized
on the continuum mesh, in the line-only column ${\rm col}_9$. The parser builds the nebular continuum by
subtracting ${\rm col}_9$ from ${\rm col}_4$, which is correct only if ${\rm col}_9$ contains exactly the energy of
the lines. For every listed line of every kept run, the energy of ${\rm col}_9$ is summed over $\pm 3$ bins
around the line (energy of a cell: value times cell width in $\ln\lambda$) and divided by the sum of the listed
fluxes of all the lines whose bins fall in the same window, to account for neighbouring lines that share it. The
listed wavelengths are converted from air to vacuum before being placed on the mesh. The raster must hold at
least the listed energy; it may hold more, because ${\rm col}_9$ also contains the lines that are not in the
master list, so the upper side is graded only for strong lines, at least 10% of H$\beta$, for which unlisted
weak lines are negligible.

- *Pass*: raster/listed $\ge 0.98$ for every line, and $\le 1.02$ for the lines above $0.1\,{\rm H}\beta$.

### 6. Dust and chemistry

**`dust_mass_scaling`**. The dust mass of a model can be predicted from the Strömgren sphere that defines its
stopping criterion (see [Sampling](gen_hii.md#derived-quantities)). The cloud stops at the hydrogen column
$N_{\rm stop} = (1 - f_{\rm esc})\, U c / \alpha_B$, and its illuminated face, per $M_\odot$ of SSP, has the area
$s_k = Q_H / (U n_{\rm H} c)$. The number of hydrogen atoms per $M_\odot$ formed is then

$$
N_{\rm stop}\, s_k = \frac{(1 - f_{\rm esc})\, Q_H}{\alpha_B\, n_{\rm H}} ,
$$

in which the ionization parameter cancels. With a dust-to-gas ratio proportional to $\xi_d Z$, the dust mass per
$M_\odot$ formed scales as

$$
\frac{M_d}{M_{\rm SSP}} \;\propto\; (1 - f_{\rm esc})\, \frac{Z}{Z_{\rm GC}}\, g(\xi_d)\, \frac{Q_H}{n_{\rm H}},
\qquad g(\xi_d) = \frac{\xi_d}{\xi_{d,{\rm MW}}}
$$

with $Z(\zeta_O)$ and $g$ from the [chemistry module](abundances.md) and $Q_H$ from the SSP node of the model. The
constant of proportionality ($\alpha_B$, the mean mass per H atom, the conversion between metal mass and grain
mass) is not computed: it is fitted as the median residual, and the check grades the scatter around it. This
tests at once the integration of the dust column, the normalisation $s_k$ and the grain scaling of the decks: a
factor two from a duplicated block of zones, or a wrong $s_k$, would move whole groups of models off the relation.

The relation is not exact. The cloud may stop at $T = 4000$ K before reaching $N_{\rm stop}$ (efficient cooling at
high metallicity); at high $U$ dust competes with the gas for the ionizing photons and shrinks the ionized
column; and $\alpha_B$ depends on the electron temperature, which varies with metallicity. Hence the loose
tolerance, and the report of the rank correlation between the residual and $\log U$, which shows how much of
the scatter is due to dust absorption.

- *Pass*: at least 90% of the converged models within 0.2 dex of the fitted relation.

**`rw|abundance_echo`** *(blocking)*. CLOUDY prints in its main output the gas-phase composition it actually used,
after the `metals`, `element scale factor` and `metals deplete jenkins 2009` commands of the deck. The check reads
this echo for every kept run and verifies:

1. **thirty elements**, H to Zn. An element absent from the composition is switched off silently by CLOUDY,
   removing its lines and its contribution to the cooling;
2. **helium**: He/H equal, within 0.01 dex, to the value of the Galactic Concordance pattern at the $\log\zeta_O$ of
   the run (Nicholls et al. 2017). Helium is a noble gas and is not depleted onto grains, so its gas-phase
   abundance must coincide with the total one;
3. **no element above its total abundance**: depletion can only move atoms from the gas into the grains, so no
   gas-phase abundance may exceed the undepleted pattern (beyond the print precision, $10^{-4}$ dex);
4. **relative pattern at fixed depletion**: in the Jenkins (2009) prescription the depletion of each element is a
   fixed offset set by $F_*$. Among runs with the same $F_*$, the difference between the echo of a run and that
   of the run closest to $\log\zeta_O = 0$ must therefore equal the difference of the logarithmic abundance
   patterns: the depletion cancels, and the scaling of the abundances with metallicity is verified without
   re-implementing CLOUDY's depletion table.

Items 2–4 need the chemistry module; when it is not available only the element count is checked.

- *Pass*: 1–4 in every run, item 4 within $10^{-3}$ dex plus one print step. `PARTIAL` when items 2–4 could not
  be graded.

!!! note
    Item 4 needs runs with the same $F_*$ and different $\zeta_O$. In a continuous LHS the only runs that share
    $F_*$ are the SSP nodes of the same sample, which also share $\zeta_O$: on a production grid item 4 does not
    run and the check ends `PARTIAL`. It is graded on dedicated test grids with repeated values of $F_*$.

### 7. Coverage of the parameter space

An emulator interpolates: it is reliable only where the grid has models. These checks verify that the design
covers the space as planned, and that the losses (non-converged runs) did not carve holes in it.

**`sampling_design_uniformity`**. The continuous LHS sample, before its crossing with the SSP nodes, is compared
axis by axis with a uniform distribution between the bounds declared in the charter, with a Kolmogorov–Smirnov
test. A Latin hypercube is stratified by construction, so its marginals are uniform to within one stratum: a low
$p$-value cannot be a statistical fluctuation, and points to a sampling bug, to axes in a different order from the
charter, or to a charter different from the one used to sample. Values outside the bounds are counted, and a
sample with a number of columns different from the number of axes fails at once. The test assumes the `uniform`
priors of the fiducial charter.

- *Pass*: every $p$-value $> 0.01$, and no value outside the bounds.

**`escape_fraction_coverage`**. The target escape fraction sets the stopping column through the Strömgren
estimate; the measured one, $f_{\rm esc}^{\rm meas}$, is the fraction of ionizing photons that actually crosses the
cloud (see [Parse HII](parse_hii.md#4-derived-quantities)). The two differ, since the Strömgren estimate ignores
the absorption by dust and helium, the diffuse field and the temperature dependence of $\alpha_B$. The check does
not require them to be equal, but requires:

- **coverage**: the measured values of the converged models, inside $[0, 0.9]$, must leave no empty interval
  wider than 0.1, edges included. Dust absorption systematically lowers $f_{\rm esc}^{\rm meas}$, and could leave
  the high-escape end of the range empty;
- **monotonicity**: the Spearman rank correlation between target and measured values must exceed 0.9, i.e. the
  target must still control the escape of the photons, even if not exactly.

- *Pass*: widest gap $\le 0.1$ and rank correlation $> 0.9$.

**`depletion(jk)_axis_coverage`**. The depletion strength $F_*$ is sampled uniformly, but the models that survive
are only the converged ones: if the convergence failures depended on $F_*$ (e.g. strongly depleted or undepleted
gas with extreme cooling), the surviving distribution would be biased, and the emulator under-trained in part
of the axis. The check applies a Kolmogorov–Smirnov test of uniformity to the $F_*$ of the converged models,
between the bounds of the charter. Each LHS draw is repeated over all the SSP nodes, so the models are not
independent draws: the significance is computed with the number of **distinct** values of $F_*$, not with the
number of models, which would overstate it by the number of nodes.

- *Pass*: $p$-value $> 0.01$.

### 8. The interface with the PDR

**`fuv_terms`**. The HII region illuminates the photodissociation region behind it with the FUV it transmits. The
PDR grid describes that field with a scalar, $G_0$, scaled by the FUV transmittance $T_{\rm FUV}$ of the ionized
layer and applied to the shape of the stellar spectrum. This is correct only if the transmission is **grey**
across the Habing band. The check verifies:

- (a) **signs**: $0 < T_{\rm FUV} \le 1$ and $N_{\rm FUV} > 0$ for every converged model. The ionized layer cannot
  amplify the incident FUV, and always emits some FUV of its own (two-photon continuum, Ly$\alpha$);
- (c) **spectral tilt**: in the models that transmit enough to matter, $T_{\rm FUV} > 10^{-2}$, $\log_{10} T(\lambda)$
  is fitted with a straight line in $\log_{10}\lambda$ across the band, and the tilt is the difference between
  its values at the red and blue edges,

  $$
  \Delta_{\rm tilt} = \frac{d\log_{10} T}{d\log_{10}\lambda}\;\log_{10}\frac{2066\,{\rm Å}}{911.76\,{\rm Å}}
  \qquad [{\rm dex}]
  $$

  The bins of the H I Lyman series (Ly$\alpha$ to $n = 40$) are excluded: they are narrow absorption features,
  not the continuum shape. Dust extinction rises towards short wavelengths in the FUV, so a dusty ionized layer
  reddens the field it transmits.

A tilt above 0.1 dex (26% between the two edges) is not a bug but a modelling decision: the scalar $G_0$ is then
not enough, and the PDR grid needs an additional axis for the tilt of the incident FUV.

- *Pass*: (a) for every model, and $\max|\Delta_{\rm tilt}| < 0.1$ dex.

---

## Thresholds

All the thresholds are collected in a single table of the module, and copied into every JSON report, so that a
verdict can always be traced back to the numbers that produced it.

| Check | Threshold | Value |
| :--- | :--- | :--- |
| `convergence` | Minimum converged fraction of the planned runs | $0.99$ |
| `energy_balance` | Minimum absorbed fraction for a model to be graded | $2\times10^{-3}$ |
| | Maximum observable excess | $0.02$ |
| | Minimum fraction of graded models within it | $0.95$ |
| `rw|energy_balance_from_raw_output` | Maximum median imbalance | $0.01$ |
| | Maximum difference between stored and recomputed imbalance | $2\times10^{-3}$ |
| `ionizing_photon_budget` | Maximum relative difference of $Q_H$ | $0.05$ |
| `balmer_decrement` | Case B reference, maximum relative deviation | $2.86$, $0.05$ |
| | Window | $\log U = -3 \pm 0.25$, $z_{\rm CMB} \le 1$ |
| `bpt_star_forming_locus` | Kewley et al. (2001), Kauffmann et al. (2003) $(a, x_0, b)$ | $(0.61, 0.47, 1.19)$, $(0.61, 0.05, 1.30)$ |
| | Maximum $\log U$, minimum fraction below Kewley | $-2$, $0.95$ |
| `sampling_design_uniformity` | Minimum KS $p$-value | $0.01$ |
| `grain_continuum_sanity` | Peak window, UV limit, maximum UV/peak | $5$–$500\,\mu$m, $0.3\,\mu$m, $10^{-3}$ |
| | Far-IR limit, window of the median ratio | $20\,\mu$m, $[0.1, 10]$ |
| | Minimum passing fraction | $0.99$ |
| `line_free_continua` | Spike excess, maximum flagged fraction | $0.10$, $0.01$ |
| `rw|line_raster_closure` | Half window, maximum deviation, strong-line limit | $3$ bins, $0.02$, $0.1\,{\rm H}\beta$ |
| `oiii_doublet_ratio` | Reference, maximum relative deviation | $0.346$, $0.01$ |
| `dust_mass_scaling` | Maximum residual, minimum fraction within it | $0.2$ dex, $0.9$ |
| `old_node_ionizing_output` | Ages, maximum $Q_H$ ratio to the youngest node | $70$, $100$ Myr, $0.01$ |
| `escape_fraction_coverage` | Range, maximum gap, minimum Spearman correlation | $[0, 0.9]$, $0.1$, $0.9$ |
| `depletion(jk)_axis_coverage` | Minimum KS $p$-value | $0.01$ |
| `fuv_terms` | Minimum $T_{\rm FUV}$ for the tilt, maximum tilt | $10^{-2}$, $0.1$ dex |
| `transparency_outside_sed_support` | Tolerance on $T$ | $10^{-6}$ |
| `rw|abundance_echo` | Elements, print step, He/H tolerance, pattern tolerance | $30$, $10^{-4}$ dex, $0.01$ dex, $10^{-3}$ dex |
| `rw|incident_continuum_matches_sed` | Tolerance on the two shape numbers | $0.05$ |
| `rw|runtime_per_model` | Budget | $5400$ core hours |
