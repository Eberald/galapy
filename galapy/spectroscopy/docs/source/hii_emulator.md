# HII region: training the emulator

CloudIA implements the module `utils/hii/train_emulator_hii.py`, run by the console script `galapy-train-hii`, to
train the **HII emulator**: a set of Sparse Variational Gaussian Processes (SVGPs) that replace CLOUDY inside a
fit. Given the 9 inputs of an HII model, the emulator returns, per solar mass of SSP formed, the emergent lines,
the nebular and grain continua, the transmission, the dust mass and the FUV terms, each with its predictive
uncertainty.

Three more modules complete the emulator:

| Module | Role |
| :--- | :--- |
| `svgp.py` | The SVGP model shared by the HII and PDR sectors, its closed-form prediction and the decoding to physical units. |
| `hii_emulator.py` | `HIIEmulator`, the inference class: it loads the artifacts and predicts batches of models. |
| `utils/validate_emulator.py` | `galapy-validate-emul`, the validation of a trained emulator on its holdout. |

The constants of the emulators (inputs, families, floors, calibration, validation thresholds) are collected in
`galapy.internal.emulators`, imported as `EMUL`.

!!! warning
    Training and inference need `torch`, `gpytorch` and `scikit-learn`, which are not dependencies of the core of
    GalaPy. The artifacts are always saved on CPU: an emulator trained on a GPU is used without one.

---

## Inputs

The training reads the corpus written by [`galapy-merge-h5 --hii`](merge_grid.md), by default
`data/products/hii_grid.h5`. `load_corpus` checks it before using it:

- the root group `units` must be exactly the table `CLOUDY.Points_Units_HII`: a corpus in other units is refused;
- the models flagged `not_converged` are excluded;
- every model must declare `pumping_removed`, with the same value for all of them: a corpus written by an older
  parser, or mixing models with and without the pumping removed, is refused;
- $Q_H$ per $M_\odot$ of every SSP node is recovered from the stored geometric factor,
  $Q_{H,\rm unit} = s_k\, U\, n_H\, c$, and must be the same for all the models of the node (relative spread
  $< 10^{-5}$). The table of the 48 nodes travels inside the artifacts, so the emulator does not need the SSP
  metadata at runtime.

The 9 inputs are `EMUL.INPUT_KEYS_HII`; the two SSP node coordinates enter in $\log_{10}$:

| Input | Meaning | Scale |
| :--- | :--- | :--- |
| `logU` | Ionization parameter at the illuminated face | as stored (dex) |
| `lognH_HII` | Hydrogen density | as stored (dex) |
| `log_zeta_O` | Oxygen abundance relative to GC, $\zeta_O$ | as stored (dex) |
| `z_CMB` | Redshift of the CMB of the run | linear |
| `xi_d` | Dust-to-metal mass ratio | linear |
| `f_esc_meas` | **Measured** escape fraction of the H-ionizing photons | linear |
| `F_star` | Depletion strength $F_*$ (Jenkins 2009) | linear |
| `tau_SSP` | Age of the SSP node | $\log_{10}$ (yr) |
| `Z_star` | Metallicity of the SSP node | $\log_{10}$ |

The emulator is trained on the *measured* escape fraction, not on the target of the sampling: the parser measures
it from the transmitted continuum, and it is the quantity a fit constrains.

The outputs are split into 7 **families**, `EMUL.FAMILIES_HII`, each trained as one batch of independent SVGPs:

| Family | Corpus dataset | Outputs | Compression | Inducing points |
| :--- | :--- | :--- | :--- | :--- |
| `lines` | `lines_emergent/fluxes` | one per line of the list | none | `--inducing-lines` |
| `cont` | `continuum/nebular_emission_per_Msun` | PCA components | PCA | `--inducing` |
| `grain` | `continuum/grain_diag_per_Msun` | PCA components | PCA | `--inducing` |
| `trans` | `continuum/transmission` | PCA components | PCA | `--inducing` |
| `dust` | `dust_mass_per_Msun` | 1 | none | `--inducing-lines` |
| `fuv_t` | `T_fuv_hii` | 1 | none | `--inducing-lines` |
| `fuv_n` | `N_fuv_hii` | 1 (offset $10^{-6}$, since it can be 0) | none | `--inducing-lines` |

---

## Command Line Interface (CLI)

The command `galapy-train-hii` runs the whole training: split, preprocessing, SVGPs, calibration and artifacts.

### Usage

```text
usage: galapy-train-hii [-h] [--corpus CORPUS] [--out OUT] [--meta META] [--inducing INDUCING]
                        [--inducing-lines INDUCING_LINES] [--var-target VAR_TARGET]
                        [--max-pca-components MAX_PCA_COMPONENTS]
                        [--rel-floor [FAM=]VALUE [[FAM=]VALUE ...]]
                        [--rel-floor-lyc [FAM=]VALUE [[FAM=]VALUE ...]] [--line-targets {ratio,absolute}]
                        [--ratio-floor RATIO_FLOOR] [--size-budget-mib SIZE_BUDGET_MIB] [--epochs EPOCHS]
                        [--batch-size BATCH_SIZE] [--lr LR] [--seed SEED] [--holdout-frac HOLDOUT_FRAC]
                        [--holdout-seed HOLDOUT_SEED] [--eval-every EVAL_EVERY] [--calib-frac CALIB_FRAC]
                        [--calib-families CALIB_FAMILIES] [--calib-quantile CALIB_QUANTILE]
                        [--device {auto,cpu,cuda}]
```

### Options

| Flag | Argument | Description |
| :--- | :--- | :--- |
| `-h`, `--help` | None | Show help message and exit. |
| `--corpus` | `CORPUS` | Merged HII corpus. <br> Default: `data/products/hii_grid.h5`. |
| `--out` | `OUT` | Stem of the artifacts: writes `<stem>_{lines,cont,grain,trans,scalars}.pt` and the companion files. <br> Default: `data/emulators/hii.pt`. |
| `--meta` | `META` | JSON manifest of the training. <br> Default: `data/emulators/hii_meta.json`. |
| `--inducing` | `INDUCING` | Inducing points of the PCA families. <br> Default: `512`. |
| `--inducing-lines` | `INDUCING_LINES` | Inducing points of the lines and of the scalars. <br> Default: `256`. |
| `--var-target` | `VAR_TARGET` | Variance the PCA components must explain. <br> Default: `0.9999`. |
| `--max-pca-components` | `MAX_PCA_COMPONENTS` | Total PCA components, shared among the PCA families. <br> Default: `100`. |
| `--rel-floor` | `[FAM=]VALUE ...` | Floor before the $\log_{10}$, relative to the peak of each model; `FAM=VALUE` sets one family, a bare `VALUE` all of them, `0` disables it. <br> Default: `lines=0 cont=1e-10 grain=1e-10 trans=1e-05` (`lines=1e-10` with `--line-targets absolute`). |
| `--rel-floor-lyc` | `[FAM=]VALUE ...` | Relative floor that replaces `--rel-floor` below the Lyman limit (911.76 Å) for the families on the wavelength grid. <br> Default: `cont=1e-05 grain=0 trans=0`. |
| `--line-targets` | `{ratio,absolute}` | Targets of the lines family: ratios to H$\beta$ (see below) or $\log_{10}$ of the fluxes. <br> Default: `ratio`. |
| `--ratio-floor` | `RATIO_FLOOR` | Minimum of $L_i/L_{{\rm H}\beta}$ with `--line-targets ratio` (`0` disables it). <br> Default: `1e-05` (`EMUL.RATIO_FLOOR`). |
| `--size-budget-mib` | `SIZE_BUDGET_MIB` | Maximum size of every artifact file, in MiB. <br> Default: `100`. |
| `--epochs` | `EPOCHS` | Training epochs of every family. <br> Default: `200`. |
| `--batch-size` | `BATCH_SIZE` | Mini-batch size. <br> Default: `1024`. |
| `--lr` | `LR` | Learning rate of AdamW. <br> Default: `5e-3`. |
| `--seed` | `SEED` | Seed of the training (inducing points, mini-batches). <br> Default: `42`. |
| `--holdout-frac` | `HOLDOUT_FRAC` | Fraction of the **clouds** kept out as holdout. <br> Default: `0.1`. |
| `--holdout-seed` | `HOLDOUT_SEED` | Seed of the holdout split (the calibration split uses `HOLDOUT_SEED + 1`). <br> Default: `42`. |
| `--eval-every` | `EVAL_EVERY` | Epochs between two evaluations on the holdout. <br> Default: `10`. |
| `--calib-frac` | `CALIB_FRAC` | Fraction of **all** the clouds kept out of training, disjoint from the holdout, to calibrate the sigma; `0` disables the calibration. <br> Default: `0.05`. |
| `--calib-families` | `CALIB_FAMILIES` | Comma-separated families whose sigma is calibrated. <br> Default: `lines,cont,grain,trans,dust,fuv_t,fuv_n`. |
| `--calib-quantile` | `CALIB_QUANTILE` | Quantile $q$ of $\lvert r\rvert$ of the calibration estimator (`0.5` is the median estimator). <br> Default: `0.95`. |
| `--device` | `{auto,cpu,cuda}` | Device of the training; `auto` takes the GPU when available. <br> Default: `auto`. |

### Examples

Train with the defaults, from the root of the data tree:

```bash
galapy-train-hii
```

Train a variant with more inducing points for the lines and scalars, in its own files, raising the size budget
(the lines file grows as $M^2$):

```bash
galapy-train-hii --inducing-lines 1024 --size-budget-mib 1000 \
    --out data/emulators/hii_L1024.pt --meta data/emulators/hii_L1024_meta.json
```

---

## From the corpus to the emulator

### 1. Holdout and calibration clouds

The models are split by **cloud**: the 48 SSP nodes of a cloud (same sampled axes, different `tau_SSP`,
`Z_star`) always end on the same side, so the holdout measures the interpolation to new clouds and not between
nodes of a cloud already seen. `holdout_split` averages the three dominant axes (`logU`, `lognH_HII`,
`log_zeta_O`, `EMUL.AXES_DOMINANT_KEYS_HII`) over the nodes of every cloud, assigns each cloud to an octant of
those axes around their medians, and draws the same fraction of clouds in every octant. The split is
deterministic for a given seed.

Two disjoint sets are drawn: the **holdout** (`--holdout-frac` of the clouds), never touched by the training, and
the **calibration** set (`--calib-frac` of all the clouds, drawn among the remaining ones), used only to
calibrate the sigma. The input standardization, $x_z = (x - \bar x)/\sigma_x$, uses every model but the holdout:
it reads the input positions of the calibration set, never its targets.

### 2. The targets

Every family is turned into well-scaled targets before the GP sees it:

$$
\tilde y = \log_{10}\max\!\big(y + o,\ f\big), \qquad
f = \max\!\big(f_{\rm rel}\,\max_\lambda y,\ 10^{-35}\big), \qquad
z = \frac{\tilde y - \mu}{\max(\sigma, 0.01)} ,
$$

with $o$ the offset of the family, $f_{\rm rel}$ the relative floor of the model (row) and $\mu, \sigma$ the
mean and standard deviation of every column over the training models. The floor fixes the depth where a spectrum
goes to zero: a transparent/opaque transition is a jump of $-\log_{10} f_{\rm rel}$ dex, which a stationary
kernel smooths over, so the floor is as high as the physics allows. The transmission gets $10^{-5}$ (a cloud with
$T < 10^{-5}$ is opaque); the continua keep $10^{-10}$ because their radio tail spans many decades, except below
the Lyman limit, where the nebular continuum gets $10^{-5}$ (`--rel-floor-lyc`). The minimum standard deviation of
0.01 dex keeps a nearly constant column from blowing numerical noise up to unit variance.

The three continua are then compressed with a PCA. The number of components of every family is the smallest
that explains `--var-target` of the variance; when the total exceeds `--max-pca-components`, the budget is shared
in proportion (`allocate_pca_budget`). The scores of each component are **whitened**, divided by their standard
deviation, so that every GP starts at unit variance.

### 3. The lines in ratio form

With `--line-targets ratio` (the default) the lines are emulated as ratios, which removes the orders of magnitude
carried by the ionizing photons:

$$
t_{{\rm H}\beta} = \log_{10} L_{{\rm H}\beta} - \log_{10}\!\big[Q_{H,\rm unit}\,(1 - f_{\rm esc})\big], \qquad
t_i = \max\!\Big(\log_{10}\frac{L_i}{L_{{\rm H}\beta}},\ \log_{10} f_{\rm ratio}\Big),
$$

where the scale $Q_{H,\rm unit}(1-f_{\rm esc})$ is analytic, computed from the inputs with the node table
(nearest node in $\tau$, linear in $Z_*$, $1-f_{\rm esc}$ floored at $10^{-4}$), and the same at training and at
inference. The reference line is H$\beta$ at 4862.7 Å (vacuum, `EMUL.RATIO_REF`), which must appear once in the
line list.

The ratio floor $f_{\rm ratio}$ (`--ratio-floor`, default $10^{-5}$) sets the faintest ratio the emulator
follows: below it a line is emulated at the floor. $10^{-5}$ lies below the faintest line ever measured in a
nebular spectrum, about $2\times10^{-5}\,I({\rm H}\beta)$ in the Orion nebula with VLT/UVES (Esteban et al. 2004),
and far below what galaxy spectra reach: about $3\times10^{-3}\,{\rm H}\beta$ in the deepest JWST composite of
$z\sim2$–3 galaxies (Strom et al. 2023), a few $10^{-2}\,{\rm H}\beta$ in single galaxies at $z = 6$–8
(Rai et al. 2026). A lower floor makes the GPs spend their capacity on unobservable ratios spanning many decades:
on the HII holdout, $10^{-10}$ instead of $10^{-4}$ worsened the metal lines above $10^{-3}\,{\rm H}\beta$ by
10–25%.

### 4. The SVGPs

Every output is an independent SVGP (`svgp.BatchedSVGPS`) on the 9 standardized inputs, with a constant mean and
a scaled Matérn-5/2 kernel with one lengthscale per input (ARD):

$$
k(x, x') = \sigma_f^2 \left(1 + \sqrt5\, r + \tfrac53 r^2\right) e^{-\sqrt5\, r}, \qquad
r^2 = \sum_{d=1}^{9} \frac{(x_d - x'_d)^2}{\ell_d^2} .
$$

The $M$ inducing points are initialized at the k-means centres of the training inputs and then optimized with
the rest. All the outputs of a family are trained together, as one batch, by maximizing the variational ELBO with
AdamW on mini-batches. The log reports, every `--eval-every` epochs, the score on the holdout in the GP space,
with the latent variance (no likelihood noise), as the emulator predicts:

```text
[hii-train] TE epoch  250/250  rmse_z 0.5087 (constant predictor 0.9894, good << it)  explained 73.6%  cov68 0.423 (calibrated 0.68)
```

`rmse_z` is the RMSE over all the outputs of the family, the *constant predictor* is the RMSE of predicting the
training mean ($z = 0$), and `explained` is $1 - {\rm MSE}/{\rm MSE}_{\rm const}$, an $R^2$ averaged over the
outputs with equal weight. `cov68` is the fraction of holdout values within $1\sigma$, before the calibration.

For the PCA families the whitening gives the same weight to the first component, which carries most of the
spectrum, and to the last ones, which carry almost nothing and are close to noise. At the end of each PCA family
the training therefore prints the holdout score of **every** component and the explained variance **weighted**
by the variance each component carries in the spectrum:

$$
{\rm explained_w} = 1 - \frac{\sum_k s_k^2\, {\rm MSE}_k}{\sum_k s_k^2\, {\rm MSE}_{{\rm const},k}} ,
$$

with $s_k$ the standard deviation of the scores of component $k$. Since the components are orthonormal, it is the
fraction of the spectral variance (within the retained components) that the emulator reproduces. On the HII
pilot, the poorly predicted last components of the nebular continuum pull the equal-weight score down to 73%,
while they carry almost none of the spectrum:

```text
[hii-train] holdout per PC, cont: rmse_z (constant predictor ~1), explained, share of the spectral variance carried by the PC
[hii-train]   PC   1  rmse_z 0.0337  explained  99.89%  variance share  87.862%
[hii-train]   PC   2  rmse_z 0.0864  explained  99.27%  variance share   6.627%
...
[hii-train]   PC  37  rmse_z 0.7221  explained  46.59%  variance share   0.001%
[hii-train] cont: explained 73.3% (equal weight per PC), 99.59% (weighted by the variance of each PC)
```

### 5. Calibration of the sigma

The SVGP sigma is not guaranteed to be honest, and it enters the likelihood of a fit as model error at every
evaluation. After each family, the training computes one scale factor $s_k$ per output from the normalized
residuals $r = (y - \mu)/\sigma$ on the calibration clouds:

$$
s_k = \frac{{\rm quantile}_q\big(\lvert r_{ik}\rvert\big)}{\Phi^{-1}\!\big(\tfrac{1+q}{2}\big)}, \qquad s_k \ge 1 ,
$$

and stores it in the artifact; at prediction time the variance is multiplied by $s_k^2$. The residuals have heavy
tails, so no single factor gives both 68% of the points within $1\sigma$ and 95% within $1.96\sigma$. The median
estimator ($q = 0.5$) fixes the core but leaves the tails overconfident (on the HII holdout, `cov95` 0.80–0.89 and
up to 3% beyond $5\sigma$); the default $q = 0.95$ gives `cov95` $= 0.95$ in every family and 0.1–0.6% beyond
$5\sigma$, with a conservative core (`cov68` 0.77–0.91). For a model error, an overconfident tail biases the
posterior while a conservative core only widens it. The clip at $s_k \ge 1$ means the calibration only ever widens
the sigma. The log reports the effect on the holdout:

```text
[hii-train] sigma calibration cont: s_k median 2.66, max 4.67 | holdout cov95 0.663 -> 0.950 (target 0.95), cov68 0.423 -> 0.789, |r|>5 0.078 -> 0.001
```

The factor corrects the average over the input space, not regions where the GP is locally worse, and it makes the
tails rarer without removing them.

---

## Inference: `HIIEmulator`

`HIIEmulator().load(stem)` reads the five artifact files, rebuilds the SVGPs and precomputes, for every family,
the quantities that do not depend on the input (`svgp.predictive_cache`): with $L$ the Cholesky factor of
$K_{ZZ}$, $m$ and $S$ the variational mean and covariance,

$$
\mu(x) = k_{xZ}\, L^{-T} m + c, \qquad
\sigma^2(x) = \big[\sigma_f^2 + k_{xZ}\, L^{-T}(S - I)\,L^{-1} k_{Zx}\big]\, s_k^2 ,
$$

so a prediction (`svgp.fast_predict`) costs two products with cached matrices instead of a pass through gpytorch.
`svgp.decode` then brings mean and variance back to physical units: it undoes the whitening and the PCA (the
variance of a wavelength is $\sum_k C_{k\lambda}^2\, s_k^2\, \sigma_k^2$), the ratio form (the variance of a line
adds that of H$\beta$), the z-score and the $\log_{10}$, with $\sigma_y = \ln 10\; y\; \sigma_{\log y}$.

`predict_batch(theta)` takes a batch of inputs in the order of `EMUL.INPUT_KEYS_HII`, with `tau_SSP` and `Z_star`
in $\log_{10}$, and returns numpy arrays:

| Key | Content | Shape |
| :--- | :--- | :--- |
| `lines_emergent` | (mean, sigma) of the line luminosities [${\rm erg\,s^{-1}\,M_\odot^{-1}}$] | `(N, n_lines)` each |
| `continuum_nebular` | (mean, sigma) of the nebular continuum, $\nu L_\nu$ [${\rm erg\,s^{-1}\,M_\odot^{-1}}$] | `(N, n_wave)` each |
| `continuum_grain` | (mean, sigma) of the grain continuum | `(N, n_wave)` each |
| `transmission` | (mean, sigma) of the transmission | `(N, n_wave)` each |
| `dust_mass_cloud` | (mean, sigma) of the dust mass [$M_\odot\,M_\odot^{-1}$] | `(N,)` each |
| `T_fuv_hii` | FUV transmittance (mean) | `(N,)` |
| `N_fuv_hii` | FUV own emission over incident (mean) | `(N,)` |

`tau_SSP` must sit on a node of the grid (within $10^{-4}$ dex), otherwise `ValueError`; `Z_star` is interpolated
linearly between nodes, as GalaPy does for the CSP. The wavelength grid, the line names and their vacuum
wavelengths are attributes of the instance (`wave_grid`, `line_names`, `line_wavelengths_rest`).

!!! note "Speed"
    One galaxy spectrum needs the 8 `tau_SSP` nodes, i.e. a batch of 8 rows. The cost of a call is dominated by
    reading the cached matrices ($d \times M \times M$ per family) from memory, so up to some size a larger batch
    costs much less than proportionally: a sampler that evaluates several points of the parameter space per call
    (e.g. nautilus with `vectorized=True`, emcee with `vectorize=True`) gains about a factor 2–3. The decoding
    products run in torch, not numpy: two thread pools (torch OpenMP and numpy OpenBLAS) alternating on the same
    cores oversubscribe them, and made a batch of 8 at 8 threads 8 times slower than with a single pool. The
    variance cost grows as $M^2$: a larger $M$ is paid at every evaluation of a fit.

---

## Validating the emulator

The command `galapy-validate-emul` runs the checks of a sector on the holdout written by the training, and
writes a JSON and a Markdown report.

### Usage

```text
usage: galapy-validate-emul [-h] --sector {hii,pdr} [--emulator EMULATOR] [--meta META] [--corpus CORPUS]
                            [--holdout HOLDOUT] [--report REPORT] [--markdown MARKDOWN]
                            [--max-points MAX_POINTS] [--n-ks N_KS] [--noise-diagnostic]
                            [--midnode-corpus MIDNODE_CORPUS] [--seed SEED] [--strict]
                            [--tp-threads TP_THREADS [TP_THREADS ...]]
                            [--tp-batches TP_BATCHES [TP_BATCHES ...]] [--tp-min-time TP_MIN_TIME]
                            [--checks CHECKS]
```

### Options

| Flag | Argument | Description |
| :--- | :--- | :--- |
| `-h`, `--help` | None | Show help message and exit. |
| `--sector` | `{hii,pdr}` | Sector of the emulator. **(Required)** |
| `--emulator` | `EMULATOR` | Stem of the artifacts. <br> Default: `data/emulators/<sector>`. |
| `--meta` | `META` | Manifest of the training. <br> Default: `<stem>_meta.json`. |
| `--corpus` | `CORPUS` | Corpus the emulator was trained on. <br> Default: `data/products/<sector>_grid.h5`. |
| `--holdout` | `HOLDOUT` | Holdout keys written by the training. <br> Default: `<stem>_holdout.json`. |
| `--report` | `REPORT` | JSON report. <br> Default: `data/validation/report_emulator_<sector>.json`. |
| `--markdown` | `MARKDOWN` | Markdown report. <br> Default: the JSON path with suffix `.md`. |
| `--max-points` | `MAX_POINTS` | Holdout points evaluated, a random subset when the holdout is larger (`0` = all). <br> Default: `8192`. |
| `--n-ks` | `N_KS` | Residuals per output in the (informative) KS test. <br> Default: `1000`. |
| `--noise-diagnostic` | None | Also report the coverage with the likelihood noise added to the latent variance. |
| `--midnode-corpus` | `MIDNODE_CORPUS` | Corpus of CLOUDY runs with `Z_star` between the nodes, for `Midnodes`. |
| `--seed` | `SEED` | Seed of the subsets drawn by the checks. <br> Default: `0`. |
| `--strict` | None | A blocking check in `SKIP` also fails the validation. |
| `--tp-threads` | `TP_THREADS ...` | Torch thread counts of the throughput grid. <br> Default: 1, 2, 4, 8 and all the available CPUs. |
| `--tp-batches` | `TP_BATCHES ...` | Rows per call of the throughput grid. <br> Default: `8 16 32 64 128`. |
| `--tp-min-time` | `TP_MIN_TIME` | Seconds of measurement per configuration of the throughput grid. <br> Default: `0.5`. |
| `--checks`, `--gates` | `CHECKS` | Run only these checks, comma-separated ids (quote ids with spaces). <br> Default: all the checks of the sector. |

### Statuses and report

Every check ends in `PASS`, `FAIL`, `SKIP` (an input is missing, the detail says which) or `ERROR` (the check
raised; the error and the tail of the traceback are reported). The validation exits with status `1` if a
blocking check ends in `FAIL` or `ERROR`, or in `SKIP` under `--strict`. The JSON report records the emulator, the
corpus, the holdout, the summary of the statuses and, for every check, its status, blocking flag, title, duration
and a detail with the numbers behind the verdict. The Markdown report shows a summary table (`Check`, `State`,
`Blocking`, `Description`, `Summary`), the tables of the checks that produce them (the throughput grid) and the
full JSON of the details.

### The checks

| Check | Blocking | Property tested | Criterion |
| :--- | :---: | :--- | :--- |
| `crit_l` | | Accuracy of the critical lines (H$\alpha$, H$\beta$, [O III] 5007, [N II] 6584, [O II] 3726, 3729, C III] 1909) | worst RMSE of $\log_{10}$, on the holdout lines above $10^{-3}\,{\rm H}\beta$, $< \log_{10}1.05 = 0.0212$ dex |
| `OIII r` | | [O III] 4959/5007 at its atomic value, which the independent GPs do not impose | every graded point within 2% of 0.346 |
| `Balmer` | | H$\alpha$/H$\beta$ follows Case B | median within 5% of 2.86, for $\log U = -3 \pm 0.25$, $z_{\rm CMB} \le 1$ |
| `Calibr.` | yes | The calibrated sigma is honest | in every family `cov95` in $[0.93, 0.97]$ and `cov68` $\ge 0.60$ |
| `Energy` | | The emulated transmission and continuum close the energy budget of the incident SED as the grid does | $\ge 95\%$ of (up to 2048) points within 1% of the grid, and $-10^{-3} \le T \le 1 + 10^{-3}$ |
| `Dust` | | Accuracy of the dust mass and its Strömgren scaling $M_d \propto (1 - f_{\rm esc})\,\xi_d\, m_Z\, Q_H / n_H$ | RMSE $< \log_{10}1.10 = 0.0414$ dex and $\ge 90\%$ of points within 0.2 dex of the scaling |
| `OOD` | | The sigma grows out of the domain | the median sigma grows monotonically at 0.25, 0.5, 1, 2 domain widths beyond the upper edge of each non-node axis, and $\ge 95\%$ of the probes exceed the in-domain sigma at the farthest step |
| `Th.put` | | Speed on CPU | $\ge 100$ galaxies/s (batch of 8 rows) at 8 threads; the grid threads × batch is reported |
| `Size` | | Size of the artifacts and memory at runtime | every file within the size budget of the training, and peak RSS $\le 1.5$ GiB to load the emulator and predict 20 batches of 8 |
| `ARD` | | The lengthscales and the response to the inputs are physical | $\ell_{F_*}$ / median $\ell$ of `logU`, `lognH_HII`, `log_zeta_O` $\le 10$ for the sulfur lines; in $\ge 90\%$ of the scans S II/H$\alpha$ and S III/H$\alpha$ decrease with $F_*$, O32 increases and S II/H$\alpha$, O I/H$\alpha$ decrease with $f_{\rm esc}$ ($\xi_d$ reported) |
| `Bins` | | Accuracy and calibration in every age node | `crit_l` and `Calibr.` pass in every `tau_SSP` node |
| `Midnodes` | | Interpolation in `Z_star` between the nodes | KS $p > 0.05$ of the line residuals and RMSE $\le 1.5\times$ that of the holdout (needs `--midnode-corpus`) |
| `GPU` | | Same predictions on GPU and CPU | maximum relative difference $\le 10^{-4}$ on 64 points (`SKIP` without a GPU) |

The criterion of `Calibr.` matches the estimator of the training: with heavy-tailed residuals a calibrated `cov95`
comes with a conservative `cov68`, so `cov68` has only a lower bound. A Kolmogorov-Smirnov test of the residuals
against $\mathcal N(0,1)$ per output is still reported, as information: with thousands of heavy-tailed residuals it
rejects gaussianity for any sigma.

`Th.put` measures `predict_batch` on a grid of torch thread counts and batch sizes, in galaxies per second, and
prints it to the log as well as to the report; the asterisk marks the cell of the criterion:

```text
[emul-val]   Galaxies per second  [check: >= 100 galaxies/s at batch 8 with 8 threads (*)]
[emul-val]    threads |    B=8   B=16   B=32   B=64  B=128
[emul-val]   ---------+-----------------------------------
[emul-val]          1 |     33     56     73     84     75
[emul-val]          4 |     77    119    166    183    165
[emul-val]          8 |    89*    140    202    219    187
[emul-val]   Best: 219 galaxies/s (1752 rows/s) with 8 threads and batch 64
```

The grid is meant to be run on the machine of the fits, to choose threads and batch when the emulator is used in
a sampler. Thread counts beyond the physical cores are usually slower. `Size` measures the memory in a fresh
process, from its own peak (`VmHWM`): `ru_maxrss` of a child started by fork+exec would also count the peak of the
validation process.

---

## Output Structure and File Formats

### 1. The artifacts

From the stem `<stem>` of `--out`:

| File | Content |
| :--- | :--- |
| `<stem>_lines.pt`, `<stem>_cont.pt`, `<stem>_grain.pt`, `<stem>_trans.pt` | State of the SVGPs and of the likelihood of one family, plus the shared `extras`. |
| `<stem>_scalars.pt` | The same for `dust`, `fuv_t` and `fuv_n`. |
| `<stem>_holdout.json` | Keys of the holdout models, number of models and clouds, fraction, seed, stratification axes, corpus. |
| `<stem>_calib.json` | Keys of the calibration models, in the same format (absent with `--calib-frac 0`). |
| `<stem>_gate_set_frozen.npz` | Frozen subset of the holdout (inputs, targets in the GP space and in physical units, families and widths), to compare a refined emulator with the base one without the corpus (`acceptance_gates`). |

The `extras` carry everything inference needs: input standardization, input keys and $\log_{10}$ inputs, node
table of $Q_{H,\rm unit}$, line names and vacuum wavelengths, wavelength grid, and per family the preprocessing
(`<fam>_prep`: means, standard deviations, floors, PCA components and score deviations, ratio form, calibration
scales `sigma_scale`), the ELBO and holdout histories and, for the PCA families, the per-component holdout scores.
Every file is checked against `--size-budget-mib`.

### 2. The manifest `--meta`

The JSON manifest records the configuration and the outcome of the training: number of models; holdout (file,
frozen set, fraction, seed, number of training and holdout models, final holdout score of every family with the
raw sigma, weighted explained variance of the PCA families); sigma calibration (file, fraction, number of models,
families, estimator and quantile, and per family the median and maximum $s_k$ with the holdout `cov68`, `cov95`
and fraction beyond $5\sigma$ before and after); input keys and their training range (`x_min`, `x_max`, used by
`OOD` and `ARD`); inducing points; artifact paths and the $\Delta M$ each file could still take within the budget;
floors, line targets and ratio floor, PCA settings; epochs, batch, learning rate, seed, timing and device; and the
outputs of every family.

### 3. Failure modes

The messages are prefixed with `[hii-train]`, `[hii-emulator]` or `[emul-val]`:

| Error | Cause |
| :--- | :--- |
| `ValueError: [hii-train] No grid points found in the corpus file` | The corpus holds no `grid_point_*` group: the fragments were not merged. |
| `ValueError: [hii-train] Units mismatch in the corpus file` | The corpus was not written with the current units table (`CLOUDY.Points_Units_HII`): parse and merge it again. |
| `ValueError: [hii-train] ... without pumping removed` | A model comes from a parser that did not remove the continuum pumping. |
| `ValueError: [hii-train] ... pumping removed for some models and not for others` | The corpus mixes fragments of different parser versions. |
| `ValueError: [hii-train] node (...): Q_H values are not consistent` | The models of an SSP node disagree on $Q_{H,\rm unit}$: the grid mixes SED extractions. |
| `ValueError: [hii-train] reference line ('h 1', 4862.7) not found once in the line list` | H$\beta$ is missing from the line list (or repeated): use another list, or `--line-targets absolute`. |
| `ValueError: [hii-train] --calib-frac ... too large with --holdout-frac ...` | Holdout and calibration together leave no clouds for training. |
| `ValueError: [hii-train] --calib-families: unknown families` / `--rel-floor...: unknown family` | A family name is not one of the 7 families. |
| `SystemExit: [hii-train] Files exceeding the size budget` | An artifact is larger than `--size-budget-mib` (e.g. a larger $M$): raise the budget. |
| `FileNotFoundError: [hii-emulator] None of the files ... exist` | The stem does not point to a complete set of artifacts. |
| `ValueError: [hii-emulator] log10 tau_SSP ... out of nodes range` | An input age is not a node of the grid. |
| `ValueError: [hii-emulator] Q_H, unit absent for the requested node` | The corpus did not cover all the 48 nodes. |
| `RuntimeError: footprint subprocess failed` | The process of `Size` could not load the emulator; its error follows. |

---

# Python API

::: galapy.spectroscopy.utils.hii.train_emulator_hii.main
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.train_emulator_hii.load_corpus
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.train_emulator_hii.preprocess_targets
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.train_emulator_hii.allocate_pca_budget
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.train_emulator_hii.train_family
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.hii.train_emulator_hii.report_per_pc
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.hii_emulator.HIIEmulator
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.svgp.BatchedSVGPS
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.svgp.predictive_cache
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.svgp.fast_predict
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.svgp.decode
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.svgp.lines_to_ratio
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.validate_emulator.main
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.validate_emulator.holdout_split
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.validate_emulator.calibration_scale
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.validate_emulator.calibration
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.validate_emulator.throughput
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.validate_emulator.footprint
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.validate_emulator.acceptance_gates
    options:
      show_root_heading: true
      show_source: false

### Example Code

```python
import json
import numpy as np
from galapy.spectroscopy.hii_emulator import HIIEmulator

# load the emulator (CPU) and predict the 8 age nodes of one HII model
emu = HIIEmulator().load('data/emulators/hii')
print(emu.input_keys)                       # logU, lognH_HII, log_zeta_O, z_CMB, xi_d, f_esc_meas, F_star, tau_SSP, Z_star
taus = [1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 7e7, 1e8]
theta = np.array([[-2.5, 2.0, -0.3, 0.0, 0.4, 0.1, 0.5, np.log10(t), np.log10(0.004)] for t in taus])
out = emu.predict_batch(theta)

L_lines, sig_lines = out['lines_emergent']      # (8, n_lines), erg/s per Msun of SSP
L_neb, sig_neb = out['continuum_nebular']       # (8, n_wave), on emu.wave_grid
T, sig_T = out['transmission']
names = [' '.join(str(n).split()) for n in emu.line_names]
j = [i for i, (n, w) in enumerate(zip(names, emu.line_wavelengths_rest)) if n == 'H 1' and abs(w - 4862.7) < 1][0]
print('H-beta per Msun:', L_lines[:, j], '+-', sig_lines[:, j])

# the training manifest: holdout score and calibration of every family
meta = json.load(open('data/emulators/hii_meta.json'))
print(meta['holdout']['final']['lines'], meta['holdout']['explained_weighted'])
print({f: (d['holdout_cov95_after'], d['s_median']) for f, d in meta['sigma_calibration']['per_family'].items()})
```
