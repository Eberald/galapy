# HII region: running CLOUDY

CloudIA implements the modules `utils/run_core.py` and `utils/hii/run_hii.py` to execute the CLOUDY decks
rendered by [`galapy-gen-hii`](gen_hii.md). The two modules split the work as follows:

1. **Engine** — `run_core` is sector-agnostic, and the PDR sector uses it too. It plans the jobs, stages each deck
   in its own working directory, launches CLOUDY (serially or on a process pool), classifies the outcome of each
   run, and writes a JSON report of the whole launch.
2. **Driver** — `run_hii` is the thin HII entrypoint: it binds the engine to the HII sector, i.e. to the default
   deck directory `data/hii` and to the two input files that the parser needs.

By default each successful run is **parsed in the same worker** that ran it, producing one HDF5 fragment
`<job_id>.h5` per model (see [Parse HII](parse_hii.md)). The fragments are then merged into the grid corpus by
`galapy-h5`.

The HII production chain is therefore:

| Step | Command | Produces |
| :--- | :--- | :--- |
| 1 | [`galapy-sed-cloudy-extract`](ssp_extraction.md) | `data/cloudy_seds/*.sed`, `ssp_metadata.json` |
| 2 | [`galapy-gen-lhs-hii`](gen_hii.md) | `data/grids/hii/hii_grid_spec.h5` |
| 3 | [`galapy-gen-hii`](gen_hii.md) | `data/hii/hii_<job_id>.in`, `data/hii/SED/`, `data/hii/jobs_hii.txt` |
| 4 | [`galapy-gen-linelist --hii`](line_list.md) | `data/lines/hii_lines.dat` |
| 5 | `galapy-run-cloudy-hii` | `data/hii/work/<job_id>/`, `data/hii/parsed/<job_id>.h5` |
| 6 | `galapy-h5 --hii` | `data/products/hii_grid.h5` |

The defaults of every step are chained, so the whole grid can be produced from the project root without passing
any path.

!!! warning
    Running the models requires CLOUDY `C25.00` (see [CLOUDY installation](cloudy_installation.md)). Only
    `--dry-run` and `--parse-only` work without it.

---

## The HII sector

`run_core` knows about a sector only through the frozen dataclass `Sector`, registered in `run_core.SECTORS`:

| Field | HII value | Role |
| :--- | :--- | :--- |
| `name` | `hii` | Prefix of decks (`hii_<job_id>.in`), outputs and manifest (`jobs_hii.txt`). |
| `saves` | `.con`, `.con_grain`, `.lines`, `.dusa`, `.grain_temp` | Outputs that must exist for a run to count as successful. |
| `linelist` | `hii_lines.dat` | Line list staged next to the deck for `save line list`. |
| `parse_module` | `galapy.spectroscopy.utils.hii.parse_one_hii` | Parser called after each successful run. |

The five extensions are the `save last ...` commands of the `hii.in.j2` template:

| Output | CLOUDY command | Used by the parser |
| :--- | :--- | :--- |
| `hii_<job_id>.con` | `save last continuum ... units angstrom no hash` | Continuum, transmission, $f_{\rm esc}$, FUV terms |
| `hii_<job_id>.con_grain` | `save last continuum grain ... units angstrom no hash` | Grain emission (diagnostic) |
| `hii_<job_id>.lines` | `save last line list ... "hii_lines.dat" emergent absolute column no hash` | Emergent line fluxes |
| `hii_<job_id>.dusa` | `save last grain abundance ... no hash` | Dust mass |
| `hii_<job_id>.grain_temp` | `save last grain temperature ... no hash` | Not parsed, kept for diagnostics |

The PDR sector registers its own `Sector` (with `.pdr`, `.heat`, `.cool` outputs and `pdr_lines.dat`) and
runs on the same engine.

---

## Command Line Interface (CLI)

The module `utils/hii/run_hii.py` exposes the CLI entrypoint `galapy-run-cloudy-hii`.

### Usage

```text
usage: galapy-run-cloudy-hii [-h] [-r RUNS] [-w WORK] [-m MANIFEST]
                             [-i INDEX | --job-id JOB_ID [JOB_ID ...]] [-j JOBS] [-t TIMEOUT] [-l LINELISTS]
                             [--resume] [-d] [--report REPORT] [--no-version-check] [--frags FRAGS]
                             [--spec SPEC] [--ssp-meta SSP_META] [--no-parse | --parse-only]
```

### Options

| Flag | Argument | Description |
| :--- | :--- | :--- |
| `-h`, `--help` | None | Show help message and exit. |
| `-r`, `--runs` | `RUNS` | Deck directory produced by `galapy-gen-hii`. It must contain `SED/` and `jobs_hii.txt`. <br> Default: `data/hii`. |
| `-w`, `--work` | `WORK` | Root of the per-job working directories. <br> Default: `<runs>/work`. |
| `-m`, `--manifest` | `MANIFEST` | List of the job ids of the grid. <br> Default: `<runs>/jobs_hii.txt`. |
| `-i`, `--index` | `INDEX` | Select the jobs by their 0-based position in the manifest, e.g. `'0-49'` or `'0,7,12-15'`. Mutually exclusive with `--job-id`. <br> Default: `None`, i.e. all jobs. |
| `--job-id` | `JOB_ID [JOB_ID ...]` | Select the jobs by explicit id, mainly for debugging a single model. Mutually exclusive with `--index`. |
| `-j`, `--jobs` | `JOBS` | Number of parallel processes. With `1`, jobs run serially in the main process. <br> Default: `1`. |
| `-t`, `--timeout` | `TIMEOUT` | Wall-clock limit of a single CLOUDY run, in seconds. <br> Default: `7200` (2 h). |
| `-l`, `--linelists` | `LINELISTS` | Directory of `hii_lines.dat`. <br> Default: `$CLOUDIA_LINELISTS`, then `data/lines`. |
| `--resume` | None | Skip the jobs whose fragment is up to date, only parse the successful runs without a fragment, and rerun all the others. See [Resuming a launch](#resuming-a-launch). |
| `-d`, `--dry-run` | None | Print the plan of the launch and exit, without running anything. |
| `--report` | `REPORT` | Path of the JSON report of the launch. <br> Default: `<work>/run_manifest.json`, or `<frags>/parse_manifest.json` with `--parse-only`. |
| `--no-version-check` | None | Skip the check of the CLOUDY version banner. **For development only**: a grid produced this way cannot be combined with the production grids. |
| `--frags` | `FRAGS` | Output directory of the fragments `<job_id>.h5`. It is the `--frags` of `galapy-h5`. <br> Default: `<runs>/parsed`. |
| `--spec` | `SPEC` | Job specification produced by `galapy-gen-lhs-hii`, forwarded to the parser. <br> Default: `data/grids/hii/hii_grid_spec.h5`. |
| `--ssp-meta` | `SSP_META` | `ssp_metadata.json` written by `galapy-sed-cloudy-extract`. The parser reads $Q_H$ of the SSP node from it. <br> Default: `data/cloudy_seds/ssp_metadata.json`. |
| `--no-parse` | None | Run CLOUDY only. Mutually exclusive with `--parse-only`. |
| `--parse-only` | None | Do **not** run CLOUDY: parse the working directories of the successful runs. Mutually exclusive with `--no-parse`. |

All the directories default to paths relative to `--runs`, so moving a whole grid only requires changing that
one flag:

| Path | Flag | Default |
| :--- | :--- | :--- |
| Decks, `SED/` | `-r` | `data/hii` |
| Manifest | `-m` | `<runs>/jobs_hii.txt` |
| Working directories | `-w` | `<runs>/work/<job_id>` |
| Fragments | `--frags` | `<runs>/parsed/<job_id>.h5` |
| Report | `--report` | `<work>/run_manifest.json` (`<frags>/parse_manifest.json` with `--parse-only`) |

### Execution modes

| Mode | Flags | CLOUDY required | What happens to each job |
| :--- | :--- | :--- | :--- |
| `run+parse` | default | Yes | Staged, run, and parsed into `<frags>/<job_id>.h5` if the run succeeded. |
| `run` | `--no-parse` | Yes | Staged and run; no fragment is written. This is the mode of the Nextflow process `RUN_CLOUDY_HII`, where parsing is a separate process. |
| `parse-only` | `--parse-only` | No | The successful working directories are parsed again, e.g. after a change of the parser. Jobs that never completed are reported as `incomplete`. |

### Job selection

The manifest `jobs_hii.txt` fixes the order of the grid. `--index` accepts a comma-separated list of 0-based
positions and closed ranges, which are merged, deduplicated and sorted: `'0-9,20,30-32'` selects 14 jobs.
Reversed ranges (`'9-0'`), negative indexes, and indexes beyond the end of the manifest are rejected.

`--job-id` selects jobs by id. Unknown ids are rejected, and the selected jobs are run in manifest order,
whatever the order on the command line.

`--index` is the natural way to split a grid into chunks across nodes or batch jobs. Each launch writes its
report to the same default path, so give each chunk its own `--report` to keep all of them.

### Examples

Check what a launch would do, without running CLOUDY:

```bash
galapy-run-cloudy-hii -i 0-49 --dry-run
```

Run and parse the first 1000 models on 32 processes, keeping a per-chunk report:

```bash
galapy-run-cloudy-hii -i 0-999 -j 32 --report data/hii/work/run_0000-0999.json
```

Debug a single model:

```bash
galapy-run-cloudy-hii --job-id 00042_003_02
```

Resume an interrupted launch:

```bash
galapy-run-cloudy-hii -j 32 --resume
```

Regenerate all the fragments after a change of the parser, without rerunning CLOUDY:

```bash
galapy-run-cloudy-hii --parse-only -j 16
```

---

## How a job is executed

### 1. Pre-flight checks

Some checks run once per launch, before any job starts:

- the manifest must be non-empty and free of duplicate ids, and the selection must be valid;
- when parsing is enabled, `--spec` and `--ssp-meta` must exist. They are checked here once, not inside every
  worker, where they would fail N times with the same error. Their paths are then made absolute before
  being forwarded to the parser.

Any failure prints `[ERROR]: ...` on `stderr` and exits with status `2`.

`--dry-run` stops at this point. It prints the plan (mode, working directory, fragment directory, and the first 20
job ids) and exits with `0`, without looking for CLOUDY. The parser inputs are still checked, unless `--no-parse`
is given.

Otherwise, unless `--parse-only` is given, the executable is located by `require_cloudy`, and its banner must
report version `25.00`. A missing CLOUDY or a wrong version exits with status `1`. A banner that cannot be read
does not block the launch, and is reported as `(banner not read)`:

```text
[hii] CLOUDY: Cloudy 25.00 — /home/user/.galapy/cloudy/c25.00/source/cloudy.exe
[hii] 1000 job, 32 processes, mode=run+parse, work=data/hii/work, frags=data/hii/parsed
```

### 2. Staging

`stage_job` builds a self-contained working directory for the job, copying in every file the deck references:

```text
data/hii/work/00042_003_02/
├── hii_00042_003_02.in        # the deck, from <runs>/
├── SED/
│   └── ssp_tau1.000e+07_Z0.0010.sed   # from <runs>/SED/, the file named by `table SED "..."`
├── GC.abn                     # from the GalaPy database, the file named by `abundances "..."`
└── hii_lines.dat              # from -l, $CLOUDIA_LINELISTS or data/lines
```

- The **deck** `<runs>/hii_<job_id>.in` must exist.
- The **SED** is the one referenced by the `table SED "..."` command of the deck. It is copied from the `SED/`
  directory staged by `galapy-gen-hii`.
- The **abundance file** is the one referenced by the `abundances "..."` command (`GC.abn`). It is resolved from
  the `Nebular/Abundances` directory of the GalaPy database, and downloaded on first access if missing.
- The **line list** `hii_lines.dat` is looked up in the `-l`/`--linelists` directory first, then in
  `$CLOUDIA_LINELISTS`, then in `data/lines` (see [Line list](line_list.md)).

A missing file fails **the job**, not the launch: the job is reported as `failed` with the error
`FileNotFoundError in stage_job: ...` and CLOUDY is not started. The files are re-staged on every run, so a
fixed SED or line list is picked up by the next rerun.

CLOUDY looks up the SED as `SED/<file>` and the abundance file as `abundances/<file>`, falling back to the bare
file name, in every directory of its search path `CLOUDY_DATA_PATH`. The staged copies are therefore found only
if the search path contains the current directory `.`. `galapy-install-cloudy` writes the variable this way (see
[CLOUDY installation](cloudy_installation.md#environment-configuration)). If you set it by hand, keep the `.`
entry.

### 3. Running CLOUDY

Before starting, the runner deletes the outcome markers and the fragment left by any previous run of the job, so
that a new run inherits nothing from the old one. It then calls

```bash
cloudy.exe -r hii_<job_id>
```

inside the working directory, which reads `hii_<job_id>.in` and writes `hii_<job_id>.out` next to the saves. The
outcome is classified as follows:

- **Timeout.** A run exceeding `--timeout` is killed. The `TIMEOUT` marker is written and the job is reported as
  `timeout`.
- **Launch error.** An executable that cannot be started (`OSError`) writes `RUN_FAILED` and is reported as
  `failed`.
- **Convergence.** The `.out` file is searched for the caution
  `C-Iterate to convergence did not converge`. When it is present, the empty marker `NOT_CONVERGED` is written.
  A non-converged model is **flagged, not discarded**: it is still parsed, and carries `not_converged = True` in
  its fragment.
- **Warnings.** CLOUDY exits with code `2` (`ES_WARNINGS`) when the model ran to the end but printed warnings,
  i.e. the `W-` lines of the `.out`. Aborts, crashes and botched monitors have codes of their own, so code `2`
  always means a complete model. With all the five saves present, the job is reported as `warning`, and the
  marker `WARNINGS` lists the warning lines. Like non-convergence, a model with warnings is **flagged, not
  discarded**: it is parsed, carries `cloudy_warnings = True` in its fragment, and counts as complete for
  `--resume`.
- **Success.** A run is `ok` if CLOUDY returned `0` **and** all the five saves of the sector exist. Any other
  return code, or a missing save (even with code `2`), fails the run: `RUN_FAILED` records the reason, e.g.
  `returncode=1 missing=['.dusa']`.

`converged.flag` is always the **last** file written, containing `1` (converged) or `0` (not converged). It
marks that CLOUDY reached its end, so it is also written for a failed run, next to `RUN_FAILED`. It is not
written after a timeout or a launch error.

| Marker | Written when | Content |
| :--- | :--- | :--- |
| `converged.flag` | CLOUDY exited, whatever the outcome (last file written) | `1` converged, `0` not converged |
| `NOT_CONVERGED` | The convergence caution is found in `.out` | Empty |
| `WARNINGS` | CLOUDY returned `2` and wrote all the saves | The `W-` lines of `.out`, one per row |
| `RUN_FAILED` | Return code other than `0` and `2`, missing save, or CLOUDY could not start | The reason |
| `TIMEOUT` | The run exceeded `--timeout` | `timeout after <N> s` |

### 4. Parsing

When parsing is enabled and the run is `ok` or `warning`, the worker calls the parser **in its own process**. It imports
`parse_one_hii` and calls its `main` with

```text
<work>/<job_id>  --spec <abs path>  --ssp-meta <abs path>  --out <frags>/<job_id>.h5
```

with the parser's `stdout` silenced. Any exception, including `SystemExit`, is caught. The parse is reported as
`failed` with a one-line error (`<Exception> in <function>: <message>`), and the full traceback is stored in the
report. A failed parse never touches the CLOUDY outputs: after fixing the parser, `--parse-only` regenerates the
missing fragments without running CLOUDY again.

With `-j N > 1` the jobs are distributed on a `ProcessPoolExecutor` of `N` workers. Each CLOUDY run is a single
process, so `N` is typically the number of available cores. Results are printed as the jobs finish, and the
report lists them in manifest order.

---

## Resuming a launch

A fragment is **up to date** when it exists and is not older than the `converged.flag` of its working directory,
i.e. it was produced from the last run, or when its working directory was deleted after parsing (the fragment is
then all that is left of the run). A working directory is **complete** when it contains `converged.flag` and
all the five saves, and no `RUN_FAILED` marker. Each job is then handled according to the mode:

| State of the job | default | `--resume` | `--parse-only` |
| :--- | :--- | :--- | :--- |
| Fragment up to date | rerun | `skipped` | parsed again (`skipped` if `--resume` is also given) |
| Complete, fragment missing or stale | rerun | parsed only | parsed only |
| Never run, timed out, crashed, failed (`RUN_FAILED`), saves missing | run | rerun | `incomplete` |

Without `--resume`, every selected job is rerun from scratch. With `--resume`, an interrupted launch picks up
exactly where it stopped. Jobs that ran but were never parsed, e.g. because the worker died between the run and
the parse, are only parsed.

---

## Output Structure and File Formats

### 1. The console log

Each job prints one line on `stderr` when it finishes:

```text
  [ ] 00000_000_00   412.3s  parse:ok
  [ ] 00000_000_01   398.7s  [NOT CONVERGED]  parse:ok
  [W] 00000_000_05   421.9s  parse:ok
  [T] 00000_000_02  7200.0s  [NOT CONVERGED]
  [!] 00000_000_03     0.0s  (FileNotFoundError in stage_job: SED missing: data/hii/SED/ssp_tau1.000e+06_Z0.0001.sed.)
  [ ] 00000_000_04   405.1s  parse:failed (KeyError in main: '[parse_one/hii] 00000_000_04: absent in ...')
```

| Mark | Status | Meaning |
| :--- | :--- | :--- |
| ` ` | `ok` | CLOUDY returned `0` and wrote all the saves. |
| `W` | `warning` | CLOUDY returned `2` (warnings) and wrote all the saves: the model is complete and parsed. |
| `-` | `skipped` | Nothing to run: the fragment is up to date, or the run was already complete (`--resume`, `--parse-only`). |
| `!` | `failed` | Staging error, launch error, non-zero return code, or missing save. |
| `T` | `timeout` | The run exceeded `--timeout`. |
| `?` | `incomplete` | `--parse-only` on a job that never completed. |

The line ends with `[NOT CONVERGED]` when the model did not converge, and with `parse:<status>` when parsing was
attempted. A timed-out job is flagged `[NOT CONVERGED]` as well, since its convergence is not established.

At the end, a summary line is printed:

```text
[hii] ok=3 warning=1 skipped=0 failed=2 incomplete=0 not-converged=2 | parse: ok=3 skipped=0 failed=1  -> data/hii/work/run_manifest.json
```

If any parse failed, it is followed by the traceback of the **first** failure. The tracebacks of all the others
are in the report.

### 2. The report `run_manifest.json`

`write_run_manifest` records the provenance of the launch and the outcome of every job:

| Key | Description |
| :--- | :--- |
| `sector` | `hii`. |
| `cloudy_exe`, `cloudy_banner`, `cloudy_data_path` | The CLOUDY installation used (`null` with `--parse-only`). |
| `cloudy_required` | The production version, `C25.00`. |
| `host`, `timestamp_utc` | Where and when the launch ran. |
| `n_jobs` | Number of selected jobs. |
| `n_ok`, `n_warning`, `n_skipped`, `n_incomplete` | Counters of the run statuses. |
| `n_failed` | Failed **plus** timed-out jobs. |
| `n_not_converged` | Jobs with `converged = false`, timeouts included. |
| `n_parse_ok`, `n_parse_skipped`, `n_parse_failed` | Counters of the parse statuses. |
| `mode` | The execution mode: `run`, `run+parse` or `parse-only`. |
| `parse_module`, `parse_argv`, `frags` | Only when parsing is enabled: the exact parser call. |
| `jobs` | One record per job, in manifest order. |

Each job record holds `job_id`, `status`, `converged` (`true`/`false`, or `null` when unknown), `seconds`,
`workdir` and `parse`. When CLOUDY ran, it also holds `returncode` and `missing`, plus `error` on failure, or
`warnings` (the list of `W-` lines) with status `warning`. `parse`
is `null` when no parse was attempted. Otherwise it holds `status` and `seconds`, plus `error` and `traceback` on
failure:

```json
{
  "job_id": "00000_000_04",
  "status": "ok",
  "converged": true,
  "seconds": 405.1,
  "workdir": "data/hii/work/00000_000_04",
  "parse": {
    "status": "failed",
    "seconds": 0.02,
    "error": "KeyError in main: '[parse_one/hii] 00000_000_04: absent in ...'",
    "traceback": "Traceback (most recent call last): ..."
  },
  "returncode": 0,
  "missing": []
}
```

The report is rewritten by every launch. Chunked launches should use distinct `--report` paths.

### 3. Exit status

| Code | Meaning |
| :--- | :--- |
| `0` | Every selected job ended with its fragment (or with a successful run, under `--no-parse`), or was skipped. Non-convergence and CLOUDY warnings are **not** failures. |
| `1` | At least one job failed, timed out, was incomplete, or failed its parse; or CLOUDY was not found, or is not `C25.00`. |
| `2` | Setup error: unreadable or invalid manifest, invalid selection, or missing parser inputs. |

---

# Python API

::: galapy.spectroscopy.utils.hii.run_hii.main
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.run_core.parse_index_selector
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.run_core.read_manifest
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.run_core.plan_jobs
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.run_core.stage_job
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.run_core.is_complete
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.run_core.already_parsed
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.run_core.run_job
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.run_core.parse_job
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.run_core.run_all
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.run_core.write_run_manifest
    options:
      show_root_heading: true
      show_source: false

### Example Code

```python
from galapy.spectroscopy.utils import require_cloudy
from galapy.spectroscopy.utils.run_core import read_manifest, plan_jobs, run_all

inst = require_cloudy()   # raises CloudyNotFound if CLOUDY is missing or not C25.00

# first 10 jobs of the grid
jobs = plan_jobs(read_manifest('data/hii/jobs_hii.txt'), index_sel='0-9')

# the parser configuration, as galapy-run-cloudy-hii builds it
parse = {'frags': 'data/hii/parsed',
         'argv': ['--spec', 'data/grids/hii/hii_grid_spec.h5',
                  '--ssp-meta', 'data/cloudy_seds/ssp_metadata.json']}

results = run_all(jobs, 'data/hii', 'data/hii/work', 'hii', inst.exe, timeout=7200,
                  nproc=4, parse=parse)
print(sum(r['status'] == 'ok' for r in results), 'successful runs')
```
