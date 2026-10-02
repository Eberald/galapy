# Line lists for `save line list`

CloudIA implements the module `utils/gen_linelist.py` to build the `.dat` files consumed by the CLOUDY
command `save line list`. The module reads the **master line list** of a sector — a YAML file stored in the
galapy-dataset — validates every entry against the **official CLOUDY LineLists**, and produces the fixed-format
`.dat` file that each deck references at run time.

!!! warning
    The master lists (`hii_full.yaml`, `pdr_full.yaml`) and the reference LineLists live in the galapy-dataset,
    not in the code.

---

## The master line list

One master list per sector, under `Nebular/Configs/Lines` of the dataset:

| Sector | Master list | Emitted file | Lines |
| :--- | :--- | :--- | :--- |
| `hii` | `hii_full.yaml` (`CONST.HII_LINES_FILE`) | `hii_lines.dat` | 64 |
| `pdr` | `pdr_full.yaml` (`CONST.PDR_LINES_FILE`) | `pdr_lines.dat` | 24 |

The file carries a single top-level key, `lines`, holding one mapping per transition:

```yaml
lines:
  - {label: 'H  1', wl: '6562.80', unit: 'A', name: 'Halpha',     group: HI}
  - {label: 'Blnd', wl: '5875.66', unit: 'A', name: 'HeI_5876',   group: HeI}
  - {label: 'o  3', wl: '5006.84', unit: '',  name: 'OIII_5007',  group: OIII}
  - {label: 'CO',   wl: '2600.05', unit: 'm', name: 'CO_1-0',     group: CO}
```

| Key | Role |
| :--- | :--- |
| `label` | The CLOUDY species tag, **at most 4 characters, spaces included and significant** (`'H  1'`, `'he 2'`, `'Blnd'`). Matching is case-insensitive and blind to internal spaces, but the string is emitted verbatim. |
| `wl` | The wavelength, kept as a **string**: it must reproduce the official entry character by character, so `6562.80` and `6562.8` are *not* interchangeable. |
| `unit` | `A` (Angstrom), `m` or `M` (micron: CLOUDY reads the suffix case-insensitively), `c` (cm), or the empty string when the official list carries no unit (Angstrom). |
| `name` | CloudIA alias of the line. Must be unique. It is not written into the `.dat`: the parsed fragments identify the lines by CLOUDY label and wavelength, in the order of the master list (see [Parse HII](parse_hii.md)). |
| `group` | Free-form tag (`HI`, `UV`, `OIII`, `dual_origin`, ...), used only to organise the list and the diagnostics; it is **not** written into the `.dat`. |

The `dual_origin` group marks the transitions that appear in both master lists (`[CII] 158 μm`,
`[NeII] 12.8 μm`, `[SiII] 34.8 μm`, plus the HI lines shared by the two sectors), i.e. the lines whose emission
must be recombined across the HII and PDR sectors.

### The reference LineLists

`Nebular/Configs/Lines_Ref` holds a frozen copy of the two official CLOUDY lists, `LineList_HII.dat` and
`LineList_PDR.dat` (`CONST.HII_LINES_OFF`, `CONST.PDR_LINES_OFF`), together with `LineList.sha256` recording
their digest. Both files are parsed — whatever the sector — so that a PDR line declared in the HII master list
still validates.

The parser `load_official` strips the trailing `#` comment of each row and matches the remainder against

```python
LINE_RE = re.compile(r'^(.{1,4}?)\s{1,}([0-9][0-9.]*)\s*([AmcM]?)\s*(air|vacuum)?\s*$', re.I)
```

so the `air`/`vacuum` keyword is recognised and discarded, and rows that do not match (headers, change logs,
malformed leftovers) are skipped silently. The result is an index keyed by *(normalised label, wavelength
string)*, where normalisation lowercases the label and removes its spaces. An empty index aborts the run.

---

## Command Line Interface (CLI)

The module exposes the CLI entrypoint `galapy-gen-linelist` to generate the `.dat` of a sector.

### Usage

```text
usage: galapy-gen-linelist [-h] (--hii | --pdr) [--out OUT]
```

### Options

| Flag | Argument | Description                                                                                                     |
| :--- | :--- |:-----------------------------------------------------------------------------------------------------------------|
| `-h`, `--help` | None | Show help message and exit.                                                                                     |
| `--hii` | None | Generate `hii_lines.dat` from the HII master list (`hii_full.yaml`). |
| `--pdr` | None | Generate `pdr_lines.dat` from the PDR master list (`pdr_full.yaml`). |
| `--out` | `OUT` | Output directory of the `.dat`. Created if missing. <br> Default: `data/lines` (`DEFAULT_OUT_DIR`). |

`--hii` and `--pdr` form a **mutually exclusive, required** group: exactly one sector per invocation. The
master list and the reference LineLists are always resolved from the GalaPy database, and have no flag.

## Validation

Before writing anything, the entries go through two independent checks, and the `.dat` is emitted **only if
both are clean**. All the errors are collected first and reported together, so a single run is enough to fix
the master list.

### 1. Against the official LineLists (`validate`)

For each entry the *(label, wl)* key is looked up in the official index:

- a missing key means the line is not a valid CLOUDY label/wavelength pair —
  `-> Not found in CLOUDY official LineList`;
- a key that is found but whose `unit` differs from the official one (compared case-insensitively, with the
  empty string treated as a unit of its own) is reported as a unit mismatch.

### 2. Internal consistency (`check_duplicates`)

- duplicate *(label, wl)* pairs, which would make CLOUDY write the same column twice;
- duplicate `name` aliases, which would collide in the parsed tables downstream.

### Failure mode

On any error the module prints the whole report on `stderr` and exits with status `1`, leaving the previous
`.dat` untouched:

```text
[gen_line] 2 Wrong format in .../Nebular/Configs/Lines/hii_full.yaml:
  NII_5755         label='n  2' wl=5754.6 -> Not found in CLOUDY official LineList
  OIII_5007        unit 'A' != '' of official CLOUDY LineList

 no .dat written
```

A successful run reports the number of validated lines:

```text
[gen_line] 64 lines -> data/lines/hii_lines.dat all validated against CLOUDY official LineLists
```

---

## Output Structure and File Formats

### 1. The emitted `.dat`

`emit_dat` writes one line per entry, the label left-padded to the 4-character field expected by CLOUDY and
the unit glued to the wavelength:

```text
H  1 6562.80A
H  1 4861.32A
Blnd 5875.66A
o  3 5006.84
CO   2600.05m
```

The order of the master list is preserved, and it is the order of the columns saved by CLOUDY: the `.dat` is
therefore part of the grid provenance, and regenerating it with a reordered master list invalidates the
already-parsed outputs.

### 2. How the decks consume it

The deck template references the list by bare filename:

```text
save last line list "hii_{{ job_id }}.lines" "hii_lines.dat" emergent absolute column no hash
```

so the `.dat` must sit in the working directory of the CLOUDY process. Staging is handled by
`run_core.stage_job`, which copies `{sector}_lines.dat` next to the deck before launching the run and resolves
its source directory in this order:

1. the `-l`/`--linelists` flag of the runner;
2. the environment variable `CLOUDIA_LINELISTS`;
3. `DEFAULT_OUT_DIR`, i.e. `data/lines`.

A missing file makes the job fail at staging time with `FileNotFoundError`, before CLOUDY is started — see
[Running CLOUDY](run_hii.md#2-staging).

---

# Python API

::: galapy.spectroscopy.utils.gen_linelist.load_official
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.gen_linelist.validate
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.gen_linelist.check_duplicates
    options:
      show_root_heading: true
      show_source: false

::: galapy.spectroscopy.utils.gen_linelist.emit_dat
    options:
      show_root_heading: true
      show_source: false
