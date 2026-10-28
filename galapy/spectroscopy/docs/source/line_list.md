# Line lists for `save line list`

CloudIA implements the module `utils/gen_linelist.py` to build the `.dat` files consumed by the CLOUDY
command `save line list`. The module reads the **master line list** of a sector — a YAML file stored in the
galapy-dataset — validates every entry against the **official CLOUDY LineLists**, and produces the fixed-format
`.dat` files that each deck references at run time: the line list proper and, when the master list declares
lines pumped by the continuum, the companion *pump* list used to remove that pumping.

!!! warning
    The master lists (`hii_full.yaml`, `pdr_full.yaml`) and the reference LineLists live in the galapy-dataset,
    not in the code.

---

## The master line list

One master list per sector, under `Nebular/Configs/Lines` of the dataset:

| Sector | Master list | Emitted files | Lines | Pumped lines |
| :--- | :--- | :--- | :--- | :--- |
| `hii` | `hii_full.yaml` (`CLOUDY.HII_LINES_FILE`) | `hii_lines.dat`, `hii_pump.dat` | 72 | 55 (13 blends, 42 single lines) |
| `pdr` | `pdr_full.yaml` (`CLOUDY.PDR_LINES_FILE`) | `pdr_lines.dat` (`pdr_pump.dat` if it declares a `pump` key) | 24 | 0 |

The file carries the top-level key `lines`, holding one mapping per transition, and the optional key `pump`
(see [Pumped resonance blends](#pumped-resonance-blends)):

```yaml
lines:
  - {label: 'H  1', wl: '6562.80', unit: 'A', name: 'Halpha',     group: HI}
  - {label: 'Blnd', wl: '5875.66', unit: 'A', name: 'HeI_5876',   group: HeI}
  - {label: 'o  3', wl: '5006.84', unit: '',  name: 'OIII_5007',  group: OIII}
  - {label: 'CO',   wl: '2600.05', unit: 'm', name: 'CO_1-0',     group: CO}
  - {label: 'N  2', wl: '6548.05', unit: 'A', name: 'NII_6548',   group: NII, verified: 'line_labels'}
```

| Key | Role |
| :--- | :--- |
| `label` | The CLOUDY species tag, **at most 4 characters, spaces included and significant** (`'H  1'`, `'he 2'`, `'Blnd'`). Matching is case-insensitive and blind to internal spaces, but the string is emitted verbatim. |
| `wl` | The wavelength, kept as a **string**: it must reproduce the official entry character by character, so `6562.80` and `6562.8` are *not* interchangeable. |
| `unit` | `A` (Angstrom), `m` or `M` (micron: CLOUDY reads the suffix case-insensitively), `c` (cm), or the empty string when the official list carries no unit (Angstrom). |
| `name` | CloudIA alias of the line. Must be unique. It is not written into the `.dat`: the parsed fragments identify the lines by CLOUDY label and wavelength, in the order of the master list (see [Parse HII](parse_hii.md)). |
| `group` | Free-form tag (`HI`, `UV`, `OIII`, `dual_origin`, ...), used only to organise the list and the diagnostics; it is **not** written into the `.dat`. |
| `verified` | Optional. The value `'line_labels'` marks a line that is **not** in the official LineLists but whose label and wavelength were checked by hand against the output of `save line labels` of CLOUDY C25: the validation against the official lists is skipped for it (unit included), and the entry is emitted as written. |

The official LineLists are a *selection* of the lines CLOUDY predicts: lines such as `N  2 6548.05A`,
`Ne 3 3967.47A`, `O  1 6300.30A` or the high Balmer lines `H  1 3970.07A` ... `3797.90A` are computed by every
model but absent from `LineList_HII.dat`. To add one, run any model with `save line labels "x.lab" last`, copy
label and wavelength exactly as they appear there (the unit is part of the printed wavelength, e.g. `6548.05A`),
and mark the entry `verified: 'line_labels'`. A wrong label then is not caught by `galapy-gen-linelist`, but by
CLOUDY itself at run time.

The `dual_origin` group marks the transitions that appear in both master lists (`[CII] 158 μm`,
`[NeII] 12.8 μm`, `[SiII] 34.8 μm`, plus the HI lines shared by the two sectors), i.e. the lines whose emission
must be recombined across the HII and PDR sectors.

### Pumped resonance blends

Every line that CLOUDY transfers is also excited by **continuum pumping**: CLOUDY adds to its intensity the
stellar continuum absorbed in the line and re-emitted in it, and reports that share as the `"<label> Pump"`
entry. The share is large for the UV resonance doublets (N V 1240, Si IV 1397, C IV 1549, Mg II 2798) and, in old
SSP nodes, for the UV intercombination lines (Si III] 1892, C III] 1909, C II] 2326: up to 99% at 70 Myr) and the
He I lines from the metastable level (3889, 10830). It is stellar light scattered in the line, and the line-only
column of the continuum never holds it. The grid stores these lines **without** the pumped share (see the
description of `lines_emergent/fluxes` in [Parse HII](parse_hii.md#1-the-fragment-job_idh5)). The `pump` key
lists every line of the master list below 10 μm that CLOUDY transfers (type `t` in `save line labels`): a blend
with its components, a single line as its own component:

```yaml
pump:
  NV_1240:   [{label: 'N  5', wl: '1238.82', unit: 'A'}, {label: 'N  5', wl: '1242.80', unit: 'A'}]
  CIV_1549:  [{label: 'C  4', wl: '1548.19', unit: 'A'}, {label: 'C  4', wl: '1550.77', unit: 'A'}]
  SiIII]_1892: [{label: 'si 3', wl: '1892.03', unit: ''}]
```

Beyond 10 μm the `Pump` entry of a fine-structure line is the excitation by the CMB, and at high redshift and
low density it exceeds the line itself (×2 for [O III] 88 μm at $z = 10$): it is not a share of the emission,
and the contrast against the CMB is applied at runtime, so the far-infrared lines carry no `pump` entry.

Each key must be the `name` of an entry of `lines` (the blend itself, e.g. `blnd 1240`, or the line); the components carry
`label`, `wl`, `unit` with the same conventions as `lines`. From them the generator writes the pump list (see
[The emitted pump `.dat`](#2-the-emitted-pump-dat)), the parser uses its output to remove the pumped share of
each line (see [Parse HII](parse_hii.md#4-derived-quantities)).

### The reference LineLists

`Nebular/Configs/Lines_Ref` holds a frozen copy of the two official CLOUDY lists, `LineList_HII.dat` and
`LineList_PDR.dat` (`CLOUDY.HII_LINES_OFF`, `CLOUDY.PDR_LINES_OFF`), together with `LineList.sha256` recording
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
| `--hii` | None | Generate `hii_lines.dat` (and `hii_pump.dat`) from the HII master list (`hii_full.yaml`). |
| `--pdr` | None | Generate `pdr_lines.dat` (and `pdr_pump.dat`, if the list has a `pump` key) from the PDR master list (`pdr_full.yaml`). |
| `--out` | `OUT` | Output directory of the `.dat` files. Created if missing. <br> Default: `data/lines` (`DEFAULT_OUT_DIR`). |

`--hii` and `--pdr` form a **mutually exclusive, required** group: exactly one sector per invocation. The
master list and the reference LineLists are always resolved from the GalaPy database, and have no flag.

## Validation

Before writing anything, the entries go through the checks below, and the `.dat` files are emitted **only if
all of them are clean**. All the errors are collected first and reported together, so a single run is enough to fix
the master list.

### 1. Against the official LineLists (`validate`)

For each entry not marked `verified: 'line_labels'`, the *(label, wl)* key is looked up in the official index:

- a missing key means the line is not a valid CLOUDY label/wavelength pair —
  `-> Not found in CLOUDY official LineList`;
- a key that is found but whose `unit` differs from the official one (compared case-insensitively, with the
  empty string treated as a unit of its own) is reported as a unit mismatch.

### 2. Internal consistency (`check_duplicates`)

- duplicate *(label, wl)* pairs, which would make CLOUDY write the same column twice;
- duplicate `name` aliases, which would collide in the parsed tables downstream.

### 3. Pumped lines

Every key of `pump` must be the `name` of an entry of `lines` — `pump: line '<name>' not in lines`. The
components of the blends are not validated against the official LineLists.

### Failure mode

On any error the module prints the whole report on `stderr` and exits with status `1`, leaving the previous
`.dat` files untouched:

```text
[gen_line] 2 Wrong format in .../Nebular/Configs/Lines/hii_full.yaml:
  NII_5755         label='n  2' wl=5754.6 -> Not found in CLOUDY official LineList
  OIII_5007        unit 'A' != '' of official CLOUDY LineList

 no .dat written
```

The unit mismatch means the official list writes the line without a unit (Angstrom by default, e.g.
`o  3 5006.84 air`): the fix is `unit: ''`, which is the same wavelength for CLOUDY. A line reported as not found
is either mistyped, or one of the lines that CLOUDY computes but the official lists omit: in that case mark it
`verified: 'line_labels'` after checking it on `save line labels`.

A successful run reports the number of validated lines and, when present, of the pumped lines:

```text
[gen_line] 72 lines -> data/lines/hii_lines.dat all validated against CLOUDY official LineLists
[gen_line] 55 pumped lines -> data/lines/hii_pump.dat
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

### 2. The emitted pump `.dat`

`emit_pump_dat` writes, for each key of `pump`, the master-list entry itself (as in `{sector}_lines.dat`), then
each of its components followed by its `Pump` entry, the CLOUDY label `"<ion> Pump"` of the share of the line due
to continuum pumping. A single line is its own component, so it appears twice before its `Pump` entry:

```text
blnd 1240
N  5 1238.82A
"N  5 Pump" 1238.82A
N  5 1242.80A
"N  5 Pump" 1242.80A
blnd 1549
...
si 3 1892.03
si 3 1892.03
"si 3 Pump" 1892.03
```

The file is named after the line list, `{sector}_pump.dat`, and written next to it only when the master list has
a non-empty `pump` key.

### 3. How the decks consume them

The deck template references the list by bare filename:

```text
save last line list "hii_{{ job_id }}.lines" "hii_lines.dat" emergent absolute column no hash
save last line list "hii_{{ job_id }}.pump" "hii_pump.dat" absolute column no hash
```

The pump list is saved **intrinsic** (no `emergent`): the `Pump` entries are zero in the emergent intensities.
Both `.dat` must sit in the working directory of the CLOUDY process. Staging is handled by
`run_core.stage_job`, which copies `{sector}_lines.dat` and the extra lists of the sector (`Sector.extra_lists`,
i.e. `{sector}_pump.dat`) next to the deck before launching the run, and resolves their source directory in this
order:

1. the `-l`/`--linelists` flag of the runner;
2. the environment variable `CLOUDIA_LINELISTS`;
3. `DEFAULT_OUT_DIR`, i.e. `data/lines`.

A missing `{sector}_lines.dat` makes the job fail at staging time with `FileNotFoundError`, before CLOUDY is
started — see [Running CLOUDY](run_hii.md#2-staging). The extra lists are instead copied **only when present**:
a missing `hii_pump.dat` is not caught at staging, and the HII deck, which always saves the pump list, fails
inside CLOUDY. Always generate the two files together, with the same `galapy-gen-linelist --hii --out <dir>`.

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

::: galapy.spectroscopy.utils.gen_linelist.emit_pump_dat
    options:
      show_root_heading: true
      show_source: false
