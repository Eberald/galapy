# Author: Enrico Veraldi
#Generate the .dat file for CLOUDY's `save line list` from the master line
#list (YAML) stored in galapy-dataset

import argparse
import pathlib
import re
import sys

import yaml

from galapy.internal.data import DataFile
from galapy.internal.globs import LINES_YML, LINES_REF
from galapy.internal.constants import (HII_LINES_FILE, PDR_LINES_FILE,
                                       HII_LINES_OFF, PDR_LINES_OFF)

# Tag of a line of CLOUDY (<=4 char, space OK), lambda,
# unit optional, keyword air/vacuum optional, comment '#' optional.
LINE_RE = re.compile(r'^(.{1,4}?)\s{1,}([0-9][0-9.]*)\s*([AmcM]?)\s*(air|vacuum)?\s*$', re.I)

# Sector
SECTORS = {
    'hii': (HII_LINES_FILE, 'hii_lines.dat'),
    'pdr': (PDR_LINES_FILE, 'pdr_lines.dat'),
}

# Official CLOUDY LineLists in LINES_REF
OFFICIAL_FILES = (HII_LINES_OFF, PDR_LINES_OFF)
# default working output directory
DEFAULT_OUT_DIR = pathlib.Path('data') / 'lines'


def load_official(paths):
    """
    Loads a mapping of labels officially recognized by the CLOUDY software.
    Raises
    ------
    SystemExit
        If no valid labels are extracted from the provided paths.

    Parameters
    ----------
    paths : list[str]
        List of file paths containing the raw label data.

    Returns
    -------
    dict[tuple[str, str], tuple[str, str]]
        A dictionary where:
        - Keys are tuples of normalized labels and lambda values.
        - Values are tuples of the original label and associated unit.
    """
    idx = {}
    for p in paths:
        for raw in pathlib.Path(p).read_text().splitlines():
            s = raw.split('#')[0].strip()
            if not s:
                continue
            m = LINE_RE.match(s)
            if not m:
                continue
            lab, wl, unit, _kw = m.groups()
            idx[(lab.strip().lower().replace(' ', ''), wl)] = (lab.strip(), unit or '')
    if not idx:
        raise SystemExit(f"no tags read/present in {paths}")
    return idx


def validate(entries, official):
    """
    Validates entries in YAML file of line (galapy-dataset) against an official LineList
    from CLOUDY (always in galapy-datset)

    Parameters:
    entries (list): List of dictionaries where each dictionary represents an entry
        to be validated. Every entry should contain 'label', 'wl', and optionally
        'unit' keys.
    official (dict): Dictionary representing the official LineList. Keys are tuples
        of normalized label and wavelength (wl), while values are tuples containing
        the corresponding data (e.g., reference or unit).

    Returns:
    list: A list of error messages describing any discrepancies found between the
        YAML entries and the official LineList. An empty list is returned if no
        errors are detected.
    """
    errs = []
    for e in entries:
        key = (str(e['label']).strip().lower().replace(' ', ''), str(e['wl']))
        if key not in official:
            errs.append(f"  {e['name']:16s} label={e['label']!r} wl={e['wl']} "
                        f"-> Not found in CLOUDY official LineList")
            continue
        _, off_unit = official[key]
        if (e.get('unit') or '').upper() != off_unit.upper():
            errs.append(f"  {e['name']:16s} unit {e.get('unit')!r} != "
                        f"{off_unit!r} of official CLOUDY LineList")
    return errs


def check_duplicates(entries):
    """
    Check for duplicate entries in a dataset to prevent errors in further processing.

    Args:
        entries (list[dict]): A list of dictionaries where each dictionary represents
            an entry with keys 'label', 'wl', and 'name'.

    Returns:
        list[str]: A list of error messages indicating the duplicates found within the
            dataset.
    """
    errs, seen_key, seen_name = [], set(), set()
    for e in entries:
        k = (str(e['label']).strip().lower(), str(e['wl']))
        if k in seen_key:
            errs.append(f"  duplicate (label, wl): {k}")
        seen_key.add(k)
        if e['name'] in seen_name:
            errs.append(f"  duplicate alias: {e['name']}")
        seen_name.add(e['name'])
    return errs


def emit_dat(entries):
    """
    Generates a formatted string representing a list of line entries with save line list format (.dat)

    Args:
        entries (list[dict]): A list of dictionaries, each representing a line. Each dictionary must
                              include the key 'label' (str) for the line label and 'wl' (float) for the
                              wavelength. Optionally, it may include the key 'unit' (str) for the unit.

    Returns:
        str: A formatted string with each line labeled and organized as per the specified rules,
             with each entry on a new line.
    """
    out = []
    for e in entries:
        lab = f"{e['label']:<4s}" #fixed width in line name
        out.append(f"{lab} {e['wl']}{e.get('unit') or ''}")
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sector = ap.add_mutually_exclusive_group(required=True)
    sector.add_argument('--hii', dest='sector', action='store_const', const='hii',
                        help=f'HII master list ({HII_LINES_FILE}) -> hii_lines.dat')
    sector.add_argument('--pdr', dest='sector', action='store_const', const='pdr',
                        help=f'PDR master list ({PDR_LINES_FILE}) -> pdr_lines.dat')
    ap.add_argument('--out', default=str(DEFAULT_OUT_DIR),
                    help=f'output directory for the .dat of save line list '
                         f'(default: {DEFAULT_OUT_DIR})')
    args = ap.parse_args()

    yaml_name, dat_name = SECTORS[args.sector]

    # Master list and official LineLists resolved from galapy-dataset
    yaml_path = DataFile(yaml_name, LINES_YML).get_file()
    official_paths = [DataFile(f, LINES_REF).get_file() for f in OFFICIAL_FILES]

    spec = yaml.safe_load(pathlib.Path(yaml_path).read_text())
    entries = spec['lines']

    official = load_official(official_paths)
    errs = validate(entries, official) + check_duplicates(entries)
    if errs:
        print(f"[gen_line] {len(errs)} Wrong format in {yaml_path}:", file=sys.stderr)
        print('\n'.join(errs), file=sys.stderr)
        print("\n no .dat written", file=sys.stderr)
        raise SystemExit(1)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / dat_name
    out_path.write_text(emit_dat(entries))
    print(f"[gen_line] {len(entries)} lines -> {out_path} all validated against CLOUDY official LineLists", file=sys.stderr)


if __name__ == '__main__':
    main()