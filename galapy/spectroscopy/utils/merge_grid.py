# Author: Enrico Veraldi
# Merge all the single parsed jobs (parse_one) into a single .h5 file

import argparse
import os
import pathlib
import sys

import h5py
import numpy as np

ROOT_SHARED = ('continuum/wave_grid', 'line_names', 'lines_emergent/wavelengths_rest')


def merge(frag_dir, out_path, cloudy_version='C25.00', ssp_lib='parsec22.NT'):
    """
    Merges multiple .h5 fragments from a directory into a single .h5 corpus file.

    This function reads all .h5 files from the specified directory and combines
    their datasets into a single output file while validating the consistency of
    shared root datasets among the fragments.

    Args:
        frag_dir (str): The directory containing the .h5 fragment files to merge.
        out_path (str): The file path for the resulting merged .h5 corpus.
        cloudy_version (str): The version of the "cloudy" software to store as an
            attribute in the output file. Default is 'C25.00'.
        ssp_lib (str): The stellar system population library to store as an
            attribute in the output file. Default is 'parsec22.NT'.

    Returns:
        int: The total number of grid points written to the merged corpus file.

    Raises:
        ValueError: If no .h5 fragments are found in the specified directory or if
            there is a mismatch in the shared root datasets among the fragments.
        BaseException: If an unexpected error occurs during the merging process,
            ensuring the intermediate output file is deleted in such cases.
    """
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frags = sorted(p for p in pathlib.Path(frag_dir).glob('*.h5')
                   if p.resolve() != out_path.resolve())
    if not frags:
        raise ValueError(f"[merge] {frag_dir}: no fragments available")
    part = out_path.with_name(out_path.name + '.part')
    try:
        with h5py.File(part, 'w') as out:
            out.attrs['cloudy_version'] = cloudy_version
            out.attrs['ssp_lib'] = ssp_lib
            ref = None
            n_pts = 0
            for fp in frags:
                with h5py.File(fp, 'r') as fr:
                    if ref is None:
                        for d in ROOT_SHARED:
                            out.create_dataset(d, data=fr[d][:])
                        ref = {d: out[d][:] for d in ROOT_SHARED}
                    else:
                        for d in ROOT_SHARED:
                            if not np.array_equal(fr[d][:], ref[d]):
                                raise ValueError(
                                    f"[merge] {fp}: '{d}' different from first fragment ")
                    for gname in (k for k in fr.keys() if k.startswith('grid_point_')):
                        fr.copy(fr[gname], out, name=gname)
                        n_pts += 1
            out.attrs['n_grid_points'] = n_pts
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, out_path)
    print(f"[merge] {len(frags)} fragments (jobs), {n_pts} grid_point, file generated in {out_path}")
    return n_pts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    sec = ap.add_mutually_exclusive_group()
    sec.add_argument('--hii', action='store_true',
                     help='sector HII: default frags=data/hii/parsed, out=data/products/hii_grid.h5')
    sec.add_argument('--pdr', action='store_true',
                     help='sector PDR: default frags=data/pdr/parsed, out=data/products/pdr_grid.h5')

    ap.add_argument('--frags', default=None,
                    help='directory fragments for the sector, <job_id>.h5')
    ap.add_argument('--out', default=None,
                    help='path .h5 merged')
    ap.add_argument('--cloudy-version', default='C25.00',
                    help='CLOUDY version written in metadata (default: C25.00)')
    ap.add_argument('--ssp-lib', default='parsec22.NT',
                    help='library SSPs saved in metadata (default: parsec22.NT)')

    args = ap.parse_args(argv)
    sector = 'hii' if args.hii else ('pdr' if args.pdr else None)

    frags = args.frags or (f'data/{sector}/parsed' if sector else None)
    out = args.out or (f'data/products/{sector}_grid.h5' if sector else None)

    if not frags:
        ap.error("Sector to be specified (--hii or --pdr) or list of --frags.")
    if not out:
        ap.error("Sector to be specified (--hii o --pdr) put a --out.")

    merge(frags, out, cloudy_version=args.cloudy_version, ssp_lib=args.ssp_lib)
    return 0


if __name__ == '__main__':
    sys.exit(main())