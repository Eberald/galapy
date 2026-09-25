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
    ap.add_argument('--frags', required=True, help='directory fragments <job_id>.h5 '
                                                   '(es: data/hii/parsed or data/pdr/parsed')
    ap.add_argument('--out', required=True,
                    help='output name and directory es. data/products/[hii/pdr]_grid.h5')
    ap.add_argument('--cloudy-version', default='C25.00')
    args = ap.parse_args(argv)
    merge(args.frags, args.out, cloudy_version=args.cloudy_version)
    return 0


if __name__ == '__main__':
    sys.exit(main())