# Author: Enrico Veraldi
# run script for the HII regions: CLOUDY run + parse of each successful model

import sys
from galapy.spectroscopy.utils.run_core import main as _main

# The HII-specific facts of this driver
RUNS_DEFAULT = 'data/hii'

# FILE inputs of utils/hii/parse_one.py, forwarded by run_core as `flag value`:
# (flag, default, help).
PARSE_INPUTS = (
    ('--spec', 'data/grids/hii/hii_grid_spec.h5',
     'hii_grid_spec.h5: expanded job spec (gen_lhs), read by parse_one'),
    ('--ssp-meta', 'data/cloudy_seds/ssp_metadata.json',
     'ssp_metadata.json (DD-2): Q_H of the SSP node, for the s_k factor'),
)


def main(argv=None):
    """
    run_core using sector hii: run + parse (unless --no-parse / --parse-only)
    """
    return _main(argv, sector_name='hii', runs_default=RUNS_DEFAULT,
                 parse_inputs=PARSE_INPUTS)


if __name__ == '__main__':
    sys.exit(main())