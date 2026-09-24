# Author: Enrico Veraldi
# run script for the HII regions

import sys

from galapy.spectroscopy.utils.run_core import main as _main


def main(argv=None):
    """
    run_core using sector hii
    """
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)

    has_runs = any(arg == '-r' or arg == '--runs' or arg.startswith('--runs=') for arg in argv)
    if not has_runs:
        argv = ['--runs', 'data/hii'] + argv

    return _main(argv, sector_name='hii')


if __name__ == '__main__':
    sys.exit(main())