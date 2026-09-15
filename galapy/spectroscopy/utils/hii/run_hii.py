# Author: Enrico Veraldi
# run script for the HII regions

import sys

from galapy.spectroscopy.utils.run_core import main as _main


def main(argv=None):
    """
    run_core using sector hii
    """
    return _main(argv, sector_name='hii')


if __name__ == '__main__':
    sys.exit(main())