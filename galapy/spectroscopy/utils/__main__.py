# Author: Enrico Veraldi
# Entry point for python -m galapy.spectroscopy.utils --install-cloudy

import argparse
import sys

from galapy.spectroscopy.utils import (CLOUDY_REQUIRED, CloudyNotFound, detect_cloudy,
                                       ensure_cloudy)

def build_parser():
    """
    Builds an argument parser for the spectroscopy utility script.

    This function initializes and configures an instance of ArgumentParser
    to handle command-line arguments for the spectroscopy utility. It
    provides options for installing the Cloudy software and includes relevant
    metadata about the script.

    Returns
    -------
    ArgumentParser
        An instance of ArgumentParser configured with the necessary arguments
        and metadata.
    """
    ap = argparse.ArgumentParser(
        prog='python -m galapy.spectroscopy.utils',
        description=f'pipeline offline CloudIA: guided installation of CLOUDY {CLOUDY_REQUIRED}',
        epilog='For more information, visit the documentation.'
    )
    ap.add_argument('--install-cloudy', action='store_true', help='install Cloudy')
    
    return ap

def main(argv=None):
    """
    Main entry point for handling command-line arguments and executing the related actions.

    Parses command-line arguments to determine the required actions, such as checking for
    the presence of CLOUDY or ensuring its installation. Depending on the provided arguments,
    it either prints the detection status or attempts installation.

    Args:
        argv (list, optional): A list of command-line arguments. Defaults to None, in which
        case sys.argv is used.

    Returns:
        int: Exit status code. Returns 0 if commands execute successfully, or a non-zero
        value if an error occurs.
    """
    ap = build_parser()
    args = ap.parse_args(argv)

    if not args.install_cloudy:
        ap.print_help()
        detection = detect_cloudy()
        print(f"\nCLOUDY {'found' if inst.found else 'NOT found'}"
              f"{' -> ' + inst.exe if inst.found else ''}")
        return 0

    try:
        detection = ensure_cloudy()
    except CloudyNotFound as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1

    return 0 if detection.found else 1

if __name__ == '__main__':
    sys.exit(main())


