# Author: Enrico Veraldi
# install CLOUDY version from a single command line

import argparse
import sys

from galapy.spectroscopy.utils import (CLOUDY_REQUIRED, CloudyNotFound, detect_cloudy,
                                       ensure_cloudy, cloudy_banner, CloudyInstall)

def build_parser() -> argparse.ArgumentParser:
    """
    Builds and configures the argument parser for the galapy-install-cloudy utility.

    This function creates and returns an ArgumentParser object configured with the
    options required to support the utility's functionality, including checking
    for existing installations of CLOUDY, specifying installation prefixes, and
    configuring parallelism during the build process.

    Returns:
        argparse.ArgumentParser: Configured parser for the command-line arguments.
    """
    ap = argparse.ArgumentParser(
        prog='galapy-install-cloudy',
        description=(f'Installation of {CLOUDY_REQUIRED} from source'))
    ap.add_argument('--check', '-c', action='store_true',
                    help='search for CLOUDY already-present installation '
                         'Exit 0 if present and correct version, Exit 1 in the other cases.')
    ap.add_argument('--prefix', '-p', dest='prefix', type=str, default=None, metavar='DIR',
                    help='Installation prefix. DEFAULT: ~/.galapy/cloudy')
    ap.add_argument('--jobs', '-j', dest='jobs', type=int, default=None, metavar='N',
                    help='`make -jN` parallelism. DEFAULT: os.cpu_count()')
    return ap

def _report(inst: CloudyInstall, banner: str = None) -> None:
    """
    Reports the status of the 'inst' object, primarily used for determining if a
    resource, such as the 'CLOUDY' application, is found or not. Provides details
    such as the executable path, data path, banner information, and required
    version.

    Parameters:
        inst (CloudyInstall): The instance to be reported, containing attributes such as 'found',
            'exe', and 'data_path'.
        banner (str, optional): An optional banner string to display in the report.
    """
    if not inst.found:
        print(f'CLOUDY: NOT FOUND')
        print(' search in: $CLOUDY_EXE ; `cloudy` on $PATH ; '
              '$CLOUDY_DATA_PATH/../source/cloudy.exe')
        return
    print(f'CLOUDY: FOUND -> {inst.exe}')
    print(f'  CLOUDY_DATA_PATH  : {inst.data_path or "(to be set)"}')
    print(f'  banner            : {banner or "(not avaiable)"}')
    print(f'  requested version : {CLOUDY_REQUIRED}')


def main(argv: list = None) -> int:
    """
    Main entrypoint for the command line utility galapy-install-cloudy.

    This function parses the provided command-line arguments, checks for an existing
    installation of CLOUDY if requested, and otherwise proceeds to install CLOUDY.

    Parameters:
        argv (list, optional): List of command-line argument strings to parse.
            Defaults to sys.argv[1:] if None.

    Returns:
        int: Exit status code (0 for success, 1 for failure).
    """
    args = build_parser().parse_args(argv)
    inst = detect_cloudy()

    if args.check:
        banner = cloudy_banner(inst.exe) if inst.found else None
        _report(inst, banner)
        ok = bool(inst.found and banner and CLOUDY_REQUIRED.lstrip('C') in banner)
        return 0 if ok else 1

    if inst.found:
        print("CLOUDY already present: nothing to do.")
        _report(inst, cloudy_banner(inst.exe))
        return 0

    try:
        inst = ensure_cloudy(prefix=args.prefix, jobs=args.jobs)
    except CloudyNotFound as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0 if inst.found else 1


if __name__ == '__main__':
    sys.exit(main())
