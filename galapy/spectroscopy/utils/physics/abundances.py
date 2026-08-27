# Author: Enrico Veraldi
# chemistry for the HII and PDR with scaling, Galatic Concordance GC (Nicholls+17)
# NOTE: this module compute quantities for the HII and PDR models and the translation table
# NOTE: files related to scaling factors and GC are in galapy-dataset/Nebular/Abundances
# NOTE: changing chemistry means modify files in galapy-dataset and rerun this module

import argparse
import hashlib
import pathlib
import sys

import numpy as np
import yaml
from IPython.core import payload

import galapy.internal.constants as CONST
from galapy.internal.data import DataFile
from galapy.internal.globs import ABU_DIR

#=============== ZMAP DEFAULT GRID ===============
ZMAP_DEFAULT_GRID = (-3.0, 1.0, 4001)
#=================================================

class ChemistryHashMismatch(RuntimeError):
    """
    Represents a custom runtime error raised when the hash of the 
    chemistry-related file 'zeta_of_Z.npz' does not match the expected 
    hash defined by the associated chemical files.

    This exception is used to verify the integrity of chemical files to ensure data consistency 
    and correctness.

    Attributes:
        None
    """


def _sha256(path):
    """Compute SHA-256 hash of a file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file for which to compute the hash.

    Returns
    -------
    str
        Hexadecimal string representation of the SHA-256
    """
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

#CHEMISTRY=============================================================================================
class Chemistry:
    def __init__(self, abn_file=None, scale_file=None, zmap_file=None):
        """
        Reads and sets up fiducial data, scale parameters, and a
        zeta map. It processes data from given ABN file, scale file, and optionally a
        zeta map file to initialize various attributes

        Parameters:
        abn_file: Optional[str]
            Path to the ABN file. If not provided, a default file location is used.
        scale_file: Optional[str]
            Path to the scale file. If not provided, a default file location is used.
        zmap_file: Optional[str]
            Path to the zeta map file. If not provided, the zeta map is not loaded.
        """
        self._abn_path = abn_file or DataFile(CONST.ABN_FILE, ABU_DIR).get_file()
        self._scl_path = scale_file or DataFile(CONST.SCALE_FILE, ABU_DIR).get_file()
        self.fiducial = self._read_abn(self._abn_path)
        scl = yaml.safe_load(open(self._scl_path))
        self.chi0 = float(scl['breakpoints']['chi0'])
        self.chi1 = float(scl['breakpoints']['chi1'])
        self.xi_O = {k: float(v) for k, v in scl['xi_O'].items()}
        self.special = scl['special']
        self.log_OH_fid = float(scl['fiducial_log_OH'])

        #zmap check
        self.Z_GC = self.metallicity_from_zeta(0.0)
        self._zmap_logzeta = self._zmap_Z = None

        if zmap_file is not False:
            self._load_zeta_map(zmap_file)

    #==================== READING ==================================
    @staticmethod
    def _read_abn(path):
        """
        read the .abn file from galapy-dataset
        """
        inv = {v: k for k, v in CONST.CLOUDY_NAME.items()}
        out = {}
        for raw in pathlib.Path(path).read_text().splitlines():
            if raw.startswith('*'):
                break
            if raw.startswith('#'):
                continue
            if not raw.strip():
                raise ValueError(f"{path}: empty raw - CLOUDY abort it (no empty raws permitted)")
            if 'GRAINS' in raw.upper():
                raise ValueError(f"{path}: presence of GRAINS in abn - double counting (already in template grains)")
            tok = raw.split()
            sym = inv.get(tok[0].upper())
            if sym is None:
                raise ValueError(f"{path}: element not recognised (galapy.internal.constants): {tok[0]!r}")
            val = float(tok[1])
            if val <= 0.0:
                raise ValueError(f"{path}: negative abundance for {sym}")
            out[sym] = val
        if len(out) != 30:
            raise ValueError(f"{path}: {len(out)} elements, expected 30. Missing elements will be switch off")
        return {k: v / out['H'] for k, v in out.items()}

    def _load_zeta_map(self, zmap_file):
        """
        read zeta_Z map from .npz file
        """
        path = zmap_file or DataFile(CONST.ZMAP_FILE, CONST.ABU_DIR).get_file()
        d = np.load(path)
        want = (_sha256(self._abn_path), _sha256(self._scl_path))
        got = (str(d['sha_abn']), str(d['sha_scaling']))
        if got != want:
            raise ChemistryHashMismatch(
                f"{CONST.ZMAP_FILE} generated from chemistry files (.abn and .yaml) different from the loaded "
                f"ones. You can re-generate it via cli command `galapy-build-zeta-map`. "
            )
        self._zmap_logzeta = d['log_zeta_O']
        self._zmap_Z = d['Z_gas']

    # ============ CHECK ===============

    @staticmethod
    def _assert_log_zeta(lz, caller):
        """
        Asserts the validity of the provided log10(zeta_O) scalar value for subsequent
        calculations. The method ensures that the provided value is a scalar and
        that it falls within the acceptable range [-3.0, 1.0].
        This function is intended for internal use only.

        Parameters:
            lz: The scalar value of log10(zeta_O) to be validated.

            caller: The name of the calling function or method from which
                    this validation is triggered. Used to provide context
                    in case of exceptions.

        Raises:
            TypeError: If the provided lz value is not of scalar type.

            ValueError: If lz falls outside the permitted range of [-3.0, +1.0].

        Note:
            The provided value must represent log10(zeta_O). Linear zeta_O or fractional
            mass values are not accepted.
        """
        if not np.isscalar(lz):
            raise TypeError(f"{caller}: expected scalar log10(zeta_O), not {type(lz)}")
        if not (-3.0 <= float(lz) <= 1.0):
            raise ValueError(
                f"{caller}: log_zeta_O = {lz} out of range [-3, +1]."
            )

    # ===================== MAIN FUNCTIONS ===================================

    def delta_O(self, element, log_zeta_O):
        """
        Compute the oxygen abundance delta value for a given element based on the specified
        logarithmic ionization parameter (log zeta_O) Nicholls+17

        Parameters:
        element : str
            The symbol of the element for which the delta value should be calculated. Supported
            values include 'H', 'He', and 'N', or other elements defined in the instance's parameters.
        log_zeta_O : float
            The logarithmic oxygen ionization parameter (log zeta_O) in base 10 (fiducial Nicholls+17)

        Returns:
        float
            The computed delta value for the specified element as a numeric value in dex.

        Raises:
        ValueError
            If the log zeta_O value does not satisfy the constraints imposed by the internal
            assertion logic.
        KeyError
            If the specified element is not found in the necessary configuration or data dictionary.
        """
        self._assert_log_zeta(log_zeta_O, 'delta_O')
        lz = float(log_zeta_O)
        if element == 'H':
            return 0.0
        if element == 'He':
            p = self.special['He']
            he = p['A'] * (1.0 + p['B'] * 10.0**lz)
            return float(np.log10(he / self.fiducial['He']))
        if element == 'N':
            p = self.special['N']
            return float(np.log10(10.0**p['a'] + 10.0**(lz + p['b'])))
        xi = self.xi_O[element]
        if lz < self.chi0:
            return float(xi)
        if lz > self.chi1:
            return float(xi / self.chi0 * self.chi1)
        return float(xi / self.chi0 * lz)

    def abundance_pattern(self, log_zeta_O):
        """
        Calculates the abundance pattern for 30 elements at the specified oxygen
        abundance logarithm (log_zeta_O). Returns the ratios of each element to
        hydrogen (n(X)/n(H)).

        Arguments:
            log_zeta_O (float): The logarithm of the abundance ratio of oxygen.

        Returns:
            dict: A dictionary where keys represent element names (str) and values
            represent their corresponding abundance ratio to hydrogen (float),
            including hydrogen itself.
        """
        self._assert_log_zeta(log_zeta_O, 'abundance_pattern')
        lz = float(log_zeta_O)
        out = {'H': 1.0}
        for s, a0 in self.fiducial.items():
            if s == 'H':
                continue
            uniform = 0.0 if s == 'He' else lz
            out[s] = a0 * 10.0**(self.delta_O(s, lz) + uniform) # X/H = (X/H)_0 *10 ** (delta_O(X)+log_zeta_O), eq.7 Nicholls+17
        return out

    def metallicity_from_zeta(self, log_zeta_O):
        """
        Calculates the mass fraction Z_gas based on the input abundance pattern.

        Parameters:
            log_zeta_O (float): The logarithm of the oxygen abundance zeta_O.

        Returns:
            float: The calculated mass fraction Z_gas.
        """
        ab = self.abundance_pattern(log_zeta_O)
        tot = sum(CONST.ATOMIC_WEIGHT[s] * n for s, n in ab.items())
        met = sum(CONST.ATOMIC_WEIGHT[s] * n for s, n in ab.items() if s not in ('H', 'He'))
        return float(met / tot)

    def zeta_from_metallicity(self, Z_gas):
        """
        This method calculates the value of zeta (Nicholls+17) based on the provided gas metallicity
        Z_gas, using pre-tabulated values. The input metallicity must lie within the
        tabulated domain to ensure reliable interpolation. Conversion to zeta accounts
        for the absolute mass fraction without relying on solar scaling or elemental
        abundance patterns.

        Parameters:
            Z_gas: float
                The gas metallicity value (absolute mass fraction) for which the method
                computes the corresponding zeta. This must fall within the range of
                tabulated metallicity values.

        Raises:
            ValueError:
                If Z_gas lies outside the tabulated metallicity range.

        Returns:
            float: The interpolated value of zeta corresponding to the input metallicity.
        """
        if self._zmap_Z is None:
            raise RuntimeError("zeta_from_metallicity: zeta map not loaded. You can generate it via cli command `galapy-build-zeta-map`.")

        Z = float(Z_gas)
        lo, hi = self._zmap_Z[0], self._zmap_Z[-1]
        if not (lo <= Z <= hi):
            raise ValueError(
                f"zeta_from_metallicity: Z_gas = {Z:.4e} out of grid domain, increase grid size "
            )
        return float(np.interp(Z, self._zmap_Z, self._zmap_logzeta))

    # ===================== INTERFACES ===================================

    def oh_from_zeta(self, log_zeta_O):
        """
        Returns the oxygen abundance in the form 12 + log(O/H).

        This method calculates the oxygen abundance 12 + log(O/H) using the given
        logarithmic metallicity value (log_zeta_O).

        Parameters:
            log_zeta_O (float): The logarithmic metallicity value.

        Returns:
            float: The calculated oxygen abundance in the form 12 + log(O/H).
        """
        return float(self.log_OH_fid + 12.0 + float(log_zeta_O))

    def zeta_from_oh(self, oh):
        """
        Computes the zeta value from a given oxygen abundance in form 12 + log(O/H) .

        Parameters:
        oh : float
            The logarithmic oxygen-to-hydrogen ratio 12+log(O/H), expressed as a float.

        Returns:
        float
             log_zeta_O, The logarithmic metallicity value.
        """
        return float(float(oh) - 12.0 - self.log_OH_fid)

    def to_solar_units(self, log_zeta_O, reference='GC'):
        """
        Converts a given logarithmic oxygen abundance ratio (log_zeta_O) to solar units
        (log10(Z_gas/Z_sun)) based on a specified reference metallicity.

        Parameters:
            log_zeta_O: float
                The logarithmic oxygen abundance ratio to be converted.
            reference: str, default 'GC'
                The reference metallicity to be used for the conversion. Must be
                one of the values specified in the SOLAR_REFERENCES dictionary (constants).

        Returns:
            float:
                The converted metallicity in solar units.

        Raises:
            ValueError:
                If the provided reference is not found in SOLAR_REFERENCES.
        """
        if reference == 'GC':
            Z_ref = self.Z_GC #derived
        elif reference in CONST.SOLAR_REFERENCES:
            Z_ref = CONST.SOLAR_REFERENCES[reference]
        else:
            raise ValueError(f"reference not present in dictionary: {reference!r}. "
                             f"Available: {sorted(CONST.SOLAR_REFERENCES)}")

        return float(np.log10(self.metallicity_from_zeta(log_zeta_O) / Z_ref))

    # ===================== GENERATION CLOUDY CHEMISTRY + .ABN SCALED CHECK ===================================

    def element_scale_lines(self, log_zeta_O):
        """
        Generates the lines for 'element scale factor <el> <Delta> log' for the .abn
        file for CLOUDY runs

        Arguments:
            log_zeta_O (float): The logarithmic zeta_O factor to calculate scaling
                                adjustments for elements.

        Returns:
            list: A list of strings in the format 'element scale factor <el> <Delta> log'.
        """
        self._assert_log_zeta(log_zeta_O, 'element_scale_lines')
        lines = []
        for s in self.fiducial:
            if s == 'H':
                continue
            d = self.delta_O(s, log_zeta_O)
            if s != 'He' and s != 'N' and abs(self.xi_O.get(s, 0.0)) < 1e-12:
                continue  # Xi_O = 0, scale already with zeta_O
            lines.append(f"element scale factor {CONST.CLOUDY_NAME[s].lower():<11} {d:+.6f} log")
        return lines

    def write_abn_file(self, path, log_zeta_O):
        """
        Writes a .abn file with pre-scaled abundances for a specific use case.

        This method generates a file with abundances already scaled according to the
        supplied log_zeta_O parameter. It is intended as a validation tool

        Attributes
        ----------
        path : pathlib.Path
            The target path where the .abn file will be written.
        log_zeta_O : float
            The logarithmic metallicity scaling factor used for abundance scaling.

        Parameters
        ----------
        path : pathlib.Path
            The file path where the .abn output is written. A temporary file is
            created and atomically renamed to this target.
        log_zeta_O : float
            The logarithmic scaling factor for metallicity, used to calculate
            the abundance values included in the file.

        """
        ab = self.abundance_pattern(log_zeta_O)
        body = [f"#.abn scaled to log(zeta_O) = {log_zeta_O:+.6f}",
                "#generated by physics.abundances"]
        body += [f"{CONST.CLOUDY_NAME[s]}\t{ab[s]:.5E}" for s in self.fiducial]
        body.append("*" * 40)
        tmp = pathlib.Path(str(path) + '.tmp')
        tmp.write_text("\n".join(body) + "\n")
        tmp.replace(path)

#CHEMISTRY=============================================================================================

#GRAINS===============================================
def grain_scale_from_xi_d(xi_d):
    """
    Calculates the grain scaling factor dependent on xi_d.

    This function computes the xi_d-dependent part of the grain factor g(xi_d) using
    the formula g(xi_d) = xi_d / XI_D_MW. It does not return the full argument for
    'grains ISM', but only the component dependent on xi_d. The full grain factor
    is given by:

        grain_fac = (Z(zeta_O) / Z_GC) * g(xi_d)

    Here, Z(zeta_O) derived using the Chemistry.metallicity_from_zeta() method.

    Args:
        xi_d (float): Dust-to-gas mass ratio, xi_d. Must be in the range (0, 1].

    Raises:
        ValueError: If xi_d is not within the range (0, 1].

    Returns:
        float: The calculated grain scaling factor g(xi_d).
    """
    xi = float(xi_d)
    if not (0.0 < xi <= 1.0):
        raise ValueError(f"grain_scale_from_xi_d: xi_d = {xi} out of range (0, 1]")
    return xi / CONST.XI_D_MW
#GRAINS===============================================

#FORMAT===============================================
def format_cloudy_float(x, sig=6):
    """
    Formats a linear numerical argument for cloudy number processing in exponential notation.

    Parameters:
    x : float
        A numerical value to be formatted in exponential notation.
    sig : int
        The number of significant digits to retain in the exponential format
        (default is 6).

    Returns:
    str
        The numerical value represented in exponential notation, formatted
        with the specified number of significant digits.
    """
    return f"{float(x):.{sig}e}"

#GENERATE ZETA MAP GRID===============================================

def build_zeta_map(out_path, abn_file=None, scale_file=None, lz_min=ZMAP_DEFAULT_GRID[0],
                   lz_max=ZMAP_DEFAULT_GRID[1], n=ZMAP_DEFAULT_GRID[2]):
    """
    Build a zeta map that correlates logarithmic ionizing photon escape fraction values with gas metallicity.

    This function generates a mapping between log_zeta_O and Z_gas values using the given abundance
    and scale files.

    Parameters:
        out_path (str): Path where the output file containing zeta map data will be saved.
        abn_file (Optional[str]): Path to the file containing abundance data. Default is None.
        scale_file (Optional[str]): Path to the file containing scale data. Default is None.
        lz_min (float): Minimum value of logarithmic ionizing photon escape fraction log_zeta_O.
        lz_max (float): Maximum value of logarithmic ionizing photon escape fraction log_zeta_O.
        n (int): Number of points for the zeta grid. Must be at least 2.

    Returns:
        dict: A dictionary containing the generated zeta map data and metadata such as SHA-256
        hashes of the input files.

    Raises:
        ValueError: If the number of points `n` is less than 2.
        ValueError: If `lz_min` is greater than or equal to `lz_max`.
        ValueError: If the generated Z values are not monotonic.
    """
    if int(n) < 2:
        raise ValueError(f"build_zeta_map: n = {n} must be >= 2")
    if float(lz_min) >= float(lz_max):
        raise ValueError(f"build_zeta_map: lz_min = {lz_min} must be < lz_max = {lz_max}")

    chem = Chemistry(abn_file=abn_file, scale_file=scale_file, zmap_file=False)

    lz = np.linspace(float(lz_min), float(lz_max), int(n))
    Z = np.array([chem.metallicity_from_zeta(float(v)) for v in lz])

    #check map is monotonic and so invertible
    bad = np.flatnonzero(np.diff(Z) <= 0)
    if bad.size:
        i = int(bad[0])
        raise ValueError(f"build_zeta_map: Z(lz) not monotonic, lz[{i}] = {lz[i]}")

    load_dict = {'log_zeta_O' : lz, 'Z_gas' : Z, 'sha_abn' : _sha256(chem._abn_path),
                 'sha_scaling' : _sha256(chem._scl_path)}

    #Writing
    out = pathlib.Path(out_path)
    tmp = out.with_name(out.name + '.tmp')
    with open(tmp, 'wb') as fh:
        np.savez(fh, **load_dict)
    tmp.replace(out)

    return payload

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='galapy-build-zeta-map',
        description='Build a zeta map that correlates logarithmic ionizing '
                    'photon escape fraction values with gas metallicity.'
    )
    ap.add_argument('-o', '--out', default=CONST.ZMAP_FILE, help=f"output file path, default {CONST.ZMAP_FILE}")
    ap.add_argument('-a', '--abn', default=CONST.ABN_FILE, help=f"abundance file path, default {CONST.ABN_FILE}")
    ap.add_argument('-s', '--scale', default=CONST.SCALE_FILE, help=f"scale file path, default {CONST.SCALE_FILE}")
    ap.add_argument('-l', '--lz-min', default=ZMAP_DEFAULT_GRID[0], help=f"minimum log_zeta_O, default {ZMAP_DEFAULT_GRID[0]}")
    ap.add_argument('-u', '--lz-max', default=ZMAP_DEFAULT_GRID[1], help=f"maximum log_zeta_O, default {ZMAP_DEFAULT_GRID[1]}")
    ap.add_argument('-n', '--n', default=ZMAP_DEFAULT_GRID[2], help=f"number of points for the zeta grid, default {ZMAP_DEFAULT_GRID[2]}")
    args = ap.parse_args(argv)

    try:
        p = build_zeta_map(args.out, abn_file=args.abn, scale_file=args.scale, lz_min=args.lz_min, lz_max=args.lz_max, n=args.n)
    except (ValueError, OSError, ChemistryHashMismatch) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    lz, Z = p['log_zeta_O'], p['Z_gas']
    step = (lz[-1] - lz[0]) / (lz.size - 1)
    print(f"write {args.out}: {lz.size} points, log_zeta_O in [{lz[0]:+.3f}, {lz[-1]:+.3f}]"
          f"step {step:.4f}, Z_gas in [{Z[0]:.4e}, {Z[-1]:.4e}]")
    print(f"    sha_abn = {p['sha_abn']}")
    print(f"    sha_scaling = {p['sha_scaling']}")

    if lz.size < ZMAP_DEFAULT_GRID[2]:
        print(f"WARNING: {lz.size} points, not enough for {ZMAP_DEFAULT_GRID[2]} points")

    return 0

if __name__ == '__main__':
    sys.exit(main())


