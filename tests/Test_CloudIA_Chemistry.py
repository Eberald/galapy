# Author: Enrico Veraldi
# check the chemistry of the CloudIA code
# - abundances.py
# - hii regions generation files

import pathlib
import numpy as np
import pytest
import ast
from jinja2 import Environment, StrictUndefined, Template, meta

from galapy.internal.constants import (ATOMIC_WEIGHT, SOLAR_REFERENCES, XI_D_MW)
from galapy.spectroscopy.utils.physics.abundances import (ZMAP_DEFAULT_GRID, Chemistry,
    ChemistryHashMismatch, build_zeta_map, format_cloudy_float,
    grain_scale_from_xi_d)


######################## ABUNDANCES ############################

class _LazyChemistry:
    """
    Initializes a Chemistry instance upon the first attribute access rather
    than during import.

    This class delay creation of `Chemistry` until it is actually needed.
    By using this proxy class, the import remains silent, and the cost (and
    skip) is incurred only in the tests that actually use the dataset engine.

    Methods
    -------
    __getattr__(name)
        Intercepts attribute access and initializes the `Chemistry` object
        if it has not already been created.
    """
    _inst = None

    def __getattr__(self, name):
        if _LazyChemistry._inst is None:
            try:
                _LazyChemistry._inst = Chemistry()
            except Exception as exc:
                pytest.skip(f"Chemistry base of the dataset not avaiable: {exc}")
        return getattr(_LazyChemistry._inst, name)


CHEM = _LazyChemistry()
LZ_GRID = (-2.30, -1.50, -1.00, -0.50, 0.00, 0.25, 0.50) #grid of log normalised psi value for metallicity (GC)
_LZ_DECK = -1.0 #off-fiducial test-value

# Registers on the HII and PDR tests
GENERATORS = []
_DECKS = []
_DECK_CTX = {}
_GRAINS_EXPECTED = {}

@pytest.mark.unit
def test_fiducial_pattern_reproduces_published_values():
    """
    Tests whether the fiducial abundance pattern is consistent with published
    values from Nicholls et al. 2017, Table 1. This includes verification using
    four independent anchors: the total mass fraction, the major nebular
    elements, the two tabulated ratios, and helium fraction (Y).
    """
    ab = CHEM.abundance_pattern(0.0)
    assert CHEM.metallicity_from_zeta(0.0) == pytest.approx(0.014254, abs=1e-6)
    assert ab['C'] == pytest.approx(2.6485e-4, rel=1e-3)
    assert ab['N'] == pytest.approx(6.1667e-5, rel=1e-3)
    assert ab['O'] == pytest.approx(5.7544e-4, rel=1e-3)
    assert np.log10(ab['C'] / ab['O']) == pytest.approx(-0.337, abs=1e-3)
    assert np.log10(ab['N'] / ab['O']) == pytest.approx(-0.970, abs=1e-3)


@pytest.mark.unit
@pytest.mark.parametrize('lz', LZ_GRID)
def test_delta_is_zero_for_oxygen_everywhere(lz):
    """
    Test that confirms the exact behavior of the delta_O function for oxygen.

    This test ensures that the delta value for oxygen is always zero across all
    input grid configurations. It checks the strict adherence to the mathematical
    definition, rather than its approximation.

    Parameters:
        lz (Any): Represents the grid configuration input parameter for the delta_O
        function. Any test value from the LZ_GRID parameter set is validated.

    Raises:
        AssertionError: If the delta_O function does not return 0.0 for the given
        input grid configuration.
    """
    assert CHEM.delta_O('O', lz) == 0.0


@pytest.mark.unit
def test_element_scale_lines_count_and_registry():
    """
    Test the behavior and output of the element scaling lines function for chemical
    elements under specified physical conditions.

    This test ensures:
    - Only the expected number of lines are returned for the given input
    - Ensures no lines are emitted for a predefined set of muted elements
    - Proper formatting of lines
    - Confirms consistency with fiducial physical outputs for the Galactic
      Concordance model by verifying computed argument values.
    """
    muted = {'lithium', 'beryllium', 'boron', 'oxygen', 'fluorine',
             'neon', 'chlorine', 'argon'}
    for lz in LZ_GRID:
        lines = CHEM.element_scale_lines(lz)
        assert len(lines) == 21, (lz, len(lines))
        names = {l.split()[3] for l in lines}
        assert not (names & muted), names & muted
        assert all(l.endswith(' log') for l in lines)
        assert all(l.upper().count(' LOG') == 1 for l in lines)

    args = {l.split()[3]: float(l.split()[4]) for l in CHEM.element_scale_lines(0.0)}
    fit = {'nitrogen', 'helium'}
    assert all(v == 0.0 for k, v in args.items() if k not in fit), args
    assert all(abs(v) < 1e-4 for k, v in args.items() if k in fit), args


@pytest.mark.unit
def test_helium_curve_and_primordial_intercept():
    """
    Test the helium curve and primordial intercept. (Nichols+17 values)
    """
    assert CHEM.delta_O('He', 0.0) == pytest.approx(0.0, abs=1e-4)
    assert CHEM.delta_O('He', -2.3) == pytest.approx(-0.06791, abs=1e-4)
    ab0 = CHEM.abundance_pattern(0.0)
    assert ab0['He'] == pytest.approx(9.7728e-2, rel=1e-4)
    p = CHEM.special['He']
    Yp = (ATOMIC_WEIGHT['He'] * p['A']) / (ATOMIC_WEIGHT['H'] + ATOMIC_WEIGHT['He'] * p['A'])
    assert Yp == pytest.approx(0.2490, abs=1e-3)


@pytest.mark.unit
def test_helium_is_not_scaled_by_the_uniform_factor():
    """
    Test to verify the helium abundance is not scaled by the uniform factor.
    """
    assert CHEM.abundance_pattern(-2.3)['He'] == pytest.approx(8.3577e-2, rel=1e-4)
    assert CHEM.abundance_pattern(+0.5)['He'] == pytest.approx(1.28481e-1, rel=1e-4)
    # He/H should go toward the plateau, not zero
    assert CHEM.abundance_pattern(-2.9)['He'] > 8.3e-2
    # Y on a plateau poor of metals is Y_primordial.
    for lz, Y_exp in ((-2.30, 0.249), (-1.00, 0.252), (0.00, 0.276), (0.50, 0.321)):
        ab = CHEM.abundance_pattern(lz)
        tot = sum(ATOMIC_WEIGHT[s] * n for s, n in ab.items())
        assert ATOMIC_WEIGHT['He'] * ab['He'] / tot == pytest.approx(Y_exp, abs=5e-4)


@pytest.mark.unit
def test_Z_GC_is_derived_from_the_loaded_files():
    """
    Unit test for verifying that the Z_GC value in the imported module is derived correctly from
    loaded files.

    This test ensures that `Z_GC` is dynamically computed rather than being hardcoded as a module
    constant.

    Raises
    ------
    AssertionError
        If `Z_GC` is defined as a constant in the module instead of being calculated.
        If `Z_GC` does not match the expected computed value from `metallicity_from_zeta`.
        If `Z_GC` does not approximate `0.014254` within the tolerance of `1e-6`.
    """
    import galapy.spectroscopy.utils.physics.abundances as abu_mod
    assert not hasattr(abu_mod, 'Z_GC')
    assert CHEM.Z_GC == CHEM.metallicity_from_zeta(0.0)
    assert CHEM.Z_GC == pytest.approx(0.014254, abs=1e-6)


@pytest.mark.unit
def test_nitrogen_eq9_equals_eq3_at_fiducial():
    """
    Tests the equivalence between nitrogen equation 9 and equation 3 of Nichols
    at fiducial conditions.
    """
    log_OH_fid = CHEM.log_OH_fid
    for lz in LZ_GRID:
        eq3 = np.log10(10**-1.732 + 10**(log_OH_fid + lz + 2.19))
        eq3_fid = np.log10(10**-1.732 + 10**(log_OH_fid + 2.19))
        assert CHEM.delta_O('N', lz) == pytest.approx(eq3 - eq3_fid, abs=1e-3)


@pytest.mark.unit
def test_metallicity_map_roundtrip_and_monotonic():
    """
    Z(zeta_O) is monotonic -- if it were not, the runtime conversion would be ambiguous.
    """
    grid = np.linspace(-2.9, 0.9, 1000)
    Z = np.array([CHEM.metallicity_from_zeta(l) for l in grid])
    assert np.all(np.diff(Z) > 0)
    for lz in grid[::37]:
        assert CHEM.zeta_from_metallicity(CHEM.metallicity_from_zeta(lz)) == \
            pytest.approx(lz, abs=1e-6)
    with pytest.raises(ValueError):
        CHEM.zeta_from_metallicity(1.0)


@pytest.mark.unit
def test_zeta_map_hash_guard(tmp_path):
    """
    Tests the behavior of the Chemistry class when provided with a .npz file containing
    hashes that do not match expected values. Verifies that the intended exception is raised.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest fixture, used to create the test file.
    """
    bad = tmp_path / 'zeta_of_Z.npz'
    np.savez(bad, log_zeta_O=np.linspace(-3, 1, 10), Z_gas=np.linspace(1e-5, 5e-2, 10),
             sha_abn='deadbeef', sha_scaling='deadbeef')
    with pytest.raises(ChemistryHashMismatch):
        Chemistry(zmap_file=str(bad))

@pytest.mark.unit
def test_build_zeta_map_writes_a_table_its_own_files_accept(tmp_path):
    """
    Test the zeta map building functionality by asserting correct file
    creation, absence of temporary files after completion, and integrity
    of data stored in the generated output file.

    Attributes:
        tmp_path: pytest fixture providing a temporary directory unique to
                  the test invocation.
    Notes:
        The test ensures the zeta map output file corresponds to the expected
        format, content keys, and internal consistency when used with the
        Chemistry class.
    """
    out = tmp_path / 'zeta_of_Z.npz'
    p = build_zeta_map(out)
    assert out.is_file()
    assert not (tmp_path / 'zeta_of_Z.npz.tmp').exists()
    assert not (tmp_path / 'zeta_of_Z.npz.tmp.npz').exists()
    assert set(p) == {'log_zeta_O', 'Z_gas', 'sha_abn', 'sha_scaling'}
    assert p['log_zeta_O'].size == ZMAP_DEFAULT_GRID[2]
    ch = Chemistry(zmap_file=str(out))
    for lz in LZ_GRID:
        assert ch.zeta_from_metallicity(ch.metallicity_from_zeta(lz)) == \
            pytest.approx(lz, abs=1e-6)


@pytest.mark.unit
def test_build_zeta_map_grid_is_a_roundtrip_budget(tmp_path):
    """
    Test function to ensure that building and writing a zeta map grid is associated
    to specified accuracy threshold. The function compares default grid settings
    with altered grid sizes to validate the stability of the zeta-metallicity
    transformation system.

    Arguments:
    - tmp_path (Path): A pytest fixture that provides a temporary directory unique to the test function invocation.

    Assertions:
    - ZMAP_DEFAULT_GRID is correctly defined as (-3.0, 1.0, 4001).
    - The transformation from metallicity to zeta and back to metallicity ensures
      a maximum deviation below 1e-6 for the default grid.
    - A condition is enforced to verify if a grid size of 2001 points does not meet
      the same threshold, indicating changes in the default budget.
    """
    assert ZMAP_DEFAULT_GRID == (-3.0, 1.0, 4001)
    probe = np.linspace(-2.9, 0.9, 400)

    def worst(n):
        ch = Chemistry(zmap_file=str(build_and_write(tmp_path, n)))
        return max(abs(ch.zeta_from_metallicity(ch.metallicity_from_zeta(l)) - l)
                   for l in probe)

    def build_and_write(root, n):
        p = root / f'zmap_{n}.npz'
        build_zeta_map(p, n=n)
        return p

    assert worst(ZMAP_DEFAULT_GRID[2]) <= 1e-6
    assert worst(2001) > 1e-6, (
        "if also 2001 points pass, budget has changed: re-measure the default")


@pytest.mark.unit
def test_grain_fac_uses_mass_fraction_not_zeta():
    """
    Test to verify that the grain factor is computed using the mass fraction rather than zeta.

    This test ensures that the grain_scale_from_xi_d function scales the metallicity
    calculation based on the correct mass fraction derived from `metallicity_from_zeta`.
    """
    lz = -1.0
    right = CHEM.metallicity_from_zeta(lz) / CHEM.metallicity_from_zeta(0.0)
    assert right == pytest.approx(0.0816, abs=1e-3)
    assert right * grain_scale_from_xi_d(XI_D_MW) == pytest.approx(0.0816, abs=1e-3)
    assert abs(right - 10**lz) > 0.015


@pytest.mark.unit
def test_abn_file_is_well_formed_for_parser(tmp_path):
    """
    Test the generated .abn file is well formatted for CLOUDY inputs
    """
    out = tmp_path / 'test.abn'
    CHEM.write_abn_file(out, -1.0)
    lines = out.read_text().splitlines()
    body = lines[:next(i for i, l in enumerate(lines) if l.startswith('*'))]
    elems = [l for l in body if not l.startswith('#')]
    assert len(elems) == 30
    assert all(l.strip() for l in body)
    assert not any('GRAINS' in l.upper() for l in body)
    assert all(float(l.split()[1]) > 0 for l in elems)
    assert all('E' in l.split()[1].upper() for l in elems)


@pytest.mark.unit
def test_abn_truncation_is_detected(tmp_path):
    """
    Test that truncating the .abn file raises an error
    """
    out = tmp_path / 'short.abn'
    CHEM.write_abn_file(out, 0.0)
    txt = out.read_text().splitlines()
    (tmp_path / 'short2.abn').write_text("\n".join(txt[:12]) + "\n")
    with pytest.raises(ValueError, match='expected 30'):
        Chemistry._read_abn(tmp_path / 'short2.abn')


@pytest.mark.unit
def test_grain_scale_normalised_at_MW():
    """
    Test the correctness of the grain scaling factor at the Milky Way (MW) conditions.

    This test validates the implementation of the grain scaling factor, ensuring that it equals
    1.0 when computed under Milky Way metallicity conditions. The Galactic Concordance defines
    the metallicity of the local gas in the Milky Way, using the ratio (D/G)_MW / Z_gas^MW,
    where the denominator is derived as Z_GC.
    """
    assert grain_scale_from_xi_d(XI_D_MW) == pytest.approx(1.0, abs=1e-12)
    assert (1.0 / 162.0) / CHEM.Z_GC == pytest.approx(XI_D_MW, abs=5e-3)
    with pytest.raises(ValueError):
        grain_scale_from_xi_d(0.0)


@pytest.mark.unit
def test_format_cloudy_float_no_underflow():
    """
    Test the behavior of the `format_cloudy_float` function to ensure that small floats
    do not appear as underflowed values and are correctly formatted with scientific
    notation when necessary.

    @param 9.5e-6: The small float value being tested to verify that it is not incorrectly
                    formatted as '0.0000' and instead utilizes an 'e' to denote scientific
                    notation.
    @type 9.5e-6: float
    """
    assert '0.0000' not in format_cloudy_float(9.5e-6)
    assert 'e' in format_cloudy_float(9.5e-6)


@pytest.mark.unit
def test_to_solar_units_requires_explicit_reference():
    """
    Tests the `to_solar_units` function of the `CHEM` module to ensure that an explicit
    reference is required when converting 'log Z/Zsun' values.
    """
    assert 'GC' not in SOLAR_REFERENCES
    a = CHEM.to_solar_units(0.0, reference='GC')
    b = CHEM.to_solar_units(0.0, reference='Caffau11')
    assert a == 0.0
    assert abs(a - b) == pytest.approx(0.0308, abs=1e-3)
    with pytest.raises(ValueError):
        CHEM.to_solar_units(0.0, reference='whatever')

@pytest.mark.unit
def test_oh_conversion_is_exact_shift():
    """
    Test that the OH conversion functions within the CHEM module handle exact shifts
    correctly and maintain consistency when converting back and forth.

    The test iterates over a predefined set of LZ_GRID values, verifying these
    conditions:

    1. The OH value derived from `oh_from_zeta(lz)` matches the expected shift
       with a precision tolerance of 1e-9.
    2. Converting the OH value back to ζ using `zeta_from_oh` restores the
       original ζ value with a precision tolerance of 1e-12.

    Attributes:
        LZ_GRID: Iterable grid of ζ values used for validation.
    """
    for lz in LZ_GRID:
        assert CHEM.oh_from_zeta(lz) == pytest.approx(8.760 + lz, abs=1e-9)
        assert CHEM.zeta_from_oh(CHEM.oh_from_zeta(lz)) == pytest.approx(lz, abs=1e-12)


######################## HII REGIONS AND PDRS ############################

_DECK_CACHE = {}

def _template_path(kind):
    """
    Determines the filesystem path to a specific template file based on the given kind.

    This function retrieves a template file path residing in a predefined directory. If the file
    exists, its path is returned; otherwise, None is returned.
    """
    try:
        from galapy.internal.data import DataFile
        from galapy.internal.globs import NEB_TPL_DIR
        p = pathlib.Path(DataFile(f'{kind}.in.j2', NEB_TPL_DIR).get_file())
    except Exception:
        return None
    return p if p.is_file() else None


def requires_templates(fn):
    """
    Decorator that ensures required templates are present before executing a test function.

    This decorator checks if the templates for the specified decks are
    available in the dataset. If any templates are missing, the test is
    skipped with an appropriate message.

    Parameters:
        fn (Callable): The test function to be wrapped.

    Returns:
        Callable: The wrapped test function.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        missing = [k for k in _DECKS if _template_path(k) is None]
        if missing:
            pytest.skip(f"template absent in the dataset: {missing}")
        return fn(*a, **kw)
    return wrapper


def _job_hii(lz=_LZ_DECK):
    """
    build a random hii job
    """
    return {
        'job_id': '00000_003_02',
        'lognH_HII': 2.5, 'logU': -2.5, 'z_CMB': 3.0, 'F_star': 0.5,
        'log_zeta_O': lz,
        'delta_NO': CHEM.delta_O('N', lz), 'delta_CO': CHEM.delta_O('C', lz),
        'sed_file': 'ssp_003_02.sed', 'log_N_stop': 21.5,
        'element_scale_block': "\n".join(CHEM.element_scale_lines(lz)),
        'grain_scale': (CHEM.metallicity_from_zeta(lz) / CHEM.Z_GC)
                       * grain_scale_from_xi_d(XI_D_MW),
    }


def _gen_module(kind):
    """
    Generates a module function based on the specified kind (hii or pdr)
    """
    if kind == 'hii':
        from galapy.spectroscopy.utils.hii import gen_input_hii as gen
    else:
        from galapy.spectroscopy.utils.pdr import gen_input_pdr as gen
    return gen


def _deck(kind, lz=_LZ_DECK):
    """
    Generates and caches a deck file based on the given kind and lz parameters.

    This function reads a template file, validates its variables against the context
    provided by the associated job specification, and renders the template to
    generate the final deck file. The result is cached for subsequent calls with the
    same parameters.
    """
    key = (kind, lz)
    if key in _DECK_CACHE:
        return _DECK_CACHE[key]
    path = _template_path(kind)
    if path is None:
        pytest.skip(f"{kind}.in.j2 not available (dataset not available)")
    text = path.read_text()

    gen = _gen_module(kind)
    ctx = _DECK_CTX[kind](lz)

    missing = sorted(meta.find_undeclared_variables(Environment().parse(text))
                     - set(ctx))
    if missing:
        pytest.fail(f"{kind}.in.j2 ask variables that are not given by job spec or build_job: {missing}")

    tmpl = Template(text, undefined=StrictUndefined, keep_trailing_newline=True)
    deck = gen.render_one(tmpl, ctx)
    _DECK_CACHE[key] = deck
    return deck


def _commands(deck):
    """
    Parses that given a deck (cloudy) remove comments that are unecessary (lines comments)
    """
    out = []
    for raw in deck.splitlines():
        s = raw.strip()
        if not s or s[0] in '#*%' or s.startswith('//'):
            continue
        out.append(s.split('#')[0].strip().lower())
    return [c for c in out if c]


################ HII REGIONS
GENERATORS.append('galapy.spectroscopy.utils.hii.gen_input_hii')
_DECKS.append('hii')
_DECK_CTX['hii'] = _job_hii
_GRAINS_EXPECTED['hii'] = 1          #only ISM, no PAH in HII


@pytest.mark.unit
@pytest.mark.parametrize('modname', GENERATORS)
def test_scale_lines_argument_is_log_zeta_AST(modname):
    """
    Tests specific functions in various modules to ensure they have the correct scaling argument.

    The test ensures that functions such as `element_scale_lines`, `delta_O`,
    `abundance_pattern`, and `metallicity_from_zeta` correctly use 'log_zeta_O'
    or ensure conditions on other nesting patterns for the last arguments provided.
    """
    import importlib
    mod = importlib.import_module(modname)
    tree = ast.parse(pathlib.Path(mod.__file__).read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ('element_scale_lines', 'delta_O',
                                  'abundance_pattern', 'metallicity_from_zeta'):
            continue
        arg = node.args[-1]
        if isinstance(arg, ast.Name):
            assert arg.id == 'log_zeta_O', (modname, arg.id)
        else:
            assert isinstance(arg, ast.Call), (modname, ast.dump(arg))


@pytest.mark.unit
@pytest.mark.parametrize('modname', GENERATORS)
def test_grain_fac_call_site_AST(modname):
    """
    Tests the AST of grain_fac call sites in the specified module to ensure proper function calls.

    This test verifies the correct usage of 'grain_fac' values in the module's abstract syntax tree (AST).
    It checks that the 'metallicity_from_zeta' function is referenced and ensures that the 'Pow' operation is
    not used in the AST representation of the 'grain_fac' assignments.
    """
    import importlib
    mod = importlib.import_module(modname)
    tree = ast.parse(pathlib.Path(mod.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, 'id', '') == 'grain_fac' or
                (isinstance(t, ast.Subscript) and
                 getattr(getattr(t, 'slice', None), 'value', None) == 'grain_fac')
                for t in node.targets):
            dump = ast.dump(node.value)
            assert 'metallicity_from_zeta' in dump, (modname, dump)
            assert 'Pow' not in dump, (modname, dump)


@pytest.mark.unit
@requires_templates
def test_helium_line_present_in_deck():
    for kind in _DECKS:
        assert 'element scale factor helium' in _deck(kind)


@pytest.mark.unit
@requires_templates
def test_no_residual_jinja_placeholders():
    import re
    for kind in _DECKS:
        assert not re.search(r'\{\{|\}\}|\{%|%\}', _deck(kind)), kind


@pytest.mark.unit
@requires_templates
def test_metals_register_is_log_in_both_sectors():
    """
    This function verifies the behavior of metals registration across different
    sectors and ensures consistency with expected parameters. It also checks
    the correctness of grain-related commands for proper sector registration.
    The function primarily analyzes and asserts specific command structures.
    """
    for kind in _DECKS:
        cmds = _commands(_deck(kind))
        metals = [c for c in cmds
                  if c.startswith('metals') and not c.startswith('metals deplete')]
        assert len(metals) == 1, (kind, metals)
        tok = metals[0].split()
        assert tok[-1] == 'log', (kind, metals[0])
        assert float(tok[1]) == pytest.approx(_LZ_DECK, abs=1e-9), (kind, metals[0])
        assert 'linear' not in metals[0], (kind, metals[0])

    # grains
    for kind in _DECKS:
        grains = [c for c in _commands(_deck(kind)) if c.startswith('grains')]
        assert len(grains) == _GRAINS_EXPECTED[kind], (kind, grains)
        for g in grains:
            assert 'linear' in g, (kind, g)
            assert ' log' not in g, (kind, g)