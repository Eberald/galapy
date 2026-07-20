# galapy/CloudIA/__init__.py
# Author: Enrico Veraldi
# Nebular module CLOUDY-based for Galapy (CloudIA)

__all__ = [
    'HIIEmulator',
    'PDREmulator',
    'NebularCSB',
    'HIINebularProcessor',
    'PDRNebularProcessor',
    'TwoSectorCombiner',
    'PDRMapping',
    'LineList',
    'JointObservation',
    'NebularResults'
]

# dict. public name - module define it
_LAZY = {
    'HIIEmulator':          'galapy.CloudIA.hii_emulator',
    'PDREmulator':          'galapy.CloudIA.pdr_emulator',
    'NebularCSP':           'galapy.CloudIA.nebular_csp',
    'HIINebularProcessor':  'galapy.CloudIA.hii_processor',
    'PDRNebularProcessor':  'galapy.CloudIA.pdr_processor',
    'TwoSectorCombiner':    'galapy.CloudIA.two_sector_combiner',
    'PDRMapping':           'galapy.CloudIA.pdr_mapping',
    'LineList':             'galapy.CloudIA.line_list',
    'JointObservation':     'galapy.CloudIA.joint_observation',
    'NebularResults':       'galapy.CloudIA.results'
}

# hint on the module
_HINT = (
    "the CloudIA module require to be used the extra 'cloudia' - pip install galapy[cloudia]"
)

def __getattr__(name):
    """
    Handles lazy-loading of module attributes to optimize performance and reduce initial
    loading time. Dynamically imports the required modules only when the specified
    attribute is accessed. Injects the imported object into the global dictionary of
    the current module for subsequent use.

    Raises:
        AttributeError: If the requested attribute name is not found in the `_LAZY`
                        mapping.
        ImportError: If an import failure occurs due to missing dependencies or
                     other issues. Includes suggestions for resolving missing
                     dependencies related to specific packages.

    Args:
        name (str): The name of the attribute being accessed.

    Returns:
        Any: The requested attribute, dynamically imported and retrieved
             from the target module.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    import importlib
    try:
        module = importlib.import_module(target)
    except ImportError as e:
        missing = getattr(e, 'name', '') or ''
        if missing.split('.')[0] in {'torch','gpytorch','scikit-learn','pyyaml'}:
            raise ImportError(f"{_HINT} (missing: {missing})") from e
        raise
    obj = getattr(module, name)
    globals()[name] = obj #injected in the global dictionary of the module (performance)
    return obj

def __dir__():
    """
    Returns a sorted list of names available in the current module.

    The function combines the names of all globally defined variables, functions,
    classes, and any elements specified in the '__all__' variable, then sorts
    the combined list alphabetically.

    Returns:
        list[str]: A sorted list of the names available in the module.
    """
    return sorted(list(globals().keys()) + __all__)