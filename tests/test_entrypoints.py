# presence checks for level3, level4, and export entry points

import importlib

def _get_callable(mod_path, names):
    # return first callable attribute found or none
    mod = importlib.import_module(mod_path)
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    return None

def test_level3_entrypoint_exists():
    """
    ensures level3 exposes a callable entry point.
    expands later to an end-to-end call with level2 output.
    """
    fn = _get_callable("geeo.level3.level3", ["run_level3", "level3", "main"])
    assert fn is not None, "no level3 entry point found"

def test_level4_entrypoint_exists():
    fn = _get_callable("geeo.level4.level4", ["run_level4", "level4", "main"])
    assert fn is not None, "no level4 entry point found"

def test_export_entrypoint_exists():
    fn = _get_callable("geeo.export.export", ["run_export", "export", "main"])
    assert fn is not None, "no export entry point found"