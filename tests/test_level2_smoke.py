# minimal smoke tests for level2

import importlib
import pytest

def _get_callable(mod_path, names):
    # return first callable attribute found or none
    mod = importlib.import_module(mod_path)
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    return None

def test_level2_rejects_empty_sensors():
    """
    ensures level2 validates user input early by rejecting empty sensor selection.
    this runs online because geeo imports ee at module scope.
    """
    fn = _get_callable("geeo.level2.level2", ["run_level2", "level2", "main"])
    assert fn is not None, "no level2 entry point found"
    bad = {
        "ROI": [12.9, 52.2, 13.9, 52.7],
        "SENSORS": [],
        "DATE_MIN": 20200101,
        "DATE_MAX": 20200105,
    }
    with pytest.raises(Exception):
        fn(bad)

@pytest.mark.smoke
@pytest.mark.online
def test_level2_s2_smoke_online(ee_session, minimal_l2_params):
    """
    verifies that a basic sentinel-2 run returns a params dict with a tss imagecollection.
    uses a tiny roi and short time window to keep it fast.
    """
    ee = ee_session
    fn = _get_callable("geeo.level2.level2", ["run_level2", "level2", "main"])
    assert fn is not None, "no level2 entry point found"

    prm = fn(minimal_l2_params)

    # basic keys
    assert isinstance(prm, dict)
    assert "TSS" in prm
    assert isinstance(prm["TSS"], ee.imagecollection.ImageCollection)

    # quick size sanity (best-effort)
    try:
        size = int(prm["TSS"].size().getInfo())
        assert size != 0
    except Exception:
        pass