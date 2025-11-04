# smoke tests that exercise level2 -> level3 -> level4 and export (no-op start)

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

def _check_that_has_ee_img_imgcol_like(ee, dct):
    # check if any value looks like an ee image or imagecollection
    return any(
        isinstance(v, (ee.image.Image, ee.imagecollection.ImageCollection))
        for v in dct.values()
    )

@pytest.mark.smoke
@pytest.mark.online
def test_pipeline_level2_level3_level4_smoke(ee_session, rich_l2_params):
    """
    runs a small end-to-end pipeline: level2 -> level3 -> level4.
    asserts each stage returns a dict and adds at least one ee raster-like object.
    """
    ee = ee_session

    run_level2 = _get_callable("geeo.level2.level2", ["run_level2", "level2", "main"])
    assert run_level2 is not None, "no level2 entry point found"

    prm_l2 = run_level2(rich_l2_params)
    assert isinstance(prm_l2, dict)
    assert _check_that_has_ee_img_imgcol_like(ee, prm_l2), "level2 returned no ee raster-like outputs"

    run_level3 = _get_callable("geeo.level3.level3", ["run_level3", "level3", "main"])
    assert run_level3 is not None, "no level3 entry point found"

    before_keys = set(prm_l2.keys())
    prm_l3 = run_level3(prm_l2)
    assert isinstance(prm_l3, dict)
    added_l3 = set(prm_l3.keys()) - before_keys
    assert added_l3, "level3 did not add any outputs"

    run_level4 = _get_callable("geeo.level4.level4", ["run_level4", "level4", "main"])
    assert run_level4 is not None, "no level4 entry point found"

    before_keys = set(prm_l3.keys())
    prm_l4 = run_level4(prm_l3)
    assert isinstance(prm_l4, dict)
    added_l4 = set(prm_l4.keys()) - before_keys
    assert added_l4, "level4 did not add any outputs"

    # basic outputs from richer parameter settings
    # server-side computation to ensure evaluation occurs
    tsm_size = prm_l4["TSM"].size().getInfo()
    nvo_size = prm_l4["NVO"].bandNames().size().getInfo()
    tsi_size = prm_l4["TSI"].size().getInfo()
    stm_size = prm_l4["STM"].size().getInfo()
    pbc_size = prm_l4["PBC"].size().getInfo()
    lsp_size = prm_l4["LSP"].size().getInfo()
    assert isinstance(tsm_size, int) and tsm_size > 0, "TSM is empty or failed to compute"
    assert isinstance(nvo_size, int) and nvo_size > 0, "NVO is empty or failed to compute"
    assert isinstance(tsi_size, int) and tsi_size > 0, "TSI is empty or failed to compute"
    assert isinstance(stm_size, int) and stm_size > 0, "STM is empty or failed to compute"
    assert isinstance(pbc_size, int) and pbc_size > 0, "PBC is empty or failed to compute"
    assert isinstance(lsp_size, int) and lsp_size > 0, "LSP is empty or failed to compute"

# EOF