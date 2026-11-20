# shared pytest config and fixtures with required ee initialization

import os
import sys
import pytest
import copy

EE_INITIALIZED = False
EE_INIT_ERROR = None

def _ensure_ee_initialized():
    global EE_INITIALIZED, EE_INIT_ERROR
    # initialize earth engine once when this file is imported
    import ee
    proj = os.getenv("EARTHENGINE_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    try:
        if proj:
            ee.Initialize(project=proj)
        else:
            ee.Initialize()
        # verify server roundtrip
        _ = ee.String("ping").getInfo()
        EE_INITIALIZED = True
        return
    except Exception as err:
        # only attempt interactive authentication if running in an interactive terminal and not in CI
        is_interactive = sys.stdin.isatty() and "CI" not in os.environ
        if is_interactive:
            try:
                ee.Authenticate()
                if proj:
                    ee.Initialize(project=proj)
                else:
                    ee.Initialize()
                _ = ee.String("ping").getInfo()
                EE_INITIALIZED = True
                return
            except Exception as auth_err:
                EE_INIT_ERROR = str(auth_err)
        else:
            EE_INIT_ERROR = str(err)

# initialize now, before any geeo import can happen
_ensure_ee_initialized()

def pytest_configure(config):
    # register markers
    config.addinivalue_line("markers", "smoke: fast surface-level checks for core entry points")
    config.addinivalue_line("markers", "online: requires authenticated earth engine")

def pytest_runtest_setup(item):
    # automatically skip online tests when earth engine is unauthenticated
    if "online" in item.keywords and not EE_INITIALIZED:
        pytest.skip(f"Earth Engine authentication required: {EE_INIT_ERROR}")

@pytest.fixture(scope="session")
def ee_session():
    # provide ee handle for tests
    if not EE_INITIALIZED:
        pytest.skip(f"Earth Engine authentication required: {EE_INIT_ERROR}")
    import ee
    return ee

@pytest.fixture(scope="session")
def tiny_roi_berlin():
    # small bbox over berlin to keep queries light
    return [12.9, 52.2, 13.9, 52.7]

@pytest.fixture(scope="session")
def test_point(ee_session):
    # single point inside tiny_roi_berlin for fast pixel sampling
    return ee_session.Geometry.Point([13.4, 52.5])

@pytest.fixture(scope="session")
def make_synth_img(ee_session):
    # create a constant server-side ee.Image with band values, mask, and timestamp
    def _make(date_str, band_dict, valid=True):
        ee = ee_session
        names = list(band_dict.keys())
        vals = [float(band_dict[k]) for k in names]
        img = ee.Image.constant(vals).rename(names).toFloat()
        mask_val = 1 if valid else 0
        mask = ee.Image.constant(mask_val).rename("mask").toUint8()
        img = img.updateMask(mask).addBands(mask)
        img = img.setDefaultProjection("EPSG:4326", None, 30)
        t_ms = ee.Date(date_str).millis()
        return img.set({"system:time_start": t_ms, "system:index": date_str})
    return _make

@pytest.fixture(scope="session")
def sample_pixel(ee_session, test_point):
    # evaluate band values of an ee.Image at test_point on the server
    def _sample(img):
        ee = ee_session
        return img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=test_point,
            scale=30
        ).getInfo()
    return _sample

@pytest.fixture(scope="session")
def minimal_l2_params(tiny_roi_berlin):
    # minimal parameters expected to work with defaults filled by the package
    return {
        "ROI": tiny_roi_berlin,
        "YEAR_MIN": 2020,
        "YEAR_MAX": 2021,
        "ROI_SIMPLIFY_GEOM_TO_BBOX": True,
        "SENSORS": ["L9", "L8", "L7", "L5", "L4", "S2"],
        "ERODE_DILATE": True,
        "FEATURES": ["BLU","GRN","RED","NIR","SW1","SW2","NDVI"],
        "PIX_RES": 60,
        "CRS": "EPSG:4326",
        "MASKS_S2": "PROB",
        "MASKS_S2_PROB": 40
    }

@pytest.fixture(scope="session")
def rich_l2_params(tiny_roi_berlin):
    # enriched parameters with more options set
    return {
        "ROI": tiny_roi_berlin,
        "YEAR_MIN": 2020,
        "YEAR_MAX": 2021,
        "ROI_SIMPLIFY_GEOM_TO_BBOX": True,
        "SENSORS": ["L9", "L8", "L7", "L5", "L4", "S2"],
        "ERODE_DILATE": True,
        "FEATURES": ["BLU","GRN","RED","NIR","SW1","SW2","NDVI", "SR"],
        
        "CUSTOM_FORMULAS": {
            "SR": {
                "formula": "N/R",
                "variable_map": {
                    "N": "NIR",
                    "R": "RED"
                }
            }
        },

        "TSM": True,
        "TSM_BASE_IMGCOL": "TSS",
        
        "FOLD_YEAR": True,
        "FOLD_CUSTOM": {
            "month": ['1-6', '7-12']
        },

        "NVO": True,
        "NVO_FOLDING": True,

        "TSI": "1RBF",
        "TSI_BASE_IMGCOL": "TSS",
        "INIT_JANUARY_1ST": True,

        "STM": ['p5', 'mean', 'p95', 'stdDev'],
        "STM_BASE_IMGCOL": "TSM",

        "STM_FOLDING": True,

        "PBC": "NLCD",

        "LSP": 'POLAR',
        "LSP_BASE_IMGCOL": "TSI",
        "LSP_BAND": "NDVI",

        "PIX_RES": 60,
        "CRS": "EPSG:3035",
        "RESAMPLING_METHOD": "bicubic",

        "EXPORT_IMAGE": True,
        "EXPORT_TABLE": True
    }

