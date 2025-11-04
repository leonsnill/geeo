# shared pytest config and fixtures with required ee initialization

import os
import pytest
import copy

def _ensure_ee_initialized():
    # initialize earth engine once when this file is imported
    import ee
    proj = os.getenv("EARTHENGINE_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    try:
        if proj:
            ee.Initialize(project=proj)
        else:
            ee.Initialize()
    except Exception:
        # fall back to interactive auth, then initialize again
        ee.Authenticate()
        if proj:
            ee.Initialize(project=proj)
        else:
            ee.Initialize()
    # verify server roundtrip
    _ = ee.String("ping").getInfo()

# initialize now, before any geeo import can happen
_ensure_ee_initialized()

def pytest_configure(config):
    # register markers
    config.addinivalue_line("markers", "smoke: fast surface-level checks for core entry points")
    config.addinivalue_line("markers", "online: requires authenticated earth engine")

@pytest.fixture(scope="session")
def ee_session():
    # provide ee handle for tests
    import ee
    return ee

@pytest.fixture(scope="session")
def tiny_roi_berlin():
    # small bbox over berlin to keep queries light
    return [12.9, 52.2, 13.9, 52.7]

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

