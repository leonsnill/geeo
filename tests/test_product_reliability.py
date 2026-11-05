# server-side reliability tests for geeo products using synthetic images with known outputs

import math
import pytest
from geeo.main import init_param
from geeo.level2.level2 import apply_custom_formula
from geeo.level2.indices import dict_features, rnb, unmix
from geeo.level2.mosaic import mosaic_imgcol
from geeo.level3.level3 import run_level3
from geeo.level4.level4 import run_level4


@pytest.mark.online
def test_level2_indices_formulas_unmixing(make_synth_img, sample_pixel):
    """
    Tests Level-2 spectral feature generation on a single synthetic image with known band values
    (BLU=0.1, GRN=0.3, RED=0.2, NIR=0.6, SW1=0.2, SW2=0.1):
    - spectral indices:
      NDVI = (0.6 - 0.2) / (0.6 + 0.2) = 0.5
      NBR  = (0.6 - 0.1) / (0.6 + 0.1) = 5 / 7 (~0.7143)
      RNB  = 0.6 / 0.1 = 6.0
    - custom formula: SR = NIR / RED = 0.6 / 0.2 = 3.0
    - spectral unmixing: pixel (RED=0.3, NIR=0.5) is an exact 50/50 linear mixture of
      endmember_a [0.1, 0.9] and endmember_b [0.5, 0.1], so both fractions must equal 0.5.
    """
    # build synthetic image with known reflectance values
    img = make_synth_img("2023-06-15", {
        "BLU": 0.1, "GRN": 0.3, "RED": 0.2, "NIR": 0.6, "SW1": 0.2, "SW2": 0.1
    })

    # apply built-in indices and custom formula
    img = dict_features["NDVI"](img)
    img = dict_features["NBR"](img)
    img = rnb(img)
    img = apply_custom_formula(img, "N / R", {"N": "NIR", "R": "RED"}, "SR")

    px = sample_pixel(img)
    assert px["NDVI"] == pytest.approx(0.5, abs=1e-4)
    assert px["NBR"] == pytest.approx(5.0 / 7.0, abs=1e-4)
    assert px["RNB"] == pytest.approx(6.0, abs=1e-4)
    assert px["SR"] == pytest.approx(3.0, abs=1e-4)

    # test unmixing on a 50/50 mixture pixel (red=0.3, nir=0.5)
    mix_img = make_synth_img("2023-06-15", {"RED": 0.3, "NIR": 0.5})
    unmixed = unmix(
        bands=["RED", "NIR"],
        endmember_values=[[0.1, 0.9], [0.5, 0.1]],
        endmember_names=["em_a", "em_b"],
        sumToOne=True,
        nonNegative=True
    )(mix_img)

    px_umx = sample_pixel(unmixed)
    assert px_umx["em_a"] == pytest.approx(0.5, abs=1e-4)
    assert px_umx["em_b"] == pytest.approx(0.5, abs=1e-4)


@pytest.mark.online
def test_tsm_and_nvo_reliability(ee_session, tiny_roi_berlin, make_synth_img, sample_pixel):
    """
    Tests Time Series Mosaic (TSM) deduplication and Number of Valid Observations (NVO) counting:
    - input TSS has 4 scenes in June 2023:
      1) 2023-06-01T10:00:00 (NDVI=0.4, valid=True)
      2) 2023-06-01T10:05:00 (NDVI=0.4, valid=True) -> same-day swath overlap
      3) 2023-06-15T10:00:00 (NDVI=0.8, valid=True)
      4) 2023-06-30T10:00:00 (NDVI=0.9, valid=False) -> cloud-masked scene
    - expected TSM size is 3 (the two June 1 scenes merge into one mosaic).
    - expected mean NDVI on TSM (unmasked scenes: 0.4 and 0.8) is 0.6, avoiding the 0.5333
      double-counting bias of TSS.
    - expected NVO is 2 (two distinct valid observation dates: 2023-06-01 and 2023-06-15).
    """
    ee = ee_session

    # construct 4-scene tss with same-day overlap and one masked scene
    tss = ee.ImageCollection([
        make_synth_img("2023-06-01T10:00:00", {"NDVI": 0.4}, valid=True),
        make_synth_img("2023-06-01T10:05:00", {"NDVI": 0.4}, valid=True),
        make_synth_img("2023-06-15T10:00:00", {"NDVI": 0.8}, valid=True),
        make_synth_img("2023-06-30T10:00:00", {"NDVI": 0.9}, valid=False),
    ])

    tsm = mosaic_imgcol(tss)
    assert tsm.size().getInfo() == 3

    # compare mean ndvi on tss (biased by overlap) vs tsm (unbiased)
    tss_mean = sample_pixel(tss.select("NDVI").mean())["NDVI"]
    tsm_mean = sample_pixel(tsm.select("NDVI").mean())["NDVI"]
    assert tss_mean == pytest.approx((0.4 + 0.4 + 0.8) / 3.0, abs=1e-4)
    assert tsm_mean == pytest.approx(0.6, abs=1e-4)

    # run level3 nvo calculation
    prm = init_param({
        "ROI": tiny_roi_berlin,
        "YEAR_MIN": 2023,
        "YEAR_MAX": 2023,
        "FEATURES": ["NDVI"],
        "NVO": True,
    })
    prm["TSS"] = tss
    prm = run_level3(prm)

    px_nvo = sample_pixel(prm["NVO"])
    assert px_nvo["NVO"] == 2


@pytest.mark.online
def test_stm_and_folding_reliability(ee_session, tiny_roi_berlin, make_synth_img, sample_pixel):
    """
    Tests Spectral-Temporal Metrics (STM) and temporal folding:
    - input TSS has 5 valid images and 1 masked outlier in 2023:
      * Jan (month 1): 3 valid images with NDVI = [0.1, 0.2, 0.3] + 1 masked image with NDVI = 999.0
      * Jul (month 7): 2 valid images with NDVI = [0.7, 0.9]
    - global unfolded STM across the 5 valid values [0.1, 0.2, 0.3, 0.7, 0.9] must yield:
      min=0.1, max=0.9, mean=0.44, median=0.3, count=5 (masked 999.0 excluded).
    - monthly folded STM (FOLD_CUSTOM month=['1-1', '7-7']) must yield:
      Jan mean=0.2, Jul mean=0.8.
    - join-based folding (STM_FOLDING_LIST_ITER=False) and list-iteration folding
      (STM_FOLDING_LIST_ITER=True) must produce identical results.
    """
    ee = ee_session

    # build synthetic collection across january and july
    tss = ee.ImageCollection([
        make_synth_img("2023-01-05", {"NDVI": 0.1}, valid=True),
        make_synth_img("2023-01-15", {"NDVI": 0.2}, valid=True),
        make_synth_img("2023-01-25", {"NDVI": 0.3}, valid=True),
        make_synth_img("2023-01-28", {"NDVI": 999.0}, valid=False),
        make_synth_img("2023-07-10", {"NDVI": 0.7}, valid=True),
        make_synth_img("2023-07-20", {"NDVI": 0.9}, valid=True),
    ])

    # test unfolded global stm
    prm_global = init_param({
        "ROI": tiny_roi_berlin,
        "YEAR_MIN": 2023,
        "YEAR_MAX": 2023,
        "FEATURES": ["NDVI"],
        "STM": ["min", "max", "mean", "median", "count"],
        "STM_FOLDING": False,
    })
    prm_global["TSS"] = tss
    prm_global = run_level3(prm_global)

    px = sample_pixel(prm_global["STM"])
    assert px["NDVI_min"] == pytest.approx(0.1, abs=1e-4)
    assert px["NDVI_max"] == pytest.approx(0.9, abs=1e-4)
    assert px["NDVI_mean"] == pytest.approx(0.44, abs=1e-4)
    assert px["NDVI_median"] == pytest.approx(0.3, abs=1e-4)
    assert px["NDVI_count"] == 5

    # test folded stm with joins vs list iteration
    base_fold_cfg = {
        "ROI": tiny_roi_berlin,
        "YEAR_MIN": 2023,
        "YEAR_MAX": 2023,
        "FEATURES": ["NDVI"],
        "FOLD_CUSTOM": {"month": ["1-1", "7-7"]},
        "STM": ["mean", "min", "max"],
        "STM_FOLDING": True,
    }

    prm_join = init_param({**base_fold_cfg, "STM_FOLDING_LIST_ITER": False})
    prm_join["TSS"] = tss
    prm_join = run_level3(prm_join)

    prm_iter = init_param({**base_fold_cfg, "STM_FOLDING_LIST_ITER": True})
    prm_iter["TSS"] = tss
    prm_iter = run_level3(prm_iter)

    join_imgs = prm_join["STM"].toList(2)
    iter_imgs = prm_iter["STM"].toList(2)

    jan_join = sample_pixel(ee.Image(join_imgs.get(0)))
    jul_join = sample_pixel(ee.Image(join_imgs.get(1)))
    jan_iter = sample_pixel(ee.Image(iter_imgs.get(0)))
    jul_iter = sample_pixel(ee.Image(iter_imgs.get(1)))

    assert jan_join["NDVI_mean"] == pytest.approx(0.2, abs=1e-4)
    assert jul_join["NDVI_mean"] == pytest.approx(0.8, abs=1e-4)
    assert jan_join == pytest.approx(jan_iter, abs=1e-4)
    assert jul_join == pytest.approx(jul_iter, abs=1e-4)


@pytest.mark.online
def test_tsi_wlin_and_rbf_reliability(ee_session, tiny_roi_berlin, make_synth_img, sample_pixel):
    """
    Tests Time Series Interpolation (TSI) for WLIN, 1RBF, and adaptive 2RBF:
    - Scenario A (symmetric gap filling on 10-day grid: Jan 1, Jan 11, Jan 21):
      * 2023-01-01: NDVI=0.2 (valid=True)
      * 2023-01-11: NDVI=0.0 (valid=False, masked gap at middle step)
      * 2023-01-21: NDVI=0.8 (valid=True)
      At 2023-01-11, both neighbors are 10 days away, so both WLIN and 1RBF must output 0.5.
    - Scenario B (adaptive bandwidth blending in 2RBF around Jan 11 with WIN1=5, WIN2=15, BW1=2):
      * two valid observations inside WIN1 (Jan 9 and Jan 13) both have NDVI=0.4
      * one extreme observation in WIN2 (Jan 1) has NDVI=10.0
      Because WIN1 already has 2 valid observations (== BW1), weight1=1.0 and weight2=0.0,
      so 2RBF at Jan 11 must ignore the WIN2 outlier and output 0.4.
    """
    ee = ee_session

    # scenario a: symmetric gap at 2023-01-11
    tss_gap = ee.ImageCollection([
        make_synth_img("2023-01-01", {"NDVI": 0.2}, valid=True),
        make_synth_img("2023-01-11", {"NDVI": 0.0}, valid=False),
        make_synth_img("2023-01-21", {"NDVI": 0.8}, valid=True),
    ])

    for method in ["WLIN", "1RBF"]:
        prm = init_param({
            "ROI": tiny_roi_berlin,
            "DATE_MIN": 20230101,
            "DATE_MAX": 20230121,
            "INIT_JANUARY_1ST": False,
            "FEATURES": ["NDVI"],
            "TSI": method,
            "INTERVAL": 10,
            "INTERVAL_UNIT": "day",
            "WIN1": 15,
            "SIGMA1": 10,
        })
        prm["TSS"] = tss_gap
        prm = run_level3(prm)

        # index 1 corresponds to the middle timestamp (2023-01-11)
        mid_img = ee.Image(prm["TSI"].sort("system:time_start").toList(3).get(1))
        px_mid = sample_pixel(mid_img)
        assert px_mid["NDVI"] == pytest.approx(0.5, abs=1e-4)

    # scenario b: adaptive 2rbf ignores win2 outlier when win1 count >= bw1
    tss_adaptive = ee.ImageCollection([
        make_synth_img("2023-01-01", {"NDVI": 10.0}, valid=True),
        make_synth_img("2023-01-09", {"NDVI": 0.4}, valid=True),
        make_synth_img("2023-01-13", {"NDVI": 0.4}, valid=True),
        make_synth_img("2023-01-21", {"NDVI": 0.4}, valid=True),
    ])

    prm_2rbf = init_param({
        "ROI": tiny_roi_berlin,
        "DATE_MIN": 20230101,
        "DATE_MAX": 20230121,
        "INIT_JANUARY_1ST": False,
        "FEATURES": ["NDVI"],
        "TSI": "2RBF",
        "INTERVAL": 10,
        "INTERVAL_UNIT": "day",
        "WIN1": 5,
        "WIN2": 15,
        "SIGMA1": 5,
        "SIGMA2": 15,
        "BW1": 2,
    })
    prm_2rbf["TSS"] = tss_adaptive
    prm_2rbf = run_level3(prm_2rbf)

    mid_2rbf = ee.Image(prm_2rbf["TSI"].sort("system:time_start").toList(3).get(1))
    px_2rbf = sample_pixel(mid_2rbf)
    assert px_2rbf["NDVI"] == pytest.approx(0.4, abs=1e-4)


@pytest.mark.online
def test_pbc_composites_reliability(ee_session, tiny_roi_berlin, make_synth_img, sample_pixel):
    """
    Tests Pixel-Based Compositing (PBC) methods (MAX-NDVI, inverted FEATURE, NLCD, and BAP):
    - input TSS has 3 images in summer 2023 tagged via BLU = 0.1, 0.5, 0.9:
      * img 1 (2023-06-05, DOY ~156): spectral bands = 0.1, NDVI = 0.2 (minimum NDVI)
      * img 2 (2023-06-20, DOY ~171): spectral bands = 0.5, NDVI = 0.5 (exact median across all bands)
      * img 3 (2023-07-02, DOY ~182): spectral bands = 0.9, NDVI = 0.9 (maximum NDVI & exact DOY target)
    - expected selection:
      * MAX-NDVI selects img 3 (BLU=0.9, NDVI=0.9)
      * inverted NDVI (PBC='NDVI', PBC_INVERT_QUALITY_METRIC=True) selects img 1 (BLU=0.1, NDVI=0.2)
      * NLCD (closest to spectral median) selects img 2 (BLU=0.5, NDVI=0.5)
      * BAP (with target window DOY 152-212 centered at DOY 182) selects img 3 (BLU=0.9, NDVI=0.9)
    """
    ee = ee_session
    bands = ["BLU", "GRN", "RED", "NIR", "SW1", "SW2", "NDVI"]

    def _band_vals(spec_val, ndvi_val):
        return {
            "BLU": spec_val, "GRN": spec_val, "RED": spec_val,
            "NIR": spec_val, "SW1": spec_val, "SW2": spec_val,
            "NDVI": ndvi_val
        }

    # build 3 candidate scenes with distinct spectral levels and dates
    tss = ee.ImageCollection([
        make_synth_img("2023-06-05", _band_vals(0.1, 0.2), valid=True),
        make_synth_img("2023-06-20", _band_vals(0.5, 0.5), valid=True),
        make_synth_img("2023-07-02", _band_vals(0.9, 0.9), valid=True),
    ])

    def _run_pbc(extra_params):
        prm = init_param({
            "ROI": tiny_roi_berlin,
            "YEAR_MIN": 2023,
            "YEAR_MAX": 2023,
            "FEATURES": bands,
            **extra_params,
        })
        prm["TSS"] = tss
        prm = run_level3(prm)
        return sample_pixel(prm["PBC"].first())

    # max-ndvi composite
    px_max = _run_pbc({"PBC": "MAX-NDVI"})
    assert px_max["BLU"] == pytest.approx(0.9, abs=1e-4)
    assert px_max["NDVI"] == pytest.approx(0.9, abs=1e-4)

    # inverted feature composite (min-ndvi)
    px_min = _run_pbc({"PBC": "NDVI", "PBC_INVERT_QUALITY_METRIC": True})
    assert px_min["BLU"] == pytest.approx(0.1, abs=1e-4)
    assert px_min["NDVI"] == pytest.approx(0.2, abs=1e-4)

    # nlcd median-distance composite
    px_nlcd = _run_pbc({"PBC": "NLCD"})
    assert px_nlcd["BLU"] == pytest.approx(0.5, abs=1e-4)
    assert px_nlcd["NDVI"] == pytest.approx(0.5, abs=1e-4)

    # bap composite with doy fold centered at doy 182 (2023-07-02)
    px_bap = _run_pbc({
        "PBC": "BAP",
        "PBC_FOLDING": True,
        "FOLD_CUSTOM": {"doy": ["152-212"]},
    })
    assert px_bap["BLU"] == pytest.approx(0.9, abs=1e-4)


@pytest.mark.online
def test_lsp_polar_reliability(ee_session, tiny_roi_berlin, test_point, sample_pixel):
    """
    Tests Land Surface Phenology (LSP='POLAR') on a synthetic 3-year equidistant time series:
    - constructs a 16-day time series across 2021-2023 where NDVI follows a clean seasonal
      cosine curve peaking at mid-year (DOY 182.5):
      NDVI(doy) = 0.5 - 0.4 * cos(2 * pi * doy / 365)
    - expected phenometrics for the bracketed year 2022:
      * long-term peak vector RAVG is near DOY 182 (within 1 interpolation step, +-16 days)
      * middle of season (MOS, 50% cumulative NDVI) is near DOY 182 (+-16 days)
      * seasonal percentiles follow strict chronological order: SOS < SOP < MOS < EOP < EOS.
    """
    ee = ee_session

    # generate 3 years of 16-day timestamps (2021-01-01 to 2023-12-31) on the server
    start_date = ee.Date("2021-01-01")
    steps = ee.List.sequence(0, 365 * 3, 16)

    def _make_cosine_step(day_offset):
        d = start_date.advance(day_offset, "day")
        doy = d.getRelative("day", "year")
        angle = doy.divide(365.0).multiply(2.0 * math.pi)
        ndvi_val = ee.Number(0.5).subtract(angle.cos().multiply(0.4))
        img = ee.Image.constant(ndvi_val).rename("NDVI").toFloat()
        mask = ee.Image.constant(1).rename("mask").toUint8()
        return (
            img.updateMask(mask)
            .addBands(mask)
            .setDefaultProjection("EPSG:4326", None, 30)
            .set({"system:time_start": d.millis()})
        )

    tsi_synth = ee.ImageCollection.fromImages(steps.map(_make_cosine_step))

    prm = init_param({
        "ROI": tiny_roi_berlin,
        "YEAR_MIN": 2021,
        "YEAR_MAX": 2023,
        "LSP": "POLAR",
        "LSP_BASE_IMGCOL": "TSI",
        "LSP_BAND": "NDVI",
        "LSP_YEAR_MIN": 2022,
        "LSP_YEAR_MAX": 2022,
        "LSP_ADJUST_SEASONAL": True,
    })
    prm["TSI"] = tsi_synth
    prm = run_level4(prm)

    px_lsp = sample_pixel(prm["LSP"].first())

    # peak vector (ravg) and middle of season (mos) must align with mid-year peak (~doy 182)
    assert px_lsp["RAVG"] == pytest.approx(182, abs=16)
    assert px_lsp["MOS"] == pytest.approx(182, abs=16)

    # phenological stages must be strictly ordered in day-of-year
    assert px_lsp["SOS"] < px_lsp["SOP"] < px_lsp["MOS"] < px_lsp["EOP"] < px_lsp["EOS"]

