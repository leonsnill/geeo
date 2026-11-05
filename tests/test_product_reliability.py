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
    Tests Time Series Interpolation (TSI) with asymmetric weights and adaptive blending:
    - Scenario A (asymmetric WLIN on 10-day grid: Jan 1, Jan 11, Jan 21):
      * 2023-01-06 (dt = -5d from Jan 11): NDVI=0.2 (valid=True)
      * 2023-01-11 (dt =  0d):             NDVI=0.0 (valid=False, masked gap)
      * 2023-01-21 (dt = +10d from Jan 11): NDVI=0.8 (valid=True)
      Linear ratio at Jan 11 is (11 - 6) / (21 - 6) = 5 / 15 = 1/3, so
      WLIN = 0.2 + (0.8 - 0.2) * (1/3) = 0.4.
    - Scenario B (asymmetric 1RBF with SIGMA1=10, WIN1=10 at Jan 11):
      * 2023-01-11 (dt =  0d): NDVI=0.2 -> weight w0 = exp(0) = 1.0
      * 2023-01-21 (dt = 10d): NDVI=0.8 -> weight w1 = exp(-0.5 * (10/10)^2) = exp(-0.5)
      Expected 1RBF at Jan 11 = (1.0 * 0.2 + exp(-0.5) * 0.8) / (1.0 + exp(-0.5)) ~ 0.42653.
    - Scenario C (adaptive 2RBF around Jan 11 with WIN1=5, WIN2=15, SIGMA1=5, SIGMA2=10, BW1=2):
      1) Full saturation (2 valid obs in WIN1 == BW1): weight1=1.0, weight2=0.0 -> ignores WIN2
         outlier (10.0) and outputs 0.4.
      2) Partial 50/50 blend (1 valid obs in WIN1 < BW1=2): weight1=0.5 (result1=0.2) and
         weight2=0.5 (result2=0.42653) -> outputs 0.5 * 0.2 + 0.5 * 0.42653 ~ 0.31327.
    """
    ee = ee_session

    # scenario a: asymmetric wlin (dt1 = 5d, dt2 = 10d -> ratio = 1/3)
    tss_wlin = ee.ImageCollection([
        make_synth_img("2023-01-01", {"NDVI": 0.0}, valid=True),
        make_synth_img("2023-01-06", {"NDVI": 0.2}, valid=True),
        make_synth_img("2023-01-11", {"NDVI": 0.0}, valid=False),
        make_synth_img("2023-01-21", {"NDVI": 0.8}, valid=True),
    ])
    prm_wlin = init_param({
        "ROI": tiny_roi_berlin,
        "DATE_MIN": 20230101,
        "DATE_MAX": 20230121,
        "INIT_JANUARY_1ST": False,
        "FEATURES": ["NDVI"],
        "TSI": "WLIN",
        "INTERVAL": 10,
        "INTERVAL_UNIT": "day",
        "WIN1": 15,
    })
    prm_wlin["TSS"] = tss_wlin
    prm_wlin = run_level3(prm_wlin)
    mid_wlin = ee.Image(prm_wlin["TSI"].sort("system:time_start").toList(3).get(1))
    assert sample_pixel(mid_wlin)["NDVI"] == pytest.approx(0.4, abs=1e-4)

    # scenario b: asymmetric 1rbf (dt0 = 0d with w=1.0, dt1 = 10d=sigma with w=exp(-0.5))
    w_sigma = math.exp(-0.5)
    expected_rbf = (1.0 * 0.2 + w_sigma * 0.8) / (1.0 + w_sigma)

    tss_1rbf = ee.ImageCollection([
        make_synth_img("2023-01-01", {"NDVI": 0.0}, valid=False),
        make_synth_img("2023-01-11", {"NDVI": 0.2}, valid=True),
        make_synth_img("2023-01-21", {"NDVI": 0.8}, valid=True),
    ])
    prm_1rbf = init_param({
        "ROI": tiny_roi_berlin,
        "DATE_MIN": 20230101,
        "DATE_MAX": 20230121,
        "INIT_JANUARY_1ST": False,
        "FEATURES": ["NDVI"],
        "TSI": "1RBF",
        "INTERVAL": 10,
        "INTERVAL_UNIT": "day",
        "WIN1": 10,
        "SIGMA1": 10,
    })
    prm_1rbf["TSS"] = tss_1rbf
    prm_1rbf = run_level3(prm_1rbf)
    mid_1rbf = ee.Image(prm_1rbf["TSI"].sort("system:time_start").toList(3).get(1))
    assert sample_pixel(mid_1rbf)["NDVI"] == pytest.approx(expected_rbf, abs=1e-4)

    # scenario c1: adaptive 2rbf full saturation (2 obs in win1 == bw1 -> weight2 = 0)
    tss_2rbf_full = ee.ImageCollection([
        make_synth_img("2023-01-01", {"NDVI": 10.0}, valid=True),
        make_synth_img("2023-01-09", {"NDVI": 0.4}, valid=True),
        make_synth_img("2023-01-13", {"NDVI": 0.4}, valid=True),
        make_synth_img("2023-01-21", {"NDVI": 0.4}, valid=True),
    ])
    base_2rbf = {
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
        "SIGMA2": 10,
        "BW1": 2,
    }
    prm_full = init_param(base_2rbf)
    prm_full["TSS"] = tss_2rbf_full
    prm_full = run_level3(prm_full)
    mid_full = ee.Image(prm_full["TSI"].sort("system:time_start").toList(3).get(1))
    assert sample_pixel(mid_full)["NDVI"] == pytest.approx(0.4, abs=1e-4)

    # scenario c2: adaptive 2rbf partial blend (1 obs in win1 < bw1=2 -> weight1=0.5, weight2=0.5)
    prm_blend = init_param(base_2rbf)
    prm_blend["TSS"] = tss_1rbf
    prm_blend = run_level3(prm_blend)
    mid_blend = ee.Image(prm_blend["TSI"].sort("system:time_start").toList(3).get(1))
    expected_blend = 0.5 * 0.2 + 0.5 * expected_rbf
    assert sample_pixel(mid_blend)["NDVI"] == pytest.approx(expected_blend, abs=1e-4)


@pytest.mark.online
def test_pbc_composites_reliability(ee_session, tiny_roi_berlin, make_synth_img, sample_pixel):
    """
    Tests Pixel-Based Compositing (PBC) methods (MAX-NDVI, inverted FEATURE, NLCD, and BAP):
    - Part 1 (MAX-NDVI, MIN-NDVI, NLCD on 3 summer 2023 scenes tagged via BLU = 0.1, 0.5, 0.9):
      * img 1 (2023-06-05): spectral bands = 0.1, NDVI = 0.2 (minimum NDVI)
      * img 2 (2023-06-20): spectral bands = 0.5, NDVI = 0.5 (exact median across all bands)
      * img 3 (2023-07-02): spectral bands = 0.9, NDVI = 0.9 (maximum NDVI)
    - Part 2 (BAP trade-off between DOY offset and YEAR offset via PBC_BAP_DOY_EQ_YEAR):
      Target window is 2021-2023 (year_center=2022, year_offset=1) and DOY 152-212 (doy_center=182):
      * scene A (2022-07-22, target year 2022, DOY ~202 -> 20 days off target DOY): BLU=0.2
      * scene B (2021-07-02, 1 year off target year, DOY ~182 -> 0 days off target DOY): BLU=0.8
      When PBC_BAP_DOY_EQ_YEAR=30 (1 year offset penalized like 30 DOY days > 20d), scene A wins (0.2).
      When PBC_BAP_DOY_EQ_YEAR=10 (1 year offset penalized like 10 DOY days < 20d), scene B wins (0.8).
    """
    ee = ee_session
    bands = ["BLU", "GRN", "RED", "NIR", "SW1", "SW2", "NDVI"]

    def _band_vals(spec_val, ndvi_val):
        return {
            "BLU": spec_val, "GRN": spec_val, "RED": spec_val,
            "NIR": spec_val, "SW1": spec_val, "SW2": spec_val,
            "NDVI": ndvi_val
        }

    # part 1: max-ndvi, inverted feature, and nlcd median composite
    tss = ee.ImageCollection([
        make_synth_img("2023-06-05", _band_vals(0.1, 0.2), valid=True),
        make_synth_img("2023-06-20", _band_vals(0.5, 0.5), valid=True),
        make_synth_img("2023-07-02", _band_vals(0.9, 0.9), valid=True),
    ])

    def _run_pbc(imgcol, year_min, year_max, extra_params):
        prm = init_param({
            "ROI": tiny_roi_berlin,
            "YEAR_MIN": year_min,
            "YEAR_MAX": year_max,
            "FEATURES": bands,
            **extra_params,
        })
        prm["TSS"] = imgcol
        prm = run_level3(prm)
        return sample_pixel(prm["PBC"].first())

    px_max = _run_pbc(tss, 2023, 2023, {"PBC": "MAX-NDVI"})
    assert px_max["BLU"] == pytest.approx(0.9, abs=1e-4)

    px_min = _run_pbc(tss, 2023, 2023, {"PBC": "NDVI", "PBC_INVERT_QUALITY_METRIC": True})
    assert px_min["BLU"] == pytest.approx(0.1, abs=1e-4)

    px_nlcd = _run_pbc(tss, 2023, 2023, {"PBC": "NLCD"})
    assert px_nlcd["BLU"] == pytest.approx(0.5, abs=1e-4)

    # part 2: bap doy-vs-year score trade-off
    tss_bap = ee.ImageCollection([
        make_synth_img("2022-07-22", _band_vals(0.2, 0.5), valid=True),
        make_synth_img("2021-07-02", _band_vals(0.8, 0.5), valid=True),
    ])
    bap_base = {
        "PBC": "BAP",
        "PBC_FOLDING": True,
        "FOLD_CUSTOM": {"doy": ["152-212"]},
        "PBC_BAP_WEIGHT_DOY": 0.4,
        "PBC_BAP_WEIGHT_YEAR": 0.4,
        "PBC_BAP_WEIGHT_CLOUD": 0.2,
    }

    # doy_eq_year=30 penalizes 1-year offset more than 20-day doy offset -> 2022 scene (0.2) wins
    px_bap_30 = _run_pbc(tss_bap, 2021, 2023, {**bap_base, "PBC_BAP_DOY_EQ_YEAR": 30})
    assert px_bap_30["BLU"] == pytest.approx(0.2, abs=1e-4)

    # doy_eq_year=10 penalizes 1-year offset less than 20-day doy offset -> 2021 scene (0.8) wins
    px_bap_10 = _run_pbc(tss_bap, 2021, 2023, {**bap_base, "PBC_BAP_DOY_EQ_YEAR": 10})
    assert px_bap_10["BLU"] == pytest.approx(0.8, abs=1e-4)


@pytest.mark.online
def test_lsp_polar_reliability(ee_session, tiny_roi_berlin, sample_pixel):
    """
    Tests Land Surface Phenology (LSP='POLAR') on a 3-year constant-rate boxcar growing season:
    - constructs a 5-day time series across 2021-2023 where NDVI = 1.0 during the 100-day
      window [DOY 100, DOY 200] (centered at DOY 150) and NDVI = 0.0 outside [100, 200].
    - because cumulative NDVI grows strictly linearly from 0% at DOY 100 to 100% at DOY 200,
      every phenometric in 2022 has an exact analytical DOY expectation (within 1 step = +-5 days):
      * RAVG (peak vector) = 150
      * SOS  (15th percentile) = 115
      * SOP  (25th percentile) = 125
      * MOS  (50th percentile) = 150
      * EOP  (75th percentile) = 175
      * EOS  (80th percentile) = 180
    """
    ee = ee_session

    # generate 3 years of 5-day steps (2021-01-01 to 2023-12-31) with boxcar ndvi in [doy 100, 200]
    start_date = ee.Date("2021-01-01")
    steps = ee.List.sequence(0, 365 * 3, 5)

    def _make_boxcar_step(day_offset):
        d = start_date.advance(day_offset, "day")
        doy = d.getRelative("day", "year")
        in_season = doy.gte(100).And(doy.lte(200))
        img = ee.Image.constant(in_season).rename("NDVI").toFloat()
        mask = ee.Image.constant(1).rename("mask").toUint8()
        return (
            img.updateMask(mask)
            .addBands(mask)
            .setDefaultProjection("EPSG:4326", None, 30)
            .set({"system:time_start": d.millis()})
        )

    tsi_synth = ee.ImageCollection.fromImages(steps.map(_make_boxcar_step))

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

    # verify exact analytical doy quantiles (within 1 sampling step = 5 days)
    assert px_lsp["RAVG"] == pytest.approx(150, abs=5)
    assert px_lsp["SOS"] == pytest.approx(115, abs=5)
    assert px_lsp["SOP"] == pytest.approx(125, abs=5)
    assert px_lsp["MOS"] == pytest.approx(150, abs=5)
    assert px_lsp["EOP"] == pytest.approx(175, abs=5)
    assert px_lsp["EOS"] == pytest.approx(180, abs=5)

