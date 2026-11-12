# tests for spatial tiling functionality

import pytest
import geopandas as gpd
from shapely.geometry import box, Polygon

from geeo.misc.tiling import (
    parse_glance_preset,
    resolve_grid,
    filter_grid_by_target,
    get_tile_ids,
    extract_tile_geom,
    run_tiled
)


@pytest.fixture
def sample_grid_gdf():
    # create small two-tile geodataframe in epsg:4326
    poly1 = box(13.0, 52.0, 13.5, 52.5)
    poly2 = box(13.5, 52.0, 14.0, 52.5)
    return gpd.GeoDataFrame({
        "tile_id": ["tile_a", "tile_b"],
        "geometry": [poly1, poly2]
    }, crs="EPSG:4326")


def test_parse_glance_preset():
    # valid presets
    assert parse_glance_preset("EU_300") == ("EU", 300000, "ID300")
    assert parse_glance_preset("glance_eu_600") == ("EU", 600000, "ID600")
    assert parse_glance_preset("NA_150KM") == ("NA", 150000, "ID150")

    # invalid presets
    assert parse_glance_preset("UNKNOWN_300") is None
    assert parse_glance_preset("EU_999") is None
    assert parse_glance_preset("true") is None


def test_resolve_grid_preset():
    # test glance preset resolution
    prm = {"ROI": [13.0, 52.0, 14.0, 52.5]}
    grid, col, is_external = resolve_grid("EU_300", prm)
    
    assert isinstance(grid, gpd.GeoDataFrame)
    assert col == "ID300"
    assert is_external is True
    assert "ID300" in grid.columns


def test_resolve_grid_backward_compatible(sample_grid_gdf):
    # test boolean true with roi as grid
    prm = {
        "ROI": sample_grid_gdf,
        "ROI_TILES_ATTRIBUTE_COLUMN": "tile_id"
    }
    grid, col, is_external = resolve_grid(True, prm)
    
    assert grid is sample_grid_gdf
    assert col == "tile_id"
    assert is_external is False


def test_filter_grid_by_target(sample_grid_gdf):
    # target bbox overlapping only tile_a
    target_bbox = [13.1, 52.1, 13.4, 52.4]
    filtered = filter_grid_by_target(sample_grid_gdf, target_bbox)
    
    assert len(filtered) == 1
    assert filtered["tile_id"].iloc[0] == "tile_a"


def test_get_tile_ids_subset(sample_grid_gdf):
    # test subsetting tile list
    grid, ids = get_tile_ids(sample_grid_gdf, "tile_id", subset_list=["tile_b"])
    
    assert ids == ["tile_b"]
    assert len(grid) == 1


def test_run_tiled_mocked_pipeline(monkeypatch, sample_grid_gdf):
    # verify tiling loop and desc naming without running heavy ee pipelines
    called_tiles = []

    def mock_l2(prm):
        called_tiles.append((prm["EXPORT_DESC"], prm["ROI_TILES"]))
        return {"status": "l2_done", "desc": prm["EXPORT_DESC"]}

    monkeypatch.setattr("geeo.misc.tiling.run_level2", mock_l2)
    monkeypatch.setattr("geeo.misc.tiling.run_level3", lambda p: p)
    monkeypatch.setattr("geeo.misc.tiling.run_level4", lambda p: p)
    monkeypatch.setattr("geeo.misc.tiling.run_export", lambda p: p)

    prm = {
        "ROI": sample_grid_gdf,
        "ROI_TILES": True,
        "ROI_TILES_ATTRIBUTE_COLUMN": "tile_id",
        "EXPORT_DESC": "TEST_JOB"
    }

    results = run_tiled(prm)

    assert set(results.keys()) == {"tile_a", "tile_b"}
    assert len(called_tiles) == 2
    # verify export_desc was appended and roi_tiles was disabled to avoid recursion
    assert called_tiles[0] == ("TEST_JOB_tile_a", False)
    assert called_tiles[1] == ("TEST_JOB_tile_b", False)


@pytest.mark.smoke
@pytest.mark.online
def test_tiling_geom_and_ee_filtering_online(ee_session, sample_grid_gdf):
    # test server-side geometry creation and point filtering
    ee = ee_session
    
    # extract first tile geometry
    tile_obj, tile_geom = extract_tile_geom(sample_grid_gdf, "tile_id", "tile_a")
    assert isinstance(tile_geom, ee.Geometry)

    # create two test points: one inside tile_a, one outside
    pt_inside = ee.Feature(ee.Geometry.Point([13.2, 52.2]), {"name": "inside"})
    pt_outside = ee.Feature(ee.Geometry.Point([13.8, 52.2]), {"name": "outside"})
    fc = ee.FeatureCollection([pt_inside, pt_outside])

    # filter server-side with tile_geom
    filtered = fc.filterBounds(tile_geom)
    assert filtered.size().getInfo() == 1
    names = filtered.aggregate_array("name").getInfo()
    assert names == ["inside"]
