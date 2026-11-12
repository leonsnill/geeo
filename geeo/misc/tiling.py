import os
import re
import ee
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm

from geeo.level2.level2 import run_level2
from geeo.level3.level3 import run_level3
from geeo.level4.level4 import run_level4
from geeo.export.export import run_export
from geeo.misc.spacetime import create_roi, create_glance_tiles


def parse_glance_preset(preset_str):
    # parse presets like eu_300, glance_eu_300
    pattern = r'(?:GLANCE_)?([A-Z]{2})_(\d+)(?:KM)?'
    match = re.fullmatch(pattern, str(preset_str).strip().upper())
    if not match:
        return None
    continent, size_km = match.groups()
    valid_continents = ['AF', 'AN', 'AS', 'EU', 'OC', 'NA', 'SA']
    if continent not in valid_continents:
        return None
    tile_size = int(size_km) * 1000
    valid_sizes = [1200000, 600000, 300000, 150000, 75000, 30000]
    if tile_size not in valid_sizes:
        return None
    return continent, tile_size, f"ID{size_km}"


def resolve_grid(roi_tiles, prm):
    # resolve grid source and id attribute column
    col = prm.get('ROI_TILES_ATTRIBUTE_COLUMN')
    target = prm.get('ROI')

    # boolean true: features in roi are the tiles
    if isinstance(roi_tiles, bool) or (isinstance(roi_tiles, str) and roi_tiles.lower() == 'true'):
        if target is None:
            raise ValueError("ROI_TILES is True, but no ROI provided in parameters.")
        grid = target
        is_external = False

    # preset string (e.g. eu_300, glance_eu_300)
    elif isinstance(roi_tiles, str) and parse_glance_preset(roi_tiles):
        continent, tile_size, default_col = parse_glance_preset(roi_tiles)
        grid = create_glance_tiles(continent_code=continent, tile_size=tile_size, land_mask=True)
        if not col:
            col = default_col
        is_external = True

    # string path (local vector file or ee asset)
    elif isinstance(roi_tiles, str):
        if roi_tiles.startswith('projects/') or roi_tiles.startswith('users/'):
            grid = ee.FeatureCollection(roi_tiles)
        else:
            try:
                grid = gpd.read_file(roi_tiles)
            except Exception as e:
                raise ValueError(f"Could not read tiling vector file at: {roi_tiles}") from e
        is_external = (target is not None and target != roi_tiles)

    # direct object
    elif isinstance(roi_tiles, (gpd.GeoDataFrame, ee.FeatureCollection)):
        grid = roi_tiles
        is_external = (target is not None and target is not roi_tiles)

    else:
        raise ValueError(f"Unsupported ROI_TILES value: {roi_tiles}")

    # ensure local file path in roi (when roi_tiles=True) is read to geodataframe
    if isinstance(grid, str):
        if grid.startswith('projects/') or grid.startswith('users/'):
            grid = ee.FeatureCollection(grid)
        else:
            try:
                grid = gpd.read_file(grid)
            except Exception as e:
                raise ValueError(f"Could not read vector file at: {grid}") from e

    if not col:
        raise ValueError("ROI_TILES_ATTRIBUTE_COLUMN is not specified and could not be inferred.")

    return grid, col, is_external


def filter_grid_by_target(grid, target):
    # optionally clip geodataframe grid by target bbox or vector
    if not isinstance(grid, gpd.GeoDataFrame):
        return grid

    # target is bbox [minx, miny, maxx, maxy]
    if isinstance(target, (list, tuple)) and len(target) == 4 and all(isinstance(v, (int, float)) for v in target):
        bbox_poly = box(target[0], target[1], target[2], target[3])
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326").to_crs(grid.crs)
        grid = grid[grid.geometry.intersects(bbox_gdf.geometry.iloc[0])]

    # target is geodataframe
    elif isinstance(target, gpd.GeoDataFrame):
        target_reproj = target.to_crs(grid.crs)
        grid = grid[grid.geometry.intersects(target_reproj.union_all())]

    # target is path to vector
    elif isinstance(target, str) and not (target.startswith('projects/') or target.startswith('users/')) and os.path.exists(target):
        try:
            target_gdf = gpd.read_file(target).to_crs(grid.crs)
            grid = grid[grid.geometry.intersects(target_gdf.union_all())]
        except Exception:
            pass

    return grid


def get_tile_ids(grid, col, subset_list=None):
    # get list of tile ids to process
    if isinstance(grid, gpd.GeoDataFrame):
        if subset_list:
            grid = grid[grid[col].isin(subset_list)]
            if grid.empty:
                raise ValueError("ROI_TILES_ATTRIBUTE_LIST does not match any entries in the grid.")
        elif grid[col].isnull().all():
            raise ValueError(f"Grid attribute column '{col}' contains only null values.")
        tile_ids = grid[col].tolist()
        return grid, tile_ids

    elif isinstance(grid, ee.FeatureCollection):
        if subset_list:
            grid = grid.filter(ee.Filter.inList(col, subset_list))
            tile_ids = subset_list
        else:
            tile_ids = grid.aggregate_array(col).distinct().getInfo()
        if not tile_ids:
            raise ValueError(f"No features found in FeatureCollection for column '{col}'.")
        return grid, tile_ids

    else:
        raise ValueError("Grid must be a GeoDataFrame or ee.FeatureCollection.")


def extract_tile_geom(grid, col, tile_id):
    # extract tile geometry and object
    if isinstance(grid, gpd.GeoDataFrame):
        tile_gdf = grid[grid[col] == tile_id]
        roi_meta = create_roi(tile_gdf, simplify_geom_to_bbox=True)
        return tile_gdf, roi_meta['roi_geom']
    else:
        tile_fc = grid.filter(ee.Filter.eq(col, tile_id))
        return tile_fc, tile_fc.geometry()


def run_tiled(prm):
    # setup tiling run
    roi_tiles = prm.get('ROI_TILES')
    grid, col, is_external = resolve_grid(roi_tiles, prm)
    subset_list = prm.get('ROI_TILES_ATTRIBUTE_LIST')
    target = prm.get('ROI')

    # clip grid if external and target is bounding box / vector
    if is_external:
        grid = filter_grid_by_target(grid, target)

    grid, tile_ids = get_tile_ids(grid, col, subset_list)

    print("---------------------------------------------------------")
    print("            USING ROI TILE PROCESSING MODE")
    print("")
    print(f"ROI_TILES: Processing {len(tile_ids)} tiles.")
    print("---------------------------------------------------------")

    pipelines = {}
    for tile_id in tqdm(tile_ids):
        tile_obj, tile_geom = extract_tile_geom(grid, col, tile_id)

        prm_tile = prm.copy()
        prm_tile['ROI_TILES'] = False
        prm_tile['EXPORT_DESC'] = f"{prm.get('EXPORT_DESC')}_{tile_id}"  # append tile id to export description

        # handle target subsetting when using external grid
        if is_external:
            # target is ee featurecollection
            if isinstance(target, ee.FeatureCollection) or (isinstance(target, str) and (target.startswith('projects/') or target.startswith('users/'))):
                target_fc = target if isinstance(target, ee.FeatureCollection) else ee.FeatureCollection(target)
                target_tile = target_fc.filterBounds(tile_geom)
                my_size = target_tile.size().getInfo()
                print(f"Tile {tile_id}: {my_size} features in target within tile.")
                if target_tile.size().getInfo() == 0:
                    continue
                prm_tile['ROI'] = target_tile

            # target is local geodataframe
            elif isinstance(target, gpd.GeoDataFrame):
                target_reproj = target.to_crs(tile_obj.crs) if target.crs != tile_obj.crs else target
                target_tile = target[target_reproj.geometry.intersects(tile_obj.union_all())]
                if target_tile.empty:
                    continue
                prm_tile['ROI'] = target_tile

            # target is bbox or polygon: tile geometry defines roi
            else:
                prm_tile['ROI'] = tile_obj
        else:
            prm_tile['ROI'] = tile_obj

        # run pipeline
        pipeline = run_level2(prm_tile)
        pipeline = run_level3(pipeline)
        pipeline = run_level4(pipeline)
        pipeline = run_export(pipeline)
        pipelines[tile_id] = pipeline

    print("---------------------------------------------------------")
    return pipelines
