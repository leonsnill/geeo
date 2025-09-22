from geeo.utils import load_parameters, merge_parameters, load_blueprint
from geeo.level2.level2 import run_level2
from geeo.level3.level3 import run_level3
from geeo.level4.level4 import run_level4
from geeo.export.export import run_export
import geopandas as gpd
from tqdm import tqdm

def init_param(params):
    # load blueprint defaults
    default_params = load_blueprint()
    
    # determine the type of `params` and load/merge accordingly
    if isinstance(params, str):
        yaml_params = load_parameters(params)
        prm = merge_parameters(default_params, yaml_params)
    elif isinstance(params, dict):
        prm = merge_parameters(default_params, dict_params=params)
    else:
        raise ValueError("params must be either a path to a YAML file or a dictionary")
    return prm


def run_param(params):
    """
    Run the level2 and level3 process with the given parameters.

    Args:
        params (str or dict): Path to the YAML file or a dictionary of parameters.
    """

    # load blueprint defaults
    default_params = load_blueprint()
    
    # determine the type of `params` and load/merge accordingly
    if isinstance(params, str):
        # assume params is a path to a YAML file
        yaml_params = load_parameters(params)
        prm = merge_parameters(default_params, yaml_params)
    elif isinstance(params, dict):
        prm = merge_parameters(default_params, dict_params=params)
    else:
        raise ValueError("params must be either a path to a YAML file or a dictionary")
    
    # ROI datacube/tiling scheme detection
    ROI_TILES = prm.get('ROI_TILES')

    if not ROI_TILES:
    # ---------------------------------------------------------------------------------------------------
    # classic single run
        pipeline = run_level2(prm)
        pipeline = run_level3(pipeline)
        pipeline = run_level4(pipeline)
        pipeline = run_export(pipeline)
        return pipeline

    # ---------------------------------------------------------------------------------------------------
    # multiple runs for each feature in ROI vector file
    
    else:
        ROI_TILES_ATTRIBUTE_COLUMN = prm.get('ROI_TILES_ATTRIBUTE_COLUMN')
        if not ROI_TILES_ATTRIBUTE_COLUMN:
            raise ValueError("ROI_TILES is set to True, but ROI_TILES_ATTRIBUTE_COLUMN is not specified.")
        ROI_TILES_ATTRIBUTE_LIST = prm.get('ROI_TILES_ATTRIBUTE_LIST')

        roi_obj = prm.get('ROI')
        if roi_obj is None:
            raise ValueError("ROI_TILES is set to True, but no ROI provided in parameters (prm['ROI'] is None).")
        if isinstance(roi_obj, gpd.GeoDataFrame):
            roi_gdf = roi_obj
        elif isinstance(roi_obj, str):
            try:
                roi_gdf = gpd.read_file(roi_obj)
            except Exception as e:
                raise ValueError("Could not read ROI file. ROI_TILES is True, but ROI is not a valid vector file path.") from e
        else:
            raise ValueError("ROI must be either a path to a vector file or a GeoDataFrame when ROI_TILES is True.")

        # subset?
        if ROI_TILES_ATTRIBUTE_LIST:
            roi_gdf = roi_gdf[roi_gdf[ROI_TILES_ATTRIBUTE_COLUMN].isin(ROI_TILES_ATTRIBUTE_LIST)]
            if roi_gdf.empty:
                raise ValueError("The specified ROI_TILES_ATTRIBUTE_LIST does not match any entries in the ROI file.")
        elif roi_gdf[ROI_TILES_ATTRIBUTE_COLUMN].isnull().all():
            raise ValueError("The specified ROI_TILES_ATTRIBUTE_COLUMN contains only null values.")
        
        print("---------------------------------------------------------")
        print("            USING ROI TILE PROCESSING MODE")
        print("")
        print(f"ROI_TILES: ROI contains {len(roi_gdf)} features.")

        pipelines = {}
        for i in tqdm(range(len(roi_gdf))):
            TILE = roi_gdf.iloc[[i]]
            tile_id = TILE[ROI_TILES_ATTRIBUTE_COLUMN].values[0]
            prm_tile = prm.copy()
            prm_tile['ROI'] = TILE
            prm_tile['EXPORT_DESC'] = f"{prm.get('EXPORT_DESC')}_{tile_id}"
            pipeline = run_level2(prm_tile)
            pipeline = run_level3(pipeline)
            pipeline = run_level4(pipeline)
            pipeline = run_export(pipeline)
            pipelines[tile_id] = pipeline
        print("---------------------------------------------------------")
        return pipelines


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run level2 process.")
    parser.add_argument('--params', type=str, help="Path to the YAML file or dictionary of parameters in JSON format.")

    args = parser.parse_args()

    if args.params:
        try:
            import json
            params = json.loads(args.params)
        except json.JSONDecodeError:
            params = args.params
    else:
        params = None

    run_param(params)
