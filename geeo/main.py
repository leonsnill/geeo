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
    
    # sanity check of parameters to assure all parameter names (keys) are valid
    invalid_keys = [key for key in prm.keys() if key not in default_params]
    if invalid_keys:
        bullet_list = "\n".join(f"- {key}" for key in invalid_keys)
        raise ValueError(f"Invalid parameter(s):\n{bullet_list}")
    
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
    # multiple runs for each tile
    else:
        from geeo.misc.tiling import run_tiled
        return run_tiled(prm)


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
