# =============================================================================
# Lazy module loader
import importlib

class LazyLoader:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def _load_module(self):
        if self.module is None:
            #print(f"Loading module {self.module_name}")
            self.module = importlib.import_module(self.module_name)
        return self.module

    def __getattr__(self, item):
        module = self._load_module()
        return getattr(module, item)

# =============================================================================
# Create parameter file
import shutil
import os

def load_blueprint():
    """
    Load the blueprint from the YAML file located in the config directory.

    Returns:
        dict: Dictionary of blueprint parameters with their default values.
    """
    blueprint_path = os.path.join(os.path.dirname(__file__), 'config', 'parameter_blueprint.yml')
    if not os.path.exists(blueprint_path):
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")

    with open(blueprint_path, 'r') as file:
        blueprint = yaml.safe_load(file)

    return blueprint


def create_parameter_file(destination="", overwrite=False):
    """
    Copies the blueprint YAML parameter file to the specified destination.
    Handles cases where the destination is a directory, a file without .yml suffix, a full file path, 
    or the current working directory if no destination is provided.

    Args:
        destination: Path to the destination YAML file or directory.
        overwrite: If True, overwrite the existing file if it exists. Default is False.
    
    Example usage:
        create_parameter_file("new_parameters.yml")  # full file path
        create_parameter_file("/path/to/directory")  # directory path
        create_parameter_file("new_parameters")      # file name without suffix
        create_parameter_file()                      # current working directory with default name
    """
    # Define the blueprint path
    blueprint_path = os.path.join(os.path.dirname(__file__), 'config', 'parameter_blueprint.yml')
    
    # Check if the blueprint file exists
    if not os.path.exists(blueprint_path):
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    # Determine the destination
    if not destination:
        # If no destination is provided, use the current working directory with default name
        dest_filename = os.path.join(os.getcwd(), "parameter.yml")
    elif os.path.isdir(destination):
        # If a directory is provided, use it with default name
        dest_filename = os.path.join(destination, "parameter.yml")
    else:
        dest_filename = destination
        if not dest_filename.endswith(".yml"):
            dest_filename += ".yml"
    
    # Check if the destination file exists
    if os.path.exists(dest_filename) and not overwrite:
        print(f"Warning: File already exists: {dest_filename}. Set overwrite=True if you wish to replace it.")
        return
    
    # Create the destination directory if it doesn't exist
    dest_dir = os.path.dirname(dest_filename)
    if dest_dir:  # Check if dest_dir is not empty
        os.makedirs(dest_dir, exist_ok=True)

    # Copy the blueprint file to the destination
    shutil.copy(blueprint_path, dest_filename)
    print(f"Parameter file created: {dest_filename}")


# =============================================================================
# load parameter file
import yaml

def load_parameters(yaml_file=None):
    """
    Load parameters from a YAML file or use the blueprint if no file is provided.

    Args:
        yaml_file (str, optional): Path to the YAML file. Defaults to None.

    Returns:
        dict: Dictionary of parameters.
    """
    if yaml_file:
        with open(yaml_file, 'r') as file:
            parameters = yaml.safe_load(file)
    else:
        parameters = load_blueprint()
    return parameters


def merge_parameters(default_params, yaml_params=None, dict_params=None):
    """
    Merge parameters from the blueprint, YAML file, and the provided dictionary.

    Args:
        default_params (dict): Default parameters from the blueprint.
        yaml_params (dict, optional): Parameters loaded from the YAML file.
        dict_params (dict, optional): Parameters provided as a dictionary.

    Returns:
        dict: Merged dictionary of parameters.
    """
    merged_params = default_params.copy()
    if yaml_params:
        merged_params.update(yaml_params)
    if dict_params:
        merged_params.update(dict_params)
    return merged_params


# expected image size
def calculate_image_size(pixels=None, width=None, height=None, width_m=None, height_m=None, pixel_size_m=None, datatype='uint8', num_bands=1):
    """
    Calculate the number of pixels in a single-band image and the expected size in MB and GB.

    Parameters:
    pixels (int): Total number of pixels (if known).
    width (int): Width of the image in pixels (optional).
    height (int): Height of the image in pixels (optional).
    width_m (float): Width of the image in meters (optional).
    height_m (float): Height of the image in meters (optional).
    pixel_size_m (float): Pixel size in meters (optional).
    datatype (str): Data type of the image ('uint8', 'uint16', 'float32', etc.). Default is 'uint8'.
    num_bands (int): Number of bands in the image. Default is 1.

    Returns:
    dict: A dictionary containing the number of pixels, size in MB, and size in GB.
    """
    if pixels is None:
        if width is not None and height is not None:
            pixels = width * height
        elif width_m is not None and height_m is not None and pixel_size_m is not None:
            width = int(width_m / pixel_size_m)
            height = int(height_m / pixel_size_m)
            pixels = width * height
        else:
            raise ValueError("Insufficient information to calculate the number of pixels")
        
    datatype_to_bytes = {
        'uint8': 1,
        'uint16': 2,
        'uint32': 4,
        'int8': 1,
        'int16': 2,
        'int32': 4,
        'float32': 4,
        'float64': 8,
    }
    
    if datatype not in datatype_to_bytes:
        raise ValueError(f"Unsupported datatype '{datatype}'. Supported types: {list(datatype_to_bytes.keys())}")

    bytes_per_pixel = datatype_to_bytes[datatype]
    total_bytes = pixels * bytes_per_pixel * num_bands
    size_mb = total_bytes / (1024**2)
    size_gb = total_bytes / (1024**3)
    
    return {
        "pixels": pixels,
        "size_mb": size_mb,
        "size_gb": size_gb
    }


import pandas as pd
import numpy as np

def rbf_interpolation(
    df, 
    value_col='NDVI',
    mode='1RBF',  # '1RBF', '2RBF', '3RBF'
    step_days=16,
    sigma1=16, win1=16, 
    sigma2=32, win2=32, 
    sigma3=64, win3=64, 
    bw1=4, bw2=8
):
    """
    RBF interpolation for pandas DataFrame with windowing and blending.
    User specifies the interpolation step in days.
    """
    df = df.dropna(subset=[value_col])
    times = df.index.values.astype('datetime64[ns]')
    values = df[value_col].values

    t_min, t_max = times.min(), times.max()
    target_times = pd.date_range(t_min, t_max, freq=f'{step_days}D')
    target_times_np = target_times.values.astype('datetime64[ns]')

    def rbf_interp_window(times, values, target_time, sigma, window):
        deltas = (times - target_time) / np.timedelta64(1, 'D')
        mask = np.abs(deltas) <= window
        if not np.any(mask):
            return np.nan, 0
        weights = np.exp(-0.5 * (deltas[mask] / sigma) ** 2)
        if np.all(weights == 0):
            return np.nan, 0
        return np.sum(values[mask] * weights) / np.sum(weights), np.count_nonzero(mask)

    results = []
    for t in target_times_np:
        if mode == '1RBF':
            interp, _ = rbf_interp_window(times, values, t, sigma1, win1)
        elif mode == '2RBF':
            interp1, count1 = rbf_interp_window(times, values, t, sigma1, win1)
            interp2, count2 = rbf_interp_window(times, values, t, sigma2, win2)
            w1 = min(count1 / bw1, 1.0)
            w2 = 1.0 - w1
            weighted = [interp1 * w1, interp2 * w2]
            if np.all(np.isnan(weighted)):
                interp = np.nan
            else:
                interp = np.nansum(weighted)
        elif mode == '3RBF':
            interp1, count1 = rbf_interp_window(times, values, t, sigma1, win1)
            interp2, count2 = rbf_interp_window(times, values, t, sigma2, win2)
            interp3, count3 = rbf_interp_window(times, values, t, sigma3, win3)
            w1 = min(count1 / bw1, 1.0)
            remainder1 = 1.0 - w1
            w2 = min(count2 / bw2, 1.0) * remainder1
            w3 = 1.0 - (w1 + w2)
            weighted = [interp1 * w1, interp2 * w2, interp3 * w3]
            if np.all(np.isnan(weighted)):
                interp = np.nan
            else:
                interp = np.nansum(weighted)
        else:
            raise ValueError("mode must be '1RBF', '2RBF', or '3RBF'")
        results.append(interp)
    return pd.DataFrame({'time': target_times, 'rbf_interp': results}).set_index('time')

# EOF

