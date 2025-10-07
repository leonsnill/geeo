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
    Copies the bluprint YAML parameter file to the specified destination.
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
    # blueprint path
    blueprint_path = os.path.join(os.path.dirname(__file__), 'config', 'parameter_blueprint.yml')
    
    # check if the blueprint file existss
    if not os.path.exists(blueprint_path):
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    if not destination:
        dest_filename = os.path.join(os.getcwd(), "parameter.yml")
    elif os.path.isdir(destination):
        dest_filename = os.path.join(destination, "parameter.yml")
    else:
        dest_filename = destination
        if not dest_filename.endswith(".yml"):
            dest_filename += ".yml"
    
    if os.path.exists(dest_filename) and not overwrite:
        print(f"Warning: File already exists: {dest_filename}. Set overwrite=True if you wish to replace it.")
        return
    
    dest_dir = os.path.dirname(dest_filename)
    if dest_dir:  # Check if dest_dir is not empty
        os.makedirs(dest_dir, exist_ok=True)

    shutil.copy(blueprint_path, dest_filename)
    print(f"Parameter file created: {dest_filename}")


# =============================================================================
# load parameter file
import yaml
import requests

def load_parameters(yaml_file=None):
    """
    load parameters from a YAML file (local or URL) or use the blueprint if no file is provided.
    
    Returns:
        dict: Dictionary of parameters.
    """
    if yaml_file:
        if isinstance(yaml_file, str) and yaml_file.startswith(('http://', 'https://')):
            response = requests.get(yaml_file)
            response.raise_for_status()
            parameters = yaml.safe_load(response.text)
        else:
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


# EOF

