from .utils import LazyLoader

# global single import for earthengine + authenticate
import ee
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Define lazy imports for submodules
misc = LazyLoader('geeo.misc')
level2 = LazyLoader('geeo.level2')
level3 = LazyLoader('geeo.level3')
level4 = LazyLoader('geeo.level4')

# Define what should be available when using 'from geeo import *'
__all__ = [
    "misc",
    "level2",
    "level3",
    "level4"
]

import sys
from .utils import create_parameter_file, calculate_image_size
from .misc.vis import VisMap
from .main import run_param
from geeo.level2.level2 import run_level2
from geeo.level3.level3 import run_level3
from geeo.misc.export import run_export
from geeo.misc.spacetime import vector_to_chunks, create_glance_tiles
from geeo.misc.postprocess import process_ee_files

# set attributes for top-level access
setattr(sys.modules[__name__], 'create_parameter_file', create_parameter_file)
setattr(sys.modules[__name__], 'VisMap', VisMap)
setattr(sys.modules[__name__], 'run_param', run_param)
setattr(sys.modules[__name__], 'run_level2', run_level2)
setattr(sys.modules[__name__], 'run_level3', run_level3)
setattr(sys.modules[__name__], 'run_export', run_export)
setattr(sys.modules[__name__], 'vector_to_chunks', vector_to_chunks)
setattr(sys.modules[__name__], 'process_ee_files', process_ee_files)
setattr(sys.modules[__name__], 'create_glance_tiles', create_glance_tiles)
setattr(sys.modules[__name__], 'calculate_image_size', calculate_image_size)
