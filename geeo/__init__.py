from .utils import LazyLoader

# global single import for earthengine + authenticate
import ee

# do not init and/or auth; let user do it; maybe in future find robust way to set project
'''
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()
'''
    
misc = LazyLoader('geeo.misc')
level2 = LazyLoader('geeo.level2')
level3 = LazyLoader('geeo.level3')
level4 = LazyLoader('geeo.level4')

__all__ = [
    "misc",
    "level2",
    "level3",
    "level4"
]

# main modules that should be accessible from top
import sys
from .utils import create_parameter_file, calculate_image_size, load_parameters, merge_parameters, load_blueprint
from .main import run_param
from geeo.level2.level2 import run_level2
from geeo.level3.level3 import run_level3
from geeo.misc.export import run_export

# extended functions outside main functionality
from geeo.misc.spacetime import vector_to_chunks, create_glance_tiles, create_tiles, getRegion, create_roi
from .misc.vis import VisMap, plot_rbf_interpolation, plot_getRegion
from geeo.level3.lsp import lsp
from geeo.misc.postprocess import process_ee_files
from geeo.level3.interpolation import tsi_rbf_array, tsi_rbf_tif, tsi_rbf_df

# set attributes for top-level access
setattr(sys.modules[__name__], 'create_parameter_file', create_parameter_file)
setattr(sys.modules[__name__], 'load_parameters', load_parameters)
setattr(sys.modules[__name__], 'merge_parameters', merge_parameters)
setattr(sys.modules[__name__], 'VisMap', VisMap)
setattr(sys.modules[__name__], 'plot_rbf_interpolation', plot_rbf_interpolation)
setattr(sys.modules[__name__], 'plot_getRegion', plot_getRegion)
setattr(sys.modules[__name__], 'run_param', run_param)
setattr(sys.modules[__name__], 'run_level2', run_level2)
setattr(sys.modules[__name__], 'run_level3', run_level3)
setattr(sys.modules[__name__], 'run_export', run_export)
setattr(sys.modules[__name__], 'run_lsp', lsp)
setattr(sys.modules[__name__], 'vector_to_chunks', vector_to_chunks)
setattr(sys.modules[__name__], 'process_ee_files', process_ee_files)
setattr(sys.modules[__name__], 'create_glance_tiles', create_glance_tiles)
setattr(sys.modules[__name__], 'create_tiles', create_tiles)
setattr(sys.modules[__name__], 'getRegion', getRegion)
setattr(sys.modules[__name__], 'calculate_image_size', calculate_image_size)
setattr(sys.modules[__name__], 'load_blueprint', load_blueprint)
setattr(sys.modules[__name__], 'tsi_rbf_array', tsi_rbf_array)
setattr(sys.modules[__name__], 'tsi_rbf_tif', tsi_rbf_tif)
setattr(sys.modules[__name__], 'tsi_rbf_df', tsi_rbf_df)
setattr(sys.modules[__name__], 'create_roi', create_roi)
