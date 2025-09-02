from ..utils import LazyLoader

# Define lazy imports for submodules within level2
export = LazyLoader('geeo.export.export')
formatting = LazyLoader('geeo.misc.formatting')
postprocess = LazyLoader('geeo.misc.postprocess')
spacetime = LazyLoader('geeo.misc.spacetime')
vis = LazyLoader('geeo.misc.vis')

# Define what should be available when using 'from geeo.level2 import *'
__all__ = [
    "export",
    "formatting",
    "postprocess",
    "spacetime",
    "vis"
]