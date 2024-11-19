from ..utils import LazyLoader

# Define lazy imports for submodules within level2
collection = LazyLoader('geeo.level2.collection')
indices = LazyLoader('geeo.level2.indices')
level2 = LazyLoader('geeo.level2.level2')
masking = LazyLoader('geeo.level2.masking')
scale = LazyLoader('geeo.level2.scale')
mosaic = LazyLoader('geeo.level2.mosaic')

# Define what should be available when using 'from geeo.level2 import *'
__all__ = [
    "collection",
    "indices",
    "level2",
    "masking",
    "scale",
    "mosaic"
]
