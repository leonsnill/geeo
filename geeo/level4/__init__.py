from ..utils import LazyLoader

# define lazy imports for submodules within level2
level4 = LazyLoader('geeo.level4.level4')
model = LazyLoader('geeo.level4.model')

# Define what should be available when using 'from geeo.level2 import *'
__all__ = [
    "level4",
    "model"
]