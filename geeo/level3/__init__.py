from ..utils import LazyLoader

# Define lazy imports for submodules within level2
composite = LazyLoader('geeo.level3.composite')
initimgcol = LazyLoader('geeo.level3.initimgcol')
interpolation = LazyLoader('geeo.level3.interpolation')
level3 = LazyLoader('geeo.level3.level3')
stm = LazyLoader('geeo.level3.stm')
lsp = LazyLoader('geeo.level3.lsp')

# Define what should be available when using 'from geeo.level2 import *'
__all__ = [
    "composite",
    "initimgcol",
    "interpolation",
    "level3",
    "stm",
    "lsp"
]