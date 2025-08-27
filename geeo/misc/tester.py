import ee
ee.Authenticate()
ee.Initialize()
import eerepr
eerepr.initialize()

import geeo
from geeo.utils import load_parameters, merge_parameters, load_blueprint

prm = {
    'YEAR_MIN': 2020,
    'YEAR_MAX': 2023,
    'FOLD_MONTH': True,
    'FOLD_YEAR': True,
    'STM': ['p50']
}
default_params = load_blueprint()
prm = merge_parameters(default_params, prm)


lvl2 = geeo.run_level2(prm)
TSS = lvl2['TSS']