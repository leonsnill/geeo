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
    'STM': ['p50'],
    'CIC': "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
    'TSM': True,
    'TSM_BASE_IMGCOL': 'CIC',
    'EXPORT_IMAGE': True,
    'EXPORT_TSM': True,
    'RESAMPLING_METHOD': 'bicubic'
}
default_params = load_blueprint()
prm = merge_parameters(default_params, prm)


lvl2 = geeo.run_level2(prm)
TSS = lvl2['TSS']

CIC = lvl2.get('CIC')
TSM = lvl2.get('TSM')

lvl3 = geeo.run_level3(lvl2)
export = geeo.run_export(lvl3)