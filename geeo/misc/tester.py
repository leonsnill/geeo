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
    'FOLD_MONTH': False,
    'FOLD_YEAR': True,
    'STM': ['p50'],
    'TSM': False,
    'TSM_BASE_IMGCOL': 'TSS',
    'EXPORT_IMAGE': True,
    'EXPORT_TSM': False,
    'RESAMPLING_METHOD': 'bicubic',
    'PBC': 'MAX-RNB',
    'PBC_FOLDING': True,
    'TSI': '1RBF',
    'LSP': 'POLAR',
    'NVO': True,
    'NVO_FOLDING': True,
    'EXPORT_NVO': True
}
default_params = load_blueprint()
prm = merge_parameters(default_params, prm)


lvl2 = geeo.run_level2(prm)
TSS = lvl2['TSS']

CIC = lvl2.get('CIC')
TSM = lvl2.get('TSM')

lvl3 = geeo.run_level3(lvl2)
export = geeo.run_export(lvl3)

PBC = lvl3.get('PBC')
#PBC

LSP = lvl3.get('LSP')

NVO = lvl3.get('NVO')