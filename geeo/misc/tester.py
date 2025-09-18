import ee
ee.Authenticate()
ee.Initialize(project='eexnill')
import eerepr
eerepr.initialize()

import geeo
from geeo.utils import load_parameters, merge_parameters, load_blueprint

glance_eu = geeo.create_glance_tiles(continent_code='EU', tile_size=150000, land_mask=True)

prm = {
    'YEAR_MIN': 2020,
    'YEAR_MAX': 2024,
    'ROI': glance_eu, #[8.026852416992174,46.09966992876708,8.280911254882799,46.25039470654484],
    'FOLD_MONTH': False,
    'FOLD_YEAR': True,
    'STM': ['p50'],
    'TSM': False,
    'TSM_BASE_IMGCOL': 'TSS',
    'EXPORT_IMAGE': False,
    'EXPORT_TSM': False,
    'RESAMPLING_METHOD': 'bilinear',
    'PBC': 'MAX-RNB',
    'PBC_FOLDING': True,
    'TSI': '2RBF',
    'WIN2': 64,
    'LSP': 'POLAR',
    'LSP_ADJUST_SEASONAL': False,
    'EXPORT_LSP': True,
    'NVO': True,
    'NVO_FOLDING': True,
    'EXPORT_NVO': False,

    #'CRS': None
}

# init
default_params = load_blueprint()
prm = merge_parameters(default_params, prm)

# lvl2
lvl2 = geeo.run_level2(prm)

TSS = lvl2['TSS']
CIC = lvl2.get('CIC')
TSM = lvl2.get('TSM')

# lvl3
lvl3 = geeo.run_level3(lvl2)
PBC = lvl3.get('PBC')
LSP = lvl3.get('LSP')
NVO = lvl3.get('NVO')

# export
export = geeo.run_export(lvl3)


# ---------------------------------------------------------------------------------------------------------------

import ee
ee.Initialize()
from geeo.level3.interpolation import rbf_interpolation_df, rbf_time_series_array, rbf_time_series_tif_gdal

inp_file = ''

test = rbf_time_series_tif_gdal(
    src_path=inp_file,
    step_days=12,
    mode='2RBF',
    sigma1=16, win1=16,
    sigma2=32, win2=48,
    sigma3=64, win3=64,
    bw1=4, bw2=8,
    n_cores=1,
    chunk_x=None, chunk_y=None,
    creation_options=None
)
