import ee
ee.Authenticate()
ee.Initialize(project="")

import geeo

prm = {'SENSORS': ['L9', 'L8'],
       'ROI': [1.362305, 42.455888, 11.535645, 47.783635],
        'YEAR_MIN': 2023, 'YEAR_MAX': 2024, 'MONTH_MIN': 1, 'MONTH_MAX': 12, 'MAX_CLOUD': 75,
        'RESAMPLING_METHOD': 'bilinear',
        'TSM': True,
        'FOLD_MONTH': False,
        'NVO': True, 'NVO_FOLDING': False,
        'TSI': '2RBF', 'INTERVAL': 12, 'SIGMA1': 20, 'WIN1': 20, 'SIGMA2': 40, 'WIN2': 60,
        'STM': ['min', 'p5', 'p10', 'p50', 'p90', 'p95', 'max', 'stdDev'], 'STM_BASE_IMGCOL': 'TSS',  'STM_FOLDING': False}

run = geeo.run_param(prm)

# map 
import geemap

img = run['NVO']
M = geemap.Map(center=[44, 7], zoom=7)
M.add_basemap('HYBRID')
M.addLayer(img, {'min': 0, 'max': 50, 'palette': 'gnuplot2'}, name='NVO')
M