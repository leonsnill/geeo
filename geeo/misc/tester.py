import ee
ee.Authenticate()
ee.Initialize(project='eexnill')
import eerepr
eerepr.initialize()
import geemap
import geeo
from geeo.utils import load_parameters, merge_parameters, load_blueprint


M = geemap.Map(center=[44, 7], zoom=7)
M.add_basemap('HYBRID')
vis_params = {'color': 'red', 'width': 2, 'lineType': 'solid','fillColor': '00000000'}
M

point = [6.315344, 44.569117] # forest/shrub grass mosaic
point = [6.717050, 44.459844]

prm = {
    'YEAR_MIN': 2014,
    'YEAR_MAX': 2024,
    'MONTH_MIN': 1,                                 # we include all months at first, the seasonal subwindows are defined in the folding section
    'MONTH_MAX': 12,
    'ROI': point,                              # our study area, note we could also supply the path to our .gpkg here
    'SENSORS': ['L8', 'L9'],                        # Landsat-8 and Landsat-9 Collection 2 Tier 1 surface reflectance data
    'MAX_CLOUD': 75,                                # only include scenes with less or equal 75% cloud cover
    'MASKS_LANDSAT': [                              # mask all pixels of this kind
        'cloud', 'cshadow', 
        'fill', 'dilated',
        ],
    'FEATURES': ['NDVI'],              # Tasseled Cap greenness, brightness, and wetness
    'FOLD_CUSTOM': {                                # Temporal folding allows to calculate habitat metrics for custom time windows, e.g. month ranges
        'month': ['3-5', '6-8', '9-11', '12-2'],    # instead, we want a custom range of months
        },
    'TSM': True,
    'NVO': False,                                    # Number of Valid Observations (NVO) is a quality metric that measures data availability for each pixel over time
    'NVO_FOLDING': False,                            # we want to calculate the quality layer for each of the custom time windows defined in the folding section
    'STM': ['p10', 'p50', 'p90', 'stdDev'],         # statistical metrics/reducers: 10th, 50th, 90th percentiles, and the standard deviation
    'STM_BASE_IMGCOL': 'TSS',                       # collection on which to calculate the STMs; here we use the basic Time-Series-Stack (TSS)
    'STM_FOLDING': False,                            # actually apply the custom folding settings when calculating the STMs
    'STM_FOLDING_LIST_ITER': False,
    'TSI': '2RBF',
    'WIN1': 25,
    'WIN2': 50

}

run = geeo.run_param(prm)


from geeo import plot_getRegion
import matplotlib.pyplot as plt

TSM = run['TSM']
TSI = run['TSI']
fig, ax = plt.subplots(figsize=(16, 4))
plot_getRegion(TSM, band='NDVI', roi=point, scale=30, axis=ax, style='x')
plot_getRegion(TSI, band='NDVI', roi=point, scale=30, axis=ax, style='.-', color='orangered')