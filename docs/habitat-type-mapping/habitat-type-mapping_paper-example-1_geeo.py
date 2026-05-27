import ee
ee.Authenticate()
ee.Initialize(project='project-name')
import geeo

# 'classical' parameter file approach:
# 1) create parameter file
geeo.create_parameter_file('habitat-metrics.prm')
# 2) edit parameter file with desired processing settings and save
# showing here a preview of the relevant settings in parameter YAML file:
# 3) run pipeline with parameter file as input
geeo.run_pipeline('habitat-metrics.prm')

# for illustrational purposes, we can also directly define the parameters in the script and run the pipeline without a parameter file
# settings
parameters_habitat_metrics = {
    # spatial and temporal filters
    'YEAR_MIN': 2022,                                 # year range
    'YEAR_MAX': 2024,
    'ROI': [5.9, 44.5, 6.9, 45.3],                    # region of interest
    # sensors and quality masking
    'SENSORS': ['L8', 'L9'],                          # sensors: Landsat-8 and Landsat-9 Collection 2 Tier 1 surface reflectance data
    'MAX_CLOUD': 75,                                  # maximum cloud cover percentage per image to be included in the analysis
    'MASKS_LANDSAT': [                                # mask all pixels based on Landsat QA band
        'cloud', 'cshadow', 
        'fill', 'dilated'
        ],
    'FEATURES': ['TCG', 'TCB', 'TCW'],                # Tasseled Cap greenness, brightness, and wetness
    # temporal folding
    'FOLD_CUSTOM': {                                  # calculate habitat metrics for custom time windows, e.g. month ranges
            'month': ['3-5', '6-8', '9-11', '12-2'],  # here we want a custom range of months (Mar-May, Jun-Aug, Sep-Nov, Dec-Feb)
        },  
    # habitat metrics settings
    'STM': ['p10', 'p50', 'p90', 'stdDev'],           # statistical metrics/reducers per-pixel: 10th, 50th, 90th percentiles, and the standard deviation
    'STM_BASE_IMGCOL': 'TSS',                         # collection on which to calculate the STMs; here we use the fundamental Time Series Stack (TSS)
    'STM_FOLDING': True,                              # switch to apply the folding settings from above to STM calculation, i.e. to calculate the STMs for each custom time window
    # export settings
    'PIX_RES': 30,                                    # pixel resolution of the output habitat metrics in meters
    'CRS': 'EPSG:3035',                               # coordinate reference system of the output habitat metrics
    'RESAMPLING_METHOD': 'bilinear',                  # resampling method to apply for continuous Landsat data
    'EXPORT_IMAGE': True,                             # switch to trigger export of any image data
    'EXPORT_STM': True,                               # switch to trigger export of the calculated habitat metrics (STMs)
    'EXPORT_DESC': 'HABITAT_METRICS'                  # description to add to the exported file name
}

# execute pipeline
run_pipeline = geeo.run_param(parameters_habitat_metrics)