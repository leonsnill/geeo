import ee
ee.Authenticate()
ee.Initialize(project='project-name')  # replace with actual project name
import geeo

# There are two options to instruct the GEEO pipeline

# -----------------------------------------------------------------------------
# (A) using a parameter YAML file (intended use)

# 1) create a parameter file with default settings
geeo.create_parameter_file('example-1_habitat-metrics.yml')

# 2) edit parameter file with desired processing settings and save
# -> this is done outside of this script in an editor

# 3) run pipeline with parameter file as input
run_pipeline = geeo.run_param('example-1_habitat-metrics.yml')

# done, the workflow is documented and 
# the specified products are calculated and exported as requested 
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# (B) creating a python dictionary (experienced users and interactive use)

# 1) create a python dictionary containing the desired processing settings (unspecified settings will be set to default values)

parameters_habitat_metrics = {
    # spatial and time filters
    'YEAR_MIN': 2022,                       # start year
    'YEAR_MAX': 2024,                       # end year
    'ROI': [6.26, 44.35, 6.57, 44.58], # region of interest
    # sensors and masking
    'SENSORS': ['L8', 'L9'],                # Landsat-8 and -9
    'MAX_CLOUD': 75,                        # maximum image cloud cover (%)
    'MASKS_LANDSAT': [                      # Landsat QA masks
        'cloud', 'cshadow', 
        'fill', 'dilated'
        ],
    'FEATURES': ['TCG', 'TCB', 'TCW'],      # Tasseled Cap greenness, brightness, and wetness
    # temporal folds
    'FOLD_CUSTOM': {                        # custom time windows
            'month': ['3-5', 
                      '6-8', 
                      '9-11', 
                      '12-2'],
        },
    # valid observation count
    'NVO': True,                            # count valid observations per pixel
    'NVO_FOLDING': True,                    # apply custom folds to NVO
    # habitat metrics
    'STM': ['p10', 'p50', 'p90', 'stdDev'], # percentiles and standard deviation
    'STM_BASE_IMGCOL': 'TSS',               # derive metrics from the time series stack
    'STM_FOLDING': True,                    # apply custom folds to metrics
    # export settings
    'PIX_RES': 30,                          # output resolution (m)
    'CRS': 'EPSG:3035',                     # output coordinate system
    'RESAMPLING_METHOD': 'bilinear',        # resampling for continuous data
    'EXPORT_IMAGE': True,                   # export imagery
    'EXPORT_NVO': True,                     # export valid observation counts
    'EXPORT_STM': True,                     # export habitat metrics
    'EXPORT_DESC': 'HABITAT_METRICS'        # export filename description
}

# 2) run pipeline with parameter dictionary as input
run_pipeline = geeo.run_param(parameters_habitat_metrics)
# -----------------------------------------------------------------------------
