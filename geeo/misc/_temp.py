import ee
ee.Initialize(project="eexnill")

import geeo

from geeo.level3.interpolation import plot_tsi_timeseries


prm = {'SENSORS': ['L8', 'L9'], 'ROI': [6.453384, 44.44164], 
       'YEAR_MIN': 2015, 'YEAR_MAX': 2024, 'MONTH_MIN': 1, 'MONTH_MAX': 12, 'MAX_CLOUD': 75
}

fig, ax = plot_tsi_timeseries(prm, feature='NDVI', show_masked=True)