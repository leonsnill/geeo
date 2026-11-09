import ee
ee.Initialize(project="eexnill")

import geeo

from geeo.level3.interpolation import plot_tsi_timeseries


prm = {'SENSORS': ['L8', 'L9'], 
       'YEAR_MIN': 2015, 'YEAR_MAX': 2024, 'MONTH_MIN': 1, 'MONTH_MAX': 12, 'MAX_CLOUD': 75,
       'FEATURES': ['NDVI', 'NBR'],
       'TSI': '2RBF',
       'INTERVAL': 16,
       'INTERVAL_UNIT': 'day',  # day, month, year
       'INIT_JANUARY_1ST': True,  # Initialize interval on Jan 1st
       'SIGMA1': 16,  # Sigma1 of RBF function (in days)
       'SIGMA2': 32,  # Sigma2 of RBF function (in days)
       'SIGMA3': 64,  # Sigma3 of RBF function (in days)
       'WIN1': 32,  # Window1 for RBF (in days)
       'WIN2': 64,  # Window2 for RBF (in days)
       'WIN3': 64  # Window3 for RBF (in days)
}

roi = [6.453384, 44.44164]

fig, ax = plot_tsi_timeseries(prm, roi=[6.453384, 44.44164], feature='NDVI', show_masked=True, figsize=(6, 3),
                              color='orangered', density_cmap='gray_r', title='Alpine grassland',
                              ylim=[0.1, 1], 
                              savepath=r"C:\Users\leonx\Desktop\Fig-3_example-2_TSI_grassland.svg")

fig, ax = plot_tsi_timeseries(prm, roi=[6.367229, 44.603027], feature='NDVI', show_masked=True, figsize=(6, 3),
                              color='blue', density_cmap='gray_r', title='Forest-grassland ecotone',
                              ylim=[0.1, 1], 
                              savepath=r"C:\Users\leonx\Desktop\Fig-3_example-2_TSI_woodland.svg")

roi_fire_2023 = [5.630965, 44.356215]
fig, ax = plot_tsi_timeseries(prm, roi=roi_fire_2023, feature='NDVI', show_masked=True, figsize=(6, 3),
                              color='red', density_cmap='gray_r', title='Pine forest with fire disturbance',
                              ylim=[0.1, 1],
                              savepath=r"C:\Users\leonx\Desktop\Fig-3_example-2_TSI_fire_2023.svg")

roi_selective_logging = [5.840892, 44.397064]
fig, ax = plot_tsi_timeseries(prm, roi=roi_selective_logging, feature='NDVI', show_masked=True, figsize=(6, 3),
                              color='green', density_cmap='gray_r', title='Mixed forest with selective logging',
                              ylim=[0.1, 1],
                              savepath=r"C:\Users\leonx\Desktop\Fig-3_example-2_TSI_selective_logging.svg")


