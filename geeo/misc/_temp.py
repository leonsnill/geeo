import ee
ee.Initialize(project="eexnill")

import geeo



prm = {'SENSORS': ['L7', 'L8', 'L9'], 
       'YEAR_MIN': 2010, 'YEAR_MAX': 2024, 'MONTH_MIN': 1, 'MONTH_MAX': 12, 'MAX_CLOUD': 50,
       'FEATURES': ['NDVI', 'NBR'],
       'MASKS_LANDSAT': ['cloud', 'cshadow', 'fill', 'dilated'],
       'TSI': '1RBF',
       'INTERVAL': 16,
       'INTERVAL_UNIT': 'day',  # day, month, year
       'INIT_JANUARY_1ST': True,  # Initialize interval on Jan 1st
       'SIGMA1': 16,  # Sigma1 of RBF function (in days)
       'SIGMA2': 32,  # Sigma2 of RBF function (in days)
       'WIN1': 64,  # Window1 for RBF (in days)
       'WIN2': 64,  # Window2 for RBF (in days)
}

geeo.plot_tsi(prm, roi=[6.453384, 44.44164], feature='NDVI', show_masked=True, figsize=(9, 3),
              color='orangered', title='Alpine grassland', ylim=[-0.2, 1])

geeo.plot_tsi(prm, roi=[6.460557, 44.428797], feature='NDVI', show_masked=True, figsize=(9, 3),
              color='blue', title='Forest-grassland ecotone', ylim=[-0.2, 1])


roi_fire_2023 = [5.630965, 44.356215]
fig, ax = geeo.plot_tsi(prm, roi=roi_fire_2023, feature='NDVI', show_masked=True, figsize=(6, 3),
                              color='red', density_cmap='gray_r', title='Pine forest with fire disturbance',
                              ylim=[0.1, 1],
                              savepath=r"")

roi_selective_logging = [5.840892, 44.397064]
fig, ax = geeo.plot_tsi(prm, roi=roi_selective_logging, feature='NDVI', show_masked=True, figsize=(6, 3),
                              color='green', density_cmap='gray_r', title='Mixed forest with selective logging',
                              ylim=[0.1, 1],
                              savepath=r"")

geeo.run_param('../../../geeo/docs/tutorials/landsat-tsi-habitat.yml')

geeo.export_img(ee.Image(''), outname="",
                nodata=-9999, dtype="int16", px_res=60)