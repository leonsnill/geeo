import ee
ee.Authenticate()
ee.Initialize(project='eexnill')  # replace with actual project name
import geeo

# There are two options to instruct the GEEO pipeline

# -----------------------------------------------------------------------------
# (A) using a parameter YAML file (intended use)

# 1) create a parameter file with default settings
prm = 'example-2_time-series-interpolation.yml'
geeo.create_parameter_file(prm)

# 2) edit parameter file with desired processing settings and save
# -> this is done outside of this script in an editor

# create plot for two selected sites to assess the interpolation results and data availability
geeo.plot_tsi(prm, roi=[6.453384, 44.44164], feature='NDVI', show_masked=True, 
              color='orangered', title='Alpine grassland', ylim=[-0.2, 1], figsize=(9, 3), savepath=r"C:\Users\leonx\Desktop\example-2_time-series-interpolation_alpine-grassland.svg")

geeo.plot_tsi(prm, roi=[6.460557, 44.428797], 
              feature='NDVI', show_masked=True, color='blue', title='Forest-grassland ecotone', 
              ylim=[-0.2, 1], figsize=(9, 3), savepath=r"C:\Users\leonx\Desktop\example-2_time-series-interpolation_forest-grassland-ecotone.svg")

# 3) run pipeline with parameter file as input
run_pipeline = geeo.run_param(prm)

# done, the workflow is documented and 
# the specified products are calculated and exported as requested 
# -----------------------------------------------------------------------------
