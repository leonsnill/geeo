import ee
ee.Authenticate()
ee.Initialize(project='eexnill')  # replace with actual project name
import geeo

# There are two options to instruct the GEEO pipeline
# -----------------------------------------------------------------------------
# (A) using a parameter YAML file (intended use)

# 1) create a parameter file with default settings
prm = 'example-3_reference-data-to-table_single_tile.yml'
geeo.create_parameter_file(prm)

# 2) edit parameter file with desired processing settings and save
# -> this is done outside of this script in an editor

# 3) run pipeline with parameter file as input
run_pipeline = geeo.run_param(prm)

# done, the workflow is documented and 
# the specified products are calculated and exported as requested 
# -----------------------------------------------------------------------------
