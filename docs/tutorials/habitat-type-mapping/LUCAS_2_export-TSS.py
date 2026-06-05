'''
Mapping pan-European Land Cover - Part 2
This file exports Landsat-based Time Series Stack (TSS) data for the European-wide LUCAS dataset
and uploads the final datasets as Earth Engine Asset.
'''

# ---------------------------------------------------------------------------------------------------
# import packages
import ee
ee.Authenticate()
ee.Initialize(project='eexnill')
import eerepr 
eerepr.initialize()
import geeo
from tqdm import tqdm
import math

# ---------------------------------------------------------------------------------------------------
lucas = ee.FeatureCollection('projects/eexnill/assets/geeo_public/LUCAS_HARMO_V1_EO_LC')


# ---------------------------------------------------------------------------------------------------
# LANDSAT TIME SERIES STACK (TSS)

# create 300km x 300km tiles covering Europe (GLANCE system)
eu_glance = geeo.create_glance_tiles(continent_code='EU', tile_size=300000, 
                                     vector_roi=None, 
                                     output_dir=None,  # only create in-memory
                                     zone_mask=False, 
                                     land_mask=True) 

from geeo import create_parameter_file, merge_parameters, load_parameters

# create parameter file from blueprint (only needs to be done once)
create_parameter_file('lucas-landsat-tss', overwrite=False)

# load parameter settings
prm = load_parameters('lucas-landsat-tss.yml')

# iterate over GLANCE tiles and export LUCAS points within each tile
for i in tqdm(range(len(eu_glance))):
    TILE = eu_glance.iloc[[i]]
    tile_id = TILE['ID300'].values[0]
    roi_dict = geeo.misc.spacetime.create_roi(TILE, simplify_geom_to_bbox=True)  # converts geometry to server-side geometry (among other stuff)
    roi = roi_dict['roi_geom']  # server-side ee.Geometry used to filter ee.FeatureCollections for export

    # filter LUCAS and EUNIS
    lucas_roi = lucas.filterBounds(roi)

    # create output filenames
    outname_lucas = 'LUCAS_HARMO_V1_EO_LC' + '_' + tile_id

    prm_lucas = {
        'ROI': lucas_roi,
        'EXPORT_DESC': outname_lucas
    }

    if lucas_roi.size().getInfo() > 0:  # some tiles may not contain any points
        geeo.run_param(merge_parameters(prm, prm_lucas))


# ---------------------------------------------------------------------------------------------------
# ADDITIONAL LAYERS: CHELSA BIOCLIM + COPERNICUS DEM

# create 600km x 600km tiles covering Europe (GLANCE system); larger tiles because requires less processing power than TSS
eu_glance = geeo.create_glance_tiles(continent_code='EU', tile_size=600000, 
                                     vector_roi=None, 
                                     output_dir=None,  # only create in-memory
                                     zone_mask=False, 
                                     land_mask=True) 

# chelsa bioclim
chelsa = ee.Image('projects/eeleeon/assets/CHELSA/CHELSA_BIOCLIM_1981-2010_V21')

# iterate over GLANCE tiles and export LUCAS points within each tile
for i in tqdm(range(len(eu_glance))):
    TILE = eu_glance.iloc[[i]]
    tile_id = TILE['ID600'].values[0]
    roi_dict = geeo.misc.spacetime.create_roi(TILE, simplify_geom_to_bbox=True)  # converts geometry to server-side geometry (among other stuff)
    roi = roi_dict['roi_geom']  # server-side ee.Geometry used to filter ee.FeatureCollections for export

    dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').filterBounds(roi).select('DEM').map(lambda x: x.resample('bilinear'))
    dem = ee.Image(dem.mosaic().set('system:time_start', dem.first().get('system:time_start'))).setDefaultProjection(ee.Projection('EPSG:4326'), scale=30)
    dem = dem.addBands(ee.Terrain.slope(dem).rename('slope'))  # calculate slope
    aspect = ee.Terrain.aspect(dem).rename('aspect')  # calculate aspect
    # calculate eastness and northness
    aspect_rad = aspect.multiply(math.pi/180.0)
    aspect_sin = aspect_rad.sin().rename('eastness')
    aspect_cos = aspect_rad.cos().rename('northness')
    # DEM features
    dem = dem.addBands(aspect_sin).addBands(aspect_cos)
    # all additional features
    img = dem.addBands(chelsa)
    
    # run export using auxiliary function provided by GEEO
    my_export = geeo.export_table(
        img, feature=lucas.filterBounds(roi), px_res=30, nodata=-9999, drop_nodata=True,
        crs='EPSG:4326', tileScale=16,
        features=['bio1', 'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 'bio7', 'bio8',
                  'bio9', 'bio10', 'bio11', 'bio12', 'bio13', 'bio14', 'bio15',
                  'bio16', 'bio17', 'bio18', 'bio19', 'DEM', 'slope', 'eastness', 'northness'],
        outname='LUCAS_HARMO_V1_EO_LC' + '_CHELSA_DEM' + '_' + tile_id,
        out_location='Asset',
        out_dir='projects/eexnill/assets/geeo_public/landsat-tss'
    )
    
# EOF
