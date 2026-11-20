import ee
import geopandas as gpd
from shapely.geometry import Polygon

ee.Authenticate()
ee.Initialize(project='project-name')  # replace with actual project name

# -----------------------------------------------------------------------------
# input

# reference points (LUCAS harmonized land cover dataset)
roi_asset = 'projects/eexnill/assets/geeo_public/LUCAS_HARMO_V1_EO_LC'
reference_points = ee.FeatureCollection(roi_asset)

# time range
year_min, year_max = 2014, 2019

# export settings
export_desc = 'LUCAS_HARMO_V1_EO_LC'
export_scale_m = 30

# GLANCE Europe (Lambert Azimuthal Equal Area) projection
export_crs = (
    'PROJCS["BU MEaSUREs Lambert Azimuthal Equal Area - EU - V01",GEOGCS["GCS_WGS_1984",'
    'DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
    'UNIT["degree",0.0174532925199433]],PROJECTION["Lambert_Azimuthal_Equal_Area"],'
    'PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],'
    'PARAMETER["longitude_of_center",20],PARAMETER["latitude_of_center",55],UNIT["meter",1.0]]'
)

# features to extract from the time series stack
feature_names = ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2', 'NDVI', 'MDWI', 'NBR']

# -----------------------------------------------------------------------------
# 300km tiling construction for Europe (GLANCE grid)

def create_glance_eu_300km_grid():
    """
    Constructs the 300km European tiling grid based on the GLANCE definition.
    Returns a GeoDataFrame with tile geometries and identifiers (ID300).
    """
    ul_x, ul_y = -5505560.0, 3346245.0
    tile_size = 300000  # 300 km in meters
    n_rows, n_cols = 19, 27

    grid_list = []
    id_list = []

    for row in range(n_rows):
        for col in range(n_cols):
            x_min = ul_x + col * tile_size
            y_max = ul_y - row * tile_size
            x_max = x_min + tile_size
            y_min = y_max - tile_size

            poly = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
            grid_list.append(poly)
            id_list.append(f"EU_300-X{col:03d}-Y{row:03d}")

    return gpd.GeoDataFrame({'ID300': id_list, 'geometry': grid_list}, crs=export_crs)

grid = create_glance_eu_300km_grid()

# subset to specific tile(s) or process all tiles (set to None to process all tiles)
target_tiles = None #['EU_300-X018-Y005']

if target_tiles is not None:
    grid = grid[grid['ID300'].isin(target_tiles)]

# -----------------------------------------------------------------------------
# Landsat loading and preprocessing functions

# band names differ between Landsat 7 and Landsat 8/9
bands_l7 = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL', 'QA_RADSAT']
bands_l89 = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL', 'QA_RADSAT']
bands_new = ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2', 'QA_PIXEL', 'QA_RADSAT']

def load_landsat(collection_id, bands, region):
    return (
        ee.ImageCollection(collection_id)
        .filterBounds(region)
        .filter(ee.Filter.calendarRange(year_min, year_max, 'year'))
        .filter(ee.Filter.lte('CLOUD_COVER_LAND', 75))
        .filter(ee.Filter.gte('GROUND_CONTROL_POINTS_MODEL', 1))
        .select(bands, bands_new)
    )

def preprocess_image(image):
    # scale to surface reflectance values
    optical = (
        image.select(['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2'])
        .multiply(0.0000275)
        .add(-0.2)
    )
    image = image.addBands(optical, overwrite=True)

    # QA_PIXEL bits: 0 fill, 1 dilated cloud, 3 cloud, 4 cloud shadow
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 + 2 + 8 + 16).eq(0)

    # cloud and cirrus confidence (bits 8-9 and 14-15) below medium
    cloud_conf = qa.rightShift(8).bitwiseAnd(3).lt(2)
    cirrus_conf = qa.rightShift(14).bitwiseAnd(3).lt(2)

    # radiometric saturation and implausible reflectance values
    saturation = image.select('QA_RADSAT').eq(0)
    valid_min = optical.reduce(ee.Reducer.min()).gt(0)
    valid_max = optical.reduce(ee.Reducer.max()).lt(1)

    mask = (
        mask
        .And(cloud_conf)
        .And(cirrus_conf)
        .And(saturation)
        .And(valid_min)
        .And(valid_max)
    )

    # calculate indices
    ndvi = image.normalizedDifference(['NIR', 'RED']).rename('NDVI')
    mdwi = image.normalizedDifference(['GRN', 'SW1']).rename('MDWI')
    nbr = image.normalizedDifference(['NIR', 'SW2']).rename('NBR')

    # combine bands, apply quality mask, and scale to int16 format
    features = (
        image.select(['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2'])
        .addBands([ndvi, mdwi, nbr])
        .updateMask(mask)
        .multiply(10000)  # scale continuous values to preserve decimals in integer format
        .toInt16()        # cast to disc space efficient int16 format
    )

    return features.copyProperties(image, image.propertyNames())

# -----------------------------------------------------------------------------
# iterate over tiles, extract point time series, and export tables

for _, tile in grid.iterrows():
    tile_id = tile['ID300']
    tile_poly = tile['geometry']
    print(f"Processing tile {tile_id}...")

    # construct server-side geometry (convert to EPSG:4326 coordinates)
    tile_gdf_4326 = gpd.GeoDataFrame(geometry=[tile_poly], crs=export_crs).to_crs('EPSG:4326')
    tile_coords = [list(coord) for coord in tile_gdf_4326.geometry.iloc[0].exterior.coords]
    tile_geom = ee.Geometry.Polygon(tile_coords)

    # filter reference points to current tile
    tile_points = reference_points.filterBounds(tile_geom)
    point_count = tile_points.size().getInfo()
    print(f"  Found {point_count} reference points in tile {tile_id}.")
    if point_count == 0:
        continue

    # load and preprocess Landsat 7, 8, 9 data for the tile
    landsat = (
        load_landsat('LANDSAT/LC09/C02/T1_L2', bands_l89, tile_geom)
        .merge(load_landsat('LANDSAT/LC08/C02/T1_L2', bands_l89, tile_geom))
        .merge(load_landsat('LANDSAT/LE07/C02/T1_L2', bands_l7, tile_geom))
    )
    tss = landsat.map(preprocess_image).sort('system:time_start')

    # limit feature collection to memory size to avoid GEE reduceRegions bug
    tile_points_limited = tile_points.limit(point_count)

    def extract_point_observations(image):
        values = image.reduceRegions(
            collection=tile_points_limited,
            reducer=ee.Reducer.first(),
            scale=export_scale_m,
            tileScale=1,  # maybe needs adjustement for large tiles with many points
        )
        date_str = image.date().format('YYYYMMdd')
        return values.map(lambda feat: feat.set('YYYYMMDD', date_str))

    # reduce across time series stack, flatten, and drop unobserved / masked rows
    tile_table = (
        tss.map(extract_point_observations)
        .flatten()
        .filter(ee.Filter.notNull(feature_names))
    )

    # export table as CSV to Google Drive
    export_description = f"TSS_{export_desc}_{tile_id}"
    task = ee.batch.Export.table.toDrive(
        collection=tile_table,
        description=export_description,
        fileNamePrefix=export_description,
        fileFormat='CSV',
    )
    task.start()
    print(f"  Export task started: {export_description}")

# eof
