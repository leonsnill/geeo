import ee
ee.Initialize()
import os
from datetime import datetime, timedelta
from geopandas import gpd
from shapely.geometry import Point, Polygon, box
import numpy as np
import json


# --------------------------------------------------------------------------------------------
#                                           Reducers                               
# --------------------------------------------------------------------------------------------
# Define common reducers
dict_reducers = {
    'mean': ee.Reducer.mean(),
    'median': ee.Reducer.median(),
    'sum': ee.Reducer.sum(),
    'min': ee.Reducer.min(),
    'max': ee.Reducer.max(),
    'stdDev': ee.Reducer.stdDev(),
    'variance': ee.Reducer.variance(),
    'p5': ee.Reducer.percentile([5]),
    'p10': ee.Reducer.percentile([10]),
    'p15': ee.Reducer.percentile([15]),
    'p20': ee.Reducer.percentile([20]),
    'p25': ee.Reducer.percentile([25]),
    'p30': ee.Reducer.percentile([30]),
    'p35': ee.Reducer.percentile([35]),
    'p40': ee.Reducer.percentile([40]),
    'p45': ee.Reducer.percentile([45]),
    'p50': ee.Reducer.percentile([50]),
    'p55': ee.Reducer.percentile([55]),
    'p60': ee.Reducer.percentile([60]),
    'p65': ee.Reducer.percentile([65]),
    'p70': ee.Reducer.percentile([70]),
    'p75': ee.Reducer.percentile([75]),
    'p80': ee.Reducer.percentile([80]),
    'p85': ee.Reducer.percentile([85]),
    'p90': ee.Reducer.percentile([90]),
    'p95': ee.Reducer.percentile([95]),
    'skew': ee.Reducer.skew(),
    'kurtosis': ee.Reducer.kurtosis(),
    'count': ee.Reducer.count(),
    'first': ee.Reducer.first(),
    'last': ee.Reducer.last()
}

# Function to combine reducers from list
def combine_reducers(reducers_list):
    combined_reducer = dict_reducers[reducers_list[0]]
    for reducer_name in reducers_list[1:]:
        if reducer_name in dict_reducers:
            combined_reducer = combined_reducer.combine(dict_reducers[reducer_name], sharedInputs=True)
        else:
            raise ValueError(f"Reducer '{reducer_name}' is not in the reducers dictionary")
    return combined_reducer


# clear-sky-observations (CSOs)
def cso(imgcol, band=None):
    if band:
        img_cso = imgcol.select(band).count()
    else:
        img_cso = imgcol.select(0).count()
    return img_cso.rename('CSO')


# --------------------------------------------------------------------------------------------
#                                 General Spatial Functions                               
# --------------------------------------------------------------------------------------------

def input_to_gdf(input_data):
    if isinstance(input_data, list):
        # Handle single point
        if len(input_data) == 2 and all(isinstance(coord, (int, float)) for coord in input_data):
            geom = Point(input_data)
            gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        
        # Handle rectangle (bounding box)
        elif len(input_data) == 4 and all(isinstance(coord, (int, float)) for coord in input_data):
            geom = box(input_data[0], input_data[1], input_data[2], input_data[3])
            gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        
        # Handle polygon
        elif all(isinstance(point, list) and len(point) == 2 for point in input_data):
            geom = Polygon(input_data)
            gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        else:
            raise ValueError("Invalid list format for input data")
    
    elif isinstance(input_data, str):
        # Handle file input (e.g., shapefile, geopackage, geojson)
        if os.path.exists(input_data):
            gdf = gpd.read_file(input_data)
        else:
            raise FileNotFoundError(f"The file '{input_data}' does not exist.")
    
    elif isinstance(input_data, gpd.GeoDataFrame):
        # Handle already a GeoDataFrame
        gdf = input_data
    
    else:
        raise ValueError("Unsupported input data format")
    
    return gdf



def create_roi(roi_input, simplify_geom_to_bbox=False):
    """
    Handles various types of Region of Interest (ROI) inputs and returns
    the corresponding polygon geometry and feature collection.

    Parameters:
    roi_input: Can be one of the following:
               - List of coordinates in EPSG:4326
                 Length of list determines type of geometry:
                 2 = Point (will create a small buffer to form a polygon)
                 4 = Rectangle (creates a polygon)
                 > 4 Polygon (must be closed, i.e. last coordinate = first coordinate)
               - File path to a vector file (str)
               - A geopandas GeoDataFrame (gpd.geodataframe.GeoDataFrame)
               - ee.Geometry (Earth Engine geometry)
               - ee.FeatureCollection (Earth Engine feature collection)

    Returns:
    roi_geom: ee.Geometry.Polygon
    roi_featcol: ee.FeatureCollection
    """

    # simplify_geom_to_bbox:
    # "Caution: providing a large or complex collection as the geometry argument can result in poor performance. 
    # Collating the geometry of collections does not scale well; use the smallest collection (or geometry) that is required to achieve the desired outcome."

    if isinstance(roi_input, list):  # WGS84
        roi_geom = create_polygon_geometry(roi_input)
        roi_featcol = ee.FeatureCollection([ee.Feature(roi_geom)])
        roi_bbox = bounds_from_featcol(roi_featcol)
        if simplify_geom_to_bbox:
            roi_geom = roi_bbox
        roi_bbox_gdf = bbox_server_to_client(roi_bbox)
        roi_bbox_gdf = gpd.GeoDataFrame(geometry=[box(roi_bbox_gdf[0], roi_bbox_gdf[1], roi_bbox_gdf[2], roi_bbox_gdf[3])], crs="EPSG:4326")
    
    elif isinstance(roi_input, str) or isinstance(roi_input, gpd.geodataframe.GeoDataFrame):
        if isinstance(roi_input, str):
            gdf = gpd.read_file(roi_input)
        elif isinstance(roi_input, gpd.geodataframe.GeoDataFrame):
            gdf = roi_input
        try:
            roi_bbox_gdf = [float(gdf.geometry.bounds.minx.iloc[0]), float(gdf.geometry.bounds.miny.iloc[0]), 
                               float(gdf.geometry.bounds.maxx.iloc[0]), float(gdf.geometry.bounds.maxy.iloc[0])]
            roi_bbox_gdf = gpd.GeoDataFrame(geometry=[box(roi_bbox_gdf[0], roi_bbox_gdf[1], roi_bbox_gdf[2], roi_bbox_gdf[3])], crs=gdf.crs)
            roi_featcol = init_featcol_from_vector(gdf)
            roi_bbox = bounds_from_featcol(roi_featcol)
            if simplify_geom_to_bbox:
                roi_geom = roi_bbox
            else:
                roi_geom = ee.Geometry(roi_featcol.geometry())
        except Exception as e:
            raise ValueError(f"Vector file could not be initialized to FeatureCollection: {e}")
    
    elif isinstance(roi_input, ee.Geometry):  # WGS84
        roi_geom = roi_input
        roi_featcol = ee.FeatureCollection([ee.Feature(roi_geom)])
        roi_bbox = bounds_from_featcol(roi_featcol)
        if simplify_geom_to_bbox:
            roi_geom = roi_bbox
        roi_bbox_gdf = bbox_server_to_client(roi_bbox)
        roi_bbox_gdf = gpd.GeoDataFrame(geometry=[box(roi_bbox_gdf[0], roi_bbox_gdf[1], roi_bbox_gdf[2], roi_bbox_gdf[3])], crs="EPSG:4326")
    
    elif isinstance(roi_input, ee.featurecollection.FeatureCollection):  # WGS84
        roi_featcol = roi_input
        roi_bbox = bounds_from_featcol(roi_featcol)
        roi_bbox_gdf = bbox_server_to_client(roi_bbox)
        # complex feature collection with many features can create overhead
        if simplify_geom_to_bbox:
            roi_geom = roi_bbox
        else:
            roi_geom = ee.Geometry(roi_featcol.geometry())
        roi_bbox_gdf = gpd.GeoDataFrame(geometry=[box(roi_bbox_gdf[0], roi_bbox_gdf[1], roi_bbox_gdf[2], roi_bbox_gdf[3])], crs="EPSG:4326")
    
    else:
        raise ValueError("ROI incorrectly specified. Input must be a list of coordinates, a file path, ee.Geometry, or ee.FeatureCollection.")
    
    dict_roi = {
        'roi_geom': roi_geom,
        'roi_featcol': roi_featcol,
        'roi_bbox': roi_bbox,
        'roi_bbox_gdf': roi_bbox_gdf
    }

    return dict_roi


def bbox_server_to_client(geometry):
    bounds = ee.Array(ee.List(geometry.bounds().coordinates()).get(0))
    min_coords = bounds.reduce(ee.Reducer.min(), [0]).project([1]).toList()
    max_coords = bounds.reduce(ee.Reducer.max(), [0]).project([1]).toList()
    x_min = min_coords.get(0)
    y_min = min_coords.get(1)
    x_max = max_coords.get(0)
    y_max = max_coords.get(1)
    return [x_min.getInfo(), y_min.getInfo(), x_max.getInfo(), y_max.getInfo()]


def create_polygon_geometry(coords):
    """
    Create a polygon geometry from a list of coordinates.

    Parameters:
    coords: List of coordinates in EPSG:4326
            Length of list determines type of geometry:
            2 = Point (creates a small buffer polygon)
            4 = Rectangle
            > 4 Polygon (must be closed, i.e. last coordinate = first coordinate)

    Returns:
    roi: ee.Geometry.Point or ee.Geometry.Polygon
    """
    length = len(coords)
    if length == 2:
        roi = ee.Geometry.Point(coords)
    elif length == 4 and not isinstance(coords[0], list):
        xmin, ymin, xmax, ymax = coords
        roi = ee.Geometry.Polygon([
            [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]
        ])
    elif length > 3 and isinstance(coords[0], list):
        roi = ee.Geometry.Polygon(coords)
    else:
        raise ValueError("Incorrectly specified")
    return roi


def init_featcol_from_vector(f):
    """
    Initialize a FeatureCollection from a vector file.

    Parameters:
    f: File path to the vector file or a GeoDataFrame.

    Returns:
    featcol: ee.FeatureCollection
    """
    if isinstance(f, str):
        gdf = gpd.read_file(f)
    elif isinstance(f, gpd.geodataframe.GeoDataFrame):
        gdf = f

    else:
        raise ValueError("Input must be a file path or GeoDataFrame.")
    # make sure the crs is 4326
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    geo_json = gdf.to_json()
    featcol = ee.FeatureCollection(json.loads(geo_json))
    return featcol


def bounds_from_featcol(featcol):
    """
    Get the bounds of a FeatureCollection as a polygon geometry.

    Parameters:
    featcol: ee.FeatureCollection

    Returns:
    roi: ee.Geometry.Polygon
    """
    bounds = featcol.geometry().bounds()
    roi = ee.Geometry(bounds)
    return roi


def bounds_from_img(img):
    """
    Get the bounds of a FeatureCollection as a polygon geometry.

    Parameters:
    featcol: ee.FeatureCollection

    Returns:
    roi: ee.Geometry.Polygon
    """
    bounds = img.geometry().bounds()
    roi = ee.Geometry(bounds)
    return roi


def geom_to_bounds(geometry):
    """
    Convert an Earth Engine geometry to a list of bounds.
    
    Args:
        geometry (ee.Geometry): The Earth Engine geometry.
        
    Returns:
        list: List of bounds [lon_min, lat_min, lon_max, lat_max].
    """
    bounds = geometry.bounds().getInfo()['coordinates'][0]
    lon_min = bounds[0][0]
    lat_min = bounds[0][1]
    lon_max = bounds[2][0]
    lat_max = bounds[2][1]
    return [lon_min, lat_min, lon_max, lat_max]


def find_utm(lon):
    try:
        coords = lon.getInfo().get('coordinates')
        geom_type = lon.getInfo().get('type')
        if geom_type == 'Point':
            lon = coords[1]
        elif geom_type == 'Polygon':
            # convert to bounds
            bounds = geom_to_bounds(lon)
            xmin = bounds[0]
            xmax = bounds[2]
            lon = xmin+(abs(xmin-xmax)/2)
        else:
            xmin = coords[0][0][0]
            xmax = coords[0][1][0]
            lon = xmin+(abs(xmin-xmax)/2)
    except Exception:
        raise ValueError("Could not extract coordinates from geometry.")
    utm = round((lon + 180) / 6)
    utm = "{0:0=2d}".format(utm)
    return "EPSG:326"+utm

# convert ImageCollection to Image
def imgcol_to_img(imgcol, date_to_bandname=True):
    img = ee.Image(imgcol.toBands())
    if date_to_bandname:
        bandnames = get_dates_with_bands_str(imgcol)
        img = img.rename(bandnames)
    return img

# reduceRegionS: extract values from image time series (ImageCollection) at point locations
# requires 'limiting' featcol into memory due to bug in EE, as descibed by N. Gorelick
# https://gis.stackexchange.com/questions/478174/reduceregion-works-but-reduceregions-returns-null-values-for-all-reducer
def reduce_regions_imgcol(featcol, scale=30, reducer=ee.Reducer.first(), tileScale=1, **kwargs):
    def wrap(img):
        values = img.reduceRegions(
            collection=featcol.limit(featcol.size()),  # limit
            reducer=reducer,
            scale=scale,
            tileScale=tileScale, 
            **kwargs
            ).map(
                lambda x: x.set({'YYYYMMDD': img.date().format('YYYYMMdd')})
        )
        return values
    return wrap

# reduceRegionS: extract values from Image at point locations (no map)
def reduce_regions_img(img, featcol, scale=30, reducer=ee.Reducer.first(), tileScale=1, **kwargs):
    values = img.reduceRegions(
        collection=featcol.limit(featcol.size()),  # limit, see above
        reducer=reducer, 
        scale=scale,
        tileScale=tileScale,  
        **kwargs
    ).map(
        lambda x: x.set({'YYYYMMDD': img.date().format('YYYYMMdd')})  # requires system:time_start property!
    )
    return values


# reduceRegion ImgCollection
def reduce_region_imgcol(imgcol, scale=30, reducer=ee.Reducer.first(), tileScale=1, **kwargs):
    def wrap(feature):
        return imgcol.map(
            lambda img: ee.Feature(
                feature.geometry(), img.reduceRegion(
                    reducer=reducer, 
                    scale=scale,
                    tileScale=tileScale,   
                    geometry=feature.geometry(),
                    **kwargs)
                ).set({'YYYYMMDD': img.date().format('YYYYMMdd')}).copyProperties(feature)
        )
    return wrap
# consider: setOutputs(image.bandNames())
# https://stackoverflow.com/questions/55208625/missing-ndvi-values-when-reducing-landsat-image-collection-to-long-format-list-u
# .filter(ee.Filter.neq('NDVI', null))

# reduceRegion img
def reduce_region_img(img, scale=30, reducer=ee.Reducer.first(), tileScale=1, **kwargs):
    def wrap(feature):
        return ee.Feature(
            feature.geometry(), 
            img.reduceRegion(
                reducer=reducer, 
                scale=scale, 
                geometry=feature.geometry(),
                tileScale=tileScale,
                **kwargs)
            ).set({'YYYYMMDD': img.date().format('YYYYMMdd')}).copyProperties(feature)
    return wrap

# reduce
def reduction(img_or_imgcol, featcol, reduceRegions=True, buffer=None, 
           scale=30, reducer='first', tileScale=1, **kwargs):
    
    # check if input is imgcol or img
    if isinstance(img_or_imgcol, ee.imagecollection.ImageCollection):
        flatten = True
        is_imgcol = True
    elif isinstance(img_or_imgcol, ee.image.Image):
        flatten = False
        is_imgcol = False
    
    # Select the appropriate reducer
    # check if multiple reducers are desired
    if isinstance(reducer, list):
        reducer = combine_reducers(reducer)
    elif reducer in list(dict_reducers.keys()):
        reducer = dict_reducers[reducer]
    else:
        raise ValueError(f"Reducer '{reducer}' is not supported. Choose from {list(dict_reducers.keys())}.")

    # Buffer features if a buffer distance is provided
    if buffer is not None:
        featcol = featcol.map(lambda x: x.buffer(buffer))

    # Check if reduceType is reduceRegions or reduceRegion
    # reduceRegions (https://developers.google.com/earth-engine/apidocs/ee-image-reduceregions)
    if reduceRegions:
        if is_imgcol:
            extracted = ee.FeatureCollection(img_or_imgcol.map(reduce_regions_imgcol(featcol, scale=scale, reducer=reducer, tileScale=tileScale, **kwargs)))
        else:
            extracted = ee.FeatureCollection(reduce_regions_img(img_or_imgcol, featcol, scale=scale, reducer=reducer, tileScale=tileScale, **kwargs))
    # reduceRegion (https://developers.google.com/earth-engine/apidocs/ee-image-reduceregion)
    else:
        if is_imgcol:
            extracted = ee.FeatureCollection(featcol.map(reduce_region_imgcol(img_or_imgcol, scale=30, reducer=reducer, tileScale=tileScale, **kwargs)))
        else:
            extracted = ee.FeatureCollection(featcol.map(reduce_region_img(img_or_imgcol, scale=30, reducer=reducer, tileScale=tileScale, **kwargs)))
    
    # check if image or imgcol has system:time_start property
    # and then remove null values
    # currently this is no longer needed since we are using reduceRegions in combination with limit (see above and N. Gorelick answer)
    #extracted = extracted.filter(ee.Filter.neq('system:time_start', 0))

    # Flatten the feature collection if desired; necessary if input is ImageCollection
    if flatten:
        extracted = extracted.flatten()

    return extracted


# --------------------------------------------------------------------------------------------
#                                   Vector chunking                               
# --------------------------------------------------------------------------------------------

# function to subset a vector/geometry dataset into chunks
# useful for datasets covering a vast geography which is better processed in chunks
# chunks are created by geographic similarity / distance
def vector_to_chunks(filepath, output_dir=None, min_samples=10, distance=10000):
    # Read the geospatial data
    if isinstance(filepath, str):
        gdf = gpd.read_file(filepath)
    
    # Reproject to a global suitable projection for measuring cartesian distance (EPSG:3857)
    gdf = gdf.to_crs(epsg=3857)
    
    # Calculate the bounding box and divide it into blocks based on the specified distance
    minx, miny, maxx, maxy = gdf.total_bounds
    x_splits = np.arange(minx, maxx, distance)
    y_splits = np.arange(miny, maxy, distance)
    
    # Create centers of the blocks
    centers = [Point((x + distance / 2, y + distance / 2)) for x in x_splits for y in y_splits]
    
    # Assign each point to the closest block center
    gdf['block'] = gdf.geometry.apply(lambda geom: np.argmin([geom.distance(center) for center in centers]))
    
    # Ensure minimum points per block
    block_sizes = gdf['block'].value_counts()
    
    # Merge small blocks with other small blocks first
    small_blocks = block_sizes[block_sizes < min_samples].index
    merged = set()
    
    for block in small_blocks:
        if block in merged:
            continue
        small_block_points = gdf[gdf['block'] == block]
        remaining_small_blocks = small_blocks[small_blocks != block]
        
        for other_block in remaining_small_blocks:
            if other_block in merged:
                continue
            other_block_points = gdf[gdf['block'] == other_block]
            distances = small_block_points.geometry.apply(lambda geom: other_block_points.distance(geom).min())
            if distances.min() < distance:
                gdf.loc[gdf['block'] == other_block, 'block'] = block
                merged.add(other_block)
        
        # Update block sizes after merging small blocks
        block_sizes = gdf['block'].value_counts()
        small_blocks = block_sizes[block_sizes < min_samples].index

    # Merge remaining small blocks with nearest larger block
    small_blocks = block_sizes[block_sizes < min_samples].index
    for block in small_blocks:
        small_block_points = gdf[gdf['block'] == block]
        remaining_points = gdf[gdf['block'] != block]
        for i, point in small_block_points.iterrows():
            distances = remaining_points.geometry.distance(point.geometry)
            if not distances.empty:
                nearest_block = remaining_points.loc[distances.idxmin()]['block']
                gdf.at[i, 'block'] = nearest_block
    
    # Recalculate block sizes after merging small blocks
    block_sizes = gdf['block'].value_counts()
    
    # Reproject back to geographic coordinates (WGS84)
    gdf = gdf.to_crs(epsg=4326)

    # Create subchunks based on blocks
    unique_blocks = gdf['block'].unique()
    subchunks = [gdf[gdf['block'] == block] for block in unique_blocks]
    
    # Optionally write subchunks to disk
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for i, subchunk in enumerate(subchunks):
            subchunk.to_file(os.path.join(output_dir, f'subchunk_{i}.gpkg'), driver='GPKG')
    
    return subchunks

# --------------------------------------------------------------------------------------------
#                                   TILING / DATACUBE                               
# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# Copyright (c) 2019, Global LANdcover mapping and Estimation (GLANCE)

# common projections (but not included in GEE)
wkt_mollweide = ' \
  PROJCS["World_Mollweide", \
    GEOGCS["GCS_WGS_1984", \
      DATUM["WGS_1984", \
        SPHEROID["WGS_1984",6378137,298.257223563]], \
      PRIMEM["Greenwich",0], \
      UNIT["Degree",0.017453292519943295]], \
    PROJECTION["Mollweide"], \
    PARAMETER["False_Easting",0], \
    PARAMETER["False_Northing",0], \
    PARAMETER["Central_Meridian",0], \
    UNIT["Meter",1], \
    AUTHORITY["EPSG","54009"]]'

# GLANCE tiling grid
# WKT strings for the 8 regions of GLANCE
wkt_AF = 'PROJCS["BU MEaSUREs Lambert Azimuthal Equal Area - AF - V01",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["degree",0.0174532925199433]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["longitude_of_center",20],PARAMETER["latitude_of_center",5],UNIT["meter",1.0]]'
wkt_AN = 'PROJCS["BU MEaSUREs Lambert Azimuthal Equal Area - AN - V01",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["degree",0.0174532925199433]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["longitude_of_center",0],PARAMETER["latitude_of_center",-90],UNIT["meter",1.0]]'
wkt_AS = 'PROJCS["BU MEaSUREs Lambert Azimuthal Equal Area - AS - V01",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["degree",0.0174532925199433]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["longitude_of_center",100],PARAMETER["latitude_of_center",45],UNIT["meter",1.0]]'
wkt_EU = 'PROJCS["BU MEaSUREs Lambert Azimuthal Equal Area - EU - V01",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["degree",0.0174532925199433]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["longitude_of_center",20],PARAMETER["latitude_of_center",55],UNIT["meter",1.0]]'
wkt_OC = 'PROJCS["BU MEaSUREs Lambert Azimuthal Equal Area - OC - V01",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["degree",0.0174532925199433]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["longitude_of_center",135],PARAMETER["latitude_of_center",-15],UNIT["meter",1.0]]'
wkt_NA = 'PROJCS["BU MEaSUREs Lambert Azimuthal Equal Area - NA - V01",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["degree",0.0174532925199433]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["longitude_of_center",-100],PARAMETER["latitude_of_center",50],UNIT["meter",1.0]]'
wkt_SA = 'PROJCS["BU MEaSUREs Lambert Azimuthal Equal Area - SA - V01",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["degree",0.0174532925199433]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["longitude_of_center",-60],PARAMETER["latitude_of_center",-15],UNIT["meter",1.0]]'

# create dictionary with WKT strings
wkt_dict = {
    'AF': wkt_AF,
    'AN': wkt_AN,
    'AS': wkt_AS,
    'EU': wkt_EU,
    'OC': wkt_OC,
    'NA': wkt_NA,
    'SA': wkt_SA,
    'Mollweide': wkt_mollweide
}

# Projections and Tiling
WKT_LAEA_TEMPLATE = """
PROJCS["BU MEaSUREs Lambert Azimuthal Equal Area - {continent} - V01",
    GEOGCS["GCS_WGS_1984",
        DATUM["D_WGS_1984",
            SPHEROID["WGS_1984",6378137.0,298.257223563]],
        PRIMEM["Greenwich",0.0],
        UNIT["degree",0.0174532925199433]],
    PROJECTION["Lambert_Azimuthal_Equal_Area"],
    PARAMETER["false_easting",0.0],
    PARAMETER["false_northing",0.0],
    PARAMETER["longitude_of_center",{lon_of_center}],
    PARAMETER["latitude_of_center",{lat_of_center}],
    UNIT["meter",1.0]]
"""

PROJ_LAEA_CENTERS = {
    'AF': {
        'lat_of_center': 5,
        'lon_of_center': 20,
    },
    'AN': {
        'lat_of_center': -90,
        'lon_of_center': 0,
    },
    'AS': {
        'lat_of_center': 45,
        'lon_of_center': 100,
    },
    'EU': {
        'lat_of_center': 55,
        'lon_of_center': 20,
    },
    'OC': {
        'lat_of_center': -15,
        'lon_of_center': 135,
    },
    'NA': {
        'lat_of_center': 50,
        'lon_of_center': -100,
    },
    'SA': {
        'lat_of_center': -15,
        'lon_of_center': -60,
    },
}
CONTINENTS = list(PROJ_LAEA_CENTERS.keys())

GLANCE_GRID_CRS_WKT = {
    continent: WKT_LAEA_TEMPLATE.format(continent=continent,
                                        **params)
    for continent, params in PROJ_LAEA_CENTERS.items()
}

GLANCE_GRIDS_UL_XY = {
    'AF': (-5312270.00, 3707205.0),
    'AN': (-3662210.00, 5169375.0),
    'AS': (-4805840.00, 5190735.0),
    'EU': (-5505560.00, 3346245.0),
    'OC': (-6961010.00, 4078425.0),
    'NA': (-7633670.00, 5076465.0),
    'SA': (-6918770.00, 4899705.0),
}

GLANCE_GRIDS_LIMITS = {
    'AF': [(0, 60), (0, 74)],
    'AN': [(0, 56), (0, 62)],
    'AS': [(0, 66), (0, 77)],
    'EU': [(0, 38), (0, 54)],
    'OC': [(0, 78), (0, 113)],
    'NA': [(0, 64), (0, 84)],
    'SA': [(0, 68), (0, 74)]
}
# --------------------------------------------------------------------------------------------

def create_glance_tiles(continent_code, tile_size=150000, vector_roi=None, output_dir=None, land_mask=False):
    """
    Create grid GeoPackage files based on continent code and grid parameters with explicit ID naming.
    Optionally restrict the grid to land surfaces using a land mask.
    Parameters:
    - continent_code (str): The code of the continent for which to create the grid. Either AF, AN, AS, EU, OC, NA, SA or use "ALL" for all continents.
    - tile_size (int): The size of the main grid tile in meters. Must be one of [1200000, 600000, 300000, 150000, 75000, 30000].
    - vector_roi (str or GeoDataFrame): The shapefile path or GeoDataFrame to clip the grid with. Default is None.
    - output_dir (str): The directory to save the grid GeoPackage files. Default is None.
    - land_mask (bool): Whether to restrict the grid to land surfaces. Default is False (no masking).
    Returns:
    - GeoDataFrame: A GeoDataFrame containing the grid tiles with IDs for all valid tile sizes.
    Raises:
    - ValueError: If the tile_size is not one of the allowed sizes.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon
    import os

    valid_sizes = [1200000, 600000, 300000, 150000, 75000, 30000]
    if tile_size not in valid_sizes:
        raise ValueError(f"Invalid tile size. Must be one of {valid_sizes}.")

    if land_mask:
        # Load land mask GeoDataFrame
        natural_earth_land = gpd.read_file(
            os.path.join(os.path.dirname(__file__), '../data/ne_10m_land.gpkg')
        )

    if continent_code == "ALL":
        continents = CONTINENTS
    else:
        continents = [continent_code]

    result_grids = []

    for continent in continents:
        crs_wkt = GLANCE_GRID_CRS_WKT[continent]
        upper_left = GLANCE_GRIDS_UL_XY[continent]
        grid_limits = GLANCE_GRIDS_LIMITS[continent]

        ul_x, ul_y = upper_left
        (row_min, row_max), (col_min, col_max) = grid_limits

        base_tile_size = 150000  # Reference tile size
        scale_factor = base_tile_size / tile_size
        row_min_scaled = int(row_min * scale_factor)
        row_max_scaled = int(row_max * scale_factor)
        col_min_scaled = int(col_min * scale_factor)
        col_max_scaled = int(col_max * scale_factor)

        grid_list = []
        id_columns = {f"ID{size // 1000}": [] for size in valid_sizes if size >= tile_size}
        x_columns = {f"X{size // 1000}": [] for size in valid_sizes if size >= tile_size}
        y_columns = {f"Y{size // 1000}": [] for size in valid_sizes if size >= tile_size}

        for row in range(row_min_scaled, row_max_scaled):
            for col in range(col_min_scaled, col_max_scaled):
                x_min = ul_x + (col * tile_size)
                y_max = ul_y - (row * tile_size)
                x_max = x_min + tile_size
                y_min = y_max - tile_size

                grid_list.append(Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]))

                for size in valid_sizes:
                    if size >= tile_size:
                        factor = size // tile_size
                        larger_col = col // factor
                        larger_row = row // factor
                        id_columns[f"ID{size // 1000}"].append(f"{continent}_{size // 1000}-X{larger_col:03d}-Y{larger_row:03d}")
                        x_columns[f"X{size // 1000}"].append(larger_col)
                        y_columns[f"Y{size // 1000}"].append(larger_row)

        grid_data = {"geometry": grid_list}
        grid_data.update(id_columns)
        grid_data.update(x_columns)
        grid_data.update(y_columns)

        grid_gdf = gpd.GeoDataFrame(grid_data, crs=crs_wkt)

        # Apply land mask if required
        if land_mask:
            print("Filtering grid tiles using the land mask...")
            natural_earth_land = natural_earth_land.to_crs(crs_wkt)
            grid_gdf = gpd.sjoin(grid_gdf, natural_earth_land, how="inner", predicate="intersects")
            grid_gdf = grid_gdf.drop(columns="index_right").drop_duplicates(subset="geometry")

        # Clip to ROI if provided
        if vector_roi:
            if isinstance(vector_roi, str):
                user_gdf = gpd.read_file(vector_roi)
            elif isinstance(vector_roi, gpd.geodataframe.GeoDataFrame):
                user_gdf = vector_roi
            else:
                raise ValueError("User input must be a file path or a GeoDataFrame")

            # Ensure CRS consistency
            if user_gdf.crs != grid_gdf.crs:
                user_gdf = user_gdf.to_crs(grid_gdf.crs)

            # Intersect grid with ROI
            grid_gdf = grid_gdf[grid_gdf.geometry.intersects(user_gdf.unary_union)]
            grid_gdf = grid_gdf.drop_duplicates(subset="geometry")  # Remove duplicates after ROI intersection

        # Save the result to output directory if specified
        if output_dir:
            output_filename = f"{output_dir}/GLANCE_{continent}_{tile_size // 1000}km.gpkg"
            grid_gdf.to_file(output_filename, driver="GPKG")
            print(f"Grid for {continent} saved to {output_filename}")

        result_grids.append(grid_gdf)

    # Return grid(s)
    return result_grids if len(result_grids) > 1 else result_grids[0]

# -------------------------------------------------------------------------------
#                                Temporal Functions                               
# -------------------------------------------------------------------------------
def int_to_datestring(x):
    date_str = str(x)
    # Parse the string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    # Format the datetime object to the desired string format
    formatted_date = date_obj.strftime('%Y-%m-%d')
    return formatted_date

'''
def add_timeband(img):
    img = ee.Image(img)
    timeImage = img.get('system:time_start')
    timeImageMasked = timeImage.updateMask(img.select(0).mask())
    return img.addBands(timeImageMasked.rename('time')).toFloat()
'''
def add_timeband(img):
    img_time = ee.Image.constant(img.get('system:time_start')).updateMask(img.select(0).mask())
    return img.addBands(img_time.rename('time')).toFloat()

# client-side function
def days_to_milli(days):
    return days*1000*60*60*24

# get acquisition dates + bands of images as ee.List
def get_dates_with_bands_str(imgcol, format='YYYYMMdd'):
    def format_date_and_bands(img):
        # Get the date in the specified format
        date_str = img.date().format(format)
        # Get the list of band names
        band_names = img.bandNames()
        # Combine the date with each band name
        date_band_strs = band_names.map(lambda band: ee.String(band).cat('_').cat(date_str))
        # Return the combined strings as a feature collection
        return ee.Feature(None, {'date_band_strs': date_band_strs})

    # Map over the image collection to get the formatted date and band strings
    features = imgcol.map(format_date_and_bands)
    
    # Extract the list of date_band_strs from the features
    date_band_strs = features.aggregate_array('date_band_strs').flatten()
    return ee.List(date_band_strs)

def imgcol_dates_to_featcol(imgcol, format='YYYYMMdd'):
    dates = imgcol.map(
        lambda img: ee.Feature(None, {'system:time_start': img.date().millis(),
                                      'YYYYMMDD': img.date().format(format)})
    )
    return ee.FeatureCollection(dates)

# OLD code to generate folds for folding using ee.List.map()
# Helper function to generate filter keys
def generate_key(year_range, month_range, doy_range):
    year_part = f"Y{year_range[0]:04d}-{year_range[1]:04d}"
    month_part = f"M{month_range[0]:02d}-{month_range[1]:02d}"
    doy_part = f"D{doy_range[0]:03d}-{doy_range[1]:03d}"
    return f"{year_part}_{month_part}_{doy_part}"

# Function to create a list of filters used for folding ImageCollections
def create_folds(fold_custom=None, fold_year=False, fold_month=False, year_start=None, year_end=None):
    filters_list = []
    
    # Determine the global ranges if not specified
    global_year_range = [year_start, year_end] if year_start is not None and year_end is not None else [0, 0]
    global_month_range = [1, 12] if fold_month else [0, 0]
    global_doy_range = [1, 366]

    # Define a utility function to add filters based on provided ranges
    def add_filters(year_ranges, month_ranges, doy_ranges, date_ranges):
        if year_ranges:
            for year_pair in year_ranges:
                start_year, end_year = map(int, year_pair.split('-'))
                if month_ranges:
                    for month_pair in month_ranges:
                        start_month, end_month = map(int, month_pair.split('-'))
                        filter_key = generate_key([start_year, end_year], [start_month, end_month], global_doy_range)
                        filter_value = ee.Filter.And(
                            ee.Filter.calendarRange(start_year, end_year, 'year'),
                            ee.Filter.calendarRange(start_month, end_month, 'month')
                        )
                        filters_list.append(ee.Dictionary({'key': filter_key, 'filter': filter_value}))
                if doy_ranges:
                    for doy_pair in doy_ranges:
                        start_doy, end_doy = map(int, doy_pair.split('-'))
                        filter_key = generate_key([start_year, end_year], global_month_range, [start_doy, end_doy])
                        filter_value = ee.Filter.And(
                            ee.Filter.calendarRange(start_year, end_year, 'year'),
                            ee.Filter.calendarRange(start_doy, end_doy, 'day_of_year')
                        )
                        filters_list.append(ee.Dictionary({'key': filter_key, 'filter': filter_value}))
                if date_ranges:
                    for date_pair in date_ranges:
                        start_date, end_date = map(str, date_pair.split('-'))
                        filter_key = generate_key([start_year, end_year], global_month_range, global_doy_range)
                        filter_value = ee.Filter.date(start_date, end_date)
                        filters_list.append(ee.Dictionary({'key': filter_key, 'filter': filter_value}))

        elif month_ranges:
            for month_pair in month_ranges:
                start_month, end_month = map(int, month_pair.split('-'))
                if doy_ranges:
                    for doy_pair in doy_ranges:
                        start_doy, end_doy = map(int, doy_pair.split('-'))
                        filter_key = generate_key(global_year_range, [start_month, end_month], [start_doy, end_doy])
                        filter_value = ee.Filter.And(
                            ee.Filter.calendarRange(start_month, end_month, 'month'),
                            ee.Filter.calendarRange(start_doy, end_doy, 'day_of_year')
                        )
                        filters_list.append(ee.Dictionary({'key': filter_key, 'filter': filter_value}))
                if date_ranges:
                    for date_pair in date_ranges:
                        start_date, end_date = map(str, date_pair.split('-'))
                        filter_key = generate_key(global_year_range, [start_month, end_month], global_doy_range)
                        filter_value = ee.Filter.date(start_date, end_date)
                        filters_list.append(ee.Dictionary({'key': filter_key, 'filter': filter_value}))
                else:
                    filter_key = generate_key(global_year_range, [start_month, end_month], global_doy_range)
                    filter_value = ee.Filter.calendarRange(start_month, end_month, 'month')
                    filters_list.append(ee.Dictionary({'key': filter_key, 'filter': filter_value}))

        elif doy_ranges:
            for doy_pair in doy_ranges:
                start_doy, end_doy = map(int, doy_pair.split('-'))
                filter_key = generate_key(global_year_range, global_month_range, [start_doy, end_doy])
                filter_value = ee.Filter.calendarRange(start_doy, end_doy, 'day_of_year')
                filters_list.append(ee.Dictionary({'key': filter_key, 'filter': filter_value}))

        elif date_ranges:
            for date_pair in date_ranges:
                start_date, end_date = map(str, date_pair.split('-'))
                filter_key = generate_key(global_year_range, global_month_range, global_doy_range)
                filter_value = ee.Filter.date(start_date, end_date)
                filters_list.append(ee.Dictionary({'key': filter_key, 'filter': filter_value}))

    # Apply year ranges and/or month ranges and/or custom ranges
    if fold_year and year_start is not None and year_end is not None:
        add_filters([f"{year_start}-{year_end}"], None, None, None)
    if fold_month:
        add_filters(None, [f"{month:02d}-{month:02d}" for month in range(1, 13)], None, None)
    if fold_custom:
        add_filters(
            fold_custom.get('year', None),
            fold_custom.get('month', None),
            fold_custom.get('doy', None),
            fold_custom.get('date', None)
        )

    # If no specific fold is applied, use the global timeframe if provided
    if not filters_list:
        filter_key = generate_key(global_year_range, global_month_range, global_doy_range)
        filter_value = ee.Filter()
        filters_list.append(ee.Dictionary({'key': filter_key, 'filter': filter_value}))

    return ee.List(filters_list)


def add_time_properties_to_img(img):
    date = ee.Date(img.get('system:time_start'))
    year = date.get('year')
    month = date.get('month')
    doy = date.getRelative('day', 'year')
    formatted_date = date.format('YYYYMMdd')
    return img.set({'year': year, 'month': month, 'doy': doy, 'date': formatted_date})


def add_date_property_to_img(img):
    date = ee.Date(img.get('system:time_start'))
    formatted_date = date.format('YYYYMMdd')
    return img.set('date', formatted_date)


# 'NEW' code to generate folds for folding using ee.Filter.listContains() and subsequent ee.Join.saveAll()
def construct_time_subwindows(YEAR_MIN, YEAR_MAX, MONTH_MIN, MONTH_MAX, DOY_MIN, DOY_MAX, DATE_MIN, DATE_MAX, FOLD_YEAR, FOLD_MONTH, FOLD_CUSTOM):
    
    def generate_custom_ranges(custom_range, unit_min, unit_max, cyclic=False):
        if custom_range is None:
            return []
        ranges = []
        if isinstance(custom_range, list):
            for r in custom_range:
                if '+-' in r:
                    target, offset = map(int, r.split('+-'))
                    start = target - offset
                    end = target + offset
                    if cyclic:
                        start = ((start - 1) % unit_max) + 1
                        end = ((end - 1) % unit_max) + 1
                    ranges.append((start, end))
                elif '-' in r:
                    start, end = map(int, r.split('-'))
                    ranges.append((start, end))
        return ranges

    def format_date(date_str):
        return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")

    def date_to_milliseconds(date):
        if isinstance(date, str):
            dt = datetime.strptime(date, "%Y%m%d")
        else:
            dt = date
        epoch = datetime(1970, 1, 1)
        return int((dt - epoch).total_seconds() * 1000)

    def calculate_time_start(year, month=1, day=1, doy=None):
        if doy is not None:
            # Convert DOY to month and day
            date = datetime(year, 1, 1) + timedelta(days=doy - 1)
            month = date.month
            day = date.day
        elif month is None:
            month = 1
            day = 1
        elif day is None:
            day = 1
            
        date_str = f"{year:04d}{month:02d}{day:02d}"
        return date_to_milliseconds(date_str)

    def generate_year_list(year_range):
        return list(range(year_range[0], year_range[1] + 1))

    def generate_month_list(month_range):
        months = []
        start, end = month_range
        if start <= end:
            months = list(range(start, end + 1))
        else:
            # Cyclic handling: wrap around the year
            months = list(range(start, 13)) + list(range(1, end + 1))
        return months

    def generate_doy_list(doy_range):
        doys = []
        start, end = doy_range
        if start <= end:
            doys = list(range(start, end + 1))
        else:
            # Cyclic handling: wrap around the year
            doys = list(range(start, 367)) + list(range(1, end + 1))
        return doys

    def check_custom_within_global(custom_ranges, global_min, global_max, unit_name):
        for start, end in custom_ranges:
            if start < global_min or end > global_max:
                raise ValueError(f"Custom {unit_name} range [{start}, {end}] is outside the global {unit_name} range [{global_min}, {global_max}].")

    # Compute global date range if DATE_MIN or DATE_MAX are not specified
    if DATE_MIN is None:
        global_start_date = f"{YEAR_MIN:04d}{MONTH_MIN:02d}01"  # Start from the first day of the month
    else:
        global_start_date = DATE_MIN
        YEAR_MIN, YEAR_MAX = int(DATE_MIN[:4]), int(DATE_MAX[:4])
        MONTH_MIN, MONTH_MAX = int(DATE_MIN[4:6]), int(DATE_MAX[4:6])
    
    if DATE_MAX is None:
        # Calculate the last day of MONTH_MAX in YEAR_MAX
        if MONTH_MAX == 12:
            global_end_date = f"{YEAR_MAX:04d}{MONTH_MAX:02d}31"
        else:
            next_month = datetime(YEAR_MAX, MONTH_MAX % 12 + 1, 1)
            last_day_of_month = next_month - timedelta(days=1)
            global_end_date = last_day_of_month.strftime("%Y%m%d")
    else:
        global_end_date = DATE_MAX

    global_start_date_int = int(global_start_date)
    global_end_date_int = int(global_end_date)

    # Check validity of custom ranges against global settings
    if FOLD_CUSTOM is not None:
        custom_years = generate_custom_ranges(FOLD_CUSTOM.get('year'), YEAR_MIN, YEAR_MAX)
    
    if custom_years:
        check_custom_within_global(custom_years, YEAR_MIN, YEAR_MAX, "year")
    
    custom_months = generate_custom_ranges(FOLD_CUSTOM.get('month'), 1, 12, cyclic=True)
    if custom_months:
        check_custom_within_global(custom_months, MONTH_MIN, MONTH_MAX, "month")
    
    custom_doys = generate_custom_ranges(FOLD_CUSTOM.get('doy'), 1, 366, cyclic=True)
    if custom_doys:
        check_custom_within_global(custom_doys, DOY_MIN, DOY_MAX, "day of year")
    
    custom_date_ranges = generate_custom_ranges(FOLD_CUSTOM.get('date'), global_start_date_int, global_end_date_int)
    if custom_date_ranges:
        for start, end in custom_date_ranges:
            if int(start) < global_start_date_int or int(end) > global_end_date_int:
                raise ValueError(f"Custom date range [{start}, {end}] is outside the global date range [{global_start_date}, {global_end_date}].")
    
    # Initialize the result dictionary
    result_dict = {}

    # Helper function to add an entry to the result_dict
    def add_to_result_dict(years, months, doys, start_year, end_year, start_doy, end_doy, time_start, year_center, year_offset, doy_center, doy_offset):
        year_str = f"{years[0]}-{years[-1]}" if years else "0000-0000"
        month_str = f"{months[0]:02d}-{months[-1]:02d}" if months else "00-00"
        doy_str = f"{start_doy:03d}-{end_doy:03d}" if doys else "000-000"
        key = f"Y{year_str}_M{month_str}_D{doy_str}"
        result_dict[key] = {
            "year": years if years else None,
            "month": months if months else None,
            "doy": doys if doys else None,
            "date": None,
            "time_start": time_start,
            "year_center": year_center,
            "year_offset": year_offset,
            "doy_center": doy_center,
            "doy_offset": doy_offset
        }

    # Helper function to get center and offset values
    def get_center_offset(range_start, range_end):
        center = (range_start + range_end) // 2
        offset = (range_end - range_start) // 2
        return center, offset

    # 1. Custom date range
    if custom_date_ranges:
        for start, end in custom_date_ranges:
            start_date_str = str(start)
            end_date_str = str(end)
            start_year = int(start_date_str[:4])
            end_year = int(end_date_str[:4])
            
            # Calculate DOY for start and end dates
            start_doy = (datetime.strptime(start_date_str, "%Y%m%d") - datetime(start_year, 1, 1)).days + 1
            end_doy = (datetime.strptime(end_date_str, "%Y%m%d") - datetime(end_year, 1, 1)).days + 1
            
            key = f"Y{start_year}-{end_year}_M00-00_D{start_doy:03d}-{end_doy:03d}"
            start_date_formatted = format_date(start_date_str)
            end_date_formatted = format_date(end_date_str)
            result_dict[key] = {
                "year": list(range(start_year, end_year + 1)),
                "month": None,
                "doy": None,
                "date": [start_date_formatted, end_date_formatted],
                "time_start": date_to_milliseconds(start_date_str),
                "year_center": (start_year + end_year) // 2,
                "year_offset": (end_year - start_year) // 2,
                "doy_center": None,
                "doy_offset": None
            }
        return result_dict

    if custom_months and custom_doys:
        raise ValueError("Custom months and days of year cannot be used together.")

    if FOLD_MONTH and custom_doys:
        raise ValueError("Monthly folding and custom days of year cannot be used together.")

    # 2. Custom years + custom months + fold year + fold month
    if custom_years and custom_months and FOLD_YEAR and FOLD_MONTH:
        for custom_year_range in custom_years:
            years = generate_year_list(custom_year_range)
            for year in years:
                for month_range in custom_months:
                    months = generate_month_list(month_range)
                    for month in months:
                        time_start = calculate_time_start(year, month)
                        add_to_result_dict([year], [month], None, year, year, 0, 0, time_start, year, 0, None, None)

    # 3. Custom years + custom months + fold year
    elif custom_years and custom_months and FOLD_YEAR:
        for custom_year_range in custom_years:
            years = generate_year_list(custom_year_range)
            for year in years:
                for month_range in custom_months:
                    start_month, end_month = month_range
                    months = generate_month_list(month_range)
                    time_start = calculate_time_start(year, start_month)
                    add_to_result_dict([year], months, None, year, year, 0, 0, time_start, year, 0, None, None)

    # 4. Custom years + custom months + fold month
    elif custom_years and custom_months and FOLD_MONTH:
        for custom_year_range in custom_years:
            years = generate_year_list(custom_year_range)
            for month_range in custom_months:
                months = generate_month_list(month_range)
                time_start = calculate_time_start(years[0], months[0])
                add_to_result_dict(years, months, None, custom_year_range[0], custom_year_range[1], 0, 0, time_start, (custom_year_range[0] + custom_year_range[1]) // 2, (custom_year_range[1] - custom_year_range[0]) // 2, None, None)

    # 14. Custom years + fold year + custom DOY
    elif custom_years and custom_doys and FOLD_YEAR :
        for custom_year_range in custom_years:
            years = generate_year_list(custom_year_range)
            for year in years:
                for doy_range in custom_doys:
                    doys_in_range = generate_doy_list(doy_range)
                    start_doy = doys_in_range[0]
                    end_doy = doys_in_range[-1]
                    doy_center, doy_offset = get_center_offset(start_doy, end_doy)
                    time_start = calculate_time_start(year, MONTH_MIN, day=None, doy=start_doy)
                    add_to_result_dict([year], None, doys_in_range, year, year, start_doy, end_doy, time_start, year, 0, doy_center, doy_offset)

    # 15. Custom years + custom DOY
    elif custom_years and custom_doys:
        for custom_year_range in custom_years:
            years = generate_year_list(custom_year_range)
            year_center, year_offset = get_center_offset(custom_year_range[0], custom_year_range[1])
            for doy_range in custom_doys:
                doys_in_range = generate_doy_list(doy_range)
                start_doy = doys_in_range[0]
                end_doy = doys_in_range[-1]
                doy_center, doy_offset = get_center_offset(start_doy, end_doy)
                time_start = calculate_time_start(years[0], MONTH_MIN, day=None, doy=start_doy)
                add_to_result_dict(years, None, doys_in_range, custom_year_range[0], custom_year_range[1], start_doy, end_doy, time_start, year_center, year_offset, doy_center, doy_offset)
    
    # 16. Custom DOY + and global YEAR foldinh
    elif custom_doys and FOLD_YEAR:
        years = generate_year_list((YEAR_MIN, YEAR_MAX))
        for year in years:
            for doy_range in custom_doys:
                doys_in_range = generate_doy_list(doy_range)
                start_doy = doys_in_range[0]
                end_doy = doys_in_range[-1]
                doy_center, doy_offset = get_center_offset(start_doy, end_doy)
                time_start = calculate_time_start(year, MONTH_MIN, day=None, doy=start_doy)
                add_to_result_dict([year], None, doys_in_range, year, year, start_doy, end_doy, time_start, year, 0, doy_center, doy_offset)

    # 16. Custom DOY
    elif custom_doys:
        for doy_range in custom_doys:
            doys_in_range = generate_doy_list(doy_range)
            start_doy = doys_in_range[0]
            end_doy = doys_in_range[-1]
            doy_center, doy_offset = get_center_offset(start_doy, end_doy)
            time_start = calculate_time_start(YEAR_MIN, MONTH_MIN, day=None, doy=start_doy)
            add_to_result_dict(None, None, doys_in_range, YEAR_MIN, YEAR_MAX, start_doy, end_doy, time_start, (YEAR_MIN + YEAR_MAX) // 2, (YEAR_MAX - YEAR_MIN) // 2, doy_center, doy_offset)

    # 5. Custom years + fold year + fold month
    elif custom_years and FOLD_YEAR and FOLD_MONTH:
        for custom_year_range in custom_years:
            years = generate_year_list(custom_year_range)
            months = generate_month_list((MONTH_MIN, MONTH_MAX))
            for year in years:
                for month in months:
                    time_start = calculate_time_start(year, month)
                    add_to_result_dict([year], [month], None, year, year, 0, 0, time_start, year, 0, None, None)

    # 6. Custom years + fold year
    elif custom_years and FOLD_YEAR:
        for custom_year_range in custom_years:
            years = generate_year_list(custom_year_range)
            for year in years:
                time_start = calculate_time_start(year)
                add_to_result_dict([year], None, None, year, year, 0, 0, time_start, year, 0, None, None)

    # 7. Custom years + fold month
    elif custom_years and FOLD_MONTH:
        for custom_year_range in custom_years:
            years = generate_year_list(custom_year_range)
            months = generate_month_list((MONTH_MIN, MONTH_MAX))
            for year in years:
                for month in months:
                    time_start = calculate_time_start(year, month)
                    add_to_result_dict([year], [month], None, year, year, 0, 0, time_start, year, 0, None, None)

    # 8. Custom years
    elif custom_years:
        for custom_year_range in custom_years:
            years = generate_year_list(custom_year_range)
            year_center, year_offset = get_center_offset(custom_year_range[0], custom_year_range[1])
            time_start = calculate_time_start(years[0])
            add_to_result_dict(years, None, None, custom_year_range[0], custom_year_range[1], 0, 0, time_start, year_center, year_offset, None, None)

    # 9. Custom months + fold month
    elif custom_months and FOLD_YEAR and FOLD_MONTH:
        years = generate_year_list((YEAR_MIN, YEAR_MAX))
        for year in years:
            for month_range in custom_months:
                months = generate_month_list(month_range)
                for month in months:
                    time_start = calculate_time_start(year, month)
                    add_to_result_dict([year], [month], None, year, year, 0, 0, time_start, year, 0, None, None)

    # 10. Custom months + fold year
    elif custom_months and FOLD_YEAR:
        years = generate_year_list((YEAR_MIN, YEAR_MAX))
        for year in years:
            for month_range in custom_months:
                months = generate_month_list(month_range)
                time_start = calculate_time_start(year, months[0])
                add_to_result_dict([year], months, None, year, year, 0, 0, time_start, year, 0, None, None)

    # 11. Custom months + fold month
    elif custom_months and FOLD_MONTH:
        for month_range in custom_months:
            months = generate_month_list(month_range)
            for month in months:
                time_start = calculate_time_start(YEAR_MIN, month)
                add_to_result_dict(generate_year_list((YEAR_MIN, YEAR_MAX)), [month], None, YEAR_MIN, YEAR_MAX, 0, 0, time_start, (YEAR_MIN + YEAR_MAX) // 2, (YEAR_MAX - YEAR_MIN) // 2, None, None)

    # 12. Custom months
    elif custom_months:
        for month_range in custom_months:
            months = generate_month_list(month_range)
            time_start = calculate_time_start(YEAR_MIN, months[0])
            add_to_result_dict(generate_year_list((YEAR_MIN, YEAR_MAX)), months, None, YEAR_MIN, YEAR_MAX, 0, 0, time_start, (YEAR_MIN + YEAR_MAX) // 2, (YEAR_MAX - YEAR_MIN) // 2, None, None)

    # 13. Fold year + fold month
    elif FOLD_YEAR and FOLD_MONTH:
        years = generate_year_list((YEAR_MIN, YEAR_MAX))
        months = generate_month_list((MONTH_MIN, MONTH_MAX))
        for year in years:
            for month in months:
                time_start = calculate_time_start(year, month)
                add_to_result_dict([year], [month], None, year, year, 0, 0, time_start, year, 0, None, None)

    # 13. Fold year + fold month
    elif FOLD_YEAR:
        years = generate_year_list((YEAR_MIN, YEAR_MAX))
        for year in years:
            time_start = calculate_time_start(year, MONTH_MIN)
            add_to_result_dict([year], None, None, year, year, 0, 0, time_start, year, 0, None, None)

    elif FOLD_MONTH:
        months = generate_month_list((MONTH_MIN, MONTH_MAX))
        for month in months:
            time_start = calculate_time_start(YEAR_MIN, month)
            add_to_result_dict(None, [month], None, YEAR_MIN, YEAR_MAX, 0, 0, time_start, None, 0, None, None)
    
    else:
        return None

    return result_dict




# EOF