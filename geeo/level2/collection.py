import ee
from geeo.misc.formatting import scale_bands, rename_bands_l4, rename_bands_l5, rename_bands_l7, rename_bands_l8, rename_bands_l9, rename_bands_s2
from geeo.misc.spacetime import create_roi

# --------------------------------------------------------------------------------------------------------------
# Landsat

def get_landsat_imgcol(roi, sensors=['L9', 'L8', 'L7','L5', 'L4'], cloudmax=75, time=None,
                       exclude_slc_off=False, gcp_filter=0):
    """
    Retrieves Landsat data from Google Earth Engine and applies data homogenisation: band names, scaling.

    Args:
        sensors (list, optional): List of Landsat sensors to retrieve data from. Defaults to ['L9', 'L8', 'L7','L5', 'L4'].
        cloudmax (int, optional): Maximum cloud cover percentage allowed. Defaults to 75.
        roi (ee.Geometry, optional): Region of Interest geometry. List of corner rectangle coordinates or server-side geometry. Defaults to None.
        time (ee.Filter, optional): Temporal filter dictionary specifying the time range of the data to retrieve. Defaults to None.
        exclude_slc_off (bool, optional): Whether to exclude SLC-off images for Landsat-7. Defaults to False.
        gcp_filter (int, optional): Minimum number of ground control points required for the image. Defaults to 0.

    Returns:
        ee.ImageCollection: Landsat image collection after applying the specified routines.
    """
    # check if time and roi are provided, otherwise set defaults
    if time is None:
        time = ee.Filter.date('1980-01-01', '2025-01-01')
    # check if roi is provided
    if roi is not None:
        if not isinstance(roi, ee.Geometry) or not isinstance(roi, ee.featurecollection.FeatureCollection):
            dict_roi = create_roi(roi)
            roi_geom = dict_roi['roi_geom']
        else:
            roi_geom = roi
    else:
        # throw an error if roi is not provided
        raise ValueError("Region of Interest (roi) must be provided.")

    # check if SLC-off should be excluded and adjust time for Landsat-7 accordingly
    if exclude_slc_off:
        time_l7 = ee.Filter.And(
            ee.Filter.date('1999-04-15', '2003-05-30'),
            time
        )
    else:
        time_l7 = time

    # dictonary containing ee.ImageCollections for each Landsat sensor
    # The collections are Landsat Collection 2 Tier 1 Level 2 products
    dict_imgcols = {
        'L9': (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
               .filterBounds(roi_geom) \
               .filter(ee.Filter.lte('CLOUD_COVER_LAND', cloudmax)) \
               .filter(ee.Filter.gte('GROUND_CONTROL_POINTS_MODEL', gcp_filter)) \
               .filter(time) \
               .map(rename_bands_l9) \
               .map(scale_bands(
                    ['CLU', 'BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2'], 
                    scale=0.0000275, offset=-0.2)
               ) \
               .map(scale_bands(
                    ['LST'], 
                    scale=0.00341802, offset=149)
               )),

        'L8': (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
               .filterBounds(roi_geom) \
               .filter(ee.Filter.lte('CLOUD_COVER_LAND', cloudmax)) \
               .filter(ee.Filter.gte('GROUND_CONTROL_POINTS_MODEL', gcp_filter)) \
               .filter(time) \
               .map(rename_bands_l8) \
               .map(scale_bands(
                    ['CLU', 'BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2'], 
                    scale=0.0000275, offset=-0.2)
               ) \
               .map(scale_bands(
                    ['LST'], 
                    scale=0.00341802, offset=149)
               )),

        'L7': (ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
               .filterBounds(roi_geom) \
               .filter(ee.Filter.lte('CLOUD_COVER_LAND', cloudmax)) \
               .filter(ee.Filter.gte('GROUND_CONTROL_POINTS_MODEL', gcp_filter)) \
               .filter(time_l7) \
               .map(rename_bands_l7) \
               .map(scale_bands(
                    ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2'], 
                    scale=0.0000275, offset=-0.2)
               ) \
               .map(scale_bands(
                    ['LST'], 
                    scale=0.00341802, offset=149)
               )),

        'L5': (ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
               .filterBounds(roi_geom) \
               .filter(ee.Filter.lte('CLOUD_COVER_LAND', cloudmax)) \
               .filter(ee.Filter.gte('GROUND_CONTROL_POINTS_MODEL', gcp_filter)) \
               .filter(time) \
               .map(rename_bands_l5) \
               .map(scale_bands(
                    ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2'], 
                    scale=0.0000275, offset=-0.2)
               ) \
               .map(scale_bands(
                    ['LST'], 
                    scale=0.00341802, offset=149)
               )),

        'L4': (ee.ImageCollection('LANDSAT/LT04/C02/T1_L2') \
               .filterBounds(roi_geom) \
               .filter(ee.Filter.lte('CLOUD_COVER_LAND', cloudmax)) \
               .filter(ee.Filter.gte('GROUND_CONTROL_POINTS_MODEL', gcp_filter)) \
               .filter(time) \
               .map(rename_bands_l4) \
               .map(scale_bands(
                    ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2'], 
                    scale=0.0000275, offset=-0.2)
               ))
    }

    # if multiple sensors are chosen, merge selected collections
    imgcol = dict_imgcols[sensors[0]]
    if len(sensors) > 1:
        for i in range(len(sensors)-1):
            imgcol = imgcol.merge(dict_imgcols[sensors[i+1]])
        imgcol = imgcol.sort("system:time_start")  # sorting is a very costly operation!

    return imgcol
    

# --------------------------------------------------------------------------------------------------------------
# Sentinel-2

def get_sentinel2_imgcol(roi, cloudmax=75, link_mask_collection='CPLUS', time=None):
    """
    time: Temporal filter dictonary of format: {'years': range(2020, 2020), months: '[6, 7, 8, 9]', 'doys': None, 'doys_off': None}
    roi: Region of Interest geometry. List of corner rectangle coords or server side geometry.
    masks: Cloud masking procedure. Either 'default' or list of 'add_mask_*' defined functions
    """
    # check if time and roi are provided, otherwise set defaults for Sentinel-2
    if time is None:
        time = ee.Filter.date('2014-01-01', '2030-01-01')

    # check if roi is provided
    if roi is not None:
        if not isinstance(roi, ee.Geometry) or not isinstance(roi, ee.featurecollection.FeatureCollection):
            dict_roi = create_roi(roi)
            roi_geom = dict_roi['roi_geom']
        else:
            roi_geom = roi
    else:
        # throw an error if roi is not provided
        raise ValueError("Region of Interest (roi) must be provided.")
    
    # Import and filter S2 SR according to cloud threshold
    imgcol = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
              .filter(time) \
              .filterBounds(roi_geom) \
              .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloudmax)))

    # filter time if provided
    if time is not None:
        imgcol = imgcol.filter(time)

    # rename bands
    imgcol = imgcol.map(rename_bands_s2)

    # scale bands
    imgcol = imgcol.map(
        scale_bands(['AER', 'BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'NIR', 'RE4', 'SW1', 'SW2'], scale=0.0001, offset=0)
        )
    
    # Add bands from other collections for masking
    if link_mask_collection == 'CPLUS':
        csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        imgcol = imgcol.linkCollection(csPlus, ['cs', 'cs_cdf'])
    if link_mask_collection == 'PROB':
        s2_cloudless_col = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        imgcol = imgcol.linkCollection(s2_cloudless_col, ['probability'])

    return imgcol


# --------------------------------------------------------------------------------------------------------------
# Copernicus DEM

def get_copernicus_dem(roi=None, crs='EPSG:4326', crs_transform=None, scale=None, resample_method='bilinear'):
    """
    Retrieves the Copernicus DEM data from Google Earth Engine and applies data homogenisation: band names, scaling.

    Args:
        roi (ee.Geometry, optional): Region of Interest geometry. List of corner rectangle coordinates or server-side geometry. Defaults to None.
        crs (str, optional): Coordinate Reference System (CRS) to use for the DEM. Defaults to 'EPSG:4326'.
        crs_transform (list, optional): List of 6 numbers specifying the affine transformation to apply to the DEM. Defaults to None.

    Returns:
        ee.ImageCollection: DEM image collection after applying the specified routines.
    """
    
    dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM')

    # check if roi is provided
    if roi is not None:
        if not isinstance(roi, ee.Geometry) or not isinstance(roi, ee.featurecollection.FeatureCollection):
            dict_roi = create_roi(roi)
            roi_geom = dict_roi['roi_geom']
        else:
            roi_geom = roi
        dem = dem.filterBounds(roi_geom)

    # DEM requires a fixed projection for slope calculation if output != 'EPSG:4326'
    if crs_transform is not None:
        dem = dem.map(lambda x: x.resample(resample_method)) 
        dem = dem.mosaic().setDefaultProjection(crs, crsTransform=crs_transform)
    elif scale is not None:
        dem = dem.map(lambda x: x.resample(resample_method)) 
        dem = dem.mosaic().setDefaultProjection(crs, scale=scale)
    else:
        dem = dem.mosaic()

    return dem


# EOF
