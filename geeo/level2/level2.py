import ee
import math
from geeo.utils import load_parameters, merge_parameters, load_blueprint
from geeo.misc.spacetime import create_roi, int_to_datestring, find_utm, wkt_dict, get_crs_transform_and_img_dimensions
from geeo.level2.masking import mask_landsat, blu_filter, mask_landsat_erodil, mask_sentinel2_prob_erodil, \
                                mask_sentinel2_cplus_erodil, mask_sentinel2_prob_shadow, mask_sentinel2_cplus, \
                                mask_sentinel2_prob
from geeo.level2.collection import get_landsat_imgcol, get_sentinel2_imgcol, get_copernicus_dem
from geeo.level2.indices import dict_features, unmix
from geeo.level2.mosaic import mosaic_imgcol


def run_level2(params):
    """
    Run the level2 process with the given parameters.

    Args:
        params (str or dict): Path to the YAML file or a dictionary of parameters.
    """

    # Load blueprint defaults
    default_params = load_blueprint()
    
    # Determine the type of `params` and load/merge accordingly
    if isinstance(params, str):
        # Assume `params` is a path to a YAML file
        yaml_params = load_parameters(params)
        prm = merge_parameters(default_params, yaml_params)
    elif isinstance(params, dict):
        # Assume `params` is a dictionary of parameters
        prm = merge_parameters(default_params, dict_params=params)
    else:
        raise ValueError("params must be either a path to a YAML file or a dictionary")

    # convert level-2 parameters to variables
    ROI = prm.get('ROI')
    ROI_SIMPLIFY_GEOM_TO_BBOX = prm.get('ROI_SIMPLIFY_GEOM_TO_BBOX')
    YEAR_MIN = prm.get('YEAR_MIN')
    YEAR_MAX = prm.get('YEAR_MAX')
    MONTH_MIN = prm.get('MONTH_MIN')
    MONTH_MAX = prm.get('MONTH_MAX')
    DOY_MIN = prm.get('DOY_MIN')
    DOY_MAX = prm.get('DOY_MAX')
    DATE_MIN = prm.get('DATE_MIN')
    DATE_MAX = prm.get('DATE_MAX')
    SENSORS = prm.get('SENSORS')
    EXCLUDE_SLCOFF = prm.get('EXCLUDE_SLCOFF')
    GCP_MIN_LANDSAT = prm.get('GCP_MIN_LANDSAT')
    MAX_CLOUD = prm.get('MAX_CLOUD')
    MASKS_LANDSAT = prm.get('MASKS_LANDSAT')
    MASKS_LANDSAT_CONF = prm.get('MASKS_LANDSAT_CONF')
    MASKS_S2 = prm.get('MASKS_S2')
    MASKS_S2_CPLUS = prm.get('MASKS_S2_CPLUS')
    MASKS_S2_PROB = prm.get('MASKS_S2_PROB')
    MASKS_S2_NIR_THRESH_SHADOW = prm.get('MASKS_S2_NIR_THRESH_SHADOW')
    ERODE_DILATE = prm.get('ERODE_DILATE')
    ERODE_RADIUS = prm.get('ERODE_RADIUS')
    DILATE_RADIUS = prm.get('DILATE_RADIUS')
    ERODE_DILATE_SCALE = prm.get('ERODE_DILATE_SCALE')
    BLUE_MAX_MASKING = prm.get('BLUE_MAX_MASKING')
    FEATURES = prm.get('FEATURES')
    DEM = prm.get('DEM')
    # UNMX
    UMX = prm.get('UMX')
    UMX_SUM_TO_ONE = prm.get('UMX_SUM_TO_ONE')
    UMX_NON_NEGATIVE = prm.get('UMX_NON_NEGATIVE')
    UMX_REMOVE_INPUT_FEATURES = prm.get('UMX_REMOVE_INPUT_FEATURES')
    TSM = prm.get('TSM')
    RESAMPLING_METHOD = prm.get('RESAMPLING_METHOD')
    # CRS
    CRS = prm.get('CRS')
    PIX_RES = prm.get('PIX_RES')

    # Time Of Interest (TOI)
    # get parameters
    # check if DATE_MIN and DATE_MAX are provided
    if DATE_MIN and DATE_MAX:
        TOI = ee.Filter.date(int_to_datestring(DATE_MIN), int_to_datestring(DATE_MAX))
        # extract year range
        YEAR_MIN, YEAR_MAX = int(str(DATE_MIN)[:4]), int(str(DATE_MAX)[:4])
    # otherwise, use YEAR_MIN, YEAR_MAX, MONTH_MIN, MONTH_MAX, DOY_MIN, DOY_MAX
    else:
        TOI = ee.Filter.And(
            ee.Filter.calendarRange(YEAR_MIN, YEAR_MAX, 'year'),
            ee.Filter.calendarRange(MONTH_MIN, MONTH_MAX, 'month'),
            ee.Filter.calendarRange(DOY_MIN, DOY_MAX, 'day_of_year')
            )
    
    # Region Of Interest (ROI)
    dict_roi = create_roi(ROI, simplify_geom_to_bbox=ROI_SIMPLIFY_GEOM_TO_BBOX)
    ROI_GEOM = dict_roi['roi_geom']
    ROI_FEATCOL = dict_roi['roi_featcol']
    ROI_BBOX = dict_roi['roi_bbox']
    ROI_BBOX_GDF = dict_roi['roi_bbox_gdf']
    prm['ROI_GEOM'] = ROI_GEOM
    prm['ROI_FEATCOL'] = ROI_FEATCOL
    prm['ROI_BBOX'] = ROI_BBOX
    prm['ROI_BBOX_GDF'] = ROI_BBOX_GDF

    # CRS settings
    # UTM finder
    if CRS == 'UTM':
        CRS = find_utm(ROI_GEOM)
    elif CRS in wkt_dict.keys():
        CRS = wkt_dict[CRS]
    else:
        CRS = CRS
    prm['CRS'] = CRS
    # verify that EE accepts projection
    try:
        if ee.Projection(CRS).getInfo():
            pass
    except:
        raise ValueError('CRS not supported by GEE. Use EPSG, WKT, UTM, or GLANCE continent identifier.')
    
    # explicitly specify CRS and IMG_DIMENSIONS, and CRS_TRANSFORM for export to match grid
    # https://developers.google.com/earth-engine/guides/exporting_images
    # "to get a block of pixels precisely aligned to another data source, specify dimensions, crs and crsTransform"
    if CRS != 'EPSG:4326':
        # make sure the origin is used
        ROI_BBOX_GDF = prm.get('ROI_BBOX_GDF')
        # convert origin to output CRS and retrieve x and y translation
        ROI_BBOX_GDF = ROI_BBOX_GDF.to_crs(CRS)
        CRS_TRANSFORM, IMG_DIMENSIONS = get_crs_transform_and_img_dimensions(ROI_BBOX_GDF, PIX_RES)
        #xmin, ymin, xmax, ymax = float(ROI_BBOX_GDF.geometry.bounds.minx[0]), float(ROI_BBOX_GDF.geometry.bounds.miny[0]), float(ROI_BBOX_GDF.geometry.bounds.maxx[0]), float(ROI_BBOX_GDF.geometry.bounds.maxy[0])
        # calculate img dimensions
        #IMG_DIMENSIONS = str(int(math.ceil((xmax - xmin) // PIX_RES)))+"x"+str(int(math.ceil((ymax - ymin) // PIX_RES)))
        # construct geotransform
        #CRS_TRANSFORM = [PIX_RES, 0, xmin, 0, PIX_RES, ymin]
        # setting PIX_RES to None no longer needed because check wihtin export_img; if CRS != WGS84, then table export scale falsely specified
        #PIX_RES = None
    else:
        IMG_DIMENSIONS = None
        CRS_TRANSFORM = None
    
    prm['IMG_DIMENSIONS'] = IMG_DIMENSIONS
    prm['CRS_TRANSFORM'] = CRS_TRANSFORM
    prm['PIX_RES'] = PIX_RES

    # ImageCollections
    # Landsat
    sensors_landsat = [i for i in ['L9', 'L8', 'L7', 'L5', 'L4'] if i in SENSORS]
    if sensors_landsat:
        landsat = get_landsat_imgcol(roi=ROI_GEOM,
                                     sensors=sensors_landsat,
                                     cloudmax=MAX_CLOUD,
                                     time=TOI,
                                     exclude_slc_off=EXCLUDE_SLCOFF,
                                     gcp_filter=GCP_MIN_LANDSAT)
        
        # create user-defined masking function
        if MASKS_LANDSAT is not None:
            if ERODE_DILATE:
                mask_function = mask_landsat_erodil(MASKS_LANDSAT, conf=MASKS_LANDSAT_CONF,
                                                    kernel_erode=ee.Kernel.circle(radius=ERODE_RADIUS, units='meters'),
                                                    kernel_dilate=ee.Kernel.circle(radius=DILATE_RADIUS, units='meters'),
                                                    scale=ERODE_DILATE_SCALE)
            else:
                mask_function = mask_landsat(MASKS_LANDSAT, conf=MASKS_LANDSAT_CONF)
            # apply masking function to the Landsat ImageCollection
            landsat = landsat.map(mask_function)

        # additional masking based on blue band reflectance
        if BLUE_MAX_MASKING is not None:
            if (BLUE_MAX_MASKING < 1) and (BLUE_MAX_MASKING > 0):
                landsat = landsat.map(blu_filter(threshold=BLUE_MAX_MASKING))

    # Sentinel-2
    sensors_sentinel2 = 'S2' in SENSORS
    if sensors_sentinel2:
        # retrieve L2A collection 
        sentinel2 = get_sentinel2_imgcol(roi=ROI_GEOM,
                                         time=TOI,
                                         cloudmax=MAX_CLOUD,
                                         link_mask_collection=MASKS_S2)
        # create user-defined masking function if masking is requested
        if MASKS_S2 is not None:
            # PROB masking (https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY)
            if MASKS_S2 == 'PROB':
                # erosion and dilation; usually decreases artifacts due to clouds and cshadows
                if ERODE_DILATE:
                    mask_function = mask_sentinel2_prob_erodil(thresh_cld_prb=MASKS_S2_PROB, 
                                                                kernel_erode=ee.Kernel.circle(radius=ERODE_RADIUS, units='meters'),
                                                                kernel_dilate=ee.Kernel.circle(radius=DILATE_RADIUS, units='meters'),
                                                                scale=ERODE_DILATE_SCALE)
                else:
                    mask_function = mask_sentinel2_prob(thresh_cld_prb=MASKS_S2_PROB)
            # Cloud Score (preferable) (https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED)
            elif MASKS_S2 == 'CPLUS':
                # erosion and dilation; usually decreases artifacts due to clouds and cshadows
                if ERODE_DILATE:
                    mask_function = mask_sentinel2_cplus_erodil(band='cs', threshold=MASKS_S2_CPLUS,
                                                                kernel_erode=ee.Kernel.circle(radius=ERODE_RADIUS, units='meters'),
                                                                kernel_dilate=ee.Kernel.circle(radius=DILATE_RADIUS, units='meters'),
                                                                scale=ERODE_DILATE_SCALE)
                else:
                    mask_function = mask_sentinel2_cplus(band='cs', threshold=MASKS_S2_CPLUS)
            else:
                raise ValueError("Unknown Sentinel-2 mask selection.")

            # apply masking function to the Sentinel-2 ImageCollection
            sentinel2 = sentinel2.map(mask_function)
            # add shadows if PROB mask is used
            if (MASKS_S2_NIR_THRESH_SHADOW) and (MASKS_S2 == 'PROB'):
                sentinel2 = sentinel2.map(mask_sentinel2_prob_shadow(
                    nir_drk_thresh=MASKS_S2_NIR_THRESH_SHADOW,
                    cld_prj_dst=5, proj_scale=120))

        # additional masking based on blue band reflectance
        if BLUE_MAX_MASKING is not None:
            if (BLUE_MAX_MASKING < 1) and (BLUE_MAX_MASKING > 0):
                sentinel2 = sentinel2.map(blu_filter(threshold=BLUE_MAX_MASKING))

    # merge Landsat and Sentinel-2 ImageCollections
    if sensors_landsat and sensors_sentinel2:
        imgcol = landsat.merge(sentinel2).set('satellite', 'LSS2')
        prm['SATELLITE'] = 'LSS2'
        imgcol = imgcol.sort('system:time_start')
    elif sensors_landsat:
        imgcol = landsat.set('satellite', 'LSAT')
        prm['SATELLITE'] = 'LSAT'
    elif sensors_sentinel2:
        imgcol = sentinel2.set('satellite', 'SEN2')
        prm['SATELLITE'] = 'SEN2'
    else:
        raise ValueError("Unknown sensor selection: Please select at least one valid sensor.")

        # resampling method; called early that subsequent analyses are correctly executed
    if RESAMPLING_METHOD:
        imgcol = imgcol.map(lambda img: img.resample(RESAMPLING_METHOD))

    # bands / indices / transformations
    existing_feat = ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2', 'RE1', 'RE2', 'RE3', 'RE4', 'LST', 'mask']
    additional_feat = [feat for feat in FEATURES if feat not in existing_feat]
    for feat in additional_feat:
        imgcol = imgcol.map(dict_features[feat])  # dict_features from indices.py contains functions
    # subset to selected features
    imgcol = imgcol.select(FEATURES + ['mask'])

    # check if DEM is requested
    if DEM:
        # Digitial Elevation Model (DEM): Copernicus DEM
        copernicus_dem = get_copernicus_dem(roi=ROI_GEOM, crs=CRS, crs_transform=CRS_TRANSFORM, scale=PIX_RES, resample_method=RESAMPLING_METHOD)
        imgcol = imgcol.map(lambda img: img.addBands(copernicus_dem))
        SPECIAL_FEATURES = ['DEM']
    else:
        SPECIAL_FEATURES = []
    prm['SPECIAL_FEATURES'] = SPECIAL_FEATURES

    # UNMIXING (UMX)
    if UMX:
        UMX_VALUES = list(UMX.values())
        UMX_NAMES = list(UMX.keys())
        imgcol = imgcol.map(unmix(bands=FEATURES, 
                                  endmember_values=UMX_VALUES, 
                                  endmember_names=UMX_NAMES, 
                                  sumToOne=UMX_SUM_TO_ONE, 
                                  nonNegative=UMX_NON_NEGATIVE))
        if UMX_REMOVE_INPUT_FEATURES:
            FEATURES = UMX_NAMES
            imgcol = imgcol.select(FEATURES+['mask']+SPECIAL_FEATURES)
        else:
            FEATURES = FEATURES + UMX_NAMES
        # return updated feature names
        prm['FEATURES'] = FEATURES

    # return preprocessed time-series stack (TSS) ImageCollection
    prm['TSS'] = imgcol

    # Time-Series Mosaic (TSM)
    if TSM:
        imgcol_tsm = mosaic_imgcol(imgcol)
        prm['TSM'] = imgcol_tsm

    # return dict
    return prm


def main(params=None):
    """
    Main function to run level2 either from a YAML file or directly provided parameters.

    Args:
        params (str or dict, optional): Path to the YAML file or a dictionary of parameters.
    """
    run_level2(params)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run level2 process.")
    parser.add_argument('--params', type=str, help="Path to the YAML file or dictionary of parameters in JSON format.")

    args = parser.parse_args()

    if args.params:
        try:
            import json
            params = json.loads(args.params)
        except json.JSONDecodeError:
            params = args.params  # It's a YAML file path
    else:
        params = None

    main(params)
