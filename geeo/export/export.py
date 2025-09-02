# ----------------------------------------------------------------------------
import ee
from geeo.utils import load_parameters, merge_parameters, load_blueprint
from geeo.misc.formatting import scale_and_dtype
from geeo.misc.spacetime import imgcol_to_img, create_roi, reduction, get_time_dict_subwindows, get_spatial_metadata

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
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


def export_img(
        img,
        outname='GEE_IMG', out_location='Drive', out_dir=None,
        px_res=None, region=None, crs=None, crsTransform=None, dimensions=None, fileDimensions=None,
        adjust_crsTransform_to_region=True,  # get crsTransform using crs + px_res, and region
        resampling_method=None,  # None (nearest neighbour), bilinear, bicubic
        nodata=None, scale=1, dtype=None, export_bandnames=False):
    
    """
    Export an Earth Engine Image to Google Drive or to an Earth Engine Asset with
    flexible control over spatial referencing, pixel resolution ('scale' in EE), datatype,
    and optional band name export.
    This function wraps ee.batch.Export.image.toDrive / toAsset and attempts to infer
    missing spatial parameters (projection, scale, region, transform) from the
    input image or from the supplied region + resolution. It can also construct a
    grid-aligned transform (crsTransform) and image dimensions to ensure
    alignment with tiling schemes.
    Starts the Earth Engine export task(s) immediately.
    
    Parameters
    ----------
    img : ee.Image
        The Earth Engine image to export.
    outname : str, default 'GEE_IMG'
        Description / base filename used in the export task (and asset ID name).
    out_location : str, {'Drive','Asset'}, default 'Drive'
        Destination: Google Drive or Earth Engine Asset.
    out_dir : str or None
        Drive folder name (for Drive) or asset root path (for Asset). Required
        when exporting to an Asset (used as parent path) or to a specific Drive
        folder.
    px_res : int or float or None
        Pixel resolution in units of the target CRS (passed as scale). Ignored if
        crsTransform is provided. If None and no crsTransform, inferred from the
        first band’s nominal scale.
    region : ee.Geometry or GeoJSON-like or None
        Geometry defining the export region. If None and no crsTransform, image
        geometry is used. Ignored when both crsTransform and dimensions are
        provided (grid-based export).
    crs : str or ee.Projection or None
        Target coordinate reference system (e.g. 'EPSG:3857'). If None, inferred
        from the first band’s projection.
    crsTransform : list[float] or None
        Affine transform as a 6-element (or 9-element) list. When provided it
        overrides px_res and (in combination with dimensions) can define spatial
        extent without an explicit region.
    dimensions : str or None
        Export dimensions as 'WIDTHxHEIGHT' (e.g. '1024x2048'). Mutually exclusive
        with scale+region unless using crsTransform for grid alignment.
    fileDimensions : tuple[int,int] or None
        Size (tileWidth, tileHeight) of internal GeoTIFF shards. If None and
        dimensions is given, a padded multiple of 256 is computed.
    adjust_crsTransform_to_region : bool, default True
        When True and no crsTransform supplied, derives a transform and dimensions
        from (region, crs, px_res) using get_spatial_metadata.
    resampling_method : str or None
        Resampling: one of None (nearest), 'bilinear', 'bicubic'. Applied prior to
        export via img.resample().
    nodata : int or float or None
        Value to encode as NoData in the output GeoTIFF.
    scale : int or float, default 1
        Multiplier applied to pixel values before casting (via scale_and_dtype).
    dtype : str or None
        Target data type (e.g. 'int16', 'uint16', 'float32'). Passed to
        scale_and_dtype.
    export_bandnames : bool, default False
        If True (Drive only), also exports a CSV listing band names.
    Behavior / Logic Notes
    ----------------------
    - If neither px_res nor crsTransform is supplied, pixel size is inferred.
    - If neither region nor crsTransform is supplied, the image geometry is used.
    - Providing crsTransform nullifies px_res. If both crsTransform and dimensions
        are set, region is ignored (extent derives from transform + dimensions).
    - When dimensions provided without fileDimensions, a padded multiple of 256 is
        computed for fileDimensions (shard size).
    - scale_and_dtype and get_spatial_metadata must exist in scope.
    Returns
    -------
    tuple
        (image_task_start_result, bandnames_task_start_result)
        The function calls Task.start(...) immediately. In the EE Python API,
        Task.start() returns None, so the first element will typically be None.
        The second element is None if export_bandnames is False; otherwise the
        result of starting the table export task (also typically None).
    """
    
    # possible spatial configurations that export_img (might differ from Export.image.toDrive in assumptions, e.g.
    # Export.image.toDrive may silently use WGS84 and scale=1000 if not specified)

    # crs, px_res (scale), region (global crs transform used, region determines also dimensions of image)
    # crs, crsTransform, region (grid matching, region determines also dimensions of image)
    # crs, crsTransform, dimensions (grid matching, dimensions determine also region of image, crsTransforms informs corner coordinate of image)
    
    # crs, crsTransform (region required, because dimensions unknown; tries to infer region from img.geometry(); only works for bound image)

    # if px_res, region, and crs are all None, get them from img
    if px_res is None and crsTransform is None:
        px_res = img.select(0).projection().nominalScale().getInfo()
        
    if region is None and crsTransform is None:    
        region = img.geometry()
    
    if crs is None:
        crs = img.select(0).projection()

    # check crs
    crs = ee.Projection(crs)
    try:
        if crs.getInfo():
            pass
    except:
        raise ValueError('CRS not supported by GEE or could not be inferred from img. Use valid EPSG or WKT string.')

    # resample if not None (Nearest Neighbour)
    if resampling_method:
        img = img.resample(resampling_method)

    # scaling, dtype and nodata
    img = scale_and_dtype(img, scale=scale, dtype=dtype, nodata=nodata)

    # if crsTransform is None (and crs != 'EPSG:4326') 
    # (user uses function separately to GEEO workflow, where it will be defined if not WGS84)
    # # (roi, crs, px_res, crs_transform=None, img_dimensions=None, simplify_geom_to_bbox=True)   
    if adjust_crsTransform_to_region and crsTransform is None:
        dict_spatial_metadata = get_spatial_metadata(region, crs, px_res)
        crsTransform = dict_spatial_metadata.get('crs_transform')
        dimensions = dict_spatial_metadata.get('img_dimensions')

    # image dimensions and tiling: shardSize & fileDimensions
    # alternatively get fileDimensions from dimensions as an tuple of (width, height):
    if fileDimensions is None and dimensions is not None:
        dim_width, dim_height = tuple([int(x) for x in dimensions.split('x')])
        fileDimensions = ((round(dim_width/256)*256)+256, (round(dim_height/256)*256)+256)  # 256 is the shardSize

    # if crsTransform is not None, set px_res and region to None
    if crsTransform is not None:
        px_res = None
    
    # now it uses crsTransform to align, but not based on ROI 
    # (which might not match grid, but by precalculated dimensions using crsTransform as origin for image (not just global grid))
    if crsTransform is not None and dimensions is not None:
        region = None

    # export image
    if out_location == 'Drive':
        out_image = ee.batch.Export.image.toDrive(
            image=img, 
            description=outname,
            scale=px_res,
            dimensions=dimensions,
            fileDimensions=fileDimensions,
            region=region,
            crs=crs,
            crsTransform=crsTransform,
            folder=out_dir,
            maxPixels=1e13,
            fileFormat='GeoTIFF',
            formatOptions={'noData': nodata},
        )
    elif out_location == 'Asset':
        out_image = ee.batch.Export.image.toAsset(
            image=img, 
            description=outname,
            scale=px_res,
            dimensions=dimensions,
            region=region,
            crs=crs,
            crsTransform=crsTransform,
            assetId=out_dir+'/'+outname,
            maxPixels=1e13
        )
    else:
        raise ValueError('Invalid out_location')
    
    out_image_process = ee.batch.Task.start(out_image)

    # separate bandname export
    if export_bandnames and out_location == 'Drive':
        bandnames = ee.FeatureCollection(
            ee.List(img.bandNames()).map(
            lambda x: ee.Feature(None, {'name': x})
            )
        )
        out_table = ee.batch.Export.table.toDrive(collection=bandnames,
                                        description=outname+'_bandnames',
                                        fileFormat='CSV',
                                        folder=out_dir)

        out_table_process = ee.batch.Task.start(out_table)
    else:
        out_table_process = None
    
    return out_image_process, out_table_process


def export_table(img_or_imgcol, feature, reduceRegions=True, buffer=None, reducer='first',
                 crs=None, tileScale=1, outname='GEE_IMG', out_location='Drive', out_dir=None,
                 px_res=30, nodata=0, drop_nodata=False, features=None, scale=1, dtype='double'):
    
    """
    Export per-feature statistics from an Earth Engine Image or ImageCollection as a table
    (CSV to Google Drive or FeatureCollection asset).
    This function (1) builds / coerces a region of interest (ROI) FeatureCollection, (2) scales
    and casts the input imagery, (3) performs a per-feature reduction (optionally across all
    images), (4) filters out incomplete or NoData rows, and (5) starts an Earth Engine table
    export task.
    Parameters
    ----------
    img_or_imgcol : ee.Image or ee.ImageCollection
        The input raster data to sample / reduce.
    feature : ee.FeatureCollection or ee.Feature or ee.Geometry or dict-like
        Region(s) of interest. If not already a FeatureCollection, it is passed to create_roi()
        to generate one.
    reduceRegions : bool, default True
        If True, uses a region-based reduction across all features (and images if an
        ImageCollection). Passed through to the internal reduction() helper.
    buffer : int or float or None, default None
        Optional buffer distance (meters) applied to each feature before extraction.
    reducer : str or ee.Reducer, default 'first'
        Reduction method identifier understood by the internal reduction() helper (e.g.
        'first', 'mean', etc.) or a custom ee.Reducer.
    crs : str or None, default None
        Target coordinate reference system (e.g. 'EPSG:4326'). If None, Earth Engine defaults
        are used.
    tileScale : int, default 1
        tileScale parameter to mitigate memory / computation limits for large exports.
    outname : str, default 'GEE_IMG'
        Description / base name for the export task and output file.
    out_location : {'Drive', 'Asset'}, default 'Drive'
        Destination for the exported table: Google Drive (CSV) or Earth Engine Asset
        (FeatureCollection).
    out_dir : str or None, default None
        Drive folder name (when out_location='Drive') or parent Asset path (when
        out_location='Asset'). For Asset exports the final assetId becomes out_dir + '/' + outname.
    px_res : int or float, default 30
        Nominal pixel resolution (meters) for sampling / reduction (passed as scale).
    nodata : int or float, default 0
        Fill value applied when scaling / casting the image bands.
    drop_nodata : bool, default False
        If True, filters out any feature rows where all bands equal the nodata value.
    features : list[str] or None, default None
        Explicit list of band/property names to retain and use for completeness filtering.
        If None, band names are fetched from the first image.
    scale : int or float, default 1
        Multiplicative factor applied to pixel values via scale_and_dtype().
    dtype : str, default 'double'
        Target numeric data type string accepted by scale_and_dtype() (e.g. 'double', 'float',
        'int16').
    Raises
    ------
    ValueError
        If img_or_imgcol is not an ee.Image or ee.ImageCollection, if out_location is invalid,
        or if type expectations are not met.
    Returns
    -------
    ee.batch.Task or None
        The started export task object in typical Earth Engine usage returns None from .start().
        (Note: This function invokes ee.batch.Task.start() immediately.)
    """

    # ROI
    if not isinstance(feature, ee.featurecollection.FeatureCollection):
        dict_roi = create_roi(feature)
        ROI_FEATCOL = dict_roi['roi_featcol']
    else:
        ROI_FEATCOL = feature
    
    # scaling, dtype and nodata
    img_or_imgcol = scale_and_dtype(img_or_imgcol, scale=scale, dtype=dtype, nodata=nodata)

    # create table
    table = reduction(img_or_imgcol, ROI_FEATCOL, reduceRegions=reduceRegions, buffer=buffer, 
                      scale=px_res, reducer=reducer, tileScale=tileScale, crs=crs)
    
    # remove empty missing / nodata
    # 1) get bandnames if not provided
    # if client-side bandnames are not provided, we need to get them from the image
    if features is None:
        if isinstance(img_or_imgcol, ee.imagecollection.ImageCollection):
            features = img_or_imgcol.first().bandNames().getInfo()
        elif isinstance(img_or_imgcol, ee.image.Image):
            features = img_or_imgcol.bandNames().getInfo()
        else:
            raise ValueError('img_or_imgcol must be an Image or ImageCollection')
    # 2) check for only complete cases in "table", i.e. NOT -9999, but None (TSS) specific´
    # if imgcol/img == TSS, the images are bound and geometries outside image bound results 
    # in incomplete properties across features in collection
    # thus, we post-filter "extracted" to only contain nonNull for the desired features in table
    #props_complete_cases = features # + ['YYYYMMDD'] in theory features should be enough
    table = table.filter(ee.Filter.notNull(features))

    # if drop_nodata, remove rows with nodata
    if drop_nodata:
        l_filters = []
        for f in features:
            l_filters.append(ee.Filter.neq(f, nodata))
        combined_filter = ee.Filter.Or(*l_filters)
        table = table.filter(combined_filter)

    # export table
    if out_location == 'Drive':
        out_table = ee.batch.Export.table.toDrive(
            collection=table, 
            description=outname,
            fileFormat='CSV',
            folder=out_dir
        )
    elif out_location == 'Asset':
        out_table = ee.batch.Export.table.toAsset(
            collection=table, 
            description=outname,
            assetId=out_dir+'/'+outname
        )
    else:
        raise ValueError('Invalid out_location')
    
    return ee.batch.Task.start(out_table)


def run_export(params):
    """
    Run the export process from level3 output.
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

    # import parameters
    # general
    FEATURES = prm.get('FEATURES')
    SATELLITE = prm.get('SATELLITE')
    ROI_FEATCOL = prm.get('ROI_FEATCOL')
    ROI_BBOX = prm.get('ROI_BBOX')
    ROI_GEOM = prm.get('ROI_GEOM')
    # TSS
    TSS = prm.get('TSS')
    EXPORT_TSS = prm.get('EXPORT_TSS')
    # TSM
    TSM = prm.get('TSM')
    EXPORT_TSM = prm.get('EXPORT_TSM')
    # NVO
    NVO = prm.get('NVO')
    EXPORT_NVO = prm.get('EXPORT_NVO')
    # TSI
    TSI = prm.get('TSI')
    EXPORT_TSI = prm.get('EXPORT_TSI')
    # STM
    STM = prm.get('STM')
    STM_reducer = prm.get('STM_reducer')
    EXPORT_STM = prm.get('EXPORT_STM')
    # PBC
    PBC = prm.get('PBC')
    EXPORT_PBC = prm.get('EXPORT_PBC')
    # LSP
    LSP = prm.get('LSP')
    LSP_IMG = prm.get('LSP_IMG')
    EXPORT_LSP = prm.get('EXPORT_LSP')
    # TREND
    # Export settings
    PIX_RES = prm.get('PIX_RES')
    CRS = prm.get('CRS')
    IMG_DIMENSIONS = prm.get('IMG_DIMENSIONS')
    CRS_TRANSFORM = prm.get('CRS_TRANSFORM')
    EXPORT_TABLE_TILE_SCALE = prm.get('EXPORT_TABLE_TILE_SCALE')
    DATATYPE = prm.get('DATATYPE')
    DATATYPE_SCALE = prm.get('DATATYPE_SCALE')
    NODATA_VALUE = prm.get('NODATA_VALUE')
    EXPORT_IMAGE = prm.get('EXPORT_IMAGE')
    EXPORT_TABLE = prm.get('EXPORT_TABLE')
    EXPORT_DESC = prm.get('EXPORT_DESC')
    EXPORT_LOCATION = prm.get('EXPORT_LOCATION')
    EXPORT_DIRECTORY = prm.get('EXPORT_DIRECTORY')
    EXPORT_BANDNAMES_AS_CSV = prm.get('EXPORT_BANDNAMES_AS_CSV')
    EXPORT_TABLE_METHOD = prm.get('EXPORT_TABLE_METHOD')
    EXPORT_TABLE_BUFFER = prm.get('EXPORT_TABLE_BUFFER')
    EXPORT_TABLE_REDUCER = prm.get('EXPORT_TABLE_REDUCER')
    EXPORT_TABLE_DROP_NODATA = prm.get('EXPORT_TABLE_DROP_NODATA')
    EXPORT_PER_FEATURE = prm.get('EXPORT_PER_FEATURE')
    EXPORT_PER_TIME = prm.get('EXPORT_PER_TIME')

    # outname general settings
    time_desc = prm.get('TIME_DESC')
    
    if EXPORT_DESC:
        desc = EXPORT_DESC + "_"
    else:
        desc = ""

    # table export: reduceRegions or reduceRegion
    if EXPORT_TABLE_METHOD == 'reduceRegions':
        REDUCE_REGIONS = True
    else:
        REDUCE_REGIONS = False
    
    # ------------------------------------------------------------------------
    # check if export is requested in general
    if EXPORT_IMAGE or EXPORT_TABLE:
    
        # --------------------------------------------------------------------
        # TSS
        if EXPORT_TSS:
            print("---------------------------------------------------------")
            print("          Exporting Time-Series-Stack (TSS)")
            print("")
            # subset to desired features
            TSS = TSS.select(FEATURES)
            if EXPORT_IMAGE:
                
                # user requested separate export for each feature
                if EXPORT_PER_FEATURE:
                    for feature in FEATURES:
                        img = imgcol_to_img(TSS.select(feature), date_to_bandname=False)  # False, because duplicate feature+date exist in TSS 
                        outfile = 'TSS_' + desc + time_desc + '_' + SATELLITE + '_' + feature
                        print("->  "+outfile)
                        export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                
                # user requested separate export for each time
                elif EXPORT_PER_TIME:
                    for i in range(TSS.size().getInfo()):
                        img = ee.Image(TSS.toList(TSS.size()).get(i)).select(FEATURES)  # .select() because 'mask' still in TSS
                        img_time_desc = img.date().format('YYYYMMdd').getInfo()
                        outfile = 'TSS_' + desc + img_time_desc + '_' + SATELLITE  # since this is a single date image "TSS" is a weird wording
                        print("->  "+outfile)
                        export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                            px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                            scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                
                # user requested single export for entire TSS
                else:
                    img = imgcol_to_img(TSS, date_to_bandname=False) # False, because duplicate feature+date exist in TSS 
                    outfile = 'TSS_' + desc + time_desc + '_' + SATELLITE
                    print("->  "+outfile)
                    export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                            px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                            scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
            if EXPORT_TABLE:
                outfile = 'TSS_' + desc + time_desc + '_' + SATELLITE
                print("->  "+outfile)
                export_table(img_or_imgcol=TSS, feature=ROI_FEATCOL, reduceRegions=REDUCE_REGIONS, buffer=EXPORT_TABLE_BUFFER, reducer=EXPORT_TABLE_REDUCER, 
                            tileScale=EXPORT_TABLE_TILE_SCALE, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, px_res=PIX_RES, 
                            nodata=NODATA_VALUE, drop_nodata=EXPORT_TABLE_DROP_NODATA, features=FEATURES, 
                            scale=DATATYPE_SCALE, dtype=DATATYPE)
            print("")
            print("---------------------------------------------------------")
                    
        # --------------------------------------------------------------------
        # TSM
        if EXPORT_TSM:
            # check if TSM exists
            if TSM:
                print("---------------------------------------------------------")
                print("          Exporting Time-Series-Mosaic (TSM)")
                print("")
                # subset to desired features
                TSM = TSM.select(FEATURES)
                if EXPORT_IMAGE:
                    # user requested separate export for each feature
                    if EXPORT_PER_FEATURE:
                        for feature in FEATURES:
                            img = imgcol_to_img(TSM.select(feature))
                            outfile = 'TSM_' + desc + time_desc + '_' + SATELLITE + '_' + feature
                            print("->  "+outfile)
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                                    px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                    scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    # user requested separate export for each time
                    elif EXPORT_PER_TIME:
                        for i in range(TSM.size().getInfo()):
                            img = ee.Image(TSM.toList(TSM.size()).get(i))
                            img_time_desc = img.date().format('YYYYMMdd').getInfo()
                            outfile = 'TSM_' + desc + img_time_desc + '_' + SATELLITE
                            print("->  "+outfile)
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    else:
                        img = imgcol_to_img(TSM)
                        outfile = 'TSM_' + desc + time_desc + '_' + SATELLITE
                        print("->  "+outfile)
                        export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                if EXPORT_TABLE:
                    outfile = 'TSM_' + desc + time_desc + '_' + SATELLITE
                    print("->  "+outfile)
                    export_table(img_or_imgcol=TSM, feature=ROI_FEATCOL, reduceRegions=REDUCE_REGIONS, buffer=EXPORT_TABLE_BUFFER, reducer=EXPORT_TABLE_REDUCER, 
                                tileScale=EXPORT_TABLE_TILE_SCALE, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, px_res=PIX_RES, 
                                nodata=NODATA_VALUE, drop_nodata=EXPORT_TABLE_DROP_NODATA, features=FEATURES, 
                                scale=DATATYPE_SCALE, dtype=DATATYPE)
                print("")
                print("---------------------------------------------------------")
            else:
                raise ValueError('TSM ImageCollection not calculated. Set *TSM: true* in the dict / .yml file.')
        
        # --------------------------------------------------------------------
        # NVO
        if EXPORT_NVO:
            if NVO:
                print("---------------------------------------------------------")
                print("      Exporting Number of Valid Observations (NVO)")
                print("")
                outfile = 'NVO_' + desc + time_desc + '_' + SATELLITE
                print("->  "+outfile)
                export_img(img=NVO, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY,
                           px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                           scale=1, dtype='int16', export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                print("")
                print("---------------------------------------------------------")
            else:
                raise ValueError('NVO Image not calculated. Set *NVO: true* in the dict / .yml file.')

        # --------------------------------------------------------------------
        # TSI
        if EXPORT_TSI:
            # check if TSI exists
            if TSI:
                print("---------------------------------------------------------")
                print("        Exporting Time-Series-Interpolation (TSI)")
                print("")
                if EXPORT_IMAGE:
                    # user requested separate export for each feature
                    if EXPORT_PER_FEATURE:
                        for feature in FEATURES:
                            img = imgcol_to_img(TSI.select(feature))
                            outfile = 'TSI_' + desc + time_desc + '_' + SATELLITE + '_' + feature
                            print("->  "+outfile)
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                                    px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                    scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    # user requested separate export for each time
                    elif EXPORT_PER_TIME:
                        for i in range(TSI.size().getInfo()):
                            img = ee.Image(TSI.toList(TSI.size()).get(i))
                            img_time_desc = img.date().format('YYYYMMdd').getInfo()
                            outfile = 'TSI_' + desc + img_time_desc + '_' + SATELLITE
                            print("->  "+outfile)
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    else:
                        img = imgcol_to_img(TSI)
                        outfile = 'TSI_' + desc + time_desc + '_' + SATELLITE
                        print("->  "+outfile)
                        export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                if EXPORT_TABLE:
                    outfile = 'TSI_' + desc + time_desc + '_' + SATELLITE
                    print("->  "+outfile)
                    export_table(img_or_imgcol=TSI, feature=ROI_FEATCOL, reduceRegions=REDUCE_REGIONS, buffer=EXPORT_TABLE_BUFFER, reducer=EXPORT_TABLE_REDUCER, 
                                tileScale=EXPORT_TABLE_TILE_SCALE, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, px_res=PIX_RES, 
                                nodata=NODATA_VALUE, drop_nodata=EXPORT_TABLE_DROP_NODATA, features=FEATURES, 
                                scale=DATATYPE_SCALE, dtype=DATATYPE)
                print("")
                print("---------------------------------------------------------")       
            else:
                raise ValueError('TSI ImageCollection not calculated. Set TSI method in the dict / .yml file.')
        
        # --------------------------------------------------------------------
        # STM
        if EXPORT_STM:
            # check if STM exists
            if STM:
                print("---------------------------------------------------------")
                print("       Exporting SPECTRAL-TEMPORAL METRICS (STM) ")
                print("")
                if EXPORT_IMAGE:
                    # user requested separate export for each feature
                    if EXPORT_PER_FEATURE:
                        for feature in FEATURES:
                            # need to construct new feature names, i.e. orginal feature name + reducer name
                            features_stm = [feature + '_' + r for r in STM_reducer]
                            if isinstance(STM, ee.imagecollection.ImageCollection):
                                img = imgcol_to_img(STM.select(features_stm), date_to_bandname=False)
                            else:
                                img = STM.select(features_stm)
                            outfile = 'STM_' + desc + time_desc + '_' + SATELLITE + '_' + feature
                            print("->  "+outfile)
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                                    px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                    scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    # user requested separate export for each time (only possible if STM is an ImageCollection, i.e. FOLDING)
                    elif EXPORT_PER_TIME and isinstance(STM, ee.imagecollection.ImageCollection):
                        # time description list
                        time_dict_subwindows = get_time_dict_subwindows(prm)
                        n_subwindows = len(time_dict_subwindows)
                        for i in range(n_subwindows):
                            img = ee.Image(STM.toList(n_subwindows).get(i))
                            #img_time_desc = img.get('system:index').getInfo()
                            img_time_desc = list(time_dict_subwindows.keys())[i]
                            outfile = 'STM_' + desc + img_time_desc + '_' + SATELLITE
                            print("->  "+outfile)
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    else:
                        if isinstance(STM, ee.imagecollection.ImageCollection):
                            img = imgcol_to_img(STM, date_to_bandname=False)
                        else:
                            img = STM
                        outfile = 'STM_' + desc + time_desc + '_' + SATELLITE
                        print("->  "+outfile)
                        export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                if EXPORT_TABLE:
                    # need to construct new feature names, i.e. orginal feature name + reducer name
                    features_stm = [feature + '_' + r for r in STM_reducer for feature in FEATURES]
                    outfile = 'STM_' + desc + time_desc + '_' + SATELLITE
                    print("->  "+outfile)
                    export_table(img_or_imgcol=STM, feature=ROI_FEATCOL, reduceRegions=REDUCE_REGIONS, buffer=EXPORT_TABLE_BUFFER, reducer=EXPORT_TABLE_REDUCER, 
                                tileScale=EXPORT_TABLE_TILE_SCALE, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, px_res=PIX_RES, 
                                nodata=NODATA_VALUE, drop_nodata=EXPORT_TABLE_DROP_NODATA, features=features_stm, 
                                scale=DATATYPE_SCALE, dtype=DATATYPE)
                print("")
                print("---------------------------------------------------------")
            else:
                raise ValueError('STM ImageCollection not calculated. Set STM method in the dict / .yml file.')
        
        # --------------------------------------------------------------------
        # PBC
        if EXPORT_PBC:
            # check if PBC exists
            if PBC:
                print("---------------------------------------------------------")
                print("       Exporting PIXEL-BASED COMPOSITES (PBC) ")
                print("")
                if EXPORT_IMAGE:
                    # user requested separate export for each feature
                    if EXPORT_PER_FEATURE:
                        for feature in FEATURES:
                            img = imgcol_to_img(PBC.select(feature))
                            outfile = 'PBC_' + desc + time_desc + '_' + SATELLITE + '_' + feature
                            print("->  "+outfile)
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                                    px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                    scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    # user requested separate export for each time 
                    elif EXPORT_PER_TIME:
                        for i in range(PBC.size().getInfo()):
                            img = ee.Image(PBC.toList(PBC.size()).get(i))
                            img_time_desc = img.get('system:index').getInfo()
                            outfile = 'PBC_' + desc + img_time_desc + '_' + SATELLITE
                            print("->  "+outfile)
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    else:
                        img = imgcol_to_img(PBC)
                        outfile = 'PBC_' + desc + time_desc + '_' + SATELLITE
                        print("->  "+outfile)
                        export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                if EXPORT_TABLE:
                    outfile = 'PBC_' + desc + time_desc + '_' + SATELLITE
                    print("->  "+outfile)
                    export_table(img_or_imgcol=PBC, feature=ROI_FEATCOL, reduceRegions=REDUCE_REGIONS, buffer=EXPORT_TABLE_BUFFER, reducer=EXPORT_TABLE_REDUCER, 
                                tileScale=EXPORT_TABLE_TILE_SCALE, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, px_res=PIX_RES, 
                                nodata=NODATA_VALUE, drop_nodata=EXPORT_TABLE_DROP_NODATA, features=FEATURES, 
                                scale=DATATYPE_SCALE, dtype=DATATYPE)
                print("")
                print("---------------------------------------------------------")
            else:
                raise ValueError('PBC ImageCollection not calculated. Set PBC method in the dict / .yml file.')

        # --------------------------------------------------------------------
        # LSP
        if EXPORT_LSP:
            # check if LSP exists
            if LSP:
                print("---------------------------------------------------------")
                print("        Exporting Land Surface Phenology (LSP)")
                print("")
                if EXPORT_IMAGE:
                    outfile = 'LSP_' + desc + time_desc + '_' + SATELLITE
                    print("->  "+outfile)
                    img = prm['LSP_IMG']
                    export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, 
                            px_res=PIX_RES, crs=CRS, nodata=-9999, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                            scale=1, dtype='int16', export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                if EXPORT_TABLE:
                    outfile = 'LSP_' + desc + time_desc + '_' + SATELLITE
                    print("->  "+outfile)
                    imgcol = prm['LSP']
                    export_table(img_or_imgcol=imgcol, feature=ROI_FEATCOL, reduceRegions=REDUCE_REGIONS, buffer=EXPORT_TABLE_BUFFER, reducer=EXPORT_TABLE_REDUCER, 
                                tileScale=EXPORT_TABLE_TILE_SCALE, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_DIRECTORY, px_res=PIX_RES, 
                                nodata=-9999, drop_nodata=EXPORT_TABLE_DROP_NODATA, features=FEATURES, 
                                scale=1, dtype='int16')
                print("")
                print("---------------------------------------------------------")       
            else:
                raise ValueError('LSP ImageCollection not calculated. Set LSP method in the dict / .yml file.')

    # return dict
    return prm






