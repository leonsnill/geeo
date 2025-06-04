# ----------------------------------------------------------------------------
import ee
from geeo.utils import load_parameters, merge_parameters, load_blueprint
from geeo.misc.formatting import scale_and_dtype
from geeo.misc.spacetime import imgcol_to_img, create_roi, reduction

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


def export_img(img, outname='GEE_IMG', out_location='Drive', out_dir=None, 
               px_res=None, region=None, dimensions=None, crs=None, crsTransform=None, 
               nodata=None, scale=1, fileDimensions=None, dtype='double', export_bandnames=False):

    # scaling, dtype and nodata
    img = scale_and_dtype(img, scale=scale, dtype=dtype, nodata=nodata)

    # image dimensions and tiling: shardSize & fileDimensions
    # alternatively get fileDimensions from dimensions as an tuple of (width, height):
    if fileDimensions is None and dimensions is not None:
        dim_width, dim_height = tuple([int(x) for x in dimensions.split('x')])
        fileDimensions = ((round(dim_width/256)*256)+256, (round(dim_height/256)*256)+256)  # 256 is the shardSize

    # if crsTransform is not None, set px_res to None
    if crsTransform is not None:
        px_res = None

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
            assetId=out_dir,
            maxPixels=1e13
        )
    else:
        raise ValueError('Invalid out_location')
    
    out_image_process = ee.batch.Task.start(out_image)

    # separate bandname export
    if export_bandnames:
        bandnames = ee.FeatureCollection(
            ee.List(img.bandNames()).map(
            lambda x: ee.Feature(None, {'name': x})
            )
        )
        if out_location == 'Drive':
            out_table = ee.batch.Export.table.toDrive(collection=bandnames,
                                            description=outname+'_bandnames',
                                            fileFormat='CSV',
                                            folder=out_dir)
        elif out_location == 'Asset':
            out_table = ee.batch.Export.table.toDrive(collection=bandnames,
                                            description=outname+'_bandnames',
                                            fileFormat='CSV',
                                            folder=out_dir)
        else:
            raise ValueError('Invalid out_location')
        
        out_table_process = ee.batch.Task.start(out_table)
    else:
        out_table_process = None
    
    return out_image_process, out_table_process


def export_table(img_or_imgcol, feature, reduceRegions=True, buffer=None, reducer='first',
                 crs=None, tileScale=1, outname='GEE_IMG', out_location='Drive', out_dir=None,
                 px_res=30, nodata=0, drop_nodata=False, features=None, scale=1, dtype='double'):
    
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
            assetId=out_dir
        )
    else:
        raise ValueError('Invalid out_location')
    
    return ee.batch.Task.start(out_table)



# joins are not the way to go here, list iteration and ImageCollection.fromImages is the way to go
# https://gis.stackexchange.com/questions/340433/making-intra-annual-image-composites-for-a-series-of-years-in-google-earth-engin
# STM
    # No fold (standard) -> ImageCollection -> STM Image -> Export Image / Table (default combined, but can be per feature)
    # Fold -> Init ImageCollection -> map -> STM Image per Fold -> Export Image / Table per Fold (default combined, but can be per feature)
# PBC
    # No fold (standard) -> ImageCollection -> PBC Image -> Export Image / Table (default combined, but can be per feature)
    # Fold -> Init ImageCollection -> map -> PBC Image per Fold -> Export Image / Table per Fold (default combined, but can be per feature)


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
    # TREND
    # Export settings
    PIX_RES = prm.get('PIX_RES')
    CRS = prm.get('CRS')
    EXPORT_TABLE_TILE_SCALE = prm.get('EXPORT_TABLE_TILE_SCALE')
    DATATYPE = prm.get('DATATYPE')
    DATATYPE_SCALE = prm.get('DATATYPE_SCALE')
    NODATA_VALUE = prm.get('NODATA_VALUE')
    EXPORT_IMAGE = prm.get('EXPORT_IMAGE')
    EXPORT_TABLE = prm.get('EXPORT_TABLE')
    EXPORT_DESC = prm.get('EXPORT_DESC')
    EXPORT_LOCATION = prm.get('EXPORT_LOCATION')
    EXPORT_FOLDER = prm.get('EXPORT_FOLDER')
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
    
    # check output from level-2 and overwrite if necessary
    IMG_DIMENSIONS = prm.get('IMG_DIMENSIONS')
    CRS_TRANSFORM = prm.get('CRS_TRANSFORM')
    if CRS != 'EPSG:4326':
        ROI_BBOX = None
        prm['ROI_BBOX'] = ROI_BBOX
    else:
        IMG_DIMENSIONS = None
        CRS_TRANSFORM = None
        prm['IMG_DIMENSIONS'] = IMG_DIMENSIONS
        prm['CRS_TRANSFORM'] = CRS_TRANSFORM
    
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
                        export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                
                # user requested separate export for each time
                elif EXPORT_PER_TIME:
                    for i in range(TSS.size().getInfo()):
                        img = ee.Image(TSS.toList(TSS.size()).get(i)).select(FEATURES)  # .select() because 'mask' still in TSS
                        img_time_desc = img.date().format('YYYYMMdd').getInfo()
                        outfile = 'TSS_' + desc + img_time_desc + '_' + SATELLITE  # since this is a single date image "TSS" is a weird wording
                        print("->  "+outfile)
                        export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                            px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                            scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                
                # user requested single export for entire TSS
                else:
                    img = imgcol_to_img(TSS, date_to_bandname=False) # False, because duplicate feature+date exist in TSS 
                    outfile = 'TSS_' + desc + time_desc + '_' + SATELLITE
                    print("->  "+outfile)
                    export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                            px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                            scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
            if EXPORT_TABLE:
                outfile = 'TSS_' + desc + time_desc + '_' + SATELLITE
                print("->  "+outfile)
                export_table(img_or_imgcol=TSS, feature=ROI_FEATCOL, reduceRegions=REDUCE_REGIONS, buffer=EXPORT_TABLE_BUFFER, reducer=EXPORT_TABLE_REDUCER, 
                            tileScale=EXPORT_TABLE_TILE_SCALE, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, px_res=PIX_RES, 
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
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                                    px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                    scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    # user requested separate export for each time
                    elif EXPORT_PER_TIME:
                        for i in range(TSM.size().getInfo()):
                            img = ee.Image(TSM.toList(TSM.size()).get(i))
                            img_time_desc = img.date().format('YYYYMMdd').getInfo()
                            outfile = 'TSM_' + desc + img_time_desc + '_' + SATELLITE
                            print("->  "+outfile)
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    else:
                        img = imgcol_to_img(TSM)
                        outfile = 'TSM_' + desc + time_desc + '_' + SATELLITE
                        print("->  "+outfile)
                        export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                if EXPORT_TABLE:
                    outfile = 'TSM_' + desc + time_desc + '_' + SATELLITE
                    print("->  "+outfile)
                    export_table(img_or_imgcol=TSM, feature=ROI_FEATCOL, reduceRegions=REDUCE_REGIONS, buffer=EXPORT_TABLE_BUFFER, reducer=EXPORT_TABLE_REDUCER, 
                                tileScale=EXPORT_TABLE_TILE_SCALE, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, px_res=PIX_RES, 
                                nodata=NODATA_VALUE, drop_nodata=EXPORT_TABLE_DROP_NODATA, features=FEATURES, 
                                scale=DATATYPE_SCALE, dtype=DATATYPE)
                print("")
                print("---------------------------------------------------------")
            else:
                raise ValueError('TSM ImageCollection not calculated. Set *TSM: true* in the dict / .yml file.')
        
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
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                                    px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                    scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    # user requested separate export for each time
                    elif EXPORT_PER_TIME:
                        for i in range(TSI.size().getInfo()):
                            img = ee.Image(TSI.toList(TSI.size()).get(i))
                            img_time_desc = img.date().format('YYYYMMdd').getInfo()
                            outfile = 'TSI_' + desc + img_time_desc + '_' + SATELLITE
                            print("->  "+outfile)
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    else:
                        img = imgcol_to_img(TSI)
                        outfile = 'TSI_' + desc + time_desc + '_' + SATELLITE
                        print("->  "+outfile)
                        export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                if EXPORT_TABLE:
                    outfile = 'TSI_' + desc + time_desc + '_' + SATELLITE
                    print("->  "+outfile)
                    export_table(img_or_imgcol=TSI, feature=ROI_FEATCOL, reduceRegions=REDUCE_REGIONS, buffer=EXPORT_TABLE_BUFFER, reducer=EXPORT_TABLE_REDUCER, 
                                tileScale=EXPORT_TABLE_TILE_SCALE, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, px_res=PIX_RES, 
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
                            feature_stm = [feature + '_' + r for r in STM_reducer]
                            if isinstance(STM, ee.imagecollection.ImageCollection):
                                img = imgcol_to_img(STM.select(feature_stm), date_to_bandname=False)
                            else:
                                img = STM.select(feature_stm)
                            outfile = 'STM_' + desc + time_desc + '_' + SATELLITE + '_' + feature
                            print("->  "+outfile)
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                                    px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                    scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    # user requested separate export for each time (only possible if STM is an ImageCollection, i.e. FOLDING)
                    elif EXPORT_PER_TIME and isinstance(STM, ee.imagecollection.ImageCollection):
                        # time description list
                        for i in range(STM.size().getInfo()):
                            img = ee.Image(STM.toList(STM.size()).get(i))
                            img_time_desc = img.get('system:index').getInfo()
                            outfile = 'STM_' + desc + img_time_desc + '_' + SATELLITE
                            print("->  "+outfile)
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    else:
                        if isinstance(STM, ee.imagecollection.ImageCollection):
                            img = imgcol_to_img(STM, date_to_bandname=False)
                        else:
                            img = STM
                        outfile = 'STM_' + desc + time_desc + '_' + SATELLITE
                        print("->  "+outfile)
                        export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                if EXPORT_TABLE:
                    outfile = 'STM_' + desc + time_desc + '_' + SATELLITE
                    print("->  "+outfile)
                    export_table(img_or_imgcol=STM, feature=ROI_FEATCOL, reduceRegions=REDUCE_REGIONS, buffer=EXPORT_TABLE_BUFFER, reducer=EXPORT_TABLE_REDUCER, 
                                tileScale=EXPORT_TABLE_TILE_SCALE, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, px_res=PIX_RES, 
                                nodata=NODATA_VALUE, drop_nodata=EXPORT_TABLE_DROP_NODATA, features=FEATURES, 
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
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                                    px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                    scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    # user requested separate export for each time 
                    elif EXPORT_PER_TIME:
                        for i in range(PBC.size().getInfo()):
                            img = ee.Image(PBC.toList(PBC.size()).get(i))
                            img_time_desc = img.get('system:index').getInfo()
                            outfile = 'PBC_' + desc + img_time_desc + '_' + SATELLITE
                            print("->  "+outfile)
                            export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                    
                    else:
                        img = imgcol_to_img(PBC)
                        outfile = 'PBC_' + desc + time_desc + '_' + SATELLITE
                        print("->  "+outfile)
                        export_img(img=img, region=ROI_BBOX, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, 
                                px_res=PIX_RES, crs=CRS, nodata=NODATA_VALUE, crsTransform=CRS_TRANSFORM, dimensions=IMG_DIMENSIONS,
                                scale=DATATYPE_SCALE, dtype=DATATYPE, export_bandnames=EXPORT_BANDNAMES_AS_CSV)
                if EXPORT_TABLE:
                    outfile = 'PBC_' + desc + time_desc + '_' + SATELLITE
                    print("->  "+outfile)
                    export_table(img_or_imgcol=PBC, feature=ROI_FEATCOL, reduceRegions=REDUCE_REGIONS, buffer=EXPORT_TABLE_BUFFER, reducer=EXPORT_TABLE_REDUCER, 
                                tileScale=EXPORT_TABLE_TILE_SCALE, outname=outfile, out_location=EXPORT_LOCATION, out_dir=EXPORT_FOLDER, px_res=PIX_RES, 
                                nodata=NODATA_VALUE, drop_nodata=EXPORT_TABLE_DROP_NODATA, features=FEATURES, 
                                scale=DATATYPE_SCALE, dtype=DATATYPE)
                print("")
                print("---------------------------------------------------------")
            else:
                raise ValueError('PBC ImageCollection not calculated. Set PBC method in the dict / .yml file.')


    # return dict
    return prm






