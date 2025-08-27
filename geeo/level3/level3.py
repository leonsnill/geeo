import ee
from geeo.level2.indices import rnb
from geeo.misc.spacetime import add_timeband, combine_reducers, generate_key
from geeo.level3.interpolation import tsi_rbf, tsi_rbf_duo, tsi_rbf_trio, tsi_linear_weighted
from geeo.level3.initimgcol import init_imgcol, init_and_join
from geeo.level3.stm import stm_initimgcol, stm_iterList
from geeo.level3.composite import composite_bap, composite_feature, composite_feature_invert, composite_nlcd

def run_level3(prm):
    """
    Run the level3 process with the given parameters.

    Args:
        params (dict): Dictionary obtained from level-2 processing.
    """

    # convert parameters to variables
    YEAR_MIN = prm.get('YEAR_MIN')
    YEAR_MAX = prm.get('YEAR_MAX')
    MONTH_MIN = prm.get('MONTH_MIN')
    MONTH_MAX = prm.get('MONTH_MAX')
    DOY_MIN = prm.get('DOY_MIN')
    DOY_MAX = prm.get('DOY_MAX')
    DATE_MIN = prm.get('DATE_MIN')
    DATE_MAX = prm.get('DATE_MAX')
    FEATURES = prm.get('FEATURES')
    FOLD_YEAR = prm.get('FOLD_YEAR')
    FOLD_MONTH = prm.get('FOLD_MONTH')
    FOLD_CUSTOM = prm.get('FOLD_CUSTOM')
    # TSI    
    TSI = prm.get('TSI')
    TSI_BASE_IMGCOL = prm.get('TSI_BASE_IMGCOL')
    INTERVAL = prm.get('INTERVAL')
    INTERVAL_UNIT = prm.get('INTERVAL_UNIT')
    INIT_JANUARY_1ST = prm.get('INIT_JANUARY_1ST')
    SIGMA1 = prm.get('SIGMA1')
    SIGMA2 = prm.get('SIGMA2')
    SIGMA3 = prm.get('SIGMA3')
    WIN1 = prm.get('WIN1')
    WIN2 = prm.get('WIN2')
    WIN3 = prm.get('WIN3')
    BW1 = prm.get('BW1')
    BW2 = prm.get('BW2')
    # STM
    STM = prm.get('STM')
    STM_BASE_IMGCOL = prm.get('STM_BASE_IMGCOL')
    STM_FOLDING = prm.get('STM_FOLDING')
    STM_FOLDING_LIST_ITER = prm.get('STM_FOLDING_LIST_ITER')
    # PBC
    PBC = prm.get('PBC')
    PBC_INVERT_QUALITY_METRIC = prm.get('PBC_INVERT_QUALITY_METRIC')
    PBC_BASE_IMGCOL = prm.get('PBC_BASE_IMGCOL')
    PBC_BAP_DOY_EQ_YEAR = prm.get('PBC_BAP_DOY_EQ_YEAR')
    PBC_BAP_MIN_CLOUDDISTANCE = prm.get('PBC_BAP_MIN_CLOUDDISTANCE')
    PBC_BAP_MAX_CLOUDDISTANCE = prm.get('PBC_BAP_MAX_CLOUDDISTANCE')
    PBC_BAP_WEIGHT_DOY = prm.get('PBC_BAP_WEIGHT_DOY')
    PBC_BAP_WEIGHT_YEAR = prm.get('PBC_BAP_WEIGHT_YEAR')
    PBC_BAP_WEIGHT_CLOUD = prm.get('PBC_BAP_WEIGHT_CLOUD')
    # General
    EXPORT_DESC_DETAIL_TIME = prm.get('EXPORT_DESC_DETAIL_TIME')

    # outname general settings
    # time description (already needed here to name STMs system_index property)
    # Time Of Interest (TOI)
    # get parameters
    # check if DATE_MIN and DATE_MAX are provided
    if DATE_MIN and DATE_MAX:
        YEAR_MIN, YEAR_MAX = int(str(DATE_MIN)[:4]), int(str(DATE_MAX)[:4])
    if EXPORT_DESC_DETAIL_TIME:
        if DATE_MIN and DATE_MAX:
            time_start = str(DATE_MIN)
            time_end = str(DATE_MAX)
            time_desc = time_start + '-' + time_end
        else:
            time_desc = generate_key([YEAR_MIN, YEAR_MAX], [MONTH_MIN, MONTH_MAX], [DOY_MIN, DOY_MAX])
    else:
        time_start = str(YEAR_MIN)
        time_end = str(YEAR_MAX)
        time_desc = time_start + '-' + time_end

    prm['TIME_DESC'] = time_desc    
    

    # TIME SERIES INTERPOLATION (TSI)
    if TSI:
        tsi_base_imgcol = prm.get(TSI_BASE_IMGCOL)
        # make sure imgcol exists
        if not tsi_base_imgcol:
            raise ValueError("TSI base ImageCollection not found.")
        tsi_methods = {
            '1RBF': {
                'window': [WIN1],
                'function': tsi_rbf(SIGMA1),
                'split_window': False
            },
            '2RBF': {
                'window': [WIN1, WIN2],
                'function': tsi_rbf_duo(SIGMA1, SIGMA2, BW1),
                'split_window': False
            },
            '3RBF': {
                'window': [WIN1, WIN2, WIN3],
                'function': tsi_rbf_trio(SIGMA1, SIGMA2, SIGMA3, BW1, BW2),
                'split_window': False
            },
            'WLIN': {
                'window': WIN1,
                'function': tsi_linear_weighted(FEATURES),
                'split_window': True
            }
        }
        # check if chosen method is available and retrieve parameters
        if TSI in tsi_methods:
            method = tsi_methods[TSI]
            WINDOW = method['window']
            INTER_FUN = method['function']
            split_window = method['split_window']
            # add timeband to collection if linear interpolation
            if TSI == 'WLIN':
                tsi_base_imgcol = tsi_base_imgcol.map(add_timeband)
        else:
            raise ValueError(f"Unknown interpolation method: {TSI}")
        # initialize image collection for interpolation
        imgcol_tsi = init_imgcol(
            imgcol=tsi_base_imgcol, interval=INTERVAL, interval_unit=INTERVAL_UNIT,
            timeband=True,
            join_window=WINDOW, january_first = INIT_JANUARY_1ST,
            split_window=split_window
        )

        # apply interpolation
        imgcol_tsi = ee.ImageCollection(imgcol_tsi.map(INTER_FUN))

        # return interpolation 
        prm['TSI'] = imgcol_tsi.select(FEATURES)

    # FOLDING
    # Global or folded STMs
    if (FOLD_YEAR) or (FOLD_MONTH) or (FOLD_CUSTOM):
        FOLD = True
    else:
        FOLD = False

    # SPECTRAL TEMPORAL METRICS (STMs)
    if STM:
        # ImageCollection from which to calculate STMs
        stm_base_imgcol = prm.get(STM_BASE_IMGCOL)
        # make sure imgcol exists
        if not stm_base_imgcol:
            raise ValueError("STM base ImageCollection not found.")
        stm_reducer = combine_reducers(STM)
    
        # check if folding is required
        if STM_FOLDING and FOLD:
            
            if STM_FOLDING_LIST_ITER:
                # Option 2) ee.List iteration
                # joins might not be the way to go here, list iteration and ImageCollection.fromImages is the way to go
                # https://gis.stackexchange.com/questions/340433/making-intra-annual-image-composites-for-a-series-of-years-in-google-earth-engin
                imgcol_stm = stm_iterList(prm, stm_base_imgcol.select(FEATURES), stm_reducer)
            else:
                # Option 1) Joins
                # initialize and join image collection
                imgcol_stm = init_and_join(prm, imgcol_secondary=stm_base_imgcol.select(FEATURES))
                imgcol_stm = ee.ImageCollection(imgcol_stm.map(stm_initimgcol(stm_reducer)))
                
            # return STM as ImageCollection
            prm['STM_reducer'] = STM
            prm['STM'] = imgcol_stm
        else:
            # calculate STMs globally for entire time series
            # set system:time_start property for table export
            img_first = stm_base_imgcol.first()
            img_stm = ee.Image(stm_base_imgcol.select(FEATURES).reduce(stm_reducer)).set('system:index', time_desc, 'system:time_start', img_first.get('system:time_start'))
            # return STM image
            prm['STM_reducer'] = STM
            prm['STM'] = img_stm

    # PIXEL-BASED COMPOSITING (PBC)
    if PBC:
        pbc_base_imgcol = prm.get(PBC_BASE_IMGCOL)
        # make sure imgcol exists
        if not pbc_base_imgcol:
            raise ValueError("PBC base ImageCollection not found.")
        # check which composite algorithm to use
        if PBC == 'BAP':
            # initialize and join image collection
            imgcol_pbc = init_and_join(prm, imgcol_secondary=pbc_base_imgcol)
            # Griffiths et al. 2013 BAP
            imgcol_pbc = ee.ImageCollection(
                imgcol_pbc.map(
                    composite_bap(
                        doy_offset_eq_year=PBC_BAP_DOY_EQ_YEAR, 
                        min_clouddistance=PBC_BAP_MIN_CLOUDDISTANCE, 
                        max_clouddistance=PBC_BAP_MAX_CLOUDDISTANCE, 
                        weight_doy=PBC_BAP_WEIGHT_DOY, 
                        weight_year=PBC_BAP_WEIGHT_YEAR, 
                        weight_cloud=PBC_BAP_WEIGHT_CLOUD)
                    )
                )
        
        elif PBC == 'MAX-NDVI':
            # initialize and join image collection
            imgcol_pbc = init_and_join(prm, imgcol_secondary=pbc_base_imgcol)
            # Maximum NDVI composite
            imgcol_pbc = ee.ImageCollection(
                imgcol_pbc.map(
                    composite_feature(band='NDVI')
                )
            )

        elif PBC in FEATURES:
            # initialize and join image collection
            imgcol_pbc = init_and_join(prm, imgcol_secondary=pbc_base_imgcol)
            if PBC_INVERT_QUALITY_METRIC:
                imgcol_pbc = ee.ImageCollection(
                    imgcol_pbc.map(
                        composite_feature_invert(band=PBC)
                    )
                )
            else:
                imgcol_pbc = ee.ImageCollection(
                    imgcol_pbc.map(
                        composite_feature(band=PBC)
                    )
                )
        
        elif PBC == 'MAX-RNB':
            # add RNB band to base imgcol
            pbc_base_imgcol = pbc_base_imgcol.map(rnb)
            # initialize and join image collection
            imgcol_pbc = init_and_join(prm, imgcol_secondary=pbc_base_imgcol)
            # Maximum RNB composite
            imgcol_pbc = ee.ImageCollection(
                imgcol_pbc.map(
                    composite_feature(band='RNB')
                )
            )
        
        elif PBC == 'NLCD':
            # initialize and join image collection
            imgcol_pbc = init_and_join(prm, imgcol_secondary=pbc_base_imgcol)
            # Maximum NLCD composite
            imgcol_pbc = ee.ImageCollection(imgcol_pbc.map(composite_nlcd))

        else:
            raise ValueError(f"Unknown composite method: {PBC}. Or {PBC} not in FEATURES.")

        prm['PBC'] = imgcol_pbc.select(FEATURES)

    # return dict
    return prm

# EOF
