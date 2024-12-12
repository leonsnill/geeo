import ee
ee.Initialize()
from geeo.misc.spacetime import add_timeband, days_to_milli, add_time_properties_to_img, construct_time_subwindows


# combine functions to create imgcol from dict and join
def init_and_join(prm, imgcol_secondary=None):
    # create time_dict from parameter file / dict
    time_dict_raw = {
        "YEAR_MIN": prm.get('YEAR_MIN'),
        "YEAR_MAX": prm.get('YEAR_MAX'),
        "MONTH_MIN": prm.get('MONTH_MIN'),
        "MONTH_MAX": prm.get('MONTH_MAX'),
        "DOY_MIN": prm.get('DOY_MIN'),
        "DOY_MAX": prm.get('DOY_MAX'),
        "DATE_MIN": prm.get('DATE_MIN'),
        "DATE_MAX": prm.get('DATE_MAX'),
        "FOLD_YEAR": prm.get('FOLD_YEAR'),
        "FOLD_MONTH": prm.get('FOLD_MONTH'),
        "FOLD_CUSTOM": prm.get('FOLD_CUSTOM')
    }
    # create time dict from time_dict_raw
    time_dict = construct_time_subwindows(**time_dict_raw)
    # initialize imgcol from time_dict
    imgcol = init_imgcol_from_time(time_dict)
    # add properties to secondary imgcol
    imgcol_secondary = imgcol_secondary.map(add_time_properties_to_img)
    # join imgcols
    imgcol = join_collections_from_dict(imgcol, imgcol_secondary, time_dict)
    return imgcol


# -------------------------------------------------------------------------------
#                   Artificial ImageCollecions intialization
# -------------------------------------------------------------------------------

def init_imgcol_from_time(time_dict):
    init_image = ee.Image()

    time_list = ee.List([
        ee.Dictionary({
            'key': key,
            'time_start': value['time_start'],
            'year': ee.List(value['year']) if value['year'] is not None else ee.List([]),
            'month': ee.List(value['month']) if value['month'] is not None else ee.List([]),
            'doy': ee.List(value['doy']) if value['doy'] is not None else ee.List([]),
            'date': ee.List(value['date']) if value['date'] is not None else ee.List([]),
            'date_range': ee.DateRange(ee.Date(value['date'][0]), ee.Date(value['date'][1])) if value['date'] is not None else None,
            'year_center': ee.Number(value['year_center'] if value['year_center'] is not None else ee.Number(0)),        
            'doy_center': ee.Number(value['doy_center'] if value['doy_center'] is not None else ee.Number(0)),           
            'year_offset': ee.Number(value['year_offset'] if value['year_offset'] is not None else ee.Number(0)),            
            'doy_offset': ee.Number(value['doy_offset'] if value['doy_offset'] is not None else ee.Number(0))    
        })
        for key, value in time_dict.items()
    ])

    def create_image_from_dict(dict):
        dict = ee.Dictionary(dict)
        image = init_image.set({
            'system:index': dict.get('key'),
            'system:time_start': dict.get('time_start'),
            'year': dict.get('year'),
            'month': dict.get('month'),
            'doy': dict.get('doy'),
            'date': dict.get('date'),
            'year_center': dict.get('year_center'),
            'doy_center': dict.get('doy_center'),
            'year_offset': dict.get('year_offset'),
            'doy_offset': dict.get('doy_offset')
        })

        if dict.contains('date_range'):
            image = image.set('date_range', dict.get('date_range'))
        return image

    images = time_list.map(create_image_from_dict)
    init_imgcol = ee.ImageCollection.fromImages(images)

    return init_imgcol


def join_collections_from_dict(new_img_col, existing_img_col, time_dict):

    # define possible filters
    filter_year = ee.Filter.listContains(leftField='year', rightField='year')
    filter_month = ee.Filter.listContains(leftField='month', rightField='month')
    filter_doy = ee.Filter.listContains(leftField='doy', rightField='doy')
    filter_date = ee.Filter.dateRangeContains(leftField='date_range', rightField='system:time_start')

    # determine which filters to use based on the time_dict
    combined_filters = []
    for value in time_dict.values():
        if value['year'] is not None:
            combined_filters.append(filter_year)
        if value['month'] is not None:
            combined_filters.append(filter_month)
        if value['doy'] is not None:
            combined_filters.append(filter_doy)
        if value['date'] is not None:
            combined_filters = [filter_date]
            break  # date range filter takes precedence, no need to combine other filters

    # combine the selected filters
    if combined_filters:
        if len(combined_filters) == 1:
            combined_filter = combined_filters[0]
        else:
            combined_filter = ee.Filter.And(*combined_filters)
    else:
        raise ValueError("No valid filters found based on the provided time_dict.")

    # perform the join
    join = ee.Join.saveAll('window1', ordering = 'system:time_start', ascending = True)
    joined_col = join.apply(new_img_col, existing_img_col, combined_filter)

    return ee.ImageCollection(joined_col)


# create empty ImageCollection from exisiting imgcol and interval
def init_imgcol(imgcol, interval=5, interval_unit='day', day_offset=0,
                timeband=False, join_window=None, split_window=False,
                january_first=True, type_key='interpolated'):
    """
    Creates an empty / artificial ImageCollection based on a user-defined time input. 
    The resulting collection holds the specified bands (empty) and encompasses the specified time window.

    Arguments:
        imgcol:         (ee.ImageCollection) Optional blueprint to copy band names from. If None, no bands are copied.
        interval:       (int) Time step interval in days.
        day_offset:     (int) Starting day offset.
        timeband:       (bool) Whether to add a timeband to the images.
        join_window:    (int or list) Window size for joining images.
        split_window:   (bool) Whether to split the join window.
        january_first:  (bool) Whether to start the time interval from January 1st.
        type_key:       (str) Type key for the images.
        years:          (list) List of years to initialize the collection with.
        doys:           (list) List of Day of Year (DOY) to initialize the collection with.
    """
    
    # evaluate if imgcol is provided; if yes, extract time information
    if imgcol:
        if january_first:  # set start to january 1st
            time_start = ee.Date(ee.Date(ee.Image(imgcol.sort('system:time_start', ascending=True).first()).get('system:time_start')).format("YYYY"))
        else:  # set start to first image in imgcol
            time_start = ee.Date(ee.Image(imgcol.sort('system:time_start', ascending=True).first()).get('system:time_start'))
        time_end = ee.Date(ee.Image(imgcol.sort('system:time_start', ascending=False).first()).get('system:time_start'))
        
        # total number of days in interval (interval defined by imgcol)
        delta_days = time_end.difference(time_start, 'day')
    else:
        raise ValueError("No ImageCollection provided to initialize from.")

    # initialize empty image
    initImage = ee.Image()

    n_days_to_interpolate = ee.List.sequence(day_offset, delta_days, interval)

    def advance_interval(interval):
        img = initImage.set({'system:index': ee.Number(interval).format('%d'),
                            'system:time_start': time_start.advance(interval, interval_unit).millis(),
                            'type': type_key})
        return img
    
    initImages = n_days_to_interpolate.map(advance_interval)

    # create ImageCollection from Images
    init_imgcol = ee.ImageCollection.fromImages(initImages)

    # add timeband
    if timeband:
        init_imgcol = init_imgcol.map(add_timeband)

    # join windows if requested
    if imgcol is not None and join_window is not None:
        if isinstance(join_window, int):
            if split_window:
                init_imgcol = join_imgs_two_windows(prim_imgcol=init_imgcol, sec_imgcol=imgcol, 
                                                window_days=join_window)
            else:
                init_imgcol = join_imgs_one_window(init_imgcol, window_days=join_window)
        else:
            init_imgcol = join_imgs_windows(prim_imgcol=init_imgcol, sec_imgcol=imgcol, 
                                        windows=join_window)
    
    init_imgcol = ee.ImageCollection(init_imgcol.filter(ee.Filter.eq('type', type_key)))

    return init_imgcol


# join collections (add imgs from one imgcol as property to imgs of another imgcol)
# 1) two bracketing and exclusive windows
def join_imgs_two_windows(prim_imgcol, sec_imgcol, window_days=30):
    window = days_to_milli(window_days)
    maxDiffFilter = ee.Filter.maxDifference(
        difference = window, 
        leftField = 'system:time_start',
        rightField = 'system:time_start'
    )
    lessEqFilter = ee.Filter.lessThanOrEquals(  # lessThanOrEquals
        leftField = 'system:time_start',
        rightField = 'system:time_start'
    )
    greaterEqFilter = ee.Filter.greaterThanOrEquals(  # greaterThanOrEquals
        leftField = 'system:time_start',
        rightField = 'system:time_start'
    )
    filter_after = ee.Filter.And(maxDiffFilter, lessEqFilter)
    filter_before = ee.Filter.And(maxDiffFilter, greaterEqFilter)

    join_after = ee.Join.saveAll(matchesKey = 'after', measureKey = 'delta',
                            ordering = 'system:time_start',
                            ascending = False)
    join_before = ee.Join.saveAll(matchesKey = 'before', measureKey = 'delta',
                            ordering = 'system:time_start',
                            ascending = True)
    
    prim_imgcol = join_after.apply(primary = prim_imgcol,
                             secondary = sec_imgcol,
                             condition = filter_after)
    prim_imgcol = join_before.apply(primary = prim_imgcol,
                             secondary = sec_imgcol,
                             condition = filter_before)
    # ee.ImageCollection(prim_imgcol.filter(ee.Filter.eq('type', 'interpolated')))  # imgcol
    return prim_imgcol


# 2) single bracketing (inclusive) window
def join_imgs_one_window(imgcol, window_days=30, join_key='window1'):
    window = days_to_milli(window_days)
    maxDiffFilter = ee.Filter.maxDifference(
        difference = window, 
        leftField = 'system:time_start',
        rightField = 'system:time_start'
    )
    join = ee.Join.saveAll(matchesKey = join_key, measureKey = 'delta',
                           ordering = 'system:time_start',
                           ascending = True)
    imgcol_joined = join.apply(primary = imgcol,
                               secondary = imgcol,
                               condition = maxDiffFilter)
    return imgcol_joined
    #return ee.ImageCollection(imgcol_joined.filter(ee.Filter.eq('type', 'interpolated')))  # imgcol


# change imcol to original imgcol
# 2) single bracketing (inclusive) window
def join_imgs_windows(prim_imgcol, sec_imgcol, windows=[12, 24, 48], keys=['window1', 'window2', 'window3']):
    for i, w in enumerate(windows):
        window = days_to_milli(w)
        maxDiffFilter = ee.Filter.maxDifference(
            difference = window, 
            leftField = 'system:time_start',
            rightField = 'system:time_start'
        )
        join = ee.Join.saveAll(matchesKey = keys[i], measureKey = 'delta')
        prim_imgcol = join.apply(primary = prim_imgcol,
                                secondary = sec_imgcol,
                                condition = maxDiffFilter)

    #return ee.ImageCollection(imgcol_joined.filter(ee.Filter.eq('type', 'interpolated')))  # imgcol
    return prim_imgcol

