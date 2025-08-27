import ee
from geeo.misc.spacetime import construct_calendarRange_filter

def stm_initimgcol(reducers):
    def wrap(img):
        window1 = ee.ImageCollection.fromImages(img.get('window1'))
        key = img.get('key')
        img_stm = ee.Image(window1.reduce(reducers))
        return img_stm.set('system:index', key, 'system:time_start', img.get('system:time_start'))
    return wrap


# stm folding function for ee.List of date dictionaries
def folding_stm(imgcol, reducers):
    def wrap(fold):
        # retrieve elements for fold
        fold_filter = ee.Dictionary(fold).get('filter')
        fold_key = ee.Dictionary(fold).get('key')
        # filter imgcol
        imgcol_filtered = imgcol.filter(fold_filter)
        # get date of first image (needed for system:time_start property for table export)
        #img_first = imgcol_filtered.first()
        # calculate STM
        img_stm = ee.Image(imgcol_filtered.reduce(reducers))
        img_stm_size = img_stm.bandNames().size()
        return img_stm.set('system:index', fold_key, 'img_stm_size', img_stm_size)
    return wrap

def stm_iterList(prm, imgcol, reducers):
    fold_list = construct_calendarRange_filter(prm)
    imgcol_stm = ee.ImageCollection.fromImages(
        fold_list.map(folding_stm(imgcol, reducers))
    )
    imgcol_stm = imgcol_stm.filter(ee.Filter.gt('img_stm_size', 0))
    return imgcol_stm


# EOF
