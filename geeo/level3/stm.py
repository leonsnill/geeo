import ee

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
        img_first = imgcol_filtered.first()
        # calculate STM
        img_stm = ee.Image(imgcol_filtered.reduce(reducers))
        return img_stm.set('system:index', fold_key, 'system:time_start', img_first.get('system:time_start'))
    return wrap


# EOF
