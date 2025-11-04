import ee
from geeo.misc.spacetime import construct_calendarRange_filter

def calc_nvo(imgcol):
    imgcol = imgcol.map(lambda img: img.addBands(
        ee.Image(ee.Image.constant(ee.Number.parse(img.date().format('YYYYMMdd'))).int().rename('DATE')).updateMask(img.mask().reduce(ee.Reducer.min()))
    ))
    img_nvo = ee.Image(imgcol.select('DATE').reduce(ee.Reducer.countDistinctNonNull())).rename('NVO')
    img = imgcol.first()
    return ee.Image(img_nvo).copyProperties(source=img).set('system:time_start', img.get('system:time_start'))

def folding_nvo(imgcol):
    def wrap(fold):
        # retrieve elements for fold
        fold_filter = ee.Dictionary(fold).get('filter')
        fold_key = ee.Dictionary(fold).get('key')
        imgcol_filtered = imgcol.filter(fold_filter)
        img_nvo = calc_nvo(imgcol_filtered)
        img_size = img_nvo.bandNames().size()
        return img_nvo.set('system:index', fold_key, 'img_size', img_size)
    return wrap

def nvo_iterList(prm, imgcol):
    fold_list = construct_calendarRange_filter(prm)
    imgcol = ee.ImageCollection.fromImages(
        fold_list.map(folding_nvo(imgcol))
    )
    imgcol = imgcol.filter(ee.Filter.gt('img_size', 0))
    return imgcol