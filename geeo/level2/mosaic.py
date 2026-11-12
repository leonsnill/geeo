import ee
from geeo.misc.spacetime import days_to_milli, imgcol_dates_to_featcol

# mosaic images in imgcol using joins (way more efficient than list iteration)
'''
def mosaic_imgcol(imgcol):
    unique_dates = imgcol_dates_to_featcol(imgcol).distinct('YYYYMMDD')  # unique YYYYMMDD
    newcol = ee.ImageCollection(
        ee.Join.saveAll('images').apply(
            primary=unique_dates, secondary=imgcol,
            condition=ee.Filter.maxDifference(  
                difference=days_to_milli(0.5),
                leftField='system:time_start', rightField='system:time_start'
            )
        )
    )
    mosaics = newcol.map(
        lambda x: ee.ImageCollection(ee.List(x.get('images'))).mosaic().set('system:time_start', x.get('system:time_start'))
    )#.sort("system:time_start")  # sort is very costly
    return mosaics
'''

def mosaic_imgcol(imgcol, date_format='YYYYMMdd'):
    imgcol_dated = imgcol.map(
        lambda img: img.set('date', img.date().format(date_format))
    )
    distinct_dates = imgcol_dated.distinct(['date'])
    join = ee.Join.saveAll(matchesKey='images')
    joined = join.apply(
        primary=distinct_dates,
        secondary=imgcol_dated,
        condition=ee.Filter.equals(leftField='date', rightField='date')
    )
    mosaics = ee.ImageCollection(joined).map(
        lambda img: ee.ImageCollection.fromImages(img.get('images'))
                    .mosaic()
                    .copyProperties(img, ['system:time_start', 'date'])
    )
    return mosaics
