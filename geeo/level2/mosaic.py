import ee
from geeo.misc.spacetime import days_to_milli, imgcol_dates_to_featcol

# mosaic images in imgcol using joins (way more efficient than list iteration)
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
    ).sort("system:time_start")
    return mosaics
