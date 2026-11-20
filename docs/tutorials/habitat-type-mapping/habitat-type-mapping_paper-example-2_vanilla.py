import ee
ee.Authenticate()
ee.Initialize(project='project-name')  # replace with actual project name

# -----------------------------------------------------------------------------
# input

roi = ee.Geometry.BBox(6.26, 44.35, 6.57, 44.58)
year_min, year_max = 2010, 2024
export_description = 'TSI_HABITAT_EU_150-X029-Y029_2010-2024_NDVI'
export_scale_m = 30

# GLANCE Europe (Lambert Azimuthal Equal Area)
export_crs = (
    'PROJCS["BU MEaSUREs Lambert Azimuthal Equal Area - EU - V01",GEOGCS["GCS_WGS_1984",'
    'DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
    'UNIT["degree",0.0174532925199433]],PROJECTION["Lambert_Azimuthal_Equal_Area"],'
    'PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],'
    'PARAMETER["longitude_of_center",20],PARAMETER["latitude_of_center",55],UNIT["meter",1.0]]'
)

# interpolation settings (1RBF)
interval_days = 16
sigma_days = 16
window_days = 64

# -----------------------------------------------------------------------------
# load Landsat 7, 8, 9 Level-2 Collection 2 Tier 1 data

# band names differ between Landsat 7 and Landsat 8/9
bands_l7 = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL', 'QA_RADSAT']
bands_l89 = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL', 'QA_RADSAT']
bands_new = ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2', 'QA_PIXEL', 'QA_RADSAT']

def load_landsat(collection_id, bands):
    return (
        ee.ImageCollection(collection_id)
        .filterBounds(roi)
        .filter(ee.Filter.calendarRange(year_min, year_max, 'year'))
        .filter(ee.Filter.lte('CLOUD_COVER_LAND', 50))
        .filter(ee.Filter.gte('GROUND_CONTROL_POINTS_MODEL', 1))
        .select(bands, bands_new)
    )

landsat = (
    load_landsat('LANDSAT/LC09/C02/T1_L2', bands_l89)
    .merge(load_landsat('LANDSAT/LC08/C02/T1_L2', bands_l89))
    .merge(load_landsat('LANDSAT/LE07/C02/T1_L2', bands_l7))
)

# -----------------------------------------------------------------------------
# preprocess Landsat data: scale, mask, and calculate NDVI

def preprocess_image(image):
    # scale to surface reflectance values
    optical = image.select(['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2']).multiply(0.0000275).add(-0.2)
    image = image.addBands(optical, overwrite=True)

    # QA_PIXEL bits: 0 fill, 1 dilated cloud, 3 cloud, 4 cloud shadow
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 + 2 + 8 + 16).eq(0)

    # cloud and cirrus confidence (bits 8-9 and 14-15) below medium
    cloud_conf = qa.rightShift(8).bitwiseAnd(3).lt(2)
    cirrus_conf = qa.rightShift(14).bitwiseAnd(3).lt(2)

    # radiometric saturation and implausible reflectance values
    saturation = image.select('QA_RADSAT').eq(0)
    valid_min = optical.reduce(ee.Reducer.min()).gt(0)
    valid_max = optical.reduce(ee.Reducer.max()).lt(1)

    mask = mask.And(cloud_conf).And(cirrus_conf).And(saturation).And(valid_min).And(valid_max)

    ndvi = image.normalizedDifference(['NIR', 'RED']).rename('NDVI')

    return (
        ndvi
        .updateMask(mask)
        .copyProperties(image, image.propertyNames())
    )

tss = landsat.map(preprocess_image).sort('system:time_start')

# -----------------------------------------------------------------------------
# create time series mosaic (i.e. duplicate observations due to scene/sensor/tile overlap)

def imgcol_dates_to_featcol(imgcol, format='YYYYMMdd'):
    dates = imgcol.map(
        lambda img: ee.Feature(None, {'system:time_start': img.date().millis(),
                                      'YYYYMMDD': img.date().format(format)})
    )
    return ee.FeatureCollection(dates)

# mosaic images in imgcol using joins (way more efficient than list iteration)
def mosaic_imgcol(imgcol):
    unique_dates = imgcol_dates_to_featcol(imgcol).distinct('YYYYMMDD')  # unique YYYYMMDD
    newcol = ee.ImageCollection(
        ee.Join.saveAll('images').apply(
            primary=unique_dates, secondary=imgcol,
            condition=ee.Filter.maxDifference(
                difference=0.5*1000*60*60*24,  # 1/2 day to milliseconds
                leftField='system:time_start', rightField='system:time_start'
            )
        )
    )
    mosaics = newcol.map(
        lambda x: ee.ImageCollection(ee.List(x.get('images'))).mosaic().set('system:time_start', x.get('system:time_start'))
    ).sort("system:time_start")
    return mosaics

tsm = mosaic_imgcol(tss)

# -----------------------------------------------------------------------------
# helper functions for interpolation

def days_to_milliseconds(days):
    return days * 1000 * 60 * 60 * 24

def build_interpolation_dates(interval_days):
    # regular dates from January 1st of the first year until December 31st of the last year
    start_date = ee.Date.fromYMD(year_min, 1, 1)
    end_date = ee.Date.fromYMD(year_max, 12, 31)

    delta_days = end_date.difference(start_date, 'day')
    day_offsets = ee.List.sequence(0, delta_days, interval_days)

    def make_target_image(day_offset):
        day_offset = ee.Number(day_offset)
        target_date = start_date.advance(day_offset, 'day')
        return ee.Image().set({
            'system:index': day_offset.format('%d'),
            'system:time_start': target_date.millis()
        })

    return ee.ImageCollection.fromImages(day_offsets.map(make_target_image))


def join_window(target_collection, source_collection, window_days, join_key):
    max_diff_filter = ee.Filter.maxDifference(
        difference=days_to_milliseconds(window_days),
        leftField='system:time_start',
        rightField='system:time_start'
    )

    join = ee.Join.saveAll(
        matchesKey=join_key,
        measureKey='delta',
        ordering='system:time_start',
        ascending=True
    )

    return ee.ImageCollection(join.apply(
        primary=target_collection,
        secondary=source_collection,
        condition=max_diff_filter
    ))


def add_rbf_weight(image, sigma_days):
    '''
    weights the image by a gaussian function of the date difference to the target date
    and adds the weight as band (masked where the image is masked)
    '''
    rbf_weight = image.expression(
        'exp(-0.5 * pow(((delta / 86400000) / sigma), 2))',
        {
            'delta': ee.Number(image.get('delta')),
            'sigma': sigma_days
        }
    )
    common_mask = image.mask().reduce(ee.Reducer.min())
    rbf_weight = rbf_weight.updateMask(common_mask)
    weighted = image.multiply(rbf_weight)

    return weighted.addBands(rbf_weight.rename('rbf_weight'))


def interpolate_1rbf(target_image):
    '''
    single RBF interpolation: weighted mean of all observations within the window
    '''
    target_image = ee.Image(target_image)

    window = ee.ImageCollection.fromImages(target_image.get('window1')).map(
        lambda image: add_rbf_weight(ee.Image(image), sigma_days)
    )

    reduced = window.reduce(ee.Reducer.sum())
    interpolated = (
        reduced.select('NDVI_sum')
        .divide(reduced.select('rbf_weight_sum'))
        .rename('NDVI')
    )

    return interpolated.set('system:time_start', target_image.get('system:time_start'))

# -----------------------------------------------------------------------------
# build regular interpolation dates and join source images to each target date

interpolation_dates = build_interpolation_dates(interval_days)
interpolation_dates = join_window(interpolation_dates, tsm, window_days, join_key='window1')

# -----------------------------------------------------------------------------
# apply 1RBF interpolation

tsi_ndvi = ee.ImageCollection(
    interpolation_dates.map(interpolate_1rbf)
).sort('system:time_start')

# -----------------------------------------------------------------------------
# stack the interpolated NDVI time series into one multiband image
# band names will be NDVI_YYYYMMdd

date_band_names = tsi_ndvi.aggregate_array('system:time_start').map(
    lambda millis: ee.String('NDVI_').cat(ee.Date(millis).format('YYYYMMdd'))
)

tsi_ndvi_image = ee.Image(tsi_ndvi.toBands()).rename(date_band_names)

# -----------------------------------------------------------------------------
# export image as Earth Engine asset

export_image = (
    tsi_ndvi_image
    .multiply(10000) # scale continuous values to preserve decimals in integer format
    .unmask(-9999) # set no data value for masked pixels
    .toInt16() # cast to disc space efficient int16 format
)

task = ee.batch.Export.image.toDrive(
    image=export_image,
    description=export_description,
    region=roi,
    crs=export_crs,
    scale=export_scale_m,
    maxPixels=1e13,
)

task.start()
