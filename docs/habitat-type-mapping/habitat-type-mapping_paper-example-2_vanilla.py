import ee
ee.Authenticate()
ee.Initialize(project='project-name')

# -----------------------------------------------------------------------------
# input

roi = ee.Geometry.BBox(5.9, 44.5, 6.9, 45.3)
export_description = 'TSI_HABITAT_S2_NDVI'
export_crs = 'EPSG:3035'
export_scale_m = 10

# -----------------------------------------------------------------------------
# load Sentinel-2 Level-2A data

sentinel2 = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filter(ee.Filter.calendarRange(2021, 2024, 'year'))
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 75))
)

# link Cloud Score+ collection for masking
cloud_score_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
sentinel2 = sentinel2.linkCollection(cloud_score_plus, ['cs', 'cs_cdf'])

# -----------------------------------------------------------------------------
# preprocess Sentinel-2 data: rename, scale, mask, and calculate NDVI

def preprocess_image(image):
    image = image.select(
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'QA60', 'SCL', 'cs', 'cs_cdf'],
        ['AER', 'BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'NIR', 'RE4', 'WV', 'SW1', 'SW2', 'QA60', 'SCL', 'cs', 'cs_cdf']
    )

    # scale to surface reflectance values
    scaled_optical = (
        image.select(['AER', 'BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'NIR', 'RE4', 'SW1', 'SW2'])
        .multiply(0.0001)
    )
    image = image.addBands(scaled_optical, overwrite=True)

    # Cloud Score+ masking
    mask = image.select('cs').gte(0.6)

    # NDVI
    ndvi = image.normalizedDifference(['NIR', 'RED']).rename('NDVI')

    return (
        image
        .updateMask(mask)
        .resample('bilinear')
        .addBands(ndvi)
        .copyProperties(image, image.propertyNames())
    )

tss = sentinel2.map(preprocess_image).select(['NDVI']).sort('system:time_start')

# -----------------------------------------------------------------------------
# helper functions for interpolation

def days_to_milliseconds(days):
    return days * 1000 * 60 * 60 * 24

def build_interpolation_dates(image_collection, interval_days):
    first_image = ee.Image(image_collection.sort('system:time_start').first())
    last_image = ee.Image(image_collection.sort('system:time_start', False).first())

    first_date = ee.Date(first_image.get('system:time_start'))
    start_date = ee.Date.fromYMD(first_date.get('year'), 1, 1)
    end_date = ee.Date(last_image.get('system:time_start'))

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
    adds rbf weight to each band based on date difference to target date using a gaussian function
    '''
    rbf_weight = image.expression(
        'exp(-0.5 * pow(((delta / 86400000) / sigma), 2))',
        {
            'delta': ee.Number(image.get('delta')),
            'sigma': sigma_days
        }
    ).rename('rbf_weight')

    rbf_weight = rbf_weight.updateMask(image.select('NDVI').mask())
    weighted_ndvi = image.select('NDVI').multiply(rbf_weight).rename('weighted_ndvi')

    return image.addBands([rbf_weight, weighted_ndvi])


def interpolate_2rbf(target_image):
    '''
    implements a double RBF interpolation where the result is a weighted blend of a narrower and a wider time window 
    '''
    target_image = ee.Image(target_image)

    window1 = ee.ImageCollection.fromImages(target_image.get('window1')).map(
        lambda image: add_rbf_weight(ee.Image(image), 20)
    )
    window2 = ee.ImageCollection.fromImages(target_image.get('window2')).map(
        lambda image: add_rbf_weight(ee.Image(image), 40)
    )

    # first RBF interpolation
    ndvi_window1 = (
        window1.select('weighted_ndvi').reduce(ee.Reducer.sum())
        .divide(window1.select('rbf_weight').reduce(ee.Reducer.sum()))
        .rename('NDVI')
    )

    # second RBF interpolation
    ndvi_window2 = (
        window2.select('weighted_ndvi').reduce(ee.Reducer.sum())
        .divide(window2.select('rbf_weight').reduce(ee.Reducer.sum()))
        .rename('NDVI')
    )

    # adaptive blending based on image density in the narrower window
    nonzero1 = window1.select('rbf_weight').reduce(ee.Reducer.count()).unmask(0)
    weight1 = nonzero1.divide(4).clamp(0.0, 1.0)
    weight2 = ee.Image(1).subtract(weight1)

    interpolated = (
        ndvi_window1.multiply(weight1)
        .unmask(0)
        .add(ndvi_window2.multiply(weight2))
        .rename('NDVI')
    )

    return interpolated.set('system:time_start', target_image.get('system:time_start'))

# -----------------------------------------------------------------------------
# build regular interpolation dates and join source images to each target date

interpolation_dates = build_interpolation_dates(tss, interval_days=20) # 12
interpolation_dates = join_window(interpolation_dates, tss, window_days=20, join_key='window1')
interpolation_dates = join_window(interpolation_dates, tss, window_days=60, join_key='window2')

# -----------------------------------------------------------------------------
# apply 2RBF interpolation

tsi_ndvi = ee.ImageCollection(
    interpolation_dates.map(interpolate_2rbf)
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
    fileNamePrefix=export_description,
    region=roi,
    crs=export_crs,
    scale=export_scale_m,
    maxPixels=1e13,
    fileFormat='GeoTIFF',
    formatOptions={'noData': -9999},
)

task.start()