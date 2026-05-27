import ee
ee.Authenticate()
ee.Initialize(project='project-name')

# -----------------------------------------------------------------------------
# input

roi = ee.Geometry.BBox(5.9, 44.5, 6.9, 45.3)
export_description = 'HABITAT_METRICS'
export_crs = 'EPSG:3035'
export_scale_m = 30

# -----------------------------------------------------------------------------
# load landsat data

landsat_8 = (
    ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(roi)
    .filter(ee.Filter.calendarRange(2022, 2024, 'year'))
    .filter(ee.Filter.lte('CLOUD_COVER_LAND', 75))
)

landsat_9 = (
    ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(roi)
    .filter(ee.Filter.calendarRange(2022, 2024, 'year'))
    .filter(ee.Filter.lte('CLOUD_COVER_LAND', 75))
)

# merge collections 
landsat = landsat_8.merge(landsat_9).sort('system:time_start')

# -----------------------------------------------------------------------------
# preprocess landsat data: scale, mask, and calculate Tasseled Cap components

def preprocess_image(image):
    image = image.select(
        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL', 'QA_RADSAT'],
        ['CLU', 'BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2', 'QA_PIXEL', 'QA_RADSAT']
    )
    # scale to surface reflectance values
    scaled_optical = (
        image.select(['CLU', 'BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2'])
        .multiply(0.0000275)
        .add(-0.2)
    )
    image = image.addBands(scaled_optical, overwrite=True)

    # quality band masking
    qa = image.select('QA_PIXEL')
    fill_clear = qa.bitwiseAnd(1 << 0).eq(0)
    dilated_clear = qa.bitwiseAnd(1 << 1).eq(0)
    cloud_clear = qa.bitwiseAnd(1 << 3).eq(0)
    shadow_clear = qa.bitwiseAnd(1 << 4).eq(0)
    cloud_conf = qa.rightShift(8).bitwiseAnd(3)
    cirrus_conf = qa.rightShift(14).bitwiseAnd(3)
    cloud_conf_clear = cloud_conf.lt(2)
    cirrus_conf_clear = cirrus_conf.lt(2)
    saturation_clear = image.select('QA_RADSAT').eq(0)
    # for surface reflectance values, valid data should be between 0 and 1
    valid_min = (
        image.select(['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2'])
        .reduce(ee.Reducer.min())
        .gt(0)
    )
    valid_max = (
        image.select(['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2'])
        .reduce(ee.Reducer.max())
        .lt(1)
    )
    # apply masks
    mask = (
        fill_clear
        .And(dilated_clear)
        .And(cloud_clear)
        .And(shadow_clear)
        .And(cloud_conf_clear)
        .And(cirrus_conf_clear)
        .And(saturation_clear)
        .And(valid_min)
        .And(valid_max)
    )

    # calculate Tasseled Cap components
    tcg = image.expression(
        'B*(-0.1603) + G*(-0.2819) + R*(-0.4934) + NIR*0.7940 + SWIR1*(-0.0002) + SWIR2*(-0.1446)',
        {
            'B': image.select('BLU'),
            'G': image.select('GRN'),
            'R': image.select('RED'),
            'NIR': image.select('NIR'),
            'SWIR1': image.select('SW1'),
            'SWIR2': image.select('SW2'),
        }
    ).rename('TCG')

    tcb = image.expression(
        'B*0.2043 + G*0.4158 + R*0.5524 + NIR*0.5741 + SWIR1*0.3124 + SWIR2*0.2303',
        {
            'B': image.select('BLU'),
            'G': image.select('GRN'),
            'R': image.select('RED'),
            'NIR': image.select('NIR'),
            'SWIR1': image.select('SW1'),
            'SWIR2': image.select('SW2'),
        }
    ).rename('TCB')

    tcw = image.expression(
        'B*0.0315 + G*0.2021 + R*0.3102 + NIR*0.1594 + SWIR1*(-0.6806) + SWIR2*(-0.6109)',
        {
            'B': image.select('BLU'),
            'G': image.select('GRN'),
            'R': image.select('RED'),
            'NIR': image.select('NIR'),
            'SWIR1': image.select('SW1'),
            'SWIR2': image.select('SW2'),
        }
    ).rename('TCW')

    return (
        image
        .updateMask(mask)
        .resample('bilinear')  # calling resample early to be used for subsequent reprojections (here: export)
        .addBands([tcg, tcb, tcw])
        .copyProperties(image, image.propertyNames())
    )

tss = landsat.map(preprocess_image).select(['TCG', 'TCB', 'TCW'])

# -----------------------------------------------------------------------------
# habitat metrics: Spectral-Temporal-Metrics (STMs) 

# create a combined reducer to calculate all desired STMs in one go
stm_reducer = (
    ee.Reducer.percentile([10])
    .combine(ee.Reducer.percentile([50]), sharedInputs=True) # shared is more efficient
    .combine(ee.Reducer.percentile([90]), sharedInputs=True)
    .combine(ee.Reducer.stdDev(), sharedInputs=True)
)

# subwindows (temporal folding) for STM calculation
# filter collections
march_to_may = tss.filter(ee.Filter.calendarRange(3, 5, 'month'))
june_to_august = tss.filter(ee.Filter.calendarRange(6, 8, 'month'))
sept_to_nov = tss.filter(ee.Filter.calendarRange(9, 11, 'month'))
dec_to_feb = tss.filter(
    ee.Filter.Or(
        ee.Filter.calendarRange(12, 12, 'month'),
        ee.Filter.calendarRange(1, 2, 'month')
    )
)
# calculate STMs for each subwindow
stm_m03_05 = march_to_may.reduce(stm_reducer).rename([
    'TCG_p10_M03-05', 'TCG_p50_M03-05', 'TCG_p90_M03-05', 'TCG_stdDev_M03-05',
    'TCB_p10_M03-05', 'TCB_p50_M03-05', 'TCB_p90_M03-05', 'TCB_stdDev_M03-05',
    'TCW_p10_M03-05', 'TCW_p50_M03-05', 'TCW_p90_M03-05', 'TCW_stdDev_M03-05'
])

stm_m06_08 = june_to_august.reduce(stm_reducer).rename([
    'TCG_p10_M06-08', 'TCG_p50_M06-08', 'TCG_p90_M06-08', 'TCG_stdDev_M06-08',
    'TCB_p10_M06-08', 'TCB_p50_M06-08', 'TCB_p90_M06-08', 'TCB_stdDev_M06-08',
    'TCW_p10_M06-08', 'TCW_p50_M06-08', 'TCW_p90_M06-08', 'TCW_stdDev_M06-08'
])

stm_m09_11 = sept_to_nov.reduce(stm_reducer).rename([
    'TCG_p10_M09-11', 'TCG_p50_M09-11', 'TCG_p90_M09-11', 'TCG_stdDev_M09-11',
    'TCB_p10_M09-11', 'TCB_p50_M09-11', 'TCB_p90_M09-11', 'TCB_stdDev_M09-11',
    'TCW_p10_M09-11', 'TCW_p50_M09-11', 'TCW_p90_M09-11', 'TCW_stdDev_M09-11'
])

stm_m12_02 = dec_to_feb.reduce(stm_reducer).rename([
    'TCG_p10_M12-02', 'TCG_p50_M12-02', 'TCG_p90_M12-02', 'TCG_stdDev_M12-02',
    'TCB_p10_M12-02', 'TCB_p50_M12-02', 'TCB_p90_M12-02', 'TCB_stdDev_M12-02',
    'TCW_p10_M12-02', 'TCW_p50_M12-02', 'TCW_p90_M12-02', 'TCW_stdDev_M12-02'
])

# combine all STMs into one image
stm_image = ee.Image.cat([
    stm_m03_05,
    stm_m06_08,
    stm_m09_11,
    stm_m12_02
])

# -----------------------------------------------------------------------------
# export image as GeoTIFF

# prepare image for export: resample, scale, set no data value, and convert to integer
export_image = (
    stm_image
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