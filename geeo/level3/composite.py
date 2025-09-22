import ee

# Pixel based composite (PBC) algorithms
# Overview by Qiu et al. 2023 (https://www.sciencedirect.com/science/article/pii/S0034425722004813)
# Implementations here:
# - MAX-RNB (Qiu et al. 2023)
# - NLCD (Jin et al. 2023)
# - BAP (Griffiths et al. 2013)
# - MAX-NDVI (Holben 1986)


# ------------------------------------------------------------------------------------------------------------
#                                                  NLCD
# ------------------------------------------------------------------------------------------------------------
def composite_nlcd(img):
    imgcol = ee.ImageCollection.fromImages(img.get('window1'))
    key = img.get('key')
    # get median for all spectral bands
    median = imgcol.select(['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2']).median()
    imgcol = imgcol.map(
        lambda img: img.addBands(
            ee.Image(img.select(['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2']).subtract(median)).pow(2).reduce('sum').multiply(-1).rename('SCORE')
        )
    )
    img_composite = ee.Image(imgcol.qualityMosaic('SCORE'))
    return img_composite.set('system:index', key, 'system:time_start', img.get('system:time_start'))


# ------------------------------------------------------------------------------------------------------------
#                                               MAX-FEATURE
# ------------------------------------------------------------------------------------------------------------
def composite_feature(band='NDVI'):
    def wrap(img):
        imgcol = ee.ImageCollection.fromImages(img.get('window1'))
        key = img.get('key')
        img_composite = ee.Image(imgcol.qualityMosaic(band))
        return img_composite.set('system:index', key, 'system:time_start', img.get('system:time_start'))
    return wrap


def composite_feature_invert(band='NDVI'):
    def wrap(img):
        imgcol = ee.ImageCollection.fromImages(img.get('window1'))
        key = img.get('key')
        imgcol = imgcol.map(
            lambda img: img.addBands(img.select(band).multiply(-1).rename('SCORE'))
        )
        img_composite = ee.Image(imgcol.qualityMosaic('SCORE'))
        return img_composite.set('system:index', key, 'system:time_start', img.get('system:time_start'))
    return wrap


# ------------------------------------------------------------------------------------------------------------
#                                           BAP (Griffiths et al. 2013)
# ------------------------------------------------------------------------------------------------------------
def composite_bap(doy_offset_eq_year, min_clouddistance=0, max_clouddistance=500, weight_doy=0.5, weight_year=0.3, weight_cloud=0.2):
    def wrap(img):
        imgcol = ee.ImageCollection.fromImages(img.get('window1'))
        key = img.get('key')
        year_target = ee.Image(ee.Number(img.get('year_center')))
        year_offset = ee.Image(ee.Number(img.get('year_offset')))
        doy_target = ee.Image(ee.Number(img.get('doy_center')))
        doy_offset = ee.Image(ee.Number(img.get('doy_offset'))).divide(3)  # 3sigma ~ 99.7% 
        imgcol = imgcol.map(bap_score(year_target, year_offset, doy_target, doy_offset, doy_offset_eq_year, min_clouddistance, max_clouddistance, weight_doy, weight_year, weight_cloud))
        img_composite = ee.Image(imgcol.qualityMosaic('SCORE')).toFloat()
        return img_composite.set('system:index', key, 'system:time_start', img.get('system:time_start'))
    return wrap


def bap_score(year_target, year_offset, doy_target, doy_offset, doy_offset_eq_year, min_clouddistance=0, max_clouddistance=500, weight_doy=0.5, weight_year=0.3, weight_cloud=0.2):
    def wrap(img):
        
        mask = img.select('mask')  # 1 = valid pixel, 0 = cloud
        img_date = img.date()
        
        # DOY score
        doy = ee.Image(img_date.getRelative('day', 'year'))
        DOY_SCORE = ee.Image(
            doy.subtract(doy_target) \
                .divide(doy_offset) \
                .pow(2) \
                .multiply(-0.5) \
                .exp()
                ).rename('DOY_SCORE')
        
        # YEAR score
        # retrieve score where +-1 year has the same score as given doy_vs_year
        doy_vs_year_score = ee.Image(
            doy_target.subtract(doy_offset_eq_year).subtract(doy_target) \
                .divide(doy_offset) \
                .pow(2) \
                .multiply(-0.5) \
                .exp()
                )
        # convert inputs to ee.Number
        year = ee.Image(ee.Number.parse(img.date().format("YYYY")))
        # absolute difference between the year and target year
        year_difference = year.subtract(year_target).abs()
        # define conditions for scoring
        condition_exact = year_difference.eq(1)  # difference within +-1 year
        condition_within_offset = year_difference.gt(1).And(year_difference.lte(year_offset))

        interpolated_score = doy_vs_year_score.multiply(
            ee.Image(1).subtract(year_difference.subtract(1).divide(year_offset.subtract(1)))
        )

        # final scoring image based on conditions
        YEAR_SCORE = ee.Image(1) \
            .where(condition_exact, doy_vs_year_score) \
            .where(condition_within_offset, interpolated_score) \
            .where(year_difference.gt(year_offset), 0)  # Set score to 0 for differences greater than the offset
        YEAR_SCORE = YEAR_SCORE.rename('YEAR_SCORE')

        # CLOUD score
        distance = mask.Not().distance(ee.Kernel.euclidean(radius=max_clouddistance, units='meters'))
        # conditions for the final image
        condition1 = mask.eq(0)  # Cloud pixels
        condition2 = distance.lte(min_clouddistance)
        condition3 = distance.gt(min_clouddistance).And(distance.lte(max_clouddistance))  # Pixels in between
        interpolated = distance.subtract(ee.Image(min_clouddistance)).divide(ee.Image(max_clouddistance).subtract(min_clouddistance))

        CLOUD_SCORE = ee.Image(1) \
            .where(condition1, 0) \
            .where(condition2, 0) \
            .where(condition3, interpolated) \
            .where(distance.gt(max_clouddistance), 1)
        CLOUD_SCORE = CLOUD_SCORE.rename('CLOUD_SCORE')

        # FINAL
        BAP_SCORE = ee.Image(
            CLOUD_SCORE.multiply(weight_cloud).add(DOY_SCORE.multiply(weight_doy)).add(YEAR_SCORE.multiply(weight_year))
            ).toFloat().rename('SCORE')

        return img.addBands([BAP_SCORE, CLOUD_SCORE, YEAR_SCORE, DOY_SCORE])
    return wrap

