import ee

# --------------------------------------------------------------------------------------------------
# GENERAL

def apply_mask(bandname='mask'):
    """
    Apply a mask to an image based on a band.
    """
    def wrap(img):
        mask = img.select(bandname)
        return img.addBands(img.updateMask(mask), overwrite=True).addBands(mask, overwrite=True).copyProperties(source=img).set('system:time_start', img.get('system:time_start'))
    return wrap


def blu_filter(threshold=0.5):
    def wrap(img):
        img = ee.Image(img)
        th = img.select('BLU').lt(threshold)
        above_zero = img.select('BLU').gt(0)
        mask = th.And(above_zero)
        img = img.updateMask(mask)
        return img.copyProperties(img, ['system:time_start'])
    return wrap


# --------------------------------------------------------------------------------------------------
# LANDSAT


def extractQAbits(qa_band, bit_start, bit_end):
    """
    Helper function to extract the QA bits from a Landsat image.
    """
    numbits = ee.Number(bit_end).subtract(ee.Number(bit_start)).add(ee.Number(1))
    qa_bits = qa_band.rightShift(bit_start).mod(ee.Number(2).pow(numbits).int())
    return qa_bits


def mask_landsat(masks, conf='Medium'):
    """
    Add a mask to a Landsat image based on the QA_PIXEL and QA_RADSAT band.
    """
    dict_mask = {'cloud': ee.Number(2).pow(3).int(),
                 'cshadow': ee.Number(2).pow(4).int(),
                 'snow': ee.Number(2).pow(5).int(),
                 'fill': ee.Number(2).pow(0).int(),
                 'dilated': ee.Number(2).pow(1).int(),}

    dict_conf = {'Low': 1, 'Medium': 2, 'High': 3}
    sel_conf = ee.Number(dict_conf[conf])

    sel_masks = [dict_mask[x] for x in masks]
    bits = ee.Number(1)

    for m in sel_masks:
        bits = ee.Number(bits.add(m))
    
    def wrap(img):
        # clouds, cloud shadows, snow, fill, dilated clouds
        qa = img.select('QA_PIXEL')
        mask = qa.bitwiseAnd(bits).eq(0)
        cloud_conf = extractQAbits(qa, 8, 9)
        cloud_low = cloud_conf.gte(sel_conf).Not()
        cirr_conf = extractQAbits(qa, 14, 15)
        cirr_low = cirr_conf.gte(sel_conf).Not()
        # radiometric saturation
        saturation = img.select('QA_RADSAT').eq(0)
        # in/valid minima and maxima
        valid_min = img.select(['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2']).reduce(ee.Reducer.min()).gt(0)
        valid_max = img.select(['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2']).reduce(ee.Reducer.max()).lt(1)
        # final mask
        mask = mask.And(cloud_low).And(cirr_low).And(saturation).And(valid_min).And(valid_max)
        mask = mask.rename('mask')
        return img.updateMask(mask).addBands(mask).copyProperties(source=img).set('system:time_start', img.get('system:time_start'))
    return wrap


# erode-dilate included (requires only single map, thus faster)
def mask_landsat_erodil(masks, conf='Medium',
                        kernel_erode=ee.Kernel.circle(radius=120, units='meters'), 
                        kernel_dilate=ee.Kernel.circle(radius=100, units='meters'), 
                        scale=30):
    """
    Add a mask to a Landsat image based on the QA_PIXEL and QA_RADSAT band.
    """
    dict_mask = {'cloud': ee.Number(2).pow(3).int(),
                 'cshadow': ee.Number(2).pow(4).int(),
                 'snow': ee.Number(2).pow(5).int(),
                 'fill': ee.Number(2).pow(0).int(),
                 'dilated': ee.Number(2).pow(1).int(),}

    dict_conf = {'Low': 1, 'Medium': 2, 'High': 3}
    sel_conf = ee.Number(dict_conf[conf])

    sel_masks = [dict_mask[x] for x in masks]
    bits = ee.Number(1)

    for m in sel_masks:
        bits = ee.Number(bits.add(m))
    
    def wrap(img):
        # clouds, cloud shadows, snow, fill, dilated clouds
        qa = img.select('QA_PIXEL')
        mask = qa.bitwiseAnd(bits).eq(0)
        cloud_conf = extractQAbits(qa, 8, 9)
        cloud_low = cloud_conf.gte(sel_conf).Not()
        cirr_conf = extractQAbits(qa, 14, 15)
        cirr_low = cirr_conf.gte(sel_conf).Not()
        # radiometric saturation
        saturation = img.select('QA_RADSAT').eq(0)
        # in/valid minima and maxima
        valid_min = img.select(['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2']).reduce(ee.Reducer.min()).gt(0)
        valid_max = img.select(['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2']).reduce(ee.Reducer.max()).lt(1)
        # final mask
        mask = mask.And(cloud_low).And(cirr_low).And(saturation).And(valid_min).And(valid_max)
        mask = (mask.focalMax(kernel=kernel_erode).focalMin(kernel=kernel_dilate)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': scale})
        .rename('mask'))
        return img.updateMask(mask).addBands(mask).copyProperties(source=img).set('system:time_start', img.get('system:time_start'))
    return wrap


# --------------------------------------------------------------------------------------------------
# SENTINEL

def mask_sentinel2_cplus(band='cs', threshold=0.6):
    def wrap(img):
        mask = img.select(band).gte(threshold).rename('mask')
        return img.updateMask(mask).addBands(mask).copyProperties(source=img).set('system:time_start', img.get('system:time_start'))
    return wrap


def mask_sentinel2_cplus_erodil(band='cs', threshold=0.6,
                                kernel_erode=ee.Kernel.circle(radius=120, units='meters'),
                                kernel_dilate=ee.Kernel.circle(radius=100, units='meters'),
                                scale=30):
    def wrap(img):
        mask = img.select(band).gte(threshold)
        mask = (mask.focalMax(kernel=kernel_erode).focalMin(kernel=kernel_dilate)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': scale})
        .rename('mask'))
        return img.updateMask(mask).addBands(mask).copyProperties(source=img).set('system:time_start', img.get('system:time_start'))
    return wrap


def mask_sentinel2_prob(thresh_cld_prb=25):
    def wrap(img):
        mask = ee.Image(img.select('probability')).lt(thresh_cld_prb).rename('mask')
        return img.updateMask(mask).addBands(mask).copyProperties(source=img).set('system:time_start', img.get('system:time_start'))
    return wrap


def mask_sentinel2_prob_erodil(thresh_cld_prb=25,
                               kernel_erode=ee.Kernel.circle(radius=120, units='meters'),
                               kernel_dilate=ee.Kernel.circle(radius=100, units='meters'),
                               scale=30):
    def wrap(img):
        mask = ee.Image(img.select('probability')).lt(thresh_cld_prb)
        mask = (mask.focalMax(kernel=kernel_erode).focalMin(kernel=kernel_dilate)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': scale})
        .rename('mask'))
        return img.updateMask(mask).addBands(mask).copyProperties(source=img).set('system:time_start', img.get('system:time_start'))
    return wrap


def mask_sentinel2_prob_shadow(nir_drk_thresh=0.2, cld_prj_dst=5, proj_scale=120):
        def wrap(img):
            # identify water pixels from the SCL band.
            not_water = img.select('SCL').neq(6)
            # identify dark NIR pixels that are not water (initial potential cloud shadow pixels).
            dark_pixels = img.select('NIR').lt(nir_drk_thresh).multiply(not_water)
            # determine the direction to project cloud shadow from clouds (assumes UTM projection).
            shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
            # Project shadows from clouds for the "maximum distance (km) to search for cloud shadows from cloud edges"
            # first, convert maximum distance from km to pixels based on scale
            cld_prj_dst_px = round((cld_prj_dst*1000)/proj_scale)
            cld_prj = (img.select('mask').directionalDistanceTransform(shadow_azimuth, cld_prj_dst_px)
                .reproject(**{'crs': img.select(0).projection(), 'scale': proj_scale})
                .select('distance')
                .mask()
                .rename('cld_transform'))
            # identify the intersection of dark pixels with cloud shadow projection.
            shadows = cld_prj.multiply(dark_pixels).rename('CSW')
            # combine cloud and shadow mask
            mask = ee.Image(img.select('mask').add(shadows).gt(0))
            mask = mask.Not().rename('mask')
            return img.updateMask(mask).addBands(mask).copyProperties(source=img).set('system:time_start', img.get('system:time_start'))
        return wrap


# EOF