import ee
from geeo.misc.spacetime import add_timeband, days_to_milli
from geeo.level3.initimgcol import join_imgs_one_window

# ----------------------------------------------------------------------------------------------------------
# crude outlier detection and removal (CURRENTLY NOT USED)
def moving_average(std_multi=3, band=['*']):
    def wrap(img):
        img = ee.Image(img)
        imgcol_window = ee.ImageCollection.fromImages(img.get('window1')).select(band)
        mean_window = imgcol_window.reduce(ee.Reducer.mean()).rename('MN')
        std_window = imgcol_window.reduce(ee.Reducer.stdDev()).rename('SD')
        upper_bound = mean_window.add(std_window.multiply(std_multi)).rename('UP')
        lower_bound = mean_window.subtract(std_window.multiply(std_multi)).rename('LW')
        return img.addBands([mean_window, std_window, upper_bound, lower_bound]).copyProperties(img, ['system:time_start'])
    return wrap


def moving_average_outlier(imgcol, window=20, std_multi=3, band=['*']):
    imgcol_ = join_imgs_one_window(imgcol, window_days=window, key='window1')
    imgcol_ = ee.ImageCollection(imgcol_.map(moving_average(std_multi=std_multi, band=band)))
    return imgcol_


# ----------------------------------------------------------------------------------------------------------
# RBF smoothing

def rbf_weight(sigma):
    '''
    sigma:      (int) StdDev of RBF-kernel in days.
    '''
    def wrap(img):
        rbf_weight = img.expression(
            'exp(-0.5*pow(((delta/86400000)/t_std), 2))',
            {
                    'delta': ee.Number(img.get('delta')),
                    't_std': sigma,
            }
        )
        weighted = ee.Image(img.multiply(rbf_weight))  # weight image with weight
        weighted = ee.Image(weighted.copyProperties(img, ['system:time_start']))
        rbf_weight = rbf_weight.updateMask(img.select(0).mask())  # mask weight band!!!
        return weighted.addBands(rbf_weight.rename('rbf_weight'))
    return wrap


def tsi_rbf(sigma=16):
    '''
    sigma:      (int) StdDev of RBF-kernel in days.
    '''
    def wrap(img):
        img = ee.Image(img)
        # get collection and add weight band
        imgcol = ee.ImageCollection.fromImages(img.get('window1'))
        imgcol = imgcol.map(rbf_weight(sigma))
        # calculate weighted sum
        bandNames = imgcol.first().bandNames()
        result = ee.Image(imgcol.reduce(ee.Reducer.sum())).divide(imgcol.select('rbf_weight').reduce(ee.Reducer.sum())).rename(bandNames)
        return result.copyProperties(img, ['system:time_start'])
    return wrap


def tsi_rbf_duo(s1=8, s2=16, bw1=8):
    def wrap(img):
        img = ee.Image(img)
        # get collections for each window and weight based on RBF
        imgcol_window1 = ee.ImageCollection.fromImages(img.get('window1'))
        imgcol_window1 = imgcol_window1.map(rbf_weight(s1))
        imgcol_window2 = ee.ImageCollection.fromImages(img.get('window2'))
        imgcol_window2 = imgcol_window2.map(rbf_weight(s2))
        # get weights for each window based on data density
        # .unmask(1) ensures that there exist no mask and zero is not divided by
        nonzero1 = ee.Image(imgcol_window1.select('rbf_weight').reduce(ee.Reducer.count())).unmask(0)
        # 3) retrieve weights
        weight = ee.Image(nonzero1.divide(ee.Image(bw1))).clamp(0.0, 1.0)
        weight2 = ee.Image(1).subtract(weight)
        # get bandnames
        bandNames = imgcol_window1.first().bandNames()
        # calc results
        result1 = (ee.Image(imgcol_window1.reduce(ee.Reducer.sum())).divide(imgcol_window1.select('rbf_weight').reduce(ee.Reducer.sum()))).multiply(weight)
        result2 = (ee.Image(imgcol_window2.reduce(ee.Reducer.sum())).divide(imgcol_window2.select('rbf_weight').reduce(ee.Reducer.sum()))).multiply(weight2)
        final = ee.Image(result1.unmask(0).add(result2)).rename(bandNames)

        return final.copyProperties(img, ['system:time_start'])
    return wrap


def tsi_rbf_trio(s1=8, s2=16, s3=32, bw1=8, bw2=16):
    def wrap(img):
        img = ee.Image(img)
        # get collections for each window and weight based on RBF
        imgcol_window1 = ee.ImageCollection.fromImages(img.get('window1'))
        imgcol_window1 = imgcol_window1.map(rbf_weight(s1))
        imgcol_window2 = ee.ImageCollection.fromImages(img.get('window2'))
        imgcol_window2 = imgcol_window2.map(rbf_weight(s2))
        imgcol_window3 = ee.ImageCollection.fromImages(img.get('window3'))
        imgcol_window3 = imgcol_window3.map(rbf_weight(s3))
        # get weights for each window based on data density
        # .unmask(1) ensures that there exist no mask and zero is not divided by
        nonzero1 = ee.Image(imgcol_window1.select('rbf_weight').reduce(ee.Reducer.count())).unmask(0)
        nonzero2 = ee.Image(imgcol_window2.select('rbf_weight').reduce(ee.Reducer.count())).unmask(0)
        # 3) retrieve weights (clamping not necessary because normalized? -> yes)
        weight = ee.Image(nonzero1.divide(ee.Image(bw1))).clamp(0.0, 1.0)
        remainder1 = ee.Image(1).subtract(weight)
        weight2 = (ee.Image(nonzero2.divide(ee.Image(bw2))).clamp(0.0, 1.0)).multiply(remainder1)
        weight3 = ee.Image(1).subtract((weight.add(weight2)))
        # get bandnames
        bandNames = imgcol_window1.first().bandNames()
        # calc results
        result1 = (ee.Image(imgcol_window1.reduce(ee.Reducer.sum())).divide(imgcol_window1.select('rbf_weight').reduce(ee.Reducer.sum()))).multiply(weight)
        result2 = (ee.Image(imgcol_window2.reduce(ee.Reducer.sum())).divide(imgcol_window2.select('rbf_weight').reduce(ee.Reducer.sum()))).multiply(weight2)
        result3 = (ee.Image(imgcol_window3.reduce(ee.Reducer.sum())).divide(imgcol_window3.select('rbf_weight').reduce(ee.Reducer.sum()))).multiply(weight3)
        # weigh results according to weight band
        final = ee.Image(result1.unmask(0).add(result2.unmask(0)).add(result3))
        return final.rename(bandNames).copyProperties(img, ['system:time_start'])
    return wrap


# ----------------------------------------------------------------------------------------------------------
# Linear interpolation

# linear interpolation with weight calculated by distance
def tsi_linear_weighted(band):
    def wrap(img):
        img = ee.Image(img)
        mosaic_before = ee.ImageCollection.fromImages(img.get('before')).mosaic()
        mosaic_after = ee.ImageCollection.fromImages(img.get('after')).mosaic()
        t1 = ee.Image(mosaic_before.select('time')).rename('t1')
        t2 = ee.Image(mosaic_after.select('time')).rename('t2')
        t = ee.Image(img.get('system:time_start')).rename('t')
        time_img = ee.Image.cat([t1, t2, t])
        linear_ratio = time_img.expression(
            '(t - t1) / (t2 - t1)',
            {
                't': time_img.select('t'),
                't1': time_img.select('t1'),
                't2': time_img.select('t2'),
            }
        )
        mosaic_before = mosaic_before.select(band)
        mosaic_after = mosaic_after.select(band)
        interpolated = ee.Image(mosaic_before.add((mosaic_after.subtract(mosaic_before).multiply(linear_ratio)))).rename(band)
        interpolated = ee.Image(img.select(band).unmask(interpolated))  # replace masked pixels with interpolated data
        return interpolated.copyProperties(img, ['system:time_start'])
    return wrap

# linear interpolation with weight = 0.5 to surrounding images
def tsi_linear(img):
    img = ee.Image(img)
    mosaic_before = ee.ImageCollection.fromImages(img.get('before')).mosaic()
    mosaic_after = ee.ImageCollection.fromImages(img.get('after')).mosaic()
    interpolated = ee.Image(mosaic_before.add((mosaic_after.subtract(mosaic_before).multiply(0.5)))).int16()
    result = img.unmask(interpolated)  # replace masked pixels with interpolated data
    return result.copyProperties(img, ['system:time_start'])

# interpolation by taking average from all available surrounding images (surr. images are defined by join)
# option 1 function: Prior join
def tsi_mean(img):
    img = ee.Image(img)
    bandNames = img.bandNames()
    imgcol_before = ee.ImageCollection.fromImages(img.get('before'))
    imgcol_after = ee.ImageCollection.fromImages(img.get('after'))
    imgcol_ = imgcol_before.merge(imgcol_after)
    result = imgcol_.reduce(ee.Reducer.mean()).rename(bandNames)
    return result.copyProperties(img, ['system:time_start'])

# option 2 function: filterDate and reduce (should be faster in theory, but not tested)
def temporal_mean(imgcol, window_days=20):
    def wrap(img):
        date = img.date()
        bandNames = img.bandNames()
        result = imgcol.filterDate(date.advance(-window_days, 'day'), date.advance(window_days, 'day')).reduce(ee.Reducer.mean())
        return result.rename(bandNames).copyProperties(img, ['system:time_start'])
    return wrap

# EOF