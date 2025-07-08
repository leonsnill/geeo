import ee
import math

# get DOY as band
def add_doyband(img):
    doy = img.date().getRelative('day', 'year')
    doy_band = ee.Image(doy).uint16().rename('DOY')
    doy_band = doy_band.updateMask(img.select(0).mask())
    return img.addBands(doy_band)

def add_mask_band(img):
    mask = img.select(0).mask()
    mask_band = ee.Image(mask).rename('mask')
    return img.addBands(mask_band)

milli_year = 365.2425*1000*60*60*24  # max difference = 1 year

def add_milliband(img):
    millis =  ee.Image(img.date().millis()).rename('millis')
    return img.addBands(millis.toInt64())

# DOY to radians
def doy_to_radians(img):
    rad = ee.Image(img.select('DOY').divide(365)).multiply(2*math.pi).rename('DOY_RAD') 
    return img.addBands(rad)

# radians to DOY
def radians_to_doy(img):
    doy = ee.Image((img.multiply(365)).divide(2*math.pi)).clamp(1, 365).int16()
    return doy

# conversion function for time series into cartesian coords
def polar_coords(band='NDVI'):
    def wrap(img):
        # time series x- and y-coordinate
        timeseries = img.select(band)
        # cartesian coordinates:
        ts_x = ee.Image(timeseries.multiply(img.select('DOY_RAD').cos())).rename('POL_RTX')
        ts_y = ee.Image(timeseries.multiply(img.select('DOY_RAD').sin())).rename('POL_RTY')
        return img.addBands([ts_x, ts_y])
    return wrap


# the polar vectors need to be computed twice: global and local
# long term average vector
def polar_vector(imgcol):
    rtx_mn = imgcol.select('POL_RTX').reduce(ee.Reducer.mean())
    rty_mn = imgcol.select('POL_RTY').reduce(ee.Reducer.mean())
    # cartesian coordinates
    r = rtx_mn.atan2(rty_mn)  # THIS IS REVERTED TO COMMON ARCTAN2 ORDER!
    r = ee.Image(r.where(r.lte(0), r.add(2*math.pi))).rename('RAVG')  #.add(2pi)
    # 'v' is for now not needed since this is the global estimate of r and theta
    #v = ee.Image(x.multiply(x).add(y.multiply(y))).sqrt()
    theta = r.add(math.pi)
    theta = (theta.where(r.gte(math.pi), r.subtract(math.pi))).rename('THETA')
    # sanity for now: start of season (theta) and peak magnitude (r, v) in DOYS
    theta_doy = radians_to_doy(theta)
    r_doy = radians_to_doy(r)
    return ee.Image([theta_doy, r_doy])


def get_seasonal_adjusted_vectors(img):
    # get collection from property
    imgcol_window = ee.ImageCollection.fromImages(img.get('window'))
    avg_seasonal = ee.Image(polar_vector(imgcol_window)).rename(['THETA_s', 'RAVG_s'])
    return img.addBands(avg_seasonal)

def theta_milli_mask(theta='THETA', name_start='t_start', name_end='t_end', name_valid_firstyear='valid_firstyear'):
    def wrap(img):
        # first- last-year filter
        # "If 〈θ〉 is located in the first half of the year, we did not obtain LSP metrics for the last year...
        not_lastyear = ee.Image(img.select(theta).lt(183))
        # ...or the 1st year if otherwise."
        not_firstyear = ee.Image(not_lastyear.Not())  # where theta gt 182 -> year-1
        # converet DOY to milliseconds
        avg_longterm_milli = ee.Image(img.select(theta).multiply(1000*60*60*24))

        # the POLAR and DOY vectors are defined correctly; but depending on DOY < 182, the date/milli
        # output must be adjusted to retrieve the correct start of season.
        # E.g. if THETA = 242, then "not_lastyear = 0", then start of season for year y happens in year y-1
        
        year_target = ee.Image(img.date().get('year'))
        #year_target = year_target.add(not_firstyear.multiply(-1))  # just subtract?
        year_target = year_target.subtract(not_firstyear)  # just subtract?
        # convert to millis
        year_target = year_target.subtract(1970)
        year_target = year_target.multiply(milli_year)
        # add DOY in millis to year to define start...
        mask_start = ee.Image(year_target.add(avg_longterm_milli)).toInt64()
        # ... and add a year on top for the end
        mask_end = ee.Image(mask_start.add(milli_year)).toInt64()

        return img.addBands(
            [mask_start.rename(name_start), mask_end.rename(name_end), not_lastyear.rename(name_valid_firstyear)], 
            overwrite=True
        )
    return wrap

def theta_milli_mask_seasonal(theta='THETA', theta_s='THETA_s', max_difference=40):
    def wrap(img):
        # calculate difference between long-term and seasonal theta
        img_theta = img.select(theta)
        img_theta_s = img.select(theta_s)
        diff = img_theta_s.subtract(img_theta).add(183).mod(365).subtract(183)  # focus on shift of adjusted DOY
        
        # adjust max difference
        diff = diff.where(diff.lt(ee.Image(max_difference).multiply(-1)), ee.Image(max_difference).multiply(-1))
        diff = diff.where(diff.gt(ee.Image(max_difference)), ee.Image(max_difference))
        # convert DOY difference to milliseconds
        diff = diff.multiply(1000*60*60*24).toInt64()

        # adjust start and end
        mask_start = img.select('t_start')
        mask_end = img.select('t_end')

        mask_start_adj = mask_start.add(diff)
        mask_end_adj = mask_end.add(diff)

        return img.addBands(
            [mask_start_adj.rename('t_start_adj'), mask_end_adj.rename('t_end_adj')],
            overwrite=True
        )
    return wrap


def mask_window(propname_imgcol='window', name_start='t_start', name_end='t_end'):
        def wrap(img):
            mask_start = img.select(name_start)
            mask_end = img.select(name_end)
            imgcol_window = ee.ImageCollection.fromImages(img.get(propname_imgcol))
            
            def wrap2(img2):
                tsi_gap_mask = img2.select('mask')  # get og mask before unmasking
                img2 = img2.unmask() # unmask to avoid masking issues
                t_valid = ee.Image(
                    (img2.select('millis').gte(mask_start)).And((img2.select('millis')).lt(mask_end))
                    )
                new_mask = tsi_gap_mask.And(t_valid)  # restore mask + apply new mask
                img2 = img2.updateMask(new_mask)
                img2 = img2.addBands(tsi_gap_mask.rename('mask'), overwrite=True)
                return img2
            
            imgcol_window = imgcol_window.map(wrap2)
            l_window = imgcol_window.toList(imgcol_window.size())
            return img.set(propname_imgcol, l_window)
        return wrap


def get_lsp_metrics(band='NDVI'):
    def wrap(img):
        # get collection from property
        imgcol_window = ee.ImageCollection.fromImages(img.get('window'))

        # scale to integer
        imgcol_window = imgcol_window.map(lambda x: x.addBands(x.select(band).multiply(10000).toInt64().rename(band), overwrite=True))  #.multiply(10000).toInt64()

        # get seasonal metrics: mean and std
        ndvi_mean_s = ee.Image(imgcol_window.select(band).reduce(ee.Reducer.mean()).rename('NDVI_mean_s'))
        ndvi_stdDev_s = ee.Image(imgcol_window.select(band).reduce(ee.Reducer.stdDev()).rename('NDVI_stdDev_s'))

        # 4.2 Cumulative sum
        def accumulate(image, acc):
            image = ee.Image(image)
            acc = ee.ImageCollection(acc)
            last = ee.Image(acc.get('last'))
            value = last.add(image).unmask(last).set('system:time_start', image.get('system:time_start'))#.copyProperties(image, image.propertyNames())
            return acc.merge(ee.ImageCollection([value])).set('last', value)
        

        cumulative = ee.ImageCollection(imgcol_window.select(band).iterate(accumulate, ee.ImageCollection([]).set('last', ee.Image(0).rename(band))))
        
        # since the accumulation function starts of with an empty 0 image, many 0s might be present which
        # will falsefy the percentile functions, so we need to mask all zeros
        cumulative = cumulative.map(add_doyband)  # DOY
        cumulative = cumulative.map(add_milliband)

        #sum/max
        p100 = ee.Image(imgcol_window.select(band).sum()).toInt64().rename('p100')
        # argmax
        max_milli = cumulative.map(
            lambda x: x.select('millis').updateMask(ee.Image( x.select(band).eq(p100) ))
        )
        min_max_milli = ee.Image(max_milli.reduce(ee.Reducer.min()))
        # ts > argmax


        def mask_zeros_p100(img):
            img_ = img.select(band)
            non_zero = ee.Image(img_.neq(0))
            min_max = ee.Image(img.select('millis').lte(min_max_milli))
            final_mask = ee.Image(non_zero.And(min_max))
            img_masked = img_.updateMask(final_mask)
            return img.addBands(img_masked.rename(band), overwrite=True)


        cumulative = cumulative.map(mask_zeros_p100)

        # get percentiles
        Q_SOS = ee.Image(cumulative.select(band).reduce(ee.Reducer.percentile([15]))).rename('Q_SOS')
        Q_SOP = ee.Image(cumulative.select(band).reduce(ee.Reducer.percentile([25]))).rename('Q_SOP')
        Q_MOP = ee.Image(cumulative.select(band).reduce(ee.Reducer.percentile([50]))).rename('Q_MOP')
        Q_EOP = ee.Image(cumulative.select(band).reduce(ee.Reducer.percentile([75]))).rename('Q_EOP')
        Q_EOS = ee.Image(cumulative.select(band).reduce(ee.Reducer.percentile([80]))).rename('Q_EOS')
        maxus = ee.Image(imgcol_window.select(band).reduce(ee.Reducer.max())).rename('MAX')

        # get DOYs at percentiles
        def perc_diff(img_eq):
            SOS = ee.Image(ee.Image(img_eq.select(band).subtract(Q_SOS)).abs()).rename('SOS')
            SOP = ee.Image(ee.Image(img_eq.select(band).subtract(Q_SOP)).abs()).rename('SOP')
            MOS = ee.Image(ee.Image(img_eq.select(band).subtract(Q_MOP)).abs()).rename('MOS')
            EOP = ee.Image(ee.Image(img_eq.select(band).subtract(Q_EOP)).abs()).rename('EOP')
            EOS = ee.Image(ee.Image(img_eq.select(band).subtract(Q_EOS)).abs()).rename('EOS')
            return img_eq.addBands([SOS, SOP, MOS, EOP, EOS])


        cumulative = cumulative.map(perc_diff)

        # create images from collection    
        sos_diff = ee.Image(cumulative.select('SOS').reduce(ee.Reducer.min()))
        sop_diff = ee.Image(cumulative.select('SOP').reduce(ee.Reducer.min()))
        mos_diff = ee.Image(cumulative.select('MOS').reduce(ee.Reducer.min()))
        eop_diff = ee.Image(cumulative.select('EOP').reduce(ee.Reducer.min()))
        eos_diff = ee.Image(cumulative.select('EOS').reduce(ee.Reducer.min()))


        def doy_from_eq(img_eq):
            SOS = ee.Image(img_eq.select('DOY').multiply(ee.Image(img_eq.select('SOS').eq(sos_diff)))).rename('SOS')
            SOP = ee.Image(img_eq.select('DOY').multiply(ee.Image(img_eq.select('SOP').eq(sop_diff)))).rename('SOP')
            MOS = ee.Image(img_eq.select('DOY').multiply(ee.Image(img_eq.select('MOS').eq(mos_diff)))).rename('MOS')
            EOP = ee.Image(img_eq.select('DOY').multiply(ee.Image(img_eq.select('EOP').eq(eop_diff)))).rename('EOP')
            EOS = ee.Image(img_eq.select('DOY').multiply(ee.Image(img_eq.select('EOS').eq(eos_diff)))).rename('EOS')
            return img_eq.addBands([SOS, SOP, MOS, EOP, EOS], overwrite=True)

        # apply doy percentile function
        cumulative = cumulative.map(doy_from_eq)

        # create images from collection    
        SOS = ee.Image(cumulative.select('SOS').reduce(ee.Reducer.max())).rename('SOS')
        SOP = ee.Image(cumulative.select('SOP').reduce(ee.Reducer.max())).rename('SOP')
        MOS = ee.Image(cumulative.select('MOS').reduce(ee.Reducer.max())).rename('MOS')
        EOP = ee.Image(cumulative.select('EOP').reduce(ee.Reducer.max())).rename('EOP')
        EOS = ee.Image(cumulative.select('EOS').reduce(ee.Reducer.max())).rename('EOS')


        return img.addBands([SOS, SOP, MOS, EOP, EOS, Q_SOS, Q_SOP, Q_MOP, Q_EOP, Q_EOS, maxus, p100, ndvi_mean_s, ndvi_stdDev_s])
    return wrap


# wrap processing into one function
def lsp(imgcol, band='NDVI', adjust_seasonal=True, adjust_seasonal_max_delta_days=40,
        year_min=None, year_max=None):

    # get temporal window
    if not year_min or not year_max:
        year_min = ee.Date(imgcol.aggregate_min('system:time_start')).get('year')
        year_max = ee.Date(imgcol.aggregate_max('system:time_start')).get('year')

    # -----------------------------------------------------------------
    # (1) Long term phenological years / average vector
    imgcol = imgcol.map(add_doyband)  # DOY
    imgcol = imgcol.map(add_mask_band)  # mask band
    imgcol = imgcol.map(doy_to_radians)  # DOY to RAD
    imgcol = imgcol.map(polar_coords(band=band))  # POL COORDS
    avg_longterm = polar_vector(imgcol).rename(['THETA', 'RAVG'])  # long term average vector 
    
    # -----------------------------------------------------------------
    # (2) Initialize pheno ImageCollection
    # create temporal window
    l_years = ee.List.sequence(year_min.subtract(year_min), year_max.subtract(year_min), 1)
    init_year = ee.Date.fromYMD(year_min, 1, 1)

    initImages = l_years.map(
        lambda y: ee.Image().set({
            'year': init_year.advance(y, 'year').get('year'),
            'system:time_start': init_year.advance(y, 'year').advance(182, 'day').millis(),
            'type': 'pheno'
        })
    )
    imgcol_pheno = ee.ImageCollection.fromImages(initImages)

    # add milli-band to joining collection, i.e. imgcol
    imgcol = imgcol.map(add_milliband)

    # join images to collection
    maxDiffFilter = ee.Filter.maxDifference(
        difference = milli_year, 
        leftField = 'system:time_start',
        rightField = 'system:time_start'
    )
    join = ee.Join.saveAll(matchesKey = 'window', 
                           measureKey = 'delta',
                           ordering = 'system:time_start',
                           ascending = True)
    
    bands_to_keep = [band, 'mask', 'millis', 'DOY', 'POL_RTX', 'POL_RTY']
    imgcol_pheno = ee.ImageCollection(join.apply(primary = imgcol_pheno,
                              secondary = imgcol.select(bands_to_keep),
                              condition = maxDiffFilter))

    # add long-term bands to initialized pheno-collection
    imgcol_pheno = imgcol_pheno.map(
        lambda img: img.addBands(avg_longterm).copyProperties(source=img).set('system:time_start', img.get('system:time_start'))
    )
    
    # mask outside season bounds
    # create mask containing valid millisecond bins
    imgcol_pheno = imgcol_pheno.map(
        theta_milli_mask(theta='THETA', name_start='t_start', name_end='t_end', name_valid_firstyear='valid_firstyear')
    )

    # apply mask to image (unadjusted)
    imgcol_pheno = imgcol_pheno.map(
        mask_window(propname_imgcol='window', name_start='t_start', name_end='t_end')
    )

    # -----------------------------------------------------------------
    # 3 Seasonally-adjusted phenological years (optional, and more demanding)
    if adjust_seasonal:
        # get seasonally-adjusted vectors, RAVG_s, THETA_s
        imgcol_pheno = imgcol_pheno.map(get_seasonal_adjusted_vectors)

        # create mask based on seasonally adjusted vectors; new function to adjust based on known start
        imgcol_pheno = imgcol_pheno.map(
            theta_milli_mask_seasonal(theta='THETA', theta_s='THETA_s', max_difference=adjust_seasonal_max_delta_days)
        )

        # reapply mask to image (adjusted)
        imgcol_pheno = imgcol_pheno.map(
            mask_window(propname_imgcol='window', name_start='t_start_adj', name_end='t_end_adj')
        )

    # -----------------------------------------------------------------
    # 4 LSP metrics

    # add LSP metrics to collection
    imgcol_pheno = imgcol_pheno.map(
        get_lsp_metrics(band=band)
    )

    '''
    TO-DOs
    - check unmask in cumsum function if TSI has gaps
    - check what happens when "band" has negative values; -> sign for positive for cumulative sum?
    '''

    return imgcol_pheno

