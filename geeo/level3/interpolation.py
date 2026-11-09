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
'''
def rbf_weight(sigma):
    def wrap(img):
        # 1. Use trusted original expression for weight calculation
        rbf_weight = img.expression(
            'exp(-0.5*pow(((delta/86400000)/t_std), 2))',
            {
                'delta': ee.Number(img.get('delta')),
                't_std': sigma,
            }
        )
        weighted = ee.Image(img.multiply(rbf_weight))
        weighted = ee.Image(weighted.copyProperties(img, ['system:time_start']))
        
        # 2. Band-specific weights:
        # img.multiply(0).add(rbf_weight) creates an image with all bands,
        # where each band has value rbf_weight and is MASKED by that band's OWN mask!
        weight_bands = img.multiply(0).add(rbf_weight)
        
        # Name weight bands with '_w' suffix (e.g. BLU_w, RED_w, RE1_w, etc.)
        band_names = img.bandNames()
        weight_names = band_names.map(lambda b: ee.String(b).cat('_w'))
        weight_bands = weight_bands.rename(weight_names)
        
        return weighted.addBands(weight_bands)
    return wrap


def tsi_rbf(sigma=16):
    def wrap(img):
        img = ee.Image(img)
        imgcol = ee.ImageCollection.fromImages(img.get('window1'))
        
        band_names = ee.Image(imgcol.first()).bandNames()
        weight_names = band_names.map(lambda b: ee.String(b).cat('_w'))
        band_names_sum = band_names.map(lambda b: ee.String(b).cat('_sum'))
        weight_names_sum = weight_names.map(lambda b: ee.String(b).cat('_sum'))
        
        # add weights to each image in the collection
        imgcol = imgcol.map(rbf_weight(sigma))
        
        # reduce the collection to summed bands and summed weights
        reduced = imgcol.reduce(ee.Reducer.sum())
        # retrieve summed data and summed weights
        sum_data = reduced.select(band_names_sum)
        sum_weights = reduced.select(weight_names_sum)
        
        # divide summed data by summed weights to get the final interpolated result
        result = sum_data.divide(sum_weights).rename(band_names)
        return result.copyProperties(img, ['system:time_start'])
    return wrap

# duo and trio, work as single RBF but with additional weighting based on data density (bw1, bw2)

def tsi_rbf_duo(s1=8, s2=16, bw1=8):
    def wrap(img):
        img = ee.Image(img)
        imgcol_window1 = ee.ImageCollection.fromImages(img.get('window1'))
        imgcol_window2 = ee.ImageCollection.fromImages(img.get('window2'))
        
        band_names = ee.Image(imgcol_window1.first()).bandNames()
        weight_names = band_names.map(lambda b: ee.String(b).cat('_w'))
        band_names_sum = band_names.map(lambda b: ee.String(b).cat('_sum'))
        weight_names_sum = weight_names.map(lambda b: ee.String(b).cat('_sum'))
        
        imgcol_window1 = imgcol_window1.map(rbf_weight(s1))
        imgcol_window2 = imgcol_window2.map(rbf_weight(s2))

        nonzero1 = imgcol_window1.select(weight_names).reduce(ee.Reducer.count()).unmask(0)
        weight1 = nonzero1.divide(ee.Image(bw1)).clamp(0.0, 1.0).rename(band_names)
        weight2 = ee.Image(1).subtract(weight1)
        
        reduced1 = imgcol_window1.reduce(ee.Reducer.sum())
        sum_data1 = reduced1.select(band_names_sum)
        sum_weights1 = reduced1.select(weight_names_sum)
        result1 = sum_data1.divide(sum_weights1).rename(band_names).multiply(weight1)
        
        reduced2 = imgcol_window2.reduce(ee.Reducer.sum())
        sum_data2 = reduced2.select(band_names_sum)
        sum_weights2 = reduced2.select(weight_names_sum)
        result2 = sum_data2.divide(sum_weights2).rename(band_names).multiply(weight2)
        
        final = result1.unmask(0).add(result2).rename(band_names)
        return final.copyProperties(img, ['system:time_start'])
    return wrap


def tsi_rbf_trio(s1=8, s2=16, s3=32, bw1=8, bw2=16):
    def wrap(img):
        img = ee.Image(img)
        imgcol_window1 = ee.ImageCollection.fromImages(img.get('window1'))
        imgcol_window2 = ee.ImageCollection.fromImages(img.get('window2'))
        imgcol_window3 = ee.ImageCollection.fromImages(img.get('window3'))
        
        band_names = ee.Image(imgcol_window1.first()).bandNames()
        weight_names = band_names.map(lambda b: ee.String(b).cat('_w'))
        band_names_sum = band_names.map(lambda b: ee.String(b).cat('_sum'))
        weight_names_sum = weight_names.map(lambda b: ee.String(b).cat('_sum'))
        
        imgcol_window1 = imgcol_window1.map(rbf_weight(s1))
        imgcol_window2 = imgcol_window2.map(rbf_weight(s2))
        imgcol_window3 = imgcol_window3.map(rbf_weight(s3))
        
        nonzero1 = imgcol_window1.select(weight_names).reduce(ee.Reducer.count()).unmask(0)
        nonzero2 = imgcol_window2.select(weight_names).reduce(ee.Reducer.count()).unmask(0)
        
        weight1 = nonzero1.divide(ee.Image(bw1)).clamp(0.0, 1.0).rename(band_names)
        remainder1 = ee.Image(1).subtract(weight1)
        
        weight2 = nonzero2.divide(ee.Image(bw2)).clamp(0.0, 1.0).multiply(remainder1).rename(band_names)
        weight3 = ee.Image(1).subtract(weight1.add(weight2))
        
        reduced1 = imgcol_window1.reduce(ee.Reducer.sum())
        sum_data1 = reduced1.select(band_names_sum)
        sum_weights1 = reduced1.select(weight_names_sum)
        result1 = sum_data1.divide(sum_weights1).rename(band_names).multiply(weight1)
        
        reduced2 = imgcol_window2.reduce(ee.Reducer.sum())
        sum_data2 = reduced2.select(band_names_sum)
        sum_weights2 = reduced2.select(weight_names_sum)
        result2 = sum_data2.divide(sum_weights2).rename(band_names).multiply(weight2)
        
        reduced3 = imgcol_window3.reduce(ee.Reducer.sum())
        sum_data3 = reduced3.select(band_names_sum)
        sum_weights3 = reduced3.select(weight_names_sum)
        result3 = sum_data3.divide(sum_weights3).rename(band_names).multiply(weight3)
        
        final = result1.unmask(0).add(result2.unmask(0)).add(result3).rename(band_names)
        return final.copyProperties(img, ['system:time_start'])
    return wrap

'''
# Previous version;

def rbf_weight(sigma):
    def wrap(img):
        rbf_weight = img.expression(
            'exp(-0.5*pow(((delta/86400000)/t_std), 2))',
            {
                    'delta': ee.Number(img.get('delta')),
                    't_std': sigma,
            }
        )
        common_mask = img.mask().reduce(ee.Reducer.min())
        #rbf_weight = rbf_weight.updateMask(img.select(0).mask())  # mask weight band!!!
        rbf_weight = rbf_weight.updateMask(common_mask)  # mask weight band!!!
        weighted = ee.Image(img.multiply(rbf_weight))  # weight image with weight
        weighted = ee.Image(weighted.copyProperties(img, ['system:time_start']))
        #rbf_weight = rbf_weight.updateMask(img.select(0).mask())  # mask weight band!!!
        return weighted.addBands(rbf_weight.rename('rbf_weight'))
    return wrap

def tsi_rbf(sigma=16):
    def wrap(img):
        img = ee.Image(img)
        # get collection and add weight band
        imgcol = ee.ImageCollection.fromImages(img.get('window1'))
        imgcol = imgcol.map(rbf_weight(sigma))
        # calculate weighted sum
        bandNames = imgcol.first().bandNames()
        reduced = imgcol.reduce(ee.Reducer.sum())
        result = reduced.divide(reduced.select('rbf_weight_sum')).rename(bandNames)
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
        reduced1 = imgcol_window1.reduce(ee.Reducer.sum())
        reduced2 = imgcol_window2.reduce(ee.Reducer.sum())
        result1 = (reduced1.divide(reduced1.select('rbf_weight_sum'))).multiply(weight)
        result2 = (reduced2.divide(reduced2.select('rbf_weight_sum'))).multiply(weight2)
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
        reduced1 = imgcol_window1.reduce(ee.Reducer.sum())
        reduced2 = imgcol_window2.reduce(ee.Reducer.sum())
        reduced3 = imgcol_window3.reduce(ee.Reducer.sum())
        result1 = (reduced1.divide(reduced1.select('rbf_weight_sum'))).multiply(weight)
        result2 = (reduced2.divide(reduced2.select('rbf_weight_sum'))).multiply(weight2)
        result3 = (reduced3.divide(reduced3.select('rbf_weight_sum'))).multiply(weight3)
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
        t = ee.Image.constant(img.get('system:time_start')).rename('t').toFloat()
        time_img = ee.Image.cat([t1, t2, t])
        linear_ratio = time_img.expression(
            '(t - t1) / (t2 - t1)',
            {
                't': time_img.select('t'),
                't1': time_img.select('t1'),
                't2': time_img.select('t2'),
            }
        )
        linear_ratio = linear_ratio.where(t1.eq(t2), 0)
        mosaic_before = mosaic_before.select(band)
        mosaic_after = mosaic_after.select(band)
        interpolated = ee.Image(mosaic_before.add((mosaic_after.subtract(mosaic_before).multiply(linear_ratio)))).rename(band)
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

# --------------------------------------------------------------------------------------------------------------------------------
# client-side: tables (df) and images/arrays
import re
from datetime import datetime
import pandas as pd
import numpy as np
from osgeo import gdal
from tqdm import tqdm

# core RBF interpolation functions (used for images, arrays and dfs)

def _rbf_build_target_times(times_np, step_days):
    t_min = times_np.min().astype('datetime64[D]')
    t_max = times_np.max().astype('datetime64[D]')
    return np.arange(t_min, t_max + np.timedelta64(step_days, 'D'), np.timedelta64(step_days, 'D')).astype('datetime64[ns]')

def _rbf_build_weights(times_np, target_times_np, sigma, win):
    td = times_np.astype('datetime64[D]').astype(int)
    Td = target_times_np.astype('datetime64[D]').astype(int)
    deltas = td[:,None] - Td[None,:]          # (t,T)
    mask = np.abs(deltas) <= win
    w = np.zeros_like(deltas, dtype='float32')
    w[mask] = np.exp(-0.5 * (deltas[mask] / sigma)**2)
    return w.astype('float32'), mask

def _rbf_blend(interps, counts, bw1, bw2, mode):
    if mode == '1RBF':
        return interps[0]
    # build raw weights
    if mode == '2RBF':
        c1 = counts[0]
        w1 = np.minimum(c1 / bw1, 1.0)
        w2 = 1.0 - w1
        weights = [w1, w2]
    elif mode == '3RBF':
        c1, c2 = counts[0], counts[1]
        w1 = np.minimum(c1 / bw1, 1.0)
        rem = 1.0 - w1
        w2 = np.minimum(c2 / bw2, 1.0) * rem
        w3 = 1.0 - (w1 + w2)
        weights = [w1, w2, w3]
    else:
        raise ValueError("mode must be '1RBF','2RBF','3RBF'")

    # stack arrays for vectorized blending
    interps_stack = np.stack(interps, axis=0)
    weights_stack = np.stack(weights, axis=0)
    valid_stack = ~np.isnan(interps_stack)
    any_valid = valid_stack.any(axis=0)
    weights_stack = np.where(valid_stack, weights_stack, 0.0)

    # renormalize weights where some (but not all) interps are NaN
    weight_sum = weights_stack.sum(axis=0, keepdims=True)  # (1,...)
    weights_norm = np.divide(
        weights_stack, weight_sum,
        out=np.zeros_like(weights_stack),
        where=weight_sum > 0
    )
    interps_filled = np.where(valid_stack, interps_stack, 0.0)
    blended = (interps_filled * weights_norm).sum(axis=0)
    blended = np.where(any_valid, blended, np.nan)
    return blended

def _rbf_interp_cube(data, weights_list, mask_list, mode, bw1, bw2):
    valid = ~np.isnan(data)
    data0 = np.where(valid, data, 0)
    interps = []
    counts = []
    # detect spatial vs vector shape
    if data.ndim == 3:  # (t,y,x)
        for w, m in zip(weights_list, mask_list):
            num = np.einsum('tyx,tT->Tyx', data0, w, optimize=True)
            den = np.einsum('tyx,tT->Tyx', valid.astype('float32'), w, optimize=True)
            interp = np.divide(num, den, out=np.full_like(num, np.nan), where=den>0)
            cnt = np.einsum('tyx,tT->Tyx', valid.astype('uint16'), m.astype('uint8'), optimize=True).astype('float32')
            interps.append(interp)
            counts.append(cnt)
    else:  # (t, n_vars)
        for w, m in zip(weights_list, mask_list):
            num = (data0.T @ w)
            den = (valid.astype('float32').T @ w)
            interp = np.divide(num, den, out=np.full_like(num, np.nan), where=den>0)
            cnt = (valid.T @ m.astype('float32'))
            interps.append(interp)
            counts.append(cnt)
    return _rbf_blend(interps, counts, bw1, bw2, mode)

# DF RBF

def _rbf_interpolate_timeseries(times_np, values_2d, mode,
                                step_days, sigma1, win1, sigma2, win2, sigma3, win3, bw1, bw2,
                                target_times=None):
    if target_times is None:
        target_times = _rbf_build_target_times(times_np, step_days)
    else:
        target_times = np.array(target_times, dtype='datetime64[ns]')

    is_1d = (values_2d.ndim == 1)
    if is_1d:
        values_2d = values_2d[:, None]

    scales = [(sigma1, win1)]
    if mode in ('2RBF','3RBF'): scales.append((sigma2, win2))
    if mode == '3RBF': scales.append((sigma3, win3))
    weights_list = []
    mask_list = []
    for sigma, win in scales:
        w, m = _rbf_build_weights(times_np, target_times, sigma, win)
        weights_list.append(w); mask_list.append(m)
    out = _rbf_interp_cube(values_2d, weights_list, mask_list, mode, bw1, bw2)  # shape (n_vars,T)
    if is_1d:
        out = out[0]
    return out, target_times

def tsi_rbf_df(
    df,
    group_col=None,
    value_cols=None,
    date_col=None,
    mode='1RBF',
    step_days=16,
    sigma1=16, win1=16,
    sigma2=32, win2=32,
    sigma3=64, win3=64,
    bw1=4, bw2=8,
    target_times=None,
    nodata_value=-9999,
    drop_all_nan=True
):
    """
    Interpolate values in a DataFrame using RBF, optionally grouped by group_col.
    If group_col is None, interpolate all rows together.
    If date_col is None, the DataFrame index is used as datetime.
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    elif value_cols is not None:
        value_cols = list(value_cols)

    out_frames = []
    print('Interpolating DF ...')
    if group_col is not None:
        group_iter = df.groupby(group_col)
        if value_cols is None:
            exclude = {group_col}
            if date_col is not None:
                exclude.add(date_col)
            value_cols = [c for c in df.columns if c not in exclude]
    else:
        group_iter = [(None, df)]
        if value_cols is None:
            raise ValueError("value_cols must be specified if group_col is None")

    for g, gdf in tqdm(group_iter):
        if date_col is None:
            times_raw = gdf.index
            if not np.issubdtype(times_raw.dtype, np.datetime64):
                try:
                    times_raw = pd.to_datetime(times_raw)
                except Exception as err:
                    raise ValueError(f"Index must be datetime64 or convertible to datetime when date_col is None: {err}")
        else:
            if date_col not in gdf.columns:
                raise ValueError(f"date_col '{date_col}' not found in DataFrame columns")
            times_raw = gdf[date_col]
            if not np.issubdtype(times_raw.dtype, np.datetime64):
                try:
                    times_raw = pd.to_datetime(times_raw)
                except Exception as err:
                    raise ValueError(f"date_col '{date_col}' must be datetime64 or convertible to datetime: {err}")

        # Drop rows with null/NaT timestamps
        valid_time_mask = pd.notna(times_raw)
        if not valid_time_mask.all():
            gdf = gdf.loc[valid_time_mask]
            times_raw = times_raw[valid_time_mask]

        if gdf.empty:
            continue

        sub = gdf[value_cols].copy()
        if nodata_value is not None:
            sub = sub.replace(nodata_value, np.nan)

        if drop_all_nan:
            valid_val_mask = sub.notna().any(axis=1)
            sub = sub.loc[valid_val_mask]
            times_raw = times_raw[valid_val_mask]

        if sub.empty:
            continue

        times_np = np.asarray(times_raw, dtype='datetime64[ns]')
        arr = sub.to_numpy(dtype='float32')

        # Ensure observations are in chronological order
        if len(times_np) > 1 and not np.all(times_np[:-1] <= times_np[1:]):
            order = np.argsort(times_np)
            times_np = times_np[order]
            arr = arr[order]

        out_arr, grp_target_times = _rbf_interpolate_timeseries(
            times_np, arr, mode,
            step_days, sigma1, win1, sigma2, win2, sigma3, win3, bw1, bw2,
            target_times=target_times
        )
        data_dict = {col: out_arr[i] for i, col in enumerate(value_cols)}
        tmp = pd.DataFrame(data_dict, index=pd.to_datetime(grp_target_times))
        if group_col is not None:
            tmp.insert(0, group_col, g)
        out_frames.append(tmp)

    if not out_frames:
        if group_col is not None:
            return pd.DataFrame(columns=[group_col, *value_cols])
        else:
            return pd.DataFrame(columns=value_cols)

    long_df = pd.concat(out_frames)
    long_df.index.name = 'time'
    return long_df


# array interface

def tsi_rbf_array(data, band_dates, step_days=16,
                          mode='3RBF',
                          sigma1=16, win1=16,
                          sigma2=32, win2=32,
                          sigma3=64, win3=64,
                          bw1=4, bw2=8,
                          target_times=None,
                          nodata_value=-9999,
):
    if isinstance(band_dates[0], str):
        times = np.array([np.datetime64(datetime.strptime(d, "%Y%m%d")) for d in band_dates])
    else:
        times = np.array(band_dates, dtype='datetime64[ns]')
    if target_times is None:
        target_times = _rbf_build_target_times(times, step_days)
    else:
        target_times = np.array(target_times, dtype='datetime64[ns]')
    scales = [(sigma1, win1)]
    if mode in ('2RBF','3RBF'): scales.append((sigma2, win2))
    if mode == '3RBF': scales.append((sigma3, win3))
    weights_list = []
    mask_list = []
    for sigma, win in scales:
        w, m = _rbf_build_weights(times, target_times, sigma, win)
        weights_list.append(w)
        mask_list.append(m)
    if nodata_value is not None:
        data = data.astype('float32')
        data = np.where(data == nodata_value, np.nan, data)
    out = _rbf_interp_cube(data, weights_list, mask_list, mode, bw1, bw2)  # (T,y,x)
    new_dates = [np.datetime_as_string(t, unit='D').replace('-','') for t in target_times]
    return out, new_dates

# image/tiff interface

def _parse_dates_from_gdal(ds, prefix=None):
    dates = []
    for i in range(1, ds.RasterCount+1):
        desc = ds.GetRasterBand(i).GetDescription()
        if not desc:
            raise ValueError(f'missing band description for band {i}')
        if prefix and not desc.startswith(prefix):
            continue
        m_all = list(re.finditer(r'(\d{8})', desc))
        if not m_all:
            raise ValueError(f'no datestring in band name: {desc}')
        ym = m_all[-1].group(1)
        try:
            dt = np.datetime64(datetime.strptime(ym, "%Y%m%d"))
        except ValueError:
            raise ValueError(f'invalid date in band name: {desc}')
        dates.append((i, dt, desc))
    if not dates:
        raise ValueError('no bands matched date pattern')
    dates.sort(key=lambda x: x[1])
    band_indices = [i for i,_,_ in dates]
    times = np.array([t for _,t,_ in dates])
    if prefix is None:
        first_desc = dates[0][2]
        if re.search(r'_\d{8}$', first_desc):
            prefix = first_desc[:-9]
        else:
            prefix = first_desc
    return band_indices, times, prefix

def _iter_windows(xsize, ysize, chunk_x, chunk_y):
    for yoff in range(0, ysize, chunk_y):
        ywin = min(chunk_y, ysize - yoff)
        for xoff in range(0, xsize, chunk_x):
            xwin = min(chunk_x, xsize - xoff)
            yield xoff, yoff, xwin, ywin

def _process_block_worker(args):
    (src_path, band_indices, xoff, yoff, xwin, ywin, nodata,
     weights_list, mask_list, mode, bw1, bw2) = args
    ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    arr_list = []
    for bi in band_indices:
        a = ds.GetRasterBand(bi).ReadAsArray(xoff, yoff, xwin, ywin).astype('float32', copy=False)
        arr_list.append(a)
    cube = np.stack(arr_list, axis=0)
    if nodata is not None:
        cube[cube == nodata] = np.nan
    out_block = _rbf_interp_cube(cube, weights_list, mask_list, mode, bw1, bw2)  # (T,y,x)
    ds = None
    return (xoff, yoff, out_block)

def tsi_rbf_tif(src_path, dst_path=None, 
                             step_days=16, mode='3RBF',
                             sigma1=16, win1=16,
                             sigma2=32, win2=32,
                             sigma3=64, win3=64,
                             bw1=4, bw2=8,
                             chunk_x=None, chunk_y=None,
                             creation_options=None,
                             n_cores=1,
                             in_prefix=None, out_prefix=None,
                             nodata_value=None):

    src = gdal.Open(src_path, gdal.GA_ReadOnly)
    band_indices, times, detected_prefix = _parse_dates_from_gdal(src, prefix=in_prefix)
    if out_prefix is None:
        out_prefix = detected_prefix
    target_times = _rbf_build_target_times(times, step_days)
    scales = [(sigma1, win1)]
    if mode in ('2RBF','3RBF'): scales.append((sigma2, win2))
    if mode == '3RBF': scales.append((sigma3, win3))
    weights_list = []
    mask_list = []
    for sigma, win in scales:
        w, m = _rbf_build_weights(times, target_times, sigma, win)
        weights_list.append(w)
        mask_list.append(m)
    xsize = src.RasterXSize
    ysize = src.RasterYSize
    # set chunk defaults only if None
    if chunk_x is None or chunk_y is None:
        bx, by = src.GetRasterBand(1).GetBlockSize()
        if chunk_x is None: chunk_x = bx if bx else 512
        if chunk_y is None: chunk_y = by if by else 512
    # clamp >0
    chunk_x = max(1, min(chunk_x, xsize))
    chunk_y = max(1, min(chunk_y, ysize))
    if creation_options is None:
        creation_options = ['COMPRESS=DEFLATE']
    driver = gdal.GetDriverByName('GTiff')
    in_dtype = src.GetRasterBand(1).DataType
    if in_dtype in (gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32):
        out_dtype = in_dtype
    else:
        out_dtype = gdal.GDT_Float32
    
    if not dst_path:
        dst_path = src_path.replace('.tif', f'_{mode}.tif')

    out_ds = driver.Create(
        dst_path,
        xsize, ysize,
        len(target_times),
        out_dtype,
        options=creation_options
    )
    out_ds.SetGeoTransform(src.GetGeoTransform())
    out_ds.SetProjection(src.GetProjection())

    if nodata_value is None:
        nodata_value = src.GetRasterBand(1).GetNoDataValue()

    # process
    if n_cores <= 1:
        for xoff, yoff, xwin, ywin in _iter_windows(xsize, ysize, chunk_x, chunk_y):
            arr_list = []
            for bi in band_indices:
                a = src.GetRasterBand(bi).ReadAsArray(xoff, yoff, xwin, ywin).astype('float32', copy=False)
                arr_list.append(a)
            cube = np.stack(arr_list, axis=0)
            if nodata_value is not None:
                cube[cube == nodata_value] = np.nan
            out_block = _rbf_interp_cube(cube, weights_list, mask_list, mode, bw1, bw2)
            for ti in range(out_block.shape[0]):
                out_ds.GetRasterBand(ti+1).WriteArray(out_block[ti], xoff, yoff)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        tasks = []
        with ProcessPoolExecutor(max_workers=n_cores) as ex:
            for xoff, yoff, xwin, ywin in _iter_windows(xsize, ysize, chunk_x, chunk_y):
                args = (src_path, band_indices, xoff, yoff, xwin, ywin, nodata_value,
                        weights_list, mask_list, mode, bw1, bw2)
                tasks.append(ex.submit(_process_block_worker, args))
            for fut in as_completed(tasks):
                xoff, yoff, out_block = fut.result()
                for ti in range(out_block.shape[0]):
                    out_ds.GetRasterBand(ti+1).WriteArray(out_block[ti], xoff, yoff)
    if nodata_value is None:
        nodata_value = np.nan
    for i, t in enumerate(target_times, start=1):
        ts = np.datetime_as_string(t, unit='D').replace('-','')
        b = out_ds.GetRasterBand(i)
        b.SetDescription(f'{out_prefix}_{ts}')
        b.SetNoDataValue(nodata_value)
    out_ds.FlushCache()
    out_ds = None
    src = None
    return dst_path

# --------------------------------------------------------------------------------------------------------------------------------
# plotting: TSS observations + client-side RBF interpolation

def _to_lonlat(roi):
    """Accept [lon, lat], (lon, lat) or ee.Geometry.Point -> (lon, lat)."""
    if isinstance(roi, ee.Geometry):
        roi = roi.centroid(1).coordinates().getInfo()
    lon, lat = roi[:2]
    return float(lon), float(lat)


def _unmasked_prm(prm, roi):
    """Slim copy of prm: ROI replaced, all masking disabled, no TSI/STM/export."""
    p = dict(prm)
    p['ROI'] = list(roi)
    # masking (parameter_blueprint.yml / level2.py)
    p['MASKS_LANDSAT'] = None
    p['MASKS_S2'] = None
    p['MASKS_HLS'] = None
    p['BLUE_MAX_MASKING'] = None
    p['ERODE_DILATE'] = False
    # not needed for the unmasked sample
    p['TSI'] = None
    p['STM'] = None
    p['PBC'] = None
    p['LSP'] = None
    p['NVO'] = False
    p['UMX'] = None
    for k in list(p):
        if k.startswith('EXPORT_'):
            p[k] = False
    return p


def _tss_to_df(imgcol, roi, feature, scale):
    from geeo import getRegion  # lazy import (avoid circular import)
    df = getRegion(imgcol, list(roi), scale=scale)
    df.index = pd.to_datetime(df.index)
    # several tiles/images per day -> one value per day
    return df[[feature]].astype(float).groupby(level=0).mean().sort_index()


def plot_tsi_timeseries(
    source,
    roi=None,
    feature='NDVI',
    unmasked=None,
    scale=20,
    interp=None,          # dict overriding interpolation settings (mode, step_days, sigma1, ...)
    show_masked=True,
    ax=None,
    color='orangered',
    legend_loc='lower left',
    ylim=None,
    boxplot=False,
    density_bar=True,      # stripe showing observation density per month (12 stripes per year)
    density_window=None,   # optional override for max density scale (vmax); None = auto
    density_cmap='gray_r',   # any matplotlib colormap name/object; default black (few) to white (many)
    figsize=None,          # (width, height) in inches; if None, defaults to (6.0, auto)
    coord_lines=1,         # 1 or 2 lines for coordinate label
    title='',              # optional text (e.g. site name) placed above the coordinate label
    savepath=None,        # e.g. 'plot.pdf' (svg/png also work)
    return_data=False,
):
    """
    Plot valid observations (black), masked observations (grey) and an RBF interpolation
    for a single point, based on the Time-Series-Stack (TSS).

    source : dict | str | dict(level3) | ee.ImageCollection
        - prm dict or path to a prm json file -> run_param is called (ROI optionally overridden)
        - level3 result dict of an earlier run_param -> level3['TSS'] is used
        - ee.ImageCollection (TSS) -> used directly
    roi : [lon, lat] | ee.Geometry.Point, optional
        Point used for sampling. Required for ImageCollection / level3 inputs
        (or falls back to nothing -> error), overrides prm['ROI'] for prm inputs.
    unmasked : ee.ImageCollection | level3 dict, optional
        TSS without masking for the grey points. Created automatically for prm inputs.
    interp : dict, optional
        Overrides of RBF settings. Defaults are read from the prm (TSI, INTERVAL, SIGMA1, WIN1, ...)
        when available, otherwise 2RBF / 12 days / 20,20 / 40,60.
    """
    import json
    import matplotlib.pyplot as plt
    from geeo import run_param  # lazy import

    prm = None
    # ---- resolve input -> masked TSS (+ optional unmasked TSS) ---------------------------------
    if isinstance(source, str):
        with open(source) as f:
            source = json.load(f)

    if isinstance(source, ee.ImageCollection):
        tss = source
    elif isinstance(source, dict) and 'TSS' in source:        # already-run prm (level3 dict)
        tss = source['TSS']
        prm = source
    elif isinstance(source, dict):                             # prm
        prm = dict(source)
        if roi is not None:
            prm['ROI'] = list(_to_lonlat(roi))
        roi = _to_lonlat(prm['ROI'])                           # must be a point for reduceRegions
        prm['ROI'] = list(roi)
        tss = run_param(prm)['TSS']
        if show_masked and unmasked is None:
            unmasked = run_param(_unmasked_prm(prm, roi))['TSS']
    else:
        raise TypeError("source must be prm dict/path, level3 dict or ee.ImageCollection")

    if roi is None:
        raise ValueError("roi is required when source is an ImageCollection or an already-run level3 dict")
    roi = _to_lonlat(roi)

    if isinstance(unmasked, dict):
        unmasked = unmasked['TSS']

    # ---- to pandas ----------------------------------------------------------------------------
    df_valid = _tss_to_df(tss, roi, feature, scale)
    df_masked = None
    if show_masked and unmasked is not None:
        df_all = _tss_to_df(unmasked, roi, feature, scale)
        # masked = observed without mask, but missing/NaN with mask
        valid_idx = df_valid[feature].dropna().index
        df_masked = df_all[~df_all.index.isin(valid_idx)].dropna()
        if df_masked.empty:
            import warnings
            warnings.warn("show_masked=True but no masked observations found "
                          "(unmasked TSS identical to masked TSS?). Check the MASKS_* keys of your prm.")

    # ---- interpolation settings ---------------------------------------------------------------
    p = prm or {}
    cfg = dict(mode=p.get('TSI', '2RBF'), step_days=p.get('INTERVAL', 12),
               sigma1=p.get('SIGMA1', 20), win1=p.get('WIN1', 20),
               sigma2=p.get('SIGMA2', 40), win2=p.get('WIN2', 60),
               sigma3=p.get('SIGMA3', 64), win3=p.get('WIN3', 64),
               bw1=p.get('BW1', 8), bw2=p.get('BW2', 16))
    if not isinstance(cfg['mode'], str):
        cfg['mode'] = '2RBF'
    cfg.update(interp or {})

    df_obs = df_valid.dropna()
    df_tsi = tsi_rbf_df(df_obs, value_cols=feature, nodata_value=None, **cfg)

    # x-axis boundaries: YEAR_MIN / YEAR_MAX of the prm (Jan 1st of YEAR_MIN to Jan 1st after YEAR_MAX);
    # DATE_MIN/DATE_MAX take precedence if set; fall back to the TSI range snapped to whole years
    t_min, t_max = df_tsi.index.min(), df_tsi.index.max()
    y_first = y_last = None
    if prm:
        if prm.get('DATE_MIN') and prm.get('DATE_MAX'):
            y_first, y_last = int(str(prm['DATE_MIN'])[:4]), int(str(prm['DATE_MAX'])[:4])
        elif prm.get('YEAR_MIN') is not None and prm.get('YEAR_MAX') is not None:
            y_first, y_last = int(prm['YEAR_MIN']), int(prm['YEAR_MAX'])
    if y_first is not None:
        x0 = pd.Timestamp(f'{y_first}-01-01')
        x1 = pd.Timestamp(f'{y_last + 1}-01-01')
    else:
        x0 = pd.Timestamp(f'{t_min.year}-01-01')
        x1 = pd.Timestamp(f'{t_max.year}-01-01')
        if t_max > x1:
            x1 = pd.Timestamp(f'{t_max.year + 1}-01-01')
    years = [y for y in range(x0.year, x1.year + 1)
             if x0 <= pd.Timestamp(f'{y}-01-01') <= x1]

    # ---- data density per month (12 stripes per year) ------------------------------------------
    # Generate monthly edge boundaries from x0 to x1
    month_edges = pd.date_range(x0, x1, freq='MS')
    if month_edges[-1] < x1:
        month_edges = month_edges.append(pd.DatetimeIndex([x1]))
    # Count valid observations occurring within each monthly interval [start, end)
    m_starts = month_edges[:-1]
    m_ends = month_edges[1:]
    obs_dt = df_obs.index
    dens = np.array([int(((obs_dt >= s) & (obs_dt < e)).sum()) for s, e in zip(m_starts, m_ends)])
    dens_vmax = density_window if density_window is not None else max(int(dens.max()), 1)

    # ---- figure -------------------------------------------------------------------------------
    from matplotlib.gridspec import GridSpec
    LW = 1.5
    def _frame(a):
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(LW)
            sp.set_color('black')
    own_fig = ax is None
    ax_box = ax_coord = ax_leg = ax_dens = ax_corner = None
    if own_fig:
        fig_w = figsize[0] if (figsize is not None and len(figsize) >= 1) else 6.0
        L, R, T, B = 0.10, 0.92, 0.97, 0.09
        wr_full = [2.0, 1.0, 0.12]                       # legend : coordinates : boxplot
        box_w = fig_w * (R - L) * wr_full[2] / sum(wr_full)   # width of boxplot column in inches
        # rows: header | data plot | density bar | NVO count bar; year labels sit below count bar
        bar_h = 0.22  # height of density and NVO subboxes in inches
        if figsize is not None and len(figsize) >= 2 and figsize[1] is not None:
            tot_h = figsize[1]
            # scale the main data plot height to fill the requested figure height
            sub_h = 0.70 + (2 * bar_h if density_bar else 0.0)
            main_h = max(tot_h * (T - B) - sub_h, 0.5)
            hr = [0.70, main_h] + ([bar_h, bar_h] if density_bar else [])
        else:
            hr = [0.70, 2.2] + ([bar_h, bar_h] if density_bar else [])
            tot_h = sum(hr) / (T - B)
        fig = plt.figure(figsize=(fig_w, tot_h))
        wr = wr_full if boxplot else wr_full[:2]
        gs = GridSpec(len(hr), len(wr), figure=fig, width_ratios=wr, height_ratios=hr,
                      wspace=0, hspace=0, left=L, right=R, top=T, bottom=B)
        r_main = 1
        ax = fig.add_subplot(gs[r_main, :2])
        ax_leg = fig.add_subplot(gs[0, 0])
        ax_coord = fig.add_subplot(gs[0, 1])
        if density_bar:
            ax_dens = fig.add_subplot(gs[2, :2], sharex=ax)
            ax_nvo = fig.add_subplot(gs[3, :2], sharex=ax)
            if boxplot:
                ax_corner = None   # no frame in the empty corner below the boxplot
        if boxplot:
            ax_box = fig.add_subplot(gs[r_main, 2], sharey=ax)
    else:
        fig = ax.figure
    if df_masked is not None and len(df_masked):
        ax.plot(df_masked.index, df_masked[feature], 'x', color='grey', alpha=0.5,
                markersize=4, label='Masked Observation')
    ax.plot(df_obs.index, df_obs[feature], 'kx', markersize=4, label='Valid Observation')
    unit_str = prm.get('INTERVAL_UNIT', 'day') if prm else 'day'
    interp_label = f"Interpolated Value ({cfg['step_days']}-{unit_str})"
    ax.plot(df_tsi.index, df_tsi[feature], color=color, marker='.', markersize=5, linewidth=1,
            label=interp_label)
    ax.set_ylabel(feature)
    ax.grid(True, color='black', alpha=0.2, linewidth=1)
    _frame(ax)
    ax.tick_params(width=LW, length=6)
    # x-axis: limits snapped outwards to whole years (Jan 1st) around the interpolated (TSI) time range
    ax.set_xlim(x0, x1)
    ax.set_xticks([pd.Timestamp(f'{y}-01-01') for y in years])
    ax.set_xticklabels([str(y) for y in years])
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        v_min = df_obs[feature].min()
        v_max = df_obs[feature].max()
        rng = v_max - v_min if v_max > v_min else 1.0
        # Determine tick step (e.g. 0.25, 0.5, 1, 2, etc.)
        raw_step = rng / 5.0
        magnitude = 10 ** np.floor(np.log10(raw_step))
        residual = raw_step / magnitude
        if residual > 5:
            step = 5 * magnitude
        elif residual > 2:
            step = 2 * magnitude
        else:
            step = magnitude
        # Add buffer and round outward to nearest tick multiple
        buf = 0.5 * step
        y_lower = np.floor((v_min - buf) / step) * step
        y_upper = np.ceil((v_max + buf) / step) * step
        ax.set_ylim(y_lower, y_upper)

    lon, lat = roi
    lon_s = f"{abs(lon):.4f}°{'E' if lon >= 0 else 'W'}"
    lat_s = f"{abs(lat):.4f}°{'N' if lat >= 0 else 'S'}"
    coord_str = f"{lon_s}\n{lat_s}" if coord_lines == 2 else f"{lon_s} {lat_s}"
    handles, labels = ax.get_legend_handles_labels()
    # order: valid, masked, interpolation
    _order = sorted(range(len(labels)), key=lambda i: (0 if labels[i].startswith('Valid') else
                                                       1 if labels[i].startswith('Masked') else 2, i))
    handles = [handles[i] for i in _order]
    labels = [labels[i] for i in _order]
    if own_fig:
        # header, left cell (2/3 width): legend, one entry per line
        ax_leg.set_xticks([]); ax_leg.set_yticks([])
        leg = ax_leg.legend(handles, labels, loc='center left', bbox_to_anchor=(0.0, 0.5), ncol=1,
                            frameon=False, fontsize=9, handlelength=1.8, labelspacing=0.35,
                            handletextpad=0.6, borderaxespad=0.0)
        ax_leg.set_frame_on(False)

        # header, right cell (1/3 width): coordinates aligned to bottom legend entry
        ax_coord.set_xticks([]); ax_coord.set_yticks([])
        try:
            fig.canvas.draw_idle()
            # Get the display coordinate of the lowermost legend text item
            renderer = fig.canvas.get_renderer()
            last_text = leg.get_texts()[-1]
            bbox_disp = last_text.get_window_extent(renderer=renderer)
            y_disp = (bbox_disp.y0 + bbox_disp.y1) / 2.0
            # Transform from display to ax_coord axes fraction
            y_coord = ax_coord.transAxes.inverted().transform((0, y_disp))[1]
        except Exception:
            y_coord = 0.22
        ax_coord.text(1.0, y_coord, coord_str, ha='right', va='center', fontsize=9,
                      linespacing=1.4, transform=ax_coord.transAxes)
        if title:
            # sits directly above the coordinate label (offset depends on number of coordinate lines)
            ax_coord.annotate(title, xy=(1.0, y_coord), xycoords='axes fraction',
                              xytext=(0, 22 if coord_lines == 2 else 13), textcoords='offset points',
                              ha='right', va='bottom', fontsize=9, fontweight='bold')
        ax_coord.set_frame_on(False)
        # data density bar ("climate stripes"): 12 monthly stripes per year
        if ax_dens is not None and len(dens):
            cmap_obj = plt.get_cmap(density_cmap)
            ax_dens.set_facecolor(cmap_obj(0.0))
            ax_dens.pcolormesh(month_edges.values, [0, 1], dens[None, :], cmap=density_cmap,
                               vmin=0, vmax=dens_vmax, shading='flat',
                               edgecolor='face', linewidth=0.3)
            ax_dens.set_xlim(ax.get_xlim())
            ax_dens.set_ylim(0, 1)
            ax_dens.set_yticks([])
            ax_dens.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            _frame(ax_dens)

            # vertical dividing tick lines between years going across density bar
            for y in years:
                ax_dens.axvline(pd.Timestamp(f'{y}-01-01'), color='black', linewidth=LW)
            # Label on the left side next to density subbox
            ax_dens.text(-0.02, 0.5, 'Density', ha='right', va='center',
                         fontsize=9, color='black', transform=ax_dens.transAxes)

            # NVO bar: white box at bottom showing number of valid observations per year
            ax_nvo.set_facecolor('white')
            ax_nvo.set_xlim(ax.get_xlim())
            ax_nvo.set_ylim(0, 1)
            ax_nvo.set_yticks([])
            # year ticks/labels live on the NVO count bar (bottom of the figure)
            ax_nvo.tick_params(axis='x', bottom=True, labelbottom=True, width=LW, length=6)
            for y in years:
                ax_nvo.axvline(pd.Timestamp(f'{y}-01-01'), color='black', linewidth=LW)
            for y in range(x0.year, x1.year):
                ya, yb = pd.Timestamp(f'{y}-01-01'), pd.Timestamp(f'{y + 1}-01-01')
                n_valid = int((df_obs.index.year == y).sum())
                ax_nvo.text(ya + (yb - ya) / 2, 0.5, str(n_valid),
                            ha='center', va='center', fontsize=9, color='black')
            _frame(ax_nvo)
            # Label on the left side next to count subbox
            ax_nvo.text(-0.02, 0.5, 'Count', ha='right', va='center',
                        fontsize=9, color='black', transform=ax_nvo.transAxes)
        if ax_corner is not None:
            ax_corner.set_xticks([]); ax_corner.set_yticks([])
            _frame(ax_corner)
    else:
        ax.legend(loc=legend_loc, frameon=True, facecolor='white', edgecolor='black',
                  fancybox=False, framealpha=1.0, fontsize=8)
    if ax_box is not None:
        box = ax_box.boxplot(
            [df_tsi[feature].dropna()], vert=True, patch_artist=True, widths=0.6,
            boxprops=dict(facecolor='black', color='black', linewidth=LW),
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(color='black', linewidth=LW, linestyle='-'),
            capprops=dict(color='black', linewidth=0),
            flierprops=dict(marker='o', markerfacecolor='black', markersize=3,
                            linestyle='none', markeredgecolor='none'))
        for cap in box['caps']:
            cap.set_visible(False)
        ax_box.set_xticks([])
        ax_box.yaxis.tick_right()
        ax_box.tick_params(axis='y', width=LW, length=6)
        _frame(ax_box)
        for yt in ax.get_yticks():
            ax_box.axhline(yt, color='black', alpha=0.2, linewidth=1)
        ax_box.set_ylim(ax.get_ylim())
    if own_fig and savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=300)
    if return_data:
        m_index = pd.date_range(x0, x1, freq='MS')[:-1]
        return fig, ax, dict(valid=df_obs, masked=df_masked, tsi=df_tsi, density=pd.Series(dens, index=m_index))
    return fig, ax

# EOF