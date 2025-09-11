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
    return np.arange(times_np.min(), times_np.max()+np.timedelta64(step_days,'D'), np.timedelta64(step_days,'D'))

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
                                step_days, sigma1, win1, sigma2, win2, sigma3, win3, bw1, bw2):
    target_times = _rbf_build_target_times(times_np, step_days)
    scales = [(sigma1, win1)]
    if mode in ('2RBF','3RBF'): scales.append((sigma2, win2))
    if mode == '3RBF': scales.append((sigma3, win3))
    weights_list = []
    mask_list = []
    for sigma, win in scales:
        w, m = _rbf_build_weights(times_np, target_times, sigma, win)
        weights_list.append(w); mask_list.append(m)
    out = _rbf_interp_cube(values_2d, weights_list, mask_list, mode, bw1, bw2)  # shape (n_vars,T)
    return out, target_times

def tsi_rbf_df(
    df,
    group_col,
    value_cols,
    date_col='YYYYMMDD',
    mode='1RBF',
    step_days=16,
    sigma1=16, win1=16,
    sigma2=32, win2=32,
    sigma3=64, win3=64,
    bw1=4, bw2=8,
    nodata_value=-9999,
    drop_all_nan=True
):
    out_frames = []
    print('Interpolating DF ...')
    for g, gdf in tqdm(df.groupby(group_col)):
        sub = gdf[value_cols].copy()
        sub = sub.replace(nodata_value, np.nan)
        if drop_all_nan:
            sub = sub.loc[sub.notna().any(axis=1)]
        if sub.empty:
            continue
        t_raw = gdf.loc[sub.index, date_col].values
        if np.issubdtype(gdf[date_col].dtype, np.datetime64):
            times_np = t_raw.astype('datetime64[ns]')
        else:
            times_np = np.array([np.datetime64(datetime.strptime(str(d), "%Y%m%d")) for d in t_raw])
        vals = sub.to_numpy(dtype='float32')  # (t,n)
        arr = vals
        out_arr, target_times = _rbf_interpolate_timeseries(
            times_np, arr, mode,
            step_days, sigma1, win1, sigma2, win2, sigma3, win3, bw1, bw2
        )
        data_dict = {col: out_arr[i] for i, col in enumerate(value_cols)}
        tmp = pd.DataFrame(data_dict, index=pd.to_datetime(target_times))
        tmp.insert(0, group_col, g)
        out_frames.append(tmp)
    if not out_frames:
        return pd.DataFrame(columns=[group_col, *value_cols])
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

# EOF