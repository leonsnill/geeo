import numpy as np
from osgeo import gdal
from scipy.interpolate import interp1d
from tqdm import tqdm

# Function to fill nodata values over time
def fill_nodata_time(data, nodata_value=-9999):
    """
    Fill the nodata values in the data array over time (across the z-dimension).
    """
    filled_data = np.copy(data)
    num_bands, rows, cols = data.shape
    
    # Iterate over each pixel location
    for i in tqdm(range(rows)):
        for j in range(cols):
            time_series = data[:, i, j]
            
            # Find the indices where the data is not nodata
            valid_idx = np.where(time_series != nodata_value)[0]
            
            if valid_idx.size > 0:
                # Find the indices where the data is nodata
                nodata_idx = np.where(time_series == nodata_value)[0]
                
                if nodata_idx.size > 0:
                    # Interpolate the valid values
                    f = interp1d(valid_idx, time_series[valid_idx], kind='linear', bounds_error=False, fill_value='extrapolate')
                    filled_values = f(nodata_idx)
                    
                    # Replace nodata values with interpolated values
                    filled_data[nodata_idx, i, j] = filled_values
    
    return filled_data

# Read the GeoTIFF file
def read_geotiff(filepath):
    dataset = gdal.Open(filepath)
    if dataset is None:
        raise ValueError(f"Could not open file {filepath}")
    
    bands_data = []
    for band in range(1, dataset.RasterCount + 1):
        band_data = dataset.GetRasterBand(band).ReadAsArray()
        bands_data.append(band_data)
    
    bands_data = np.stack(bands_data)
    
    return bands_data, dataset

# Function to get band names from image
def bandnames_from_img(ds):
    bandnames = [ds.GetRasterBand(x+1).GetDescription() for x in range(ds.RasterCount)]
    return bandnames

# Write the modified data to a new GeoTIFF file
def write_geotiff(filepath, data, dataset):
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(filepath, dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount, gdal.GDT_Int16)
    
    if out_dataset is None:
        raise ValueError(f"Could not create file {filepath}")
    
    # Set the GeoTransform and Projection from the original dataset
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())

    bandnames = bandnames_from_img(dataset)

    
    # Write each band data
    for i, band in enumerate(range(data.shape[0])):
        out_band = out_dataset.GetRasterBand(band + 1)
        out_band.WriteArray(data[band])
        out_band.SetDescription(bandnames[i])
        out_band.SetNoDataValue(-9999)
    
    out_dataset.FlushCache()

# Main function to handle the process
def fill_raster_nodata_over_time(input_file, output_file):
    # Read the data
    data, dataset = read_geotiff(input_file)
    
    # Fill nodata values over time
    filled_data = fill_nodata_time(data)
    
    # Write the filled data to a new file
    write_geotiff(output_file, filled_data, dataset)

# Example usage
input_file = '/Users/leonnill/Desktop/cabo/TSI_MZQ-CABO-DELGADO_2019-2023_SEN2_TCG.tif'
output_file = '/Users/leonnill/Desktop/cabo/TSI_MZQ-CABO-DELGADO_2019-2023_SEN2_TCG_inter.tif'
fill_raster_nodata_over_time(input_file, output_file)

print("Nodata values filled over time and saved to", output_file)
