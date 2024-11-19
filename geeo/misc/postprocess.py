# Import necessary libraries
from pathlib import Path
import csv
import os
from osgeo import gdal
gdal.DontUseExceptions()
from tqdm import tqdm

# Default settings
DEFAULT_PATTERN = "*.tif"
DEFAULT_NODATA = -9999
DEFAULT_DELETE_OLD = True

# Function to get band names from CSV file
def bandames_from_csv(file):
    with open(file, 'rU') as infile:
        reader = csv.DictReader(infile)
        data = {}
        for row in reader:
            for header, value in row.items():
                try:
                    data[header].append(value)
                except KeyError:
                    data[header] = [value]
    return data['name']

# Function to get band names from image
def bandnames_from_img(ds):
    bandnames = [ds.GetRasterBand(x+1).GetDescription() for x in range(ds.RasterCount)]
    return bandnames

# Function to process files
def process_ee_files(input_path, bandname_file=False, pattern="*.tif",
                  nodata=None, dtype=None, delete_old=True,
                  calc_stats=True, compress=True, pyramids=False,
                  num_threads='ALL_CPUS'):

    # Check if input_path is a directory or a file
    input_path = Path(input_path)
    if input_path.is_dir():
        l_files = [str(f) for f in input_path.rglob(pattern)]
    else:
        l_files = [str(input_path)]

    if dtype is None:
        dtype_print = "inferred"
    if nodata is None:
        nodata_print = "inferred"
    # print user output file, and options chosen
    print("********************************************************")
    print(f"Processing files in \n{input_path}")
    print(f"    * Bandname-file: {bandname_file}")
    if input_path.is_dir():
        print(f"    * Pattern: {pattern}")
    print(f"    * Delete old files: {delete_old}")
    print(f"    * Calculate statistics: {calc_stats}")
    print(f"    * Compress: {compress}")
    print(f"    * Data type: {dtype_print}")
    print(f"    * No data value: {nodata_print}")
    print(f"    * Pyramids: {pyramids}")
    print("********************************************************")
    
    gdal.SetConfigOption('GDAL_NUM_THREADS', str(num_threads))

    for f in tqdm(l_files):
        temp_file = f.replace(".tif", "_TEMP.tif")
        os.rename(f, temp_file)
        
        # Infer data type if not specified
        if dtype is None:
            ds = gdal.Open(temp_file)
            dtype = ds.GetRasterBand(1).DataType
            ds = None
        if nodata is None:
            ds = gdal.Open(temp_file)
            nodata = ds.GetRasterBand(1).GetNoDataValue()
            ds = None

        # Build translate options / add compression if desired
        opt = gdal.TranslateOptions(
            format="GTiff", noData=nodata, outputType=dtype, creationOptions=["COMPRESS=DEFLATE"] if compress else []
        )
        
        # Translate
        ds = gdal.Open(temp_file)
        ds = gdal.Translate(f, ds, options=opt)
        ds = None
        
        if bandname_file:
            f_bandnames = f.replace('.tif', '_bandnames.csv')
            bandnames = bandames_from_csv(f_bandnames)
        else:
            bandnames = bandnames_from_img(gdal.Open(temp_file))
        
        # Rename bands and calculate statistics
        img = gdal.Open(f)
        for i in range(img.RasterCount):
            band = img.GetRasterBand(i + 1)
            band.SetDescription(bandnames[i])
            if nodata is not None:
                band.SetNoDataValue(nodata)
            if calc_stats:
                stats = band.GetStatistics(True, True)
                band.SetStatistics(stats[0], stats[1], stats[2], stats[3])
            band = None
        img = None
        
        # Add pyramids
        if pyramids:
            ds = gdal.Open(f)
            ds.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64, 128])
            ds = None

        if delete_old:
            os.remove(temp_file)

# Command-line interface (CLI) handling
if __name__ == "__main__":
    import argparse
    
    # Create the parser
    parser = argparse.ArgumentParser(description="Process TIFF files in a directory or a single file.")
    
    # Add arguments
    parser.add_argument("path", type=str, help="File or directory path to process.")
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN, help="Pattern to match files (default: '*.tif'). Used only when processing a directory.")
    parser.add_argument("--bandname_file", action="store_true", help="Use bandname CSV files if present.")
    parser.add_argument("--nodata", type=int, default=DEFAULT_NODATA, help="No data value for the output (default: -9999).")
    parser.add_argument("--dtype", type=int, help="Data type for the output (inferred if not specified).")
    parser.add_argument("--delete_old", action="store_true", default=DEFAULT_DELETE_OLD, help="Delete the temporary files after processing (default: True).")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the processing function with the arguments
    process_ee_files(input_path=args.path, bandname_file=args.bandname_file,
                  pattern=args.pattern, nodata=args.nodata, dtype=args.dtype, delete_old=args.delete_old)