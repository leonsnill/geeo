from geeo.misc.postprocess import process_ee_files

# Define the input directory
inp = r"D:\DDesktop\denali"
nodata = -9999
pattern = "*.tif"

process_ee_files(inp, bandname_file=False, pattern=pattern, nodata=nodata)
