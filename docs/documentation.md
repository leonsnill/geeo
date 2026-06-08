# GEEO Parameter Documentation

Below is the detailed documentation of all available methods and parameter options for the main workflow of **GEEO**. 

For the documentation on how to start using and interacting with GEEO, please be referred to the [Getting Started](../README.md#getting-started) and [Introduction](/docs/tutorials/tutorial_0_introducing-geeo.ipynb) sections, accordingly.

## Modules and Settings Overview
[Level-2](#level-2)
- [Space and Time](#space-and-time)
- [Sensor and Data Quality Settings (Time Series Stack - TSS)](#sensor-and-data-quality-settings-time-series-stack---tss)
- [Bands \| Indices \| Features](#bands--indices--features)
- [Custom Image Collection (CIC)](#custom-image-collection-cic)
- [Time Series Mosaic (TSM)](#time-series-mosaic-tsm)

[Level-3](#level-3)
- [Temporal Subwindows \| Folding](#temporal-subwindows--folding)
- [Number of Valid Observations (NVO)](#number-of-valid-observations-nvo)
- [Time Series Interpolation (TSI)](#time-series-interpolation-tsi)
- [Spectral-Temporal-Metrics (STM)](#spectral-temporal-metrics-stm)
- [Pixel-Based Composites (PBC)](#pixel-based-composites-pbc)

[Level-4](#level-4)
- [Land Surface Phenology (LSP)](#land-surface-phenology-lsp)

[Export](#export)
- [Image Settings](#image-settings)
- [Products to Export](#products-to-export)
- [General Export Settings](#general-export-settings)

## LEVEL-2
The Level-2 module harmonises, standardises and merges the user-specified input image collections from the Earth Engine data catalogue. 

### SPACE AND TIME
These are the global spatial and temporal filters applied to the ImageCollections. 

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| `YEAR_MIN`                 | int     | 2023         | YYYY                        | Start year (inclusive). Not used when DATE_MIN and DATE_MAX are set. |
| `YEAR_MAX`                 | int     | 2023         | YYYY                       | End year (inclusive). Not used when DATE_MIN and DATE_MAX are set. |
| `MONTH_MIN`                | int     | 1            | [1, 12]                            | Start month (inclusive). Can be greater than MONTH_MAX to wrap (e.g. 11→2). |
| `MONTH_MAX`                | int     | 12           | [1, 12]                            | End month (inclusive). Can be lower than MONTH_MIN to wrap (e.g. 11→2)|
| `DOY_MIN`                  | int     | 1            | [1, 366]                           | Start day-of-year (inclusive). Can be greater than DOY_MAX to wrap (e.g. 300→31). |
| `DOY_MAX`                  | int     | 366          | [1, 366]                           | End day-of-year (inclusive). Can be lower than DOY_MIN to wrap (e.g. 300→31). |
| `DATE_MIN`                 | str     | null         | YYYYMMDD                        | Start date. Overrides YEAR/MONTH/DOY when paired with DATE_MAX. |
| `DATE_MAX`                 | str     | null         | YYYYMMDD                        | End date. Overrides YEAR/MONTH/DOY when paired with DATE_MIN. |
| `ROI`                      | list / str / GeoDataFrame / ee object | [12.9, 52.2, 13.9, 52.7] | Point [lon, lat]; BBox [xmin, ymin, xmax, ymax]; path to .shp/.gpkg; GeoDataFrame; ee.Geometry / ee.FeatureCollection | Region of Interest acting as primary spatial filter to ImageCollection. |
| `ROI_SIMPLIFY_GEOM_TO_BBOX`| bool    | true         | true, false                     | Simplifies input ROI to its bounding box for spatial filtering. This is usually desireable, since  providing a complex collection as geometry argument may result in unnecessary poor performance. Note that a scenario where setting this to false makes sense if a simple geometry covers a large geographic area but the bounding box would include images outside the region of interest (for examle when a point dataset is used as ROI) |
| `ROI_TILES`                 | bool    | false        | true, false                     | Use features in ROI vector as independent tiles (ROIs) for processing and export. Applies only when ROI is a vector file (shp/gpkg/GeoDataFrame). Requires ROI_TILES_ATTRIBUTE_COLUMN to be specified to append to output filenames. |
| `ROI_TILES_ATTRIBUTE_COLUMN`| str     | null         | column name or null             | Attribute column containing unique per-feature IDs; appended to EXPORT_DESC for per-tile outputs; Also used for subsetting via ROI_TILES_ATTRIBUTE_LIST. |
| `ROI_TILES_ATTRIBUTE_LIST`  | list    | null         | list of values or null          | ROI_TILES_ATTRIBUTE_LIST can be used if ROI_TILES is activated to select only the matching features found in ROI_TILES_ATTRIBUTE_COLUMN. The default (null) selects all features. |

- The `YEAR_*`, `MONTH_*`, and `DOY_*` parameters are combined to construct a logical *AND* filtering. Calendar year wrapping is considered for `MONTH_*`, and `DOY_*`, i.e. `MONTH_MIN` and `DOY_MIN` may be greater than `MONTH_MAX` and `DOY_MAX`, respectively (e.g. to only process imagery between November to February, set `MONTH_MIN: 11` and `MONTH_MAX: 2`).
- If `DATE_MIN` and `DATE_MAX` are specified, `YEAR_*`, `MONTH_*`, and `DOY_*` are ignored/overwritten.
- The Region-Of-Interest (`ROI`) can be specified as a simple list of coordinates (points, polygons), a client-side shp/gpkg file or GeoPandas GeoDataFrame, as well as a server-side ee.Geometry or ee.FeatureCollection.
For the spatial filtering, it is recommended to set `ROI_SIMPLIFY_GEOM_TO_BBOX: true ` since providing a complex vector file or FeatureCollection may result in poor performance. 

### SENSOR AND DATA QUALITY SETTINGS (TIME SERIES STACK - TSS)

The following settings concern the Time-Series-Stack (TSS). The TSS is the primary output and serves as initial input to subsequent algorithms. The TSS is an image collection consisting of individual images matching the spatial tiling schemes used by NASA (Landsat) and ESA (Sentinel-2) in which the spectral bands were named and scaled identically, the collections were spatially and temporally filtered to match the specified study area and time period (see [Space and Time](#space-and-time)), the imagery was quality filtered according to user-specified criteria (e.g., cloud masking), and the desired spectral bands and spectral indices were calculated. 

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| `SENSORS`                  | list    | [L9, L8, L7, L5, L4] | L9, L8, L7, L5, L4, S2, HLS, HLSL30, HLSS30 | The primary input ee.ImageCollections to be used for processing. <br> All available options are surface reflectance products: L9/L8/L7/L5/L4 -> LANDSAT Collection 2 Tier 1 Surface Reflectance product ([LANDSAT/LXXX/C02/T1_L2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2)); S2 -> Sentinel-2 ([COPERNICUS/S2_SR_HARMONIZED](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)). HLS/HLSL30/HLSS30 -> Harmonized Landsat ([HLSL30](https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSL30_v002)), Sentinel-2 ([HLSS30](https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSS30_v002)) product, and combined (HLS) product.  <br> L9/L8/L7/L5/L4 + S2 can be collectively selected to be merged into one TSS collection, e.g. [L9, L8, S2]. |
| `MAX_CLOUD`                | int     | 75           | 0-100                           | Maximum image scene cloud cover percentage for images to be included in processing. The metadata used for the sensors are "CLOUD_COVER_LAND" (L9/L8/L7/L5/L4), "CLOUDY_PIXEL_PERCENTAGE" for S2, and "CLOUD_COVERAGE" (HLS/HLSL30/HLSS30). |
| `EXCLUDE_SLCOFF`           | bool    | false        | true, false                     | Wether to exclude [Landsat-7 ETM+ Scan Line Corrector (SLC)-off](https://www.usgs.gov/faqs/what-landsat-7-etm-slc-data) scenes (i.e., L7 scenes after 2003-05-30). These scenes have (stripy) data gaps but retain radiometric and geometric integrity as prior to SLC failure. |
| `GCP_MIN_LANDSAT`          | int     | 1            | >=1                             | Minimum Ground Control Points (GCP) threshold for Landsat scenes to be included in processing. Minim threshold of 1 highly recommended since legacy images may be incorrectly georectified while still being included in the LANDSAT Collection 2 Tier 1 Surface Reflectance products ([LANDSAT/LXXX/C02/T1_L2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2)). |
| `MASKS_LANDSAT`            | list    | [cloud, cshadow, snow, fill, dilated] | (cloud, cshadow, snow, fill, dilated, saturated); null disables. | The [Pixel Quality Assessment (QA_PIXEL) band](https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands) provided with the LANDSAT products allow to mask out pixels affected by (potentially) quality affecting issues, including clouds, cloud shadows, snow/ice, brightness saturation, or fill values. Affects [L9, L8, L7, L5, L4] under SENSORS setting. <br> See [`geeo/level2/masking.py`](../geeo/level2/masking.py) for implementation details.  |
| `MASKS_LANDSAT_CONF`       | str     | Medium       | Medium, High                    | The Pixel Quality Assessment (QA_PIXEL) band of the LANDSAT data further allows setting a confidence level for the masking procedure. GEEO uses this setting for the cloud and cirrus cloud detection. The feault setting of Medium (compared to High) applies a more conservative, aggressive masking, where cloudy pixel-remnants after masking are less likely, hence favouring an error of commission (rather exclude a non-cloudy pixel than keeping a cloudy pixel) over an error of ommission (this is generally desireable over omitting clouds from masking). |
| `MASKS_S2`                 | str     | CPLUS        | null, CPLUS, PROB, SCL          | Settings for Sentinel‑2 (S2) cloud masking. Available options are Cloud Score Plus ([CPLUS](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED#description)) (highly recommended), the s2cloudless-based cloud probability layer [PROB](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY) (cloud probability), the Sentinel-2 product's internal Scene Classification band ([SCL](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)), or no masking (null). <br> See [`geeo/level2/masking.py`](../geeo/level2/masking.py) for implementation details. |
| `MASKS_S2_CPLUS`           | float   | 0.6          | 0-1                             | Cloud Score Plus ([CPLUS](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED#description)) threshold, where 0 represents "not clear" (occluded), while 1 represents "clear" (unoccluded) observations. For more information about the Cloud Score+ detection approach, see [here](https://medium.com/google-earth/all-clear-with-cloud-score-bd6ee2e2235e). |
| `MASKS_S2_PROB`            | int     | 30           | 0-100                           | Cloud probability (%) cutoff when using [PROB](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY). See [here](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY) for MASKS_S2. 40-60% can be considered a balanced default. |
| `MASKS_S2_NIR_THRESH_SHADOW`| float  | 0.2          | 0-1                             | When using PROB cloud masking, cloud shadows are detected and matched to clouds by identifying dark objects in the NIR band using a reflectance value threshold. See [here](https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless) for a Google tutorial.|
| `MASKS_HLS`                | list    | [cloud, cshadow, snow] | cloud, cshadow, snow, fill, dilated, saturated | Same procedure as MASKS_LANDSAT. See also [page 17 in HLS Product User Guide](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf) for details. <br> See [`geeo/level2/masking.py`](../geeo/level2/masking.py) for implementation details. |
| `ERODE_DILATE`             | bool    | false        | true, false                     | Option to apply morphological erosion and/or dilation to the specified MASK_*. Dilation adds pixels to the boundaries of the masked pixels in the image increasing the mask's pixel area, while erosion removes pixels on the mask's boundaries. This allows to (a) make sure that cloud borders are more likely excluded while small, individual pixels that are likely falsely masked are retained in the analysis. See Fig. 15 in [Qiu et al. 2019](https://doi.org/10.1016/j.rse.2019.05.024) for a visual demonstration of cloud erosion and dilation.  |
| `ERODE_RADIUS`             | int     | 60           | 0-Inf                          | Radius of the circular erosion kernel in meters. Can be set to 0 while DILATE_RADIUS is >0 to only dilate (and vice versa). |
| `DILATE_RADIUS`            | int     | 120          | 0-Inf                          | Radius of the circular dilation kernel in meters. usually ≥ erosion radius. Can be set to 0 while ERODE_RADIUS is >0 to only erode (and vice versa). |
| `ERODE_DILATE_SCALE`       | int     | 60           | meters                          | Pixel scale at which to perform the erosion-dilation operation (larger value = faster, because coarser pixel grain). For this kind of operation the performance advantage of using 60-120 meter scale (even when working with Sentinel-2 10m data) is usually desirable and accurate enough.|
| `BLUE_MAX_MASKING`         | float   | null         | 0-1                             | Additional masking option which uses the blue reflectance band to exclude pixels that exceed the specified threshold (e.g. haze/glint). Option is ignored if null or set to 0. |

The masking settings are set to very typical defaults of only considering scenes with less than 75% cloud cover, masking clouds, cloud shadows, snow/ice, as well as fill values. 
As such, in practise, when having created a new parameter file or when operating with python dictionaries in interactive sessions, only setting the SENSORS setting is usually enough to guarantee a quality filtered Time-Series-Stack (TSS) output suitable for most analyses. Of course, however, we encourage being aware and explicit about the desired settings and adjusting accordingly. For example, including snow or ice pixels in the analyses might be of central interest to certain applications and should be considered when performing the masking.

### BANDS | INDICES | FEATURES

In this section the user specifies the desired bands, indices and features to be included and/or calculated. Beyond the standard bands provided with the selected `SENSORS`, GEEO includes a range of predefined indices under the `FEATURES` paramater that can be selected. Additionally, users can specify `CUSTOM_FORMULAS` to calculate their own features. Furthermore, the spectal unmixing method provided in Earth Engine is accessible at this stage of processing.

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| `FEATURES`                 | list    | [BLU, GRN, RED, NIR, SW1, SW2, NDVI] | BLU, GRN, RED, NIR, SW1, SW2 (Landsat+Sentinel-2), LST (Landsat, if LST was processed for scene), RE1, RE2, RE3, RE4 (Sentinel-2), NDVI, EVI, NBR, NDMI, NDWI, MDWI, NDB, TCG, TCB, TCW, SWR | Bands / indices / custom formula outputs kept in pipeline; include DEM or unmixing outputs here for export. |
| `CUSTOM_FORMULAS`  | dict  | See parameter file    | For example: <br> INDEX_NAME: <br> formula: "(X-Y)/(X+Y)" <br> variable_map: {X: NIR, Y: RED} | Users can specify `CUSTOM_FORMULAS` to calculate their own features. Users must provide a name to be used as bandname, the formula for calculating the feature, and the bands used within the formula matching existing bandnames in the TSS. Custom formulas can be used in chains, i.e. output (NAME) from previous formula may be called in subsequent formula. Note that to append the custom formula to the TSS, also add the name to the `FEATURES` list |
| `UMX`                     | dict    | null         | {'Class1': [100, 230, ..., 123], 'Class2': [24, ...], ...}                 | Order in endmember dictionary must match order defined in FEATURES. For example, if {'PV': [0.08, 0.1, 0.07, 0.24, 0.11, 0.08], 'Soil': [0.2, 0.3, 0.3, 0.4, 0.45, 0.55]} and FEATURES [BLU, GRN, RED, NIR, SW1, SW2], then corresponding feature-value combinations will be used (e.g. Soil 0.2 for BLU). Attention must be paid that values in UMX match scale of FEATURES. Wrapper for [ee.Image.unmix](https://developers.google.com/earth-engine/apidocs/ee-image-unmix). See also [unmixing example](https://developers.google.com/earth-engine/guides/image_transforms#spectral-unmixing) from Earth Engine. <br> See [`unmix()`](../geeo/level2/indices.py) for implementation details. |
| `UMX_SUM_TO_ONE`           | bool    | true         | true, false                     | Enforce abundance fractions sum to 1. See [ee.Image.unmix](https://developers.google.com/earth-engine/apidocs/ee-image-unmix).|
| `UMX_NON_NEGATIVE`         | bool    | true         | true, false                     | Constrain abundance fractions to ≥ 0. See [ee.Image.unmix](https://developers.google.com/earth-engine/apidocs/ee-image-unmix).|
| `UMX_REMOVE_INPUT_FEATURES`| bool    | true         | true, false                     | Keep only abundance (unmixed) bands if true; otherwise append. |

### CUSTOM IMAGE COLLECTION (CIC)
A custom image collection (CIC) can be specified to be used for Level-3 and Level-4 analyses. Note that Level-2 routines, such as cloud masking, are specifically designed for the Time Series Stack (TSS) and are not compatible with a CIC. As such, filtering (spatially, temporally, quality, etc.) must be performed by the user beforehand. Using a CIC generally requires additional attention by the user and is ideally performed in interactive mode (see [Introduction](/docs/tutorials/tutorial_0_introducing-geeo.ipynb) for interactive mode).

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| `CIC`             | str   | null    | ee.ImageCollection path | Use specified ImageCollection (instead of constructing TSS). |
| `CIC_FEATURES`    | list  | null    | band names or null      | Optional band subset; null keeps all CIC bands. |

### TIME SERIES MOSAIC (TSM)
TSM generates a harmonized spatial mosaic from images acquired on the same day. Satellite data products like Landsat or Sentinel-2 are organized in tiling schemes where image scenes can overlap, meaning the same measurement may appear in multiple images for a given date. Using TSMs helps streamline time series exports, reducing the need for post-processing and preventing bias or errors in statistical calculations—such as double-counting the same measurement in image metrics.

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| `TSM`                      | bool    | false        | true, false                     | Build a single spatial mosaic per unique acquisition date found in TSS or CIC ImageCollections. The Time Series Mosaic is closely related to the Time Series Stack, but converts spatially overlapping, individual images into mosaics containing only a single image for each unique acquisition date. <br> See [`mosaic_imgcol()`](../geeo/level2/mosaic.py) for implementation details. |
| `TSM_BASE_IMGCOL`          | str     | TSS          | TSS, CIC                        | The source collection to be used for mosaicking. |


## LEVEL-3

The Level-3 module contains algorithms to generate higher-level image products for downstream analyses that require spatial-temporal continuity.
The products that can be generated on this level are the Time Series Interpolation (TSI), Spectral-Temporal-Metrics (STM), Pixel-Based Composites (PBC), and the metadata product Number of Valid Observations (NVO).

### TEMPORAL SUBWINDOWS | FOLDING
Temporal folding is a setting that can be used for NVO, STM, and PBC. It allows to create temporal subwindows within the global time settings (SPACE AND TIME).  For example, if the user wants a mean NDVI product (= STM) for each month, setting FOLD_MONTH to true allows to later create a STM for each month within the global time settings.  

Generally, all settings below will be combined into all possible unique combinations.
For example, if the user specifies FOLD_YEAR=True, FOLD_MONTH=True, YEAR_MIN=2020, YEAR_MAX=2021, MONTH_MIN=1, MONTH_MAX=12, there are 2x12=24 subwindows created (each month for each year), for each of which a NVO, STM or PBC can be generated separately if specifed.


| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| `FOLD_YEAR`                | bool    | false        | true, false                     | Partition time series into yearly subwindows for each unique year from YEAR_MIN to YEAR_MAX. |
| `FOLD_MONTH`               | bool    | false        | true, false                     | Partition into monthly subwindows for each unique month from MONTH_MIN to MONTH_MAX. (Jan..Dec across years). |
| `FOLD_CUSTOM`              | dict    | {year: null, month: null, doy: null, date: null} | year: [YYYY-YYYY, ...] / [YYYY+-OFFSET, ...]; month: [MM-MM, ...] / [MM+-OFFSET, ...]; doy: [DOY-DOY, ...] / [DOY+-OFFSET, ...]; date: [YYYYMMDD-YYYYMMDD, ...] / [YYYYMMDD+-OFFSET, ...]  | Custom windows via ranges or target±offset lists for years, months, and day-of-years. For example, {year: [2020-2021], doy: [192+-30, 240+-30]}. |


### NUMBER OF VALID OBSERVATIONS (NVO)
Determines the number of valid (unmasked) observations per pixel within a specified temporal window, providing a measure of the reliability and robustness of resulting data products. Since clouds, snow, and shadows are masked at the pixel level, it is important to evaluate data availability for each study area and time period individually ([Lewińska et al., 2024](https://doi.org/10.1016/j.dib.2024.111054)). For instance, image features like STMs may not be reliable if they are based on only a small number of observations ([Frantz et al., 2023](https://doi.org/10.1016/j.rse.2023.113823)).

| Parameter     | Type | Default | Allowed Values / Format | Description |
|---------------|------|---------|-------------------------|-------------|
| `NVO`           | bool | false   | true, false             | Compute per-pixel count of valid (unmasked) observations from TSS. <br> See [`calc_nvo()`](../geeo/level3/nvo.py) for implementation details. |
| `NVO_FOLDING`   | bool | false   | true, false             | If true, the counts are produced per temporal fold (see [Temporal Subwindows \| Folding](#temporal-subwindows--folding)). Usefull to assess the number of observations that were used to generate STMs, for example. <br> See [`folding_nvo()`](../geeo/level3/nvo.py) for implementation details. |


### TIME SERIES INTERPOLATION (TSI)
Many applications require spatial-temporal continuity, equidistant observations, and cannot deal with missing pixel values. TSI addresses this by applying interpolation techniques to generate a gap-free, equidistant time series from the TSS, TSM, or CIC. Users define the desired time interval (e.g., every 10 days) and select an interpolation method with relevant parameters. GEEO supports two interpolation approaches: weighted linear interpolation (based on proximity to the nearest observation) and Radial Basis Function (RBF, or ‘Gaussian’) interpolation. See [`Example 2`](../docs/tutorials/habitat-type-mapping_geeo.ipynb) for a worked example of Time Series Interpolation. 

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| `TSI`                      | str     | null         | null, WLIN, 1RBF, 2RBF, 3RBF    | Interpolation method: WLIN (weighted linear) or Radial-Basis-Function (RBF) using one to three Gaussian kernels; null disables. Up to three RBF kernels with different settings can be chosen, allowing the time series to be adaptively smoothed according to data availability ([Frantz, 2019](https://doi.org/10.3390/rs11091124)). See [`Example 2`](../docs/tutorials/habitat-type-mapping_geeo.ipynb) for a worked example and visual demonstration of Time Series Interpolation. <br> See [`geeo/level3/interpolation.py`](../geeo/level3/interpolation.py) for implementation details on WLIN and RBF. |
| `TSI_BASE_IMGCOL`          | str     | TSS          | TSS, TSM, CIC                   | The collection for which to interpolate. While the TSM is a more accurate choice over the TSS (removal of multiple observations of the same date), the TSS is a much faster and efficient choice and differences are usually negligible.  |
| `INTERVAL`                 | int     | 16           | days                            | Desired temporal spacing between interpolated timestamps. |
| `INTERVAL_UNIT`            | str     | day          | day, month, year                | Unit used for INTERVAL spacing. |
| `INIT_JANUARY_1ST`         | bool    | false        | true, false                     | Force interpolation to start Jan 1 (of YEAR_MIN). |
| `SIGMA1`                   | int     | 16           | days                            | Width of the 1st RBF (Gaussian) kernel in days. |
| `SIGMA2`                   | int     | 32           | days                            | Width of the 2nd RBF (Gaussian) kernel in days (2RBF/3RBF only). |
| `SIGMA3`                   | int     | 64           | days                            | Width of the 3nrd RBF (Gaussian) kernel in days (3RBF only). |
| `WIN1`                     | int     | 16           | days                            | +- of days of maximum offset for observations in TSI_BASE_IMGCOL to be included for each INTERVAL for the RBF1 kernel.  |
| `WIN2`                     | int     | 32           | days                            | +- of days of maximum offset for observations in TSI_BASE_IMGCOL to be included for each INTERVAL for the RBF2 kernel. |
| `WIN3`                     | int     | 64           | days                            | ++- of days of maximum offset for observations in TSI_BASE_IMGCOL to be included for each INTERVAL for the RBF3 kernel. |
| `BW1`                      | int     | 4            |                                 | Expected/ideal number of observations between two timesteps (e.g. within 16 days when INTERVAL=16) used to calculate the weight for 2nd kernel (2RBF/3RBF) over the first. If there are enough observations to satisfy the ideal setting for a single kernel, the second kernel weight is set to zero, i.e. obervations further away do not impact the interpolation result. This allows to adaptively smooth the interpolation according to actual data availability: less original observations -> larger kernel used vs. plenty of observations within window -> smaller kernel used. |
| `BW2`                      | int     | 8            |                                 | Weight for 3rd kernel (3RBF), see BW1. |


### SPECTRAL-TEMPORAL-METRICS (STM)
STMs are a commonly applied dimensionality reduction in remote sensing, in which pixel-wise statistics across individual bands/features are calculated for a given set of imagery over time. They capture spectral information over time and can be retrieved for different temporal sub-windows (years, seasons, months, etc.). STMs are robust, spatially continuous, and can be efficiently calculated, and as such, underpin a huge amount of remote sensing products that are currently available. See [`Example 1`](../docs/tutorials/habitat-type-mapping_geeo.ipynb) for a worked example of Spectral Temporal Metrics (STMs).  

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| `STM`                      | list    | null         | mean, median, sum, min, max, stdDev, variance, p1, p5, p10, p15, p20, p25, p30, p35, p40, p45, p50, p55, p60, p65, p70, p75, p80, p85, p90, p95, p99, skew, kurtosis, count, first, last; For example: [mean, stdDev] | Metrics to calculate for FEATURES and TIME (possibly FOLDING if STM_FOLDING set to True); null skips STM calculation.<br> See [`geeo/level3/stm.py`](../geeo/level3/stm.py) for implementation details.  |
| `STM_BASE_IMGCOL`          | str     | TSS          | TSS, TSM, TSI, CIC              | Collection to use for STM calculation. |
| `STM_FOLDING`              | bool    | false        | true, false                     | Whether to apply temporal folding for STM calculation, i.e., compute metrics for folds specified in FOLD_YEAR, FOLD_MONTH, or FOLD_CUSTOM. |
| `STM_FOLDING_LIST_ITER`    | bool    | false        | true, false                     | Alternate per-fold implementation (list iteration). Use default (False) first. For some large area applications with few temporal folds, setting this to True might perform better. The original implementation for folding uses joins, i.e. creates an ee.ImageCollection for each desired fold, joins the images matching the temporal filter to the collection and then maps over the collection to perform the operations. Setting this to true will inesatd create a list of temporal filters, map over the list and filter the base imgcol iteratively. Usually, for many applications joins are way more efficient than list iterations (see [Coding Best Practices](https://developers.google.com/earth-engine/guides/best_practices#join_vs_map-filter) in Earth Engine)!. |


### PIXEL-BASED COMPOSITES (PBC)
PBC produces cloud-free, radiometrically, and potentially phenologically consistent imagery that is spatially continuous across large regions. Various parameters can be used to rank pixels by their suitability, guiding how the mosaic is assembled. A typical PBC method is maximum NDVI compositing, which selects imagery from periods of peak photosynthetic activity.

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| `PBC`                      | str     | null         | null, BAP, MAX-RNB, NLCD, FEATURE | Available compositing methods are: Best-Available-Pixel (BAP) compositing ([Griffiths et al. 2013](https://doi.org/10.1109/JSTARS.2012.2228167), see tutorial [here](https://eol.pages.cms.hu-berlin.de/gcg_eo/04_baps.html)); MAX-RNB ([Qiu et al. 2023](https://doi.org/10.1016/j.rse.2022.113375)); FEATURE (e.g. NDVI) (max value if not PBC_INVERT_QUALITY_METRIC); NLCD (quality-flag logic) ([Jin et al. 2023](https://doi.org/10.34133/remotesensing.0022)); null disables. <br> See [`geeo/level3/composite.py`](../geeo/level3/composite.py) for implementation details.|
| `PBC_BASE_IMGCOL`          | str     | TSS          | TSS, TSM, TSI, CIC              | Source collection for compositing. |
| `PBC_FOLDING`              | bool    | false        | true, false                     | Whether to apply temporal folding for PBC calculation, i.e., output separate composites for each temporal fold. |
| `PBC_INVERT_QUALITY_METRIC`| bool    | false        | true, false                     | Appies for FEATURE composites and onverts the quality metric, i.e., selecting the observations associated with the minimum (inverted) value instead of the maximum (-> a maximum NDVI composite would become a minimum NDVI composite). |
| `PBC_BAP_DOY_EQ_YEAR`      | int     | 30           | days                            | Only BAP: DOY offset where seasonal (DOY) score weight equals YEAR score. <br> See [`geeo/level3/composite.py -> bap_score()`](../geeo/level3/composite.py) for implementation details and [Griffiths et al. 2013](https://doi.org/10.1109/JSTARS.2012.2228167) and the tutorial [here](https://eol.pages.cms.hu-berlin.de/gcg_eo/04_baps.html) for details on the algorithm.|
| `PBC_BAP_MAX_CLOUDDISTANCE`| int     | 500          | meters                          | Only BAP: Cloud distance (m) giving CLOUD score = 1. |
| `PBC_BAP_MIN_CLOUDDISTANCE`| int     | 0            | meters                          | Only BAP: Cloud distance (m) giving CLOUD score = 0. |
| `PBC_BAP_WEIGHT_DOY`       | float   | 0.6          | 0-1                             | Only BAP: Weight for DOY (seasonal proximity) component. |
| `PBC_BAP_WEIGHT_YEAR`      | float   | 0.2          | 0-1                             | Only BAP: Weight for YEAR (temporal recency) component. |
| `PBC_BAP_WEIGHT_CLOUD`     | float   | 0.2          | 0-1                             | Only BAP: Weight for CLOUD (cloud distance) component. |


## LEVEL-4

### LAND SURFACE PHENOLOGY (LSP)
LSP metrics are intended to characterize seasonal events in vegetation life cycles. Typically, a vegetation index like NDVI is used to detect phenological stages - such as start-of-season or peak-season - by linking changes in the spectral curve over time to these stages, often through straightforward thresholding approaches.
In the current version, GEEO implements the land surfave phenology retrieval developed by [Brooks et al. 2020](https://doi.org/10.3390/f11060606) and fine-tuned to possible seasonal adjustments by [Frantz et al. 2022](https://doi.org/10.3390/rs14030597). For a concise, informative description of the algorithmic backhground, the reader is referred to section *"3.2.2. Land Surface Phenology for Vegetation Dynamics 2.0"* in Frantz et al.'s paper.

| Parameter                 | Type  | Default | Allowed Values / Format | Description |
|---------------------------|-------|---------|-------------------------|-------------|
| `LSP`                       | str   | null    | null, POLAR             | Land Surface Phenology method. Currently, only POLAR is implemented. POLAR implements the polar-coordinates-based land surfave phenology retrieval developed by [Brooks et al. 2020](https://doi.org/10.3390/f11060606) and fine-tuned to possible seasonal adjustments by [Frantz et al. 2022](https://doi.org/10.3390/rs14030597). For a concise description the reader is referred to section *"3.2.2. Land Surface Phenology for Vegetation Dynamics 2.0"* in Frantz et al.'s paper. <br> See [`geeo/level4/lsp.py`](../geeo/level4/lsp.py) for details on the code implementation. |
| `LSP_BASE_IMGCOL`           | str   | TSI     | TSI, CIC                | Source collection for LSP metric calculation (A gap-free and equidistant time series such as a RBF-interpolated TSI is highly recommended for unbiased results). |
| `LSP_BAND`                  | str   | NDVI    | Any feature band        | Band providing the (phenology) signal for phenometric calculation. Usually a vegetation index such as NDVI or EVI. |
| `LSP_YEAR_MIN`              | int   | null    | year or null            | Override global YEAR_MIN for phenometric calculation (null uses global). The approach by Brooks et al. (2020) as implemented here requires a bracketing year at the beginning or end of the time series for its metric calculation. As such, for instance, when the users desires phenometrics from 2015-2020, and sensor availability permits, it is best to set YEAR_MIN and YEAR_MAX to 2014 and 2021, and LSP_YEAR_MIN and LSP_YEAR_MAX to 2015 and 2020, accordingly. |
| `LSP_YEAR_MAX`              | int   | null    | year or null            | Override global YEAR_MAX for phenometric calculation (null uses global). See LSP_YEAR_MIN for more details. |
| `LSP_ADJUST_SEASONAL`       | bool  | false   | true, false             | Enable per-year seasonal start/end DOY adjustment as introduced by [Frantz et al. 2022](https://doi.org/10.3390/rs14030597), i.e. instead of calcualting a fixed start- and end-of-season date per pixel, the start- and end-of-season are allowed to vary from phenological year to year within a reasonable maximum (LSP_ADJUST_SEASONAL_MAX_DAYS) range, resembling more closely the inter-annual variation of phenology in reality. |
| `LSP_ADJUST_SEASONAL_MAX_DAYS` | int| 40      | days                    | Max ± days shift allowed when the seasonal adjustment (LSP_ADJUST_SEASONAL) is enabled. |

The output LSP metrics (bands) are currently: 

- The long-term (entire time series) average vector pointing to the (average) DOY of peak season (`RAVG`), the long-term average vector pointing to the (average) DOY of the off season trough (`THETA`)
- `t_start` and `t_end` (+ `t_start_adj`; `t_end_adj` for seasonal adjustment) providing the start and end point for each phenological year in days since 1970-01-01.
- The LSP metrics: start-of-season (SOS; Number of days (or radial angle) corresponding to 15% of cumulative annual LSP_BAND), start-of-peak-season (SOP; 25% cum.), mid-of-season (MOS; 50% cum.), end-of-peak season (EOP; 75% cum.), and end-of-season (EOS; 80% cum.)
- `valid_firstyear`: binary flag indicating validity of LSP metrics per pixel for first and last year. Set to 1 if first year has valid LSP metrics (then last year has not) for that pixel. 0 means that the valid LSP metrics start at y+1 and the last year has valid LSP metrics. For example: Consider a time series from 2020-2025. 0 means that valid LSP metrics were calculated for 2021-2025. 1 means that valid LSP metrics were calculated for 2020-2024. Varies on per-pixel basis.

## EXPORT
Image or table export of the above products to Google Drive or as Earth Engine Asset. Allows to specify, and takes care of, spatial and image metadata, including projection, resampling method, data type and scale.

### IMAGE SETTINGS
| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| `PIX_RES`                  | int     | 30           | meters                          | Output pixel size in meters. |
| `CRS`                      | str     | EPSG:4326    | (EPSG), (WKT-string), UTM; Mollweide; (GLANCE) AF, AS, NA, SA, OC, EU | Target projection (EPSG code, WKT string, UTM (automatically finds UTM zone), Mollweide, or [GLANCE](https://github.com/measures-glance/glance-grids) continent identifier (AF, AS, NA, SA, OC, EU)). |
| `CRS_TRANSFORM`            | list    | null         | [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation] | Explicit affine transform (required if using IMG_DIMENSIONS). |
| `IMG_DIMENSIONS`           | str     | null         | WidthxHeight, e.g. 1000x1000    | Fixed pixel dimensions (requires CRS_TRANSFORM). |
| `RESAMPLING_METHOD`        | str     | null         | null, bilinear, bicubic         | Optional resampling for all subsequent operations (null = nearest neighbour). |
| `DATATYPE`                 | str     | int16        | uint8, int8, uint16, int16, uint32, int32, float, double        | Export datatype after applying DATATYPE_SCALE. |
| `DATATYPE_SCALE`           | int     | 10000        |                                 | Multiplicative factor (e.g. reflectance * 10000 before casting). **Attention**! Must be compatible with datatype. |
| `NODATA_VALUE`             | int     | -9999        |                                 | Fill value for masked pixels in exports. |

If the ROI, CRS and PIX_RES facilitate exact matching of the pixel grid, the ouput image will perfectly align with the specified spatial extent. See the Jupyter Notebook [`tutorial_1_spatial-tiling-and-metadata.ipynb`](../docs/tutorials/tutorial_1_spatial-tiling-and-metadata.ipynb)) for an example and further descriptions.

### PRODUCTS TO EXPORT
| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| `EXPORT_IMAGE`             | bool    | false        | true, false                     | Export final image products in general. Still requires the desired products to be selected below. |
| `EXPORT_TABLE`             | bool    | false        | true, false                     | Export attribute/sample table (reduceRegions or reduceRegion). |
| `EXPORT_TSS`               | bool    | false        | true, false                     | Export preprocessed Time Series Stack (TSS). |
| `EXPORT_CIC`               | bool    | false        | true, false                     | Export custom ImageCollection (CIC). |
| `EXPORT_TSM`               | bool    | false        | true, false                     | Export Time Series Mosaic (TSM). |
| `EXPORT_NVO`               | bool    | false        | true, false                     | Export Number of Valid Observations (NVO). |
| `EXPORT_TSI`               | bool    | false        | true, false                     | Export interpolated time series (TSI). |
| `EXPORT_STM`               | bool    | false        | true, false                     | Export Spectral Temporal Metrics (STM). |
| `EXPORT_PBC`               | bool    | false        | true, false                     | Export pixel-based composites (PBC). |
| `EXPORT_LSP`               | bool    | false        | true, false                     | Export Land Surface Phenology (LSP). |

### GENERAL EXPORT SETTINGS
| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| `EXPORT_LOCATION`          | str     | Drive        | Drive, Asset                    | Destination: Google Drive folder or EE Asset collection. |
| `EXPORT_DIRECTORY`         | str     | null         |                                 | Drive subfolder name or full asset ID folder; null = default/root. |
| `EXPORT_DESC`              | str     | GEEO         |                                 | Prefix for export task & filenames. |
| `EXPORT_DESC_DETAIL_TIME`  | bool    | false        | true, false                     | Append detailed temporal window key to export names. |
| `EXPORT_BANDNAMES_AS_CSV`  | bool    | false        | true, false                     | Also export CSV listing band names (useful for +100 band images). |
| `EXPORT_TABLE_FORMAT`      | str     | CSV          | CSV, GeoJSON, KML, etc.         | Output file format for table export. |
| `EXPORT_TABLE_METHOD`      | str     | reduceRegions| reduceRegions, reduceRegion     | Multi-feature vs single-geometry reduction method. See [Earth Engine docs](https://developers.google.com/earth-engine/guides/best_practices#reduceregion_vs_reduceregions_vs_for-loop) for more info. |
| `EXPORT_TABLE_TILE_SCALE`  | float   | 1            | 0.1-16                          | Tile scale tuning: smaller tileScale -> larger tiles, poentially faster but may run out of memory. Larger tileScale = smaller tiles, may enable computations that run out of memory with the default. If after setting this to 16, the export still runs out of memory, consider a less complex (smaller) ROI, smaller extent, shorter time series, etc. I.e. run in chunks.|
| `EXPORT_TABLE_BUFFER`      | int     | null         | meters                          | Buffer distance applied to features before reduction. |
| `EXPORT_TABLE_REDUCER`     | str     | first        | first, ...                      | Reducer applied to intersecting pixels. |
| `EXPORT_TABLE_DROP_NODATA` | bool    | false        | true, false                     | Drop rows comprised entirely of nodata. |
| `EXPORT_PER_FEATURE`       | bool    | false        | true, false                     | Export separate image per feature / band (**Attention**: can result in many tasks). |
| `EXPORT_PER_TIME`          | bool    | false        | true, false                     | Export each timestamp image separately (**Attention**: can result in even more tasks). |

---
