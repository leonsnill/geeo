# GEEO Parameter Documentation
## LEVEL-2
### SPACE AND TIME
These are the global spatial and temporal filters applied to the ImageCollections. The YEAR, MONTH, and DOY parameters can be combined to
trigger a logical and filtering. Calendar year wrapping is considered for MONTH and DOY, i.e. MONTH_MIN and DOY_MIN may be greater than MONTH_MAX and DOY_MAX, respectively.
For example, to only process November to February, set MONTH_MIN=11 and MONTH_MAX=2.
If a DATE range is specified, YEAR, MONTH, and DOY are ignored.
The Region-Of-Interest (ROI) can be specified as a simple list of coordinates (point, polygon), a client-side shp/gpkg file or GeoPandas GeoDataFrame, as well as a server-side ee.Geometry or ee.FeatureCollection.
For filtering ee.ImageCollections spatially, it is recommended to leave ROI_SIMPLIFY_GEOM_TO_BBOX set to `true`, because providing a complex collection as geometry argument may result in poor performance. 


| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| YEAR_MIN                 | int     | 2023         | YYYY                        | Start year (inclusive) when DATE_* not set. |
| YEAR_MAX                 | int     | 2023         | YYYY                       | End year (inclusive); paired with MONTH/DOY filters when DATE_* absent. |
| MONTH_MIN                | int     | 1            | [1, 12]                            | First month in seasonal window; can exceed MONTH_MAX to wrap (e.g. 11→2). |
| MONTH_MAX                | int     | 12           | [1, 12]                            | Last month in window; wrapping applied when < MONTH_MIN. |
| DOY_MIN                  | int     | 1            | [1, 366]                           | First Day-Of-Year; can exceed DOY_MAX to express wrap (e.g. 300→60). |
| DOY_MAX                  | int     | 366          | [1, 366]                           | Last DOY; wrapping logic applied when < DOY_MIN. |
| DATE_MIN                 | str     | null         | YYYYMMDD                        | Absolute start date; overrides YEAR/MONTH/DOY when paired with DATE_MAX. |
| DATE_MAX                 | str     | null         | YYYYMMDD                        | Absolute end date; effective only if DATE_MIN set. |
| ROI                      | list / str / GeoDataFrame / ee object | [12.9, 52.2, 13.9, 52.7] | Point [lon, lat]; BBox [xmin, ymin, xmax, ymax]; path to .shp/.gpkg; GeoDataFrame; ee.Geometry / ee.FeatureCollection | Area of Interest; complex polygons can slow filtering and may be bbox-simplified. |
| ROI_SIMPLIFY_GEOM_TO_BBOX| bool    | true         | true, false                     | Simplifies geometry to bounding box for ee.ImageCollection filtering. Usually desireable, since  providing a complex collection as geometry argument may result in poor performance. A scenario where setting this to false makes sense if a not too complex geometry covers a large area but the bounding box would include ee.Images that are not needed (e.g. points covering distinct regions across a large area with great distances between the points.) |
| ROI_TILES                 | bool    | false        | true, false                     | Use features in ROI vector as independent tiles (ROIs) for filtering and export; applies only when ROI is a vector file (shp/gpkg/GeoDataFrame). Requires ROI_TILES_ATTRIBUTE_COLUMN to be specified to append to output filenames. |
| ROI_TILES_ATTRIBUTE_COLUMN| str     | null         | column name or null             | Attribute column containing unique per-feature IDs; appended to EXPORT_DESC for per-tile outputs; used for subsetting via ROI_TILES_ATTRIBUTE_LIST. |
| ROI_TILES_ATTRIBUTE_LIST  | list    | null         | list of values or null          | Values (matching ROI_TILES_ATTRIBUTE_COLUMN) to select a subset of features; null selects all features. |

### SENSOR AND DATA QUALITY SETTINGS (TIME SERIES STACK - TSS)

The following settings define if and which ee.ImageCollection Time-Series-Stack (TSS) to use and how to pre-process the data, most notably cloud masking. 
The masking settings are set to typical settings to only consider scenes with less than 75% cloud cover, and masking clouds, cloud shadows, snow/ice, as well as fill values. As such, in practise when having created a new parameter file or when operating with python dictionaries in interactive sessions, only setting SENSORS is often enough. 

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| SENSORS                  | list    | [L9, L8, L7, L5, L4] | L9, L8, L7, L5, L4, S2, HLS, HLSL30, HLSS30 | L9/L8/L7/L5/L4 -> LANDSAT Collection 2 Tier 1 Surface Reflectance product ([LANDSAT/LXXX/C02/T1_L2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2)); S2 -> Sentinel-2 ([COPERNICUS/S2_SR_HARMONIZED](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)). L9/L8/L7/L5/L4 + S2 can be merged, e.g. [L9, L8, S2]; HLS/HLSL30/HLSS30 -> Harmonized Landsat ([HLSL30](https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSL30_v002)), Sentinel-2 ([HLSS30](https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSS30_v002)) product, and combined (HLS) product. |
| MAX_CLOUD                | int     | 75           | 0-100                           | Scene-level metadata cloud cover (%) filter before per-pixel masking. Band used for sensors: CLOUD_COVER_LAND (L9/L8/L7/L5/L4), CLOUDY_PIXEL_PERCENTAGE (S2), and CLOUD_COVERAGE (HLS/HLSL30/HLSS30). |
| EXCLUDE_SLCOFF           | bool    | false        | true, false                     | Skip Landsat-7 ETM+ Scan Line Corrector (SLC)-off (striping) scenes (i.e., L7 scenes after 2003-05-30). |
| GCP_MIN_LANDSAT          | int     | 1            | >=1                             | Minimum Ground Control Points threshold; lower quality scenes removed. Required since legacy images may be incorrectly georectified. |
| MASKS_LANDSAT            | list    | [cloud, cshadow, snow, fill, dilated] | See description | Apply listed QA masks (cloud, cshadow, snow, fill, dilated, saturated); null disables. |
| MASKS_LANDSAT_CONF       | str     | Medium       | Medium, High                    | Confidence threshold for Landsat QA masking (Medium = more conservative; favours error of commission, i.e. cloud remnants less likely, but clear pixels may be falsely masked). |
| MASKS_S2                 | str     | CPLUS        | null, CPLUS, PROB, SCL          | Sentinel‑2 cloud masking: [CPLUS](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED#description) (Cloud Score Plus, strongly recommended for S2), [PROB](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY) (cloud probability), SCL (SCL band of [COPERNICUS/S2_SR_HARMONIZED](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) product), null (none). |
| MASKS_S2_CPLUS           | float   | 0.6          | 0-1                             | Cloud Score Plus threshold (lower masks more pixels) See [here](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED). |
| MASKS_S2_PROB            | int     | 30           | 0-100                           | Cloud probability (%) cutoff when using PROB. See [here](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY) |
| MASKS_S2_NIR_THRESH_SHADOW| float  | 0.2          | 0-1                             | Dark NIR threshold for shadow masking when using PROB cloud masking. See [here](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY) |
| MASKS_HLS                | list    | [cloud, cshadow, snow] | cloud, cshadow, snow, fill, dilated, saturated | HLS Fmask classes to remove. See MASKS_LANDSAT. See also [page 17 in HLS Product User Guide](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf) for details. |
| ERODE_DILATE             | bool    | false        | true, false                     | Apply morphological erosion and/or dilation on mask to further reduce artifacts. |
| ERODE_RADIUS             | int     | 60           | 0-Inf                          | Erosion kernel radius (meters). Can be set to 0 while DILATE_RADIUS is >0 to only dilate (and vice versa). |
| DILATE_RADIUS            | int     | 120          | 0-Inf                          | Dilation kernel radius (meters); usually ≥ erosion radius. Can be set to 0 while ERODE_RADIUS is >0 to only erode (and vice versa). |
| ERODE_DILATE_SCALE       | int     | 60           | meters                          | Pixel scale used for morphology (higher = faster, because coarser pixel grain). |
| BLUE_MAX_MASKING         | float   | null         | 0-1                             | Mask pixels whose blue reflectance exceed this threshold (e.g. haze/glint); ignored if null or 0. |

### BANDS | INDICES | FEATURES

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| FEATURES                 | list    | [BLU, GRN, RED, NIR, SW1, SW2, NDVI] | BLU, GRN, RED, NIR, SW1, SW2 (Landsat+Sentinel-2), LST (Landsat, if LST was processed for scene), RE1, RE2, RE3, RE4 (Sentinel-2), NDVI, EVI, NBR, NDMI, NDWI, MDWI, NDB, TCG, TCB, TCW, SWR | Bands / indices / custom formula outputs kept in pipeline; include DEM or unmixing outputs here for export. |
| CUSTOM_FORMULAS  | dict  | null    | {NAME: {formula: "(G-SW1)/(G+SW1)", variable_map: {G: GRN, SW1: SW1}}, ...} | Per-image EE expression definitions; add NAME to FEATURES to include resulting band. Custom formulas can be used in chains, i.e. output (NAME) from previous formula may be called in subsequent formula. |
| UMX                      | dict    | null         | {'Class1': [100, 230, ..., 123], 'Class2': [24, ...], ...}                 | Order in endmember dictionary must match order defined in FEATURES. For example, if {'PV': [0.08, 0.1, 0.07, 0.24, 0.11, 0.08], 'Soil': [0.2, 0.3, 0.3, 0.4, 0.45, 0.55]} and FEATURES [BLU, GRN, RED, NIR, SW1, SW2], then corresponding feature-value combinations will be used (e.g. Soil 0.2 for BLU). Attention must be paid that values in UMX match scale of FEATURES. Wrapper for [ee.Image.unmix](https://developers.google.com/earth-engine/apidocs/ee-image-unmix). See also [unmixing example](https://developers.google.com/earth-engine/guides/image_transforms#spectral-unmixing) from Earth Engine. |
| UMX_SUM_TO_ONE           | bool    | true         | true, false                     | Enforce abundance fractions sum to 1. See [ee.Image.unmix](https://developers.google.com/earth-engine/apidocs/ee-image-unmix).|
| UMX_NON_NEGATIVE         | bool    | true         | true, false                     | Constrain abundance fractions to ≥ 0. See [ee.Image.unmix](https://developers.google.com/earth-engine/apidocs/ee-image-unmix).|
| UMX_REMOVE_INPUT_FEATURES| bool    | true         | true, false                     | Keep only abundance (unmixed) bands if true; otherwise append. |

### CUSTOM IMAGE COLLECTION (CIC)
A custom image collection (CIC) can be specified for use in Level-3 and Level-4 analyses. However, Level-2 routines designed for the TSS—such as cloud masking—are not compatible with a CIC and must be performed by the user beforehand. Using a CIC typically requires additional attention; refer to the software documentation for CIC usage examples.

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| CIC             | str   | null    | ee.ImageCollection path | Use specified ImageCollection (instead of constructing TSS). |
| CIC_FEATURES    | list  | null    | band names or null      | Optional band subset; null keeps all CIC bands. |

### TIME SERIES MOSAIC (TSM)
TSM generates a harmonized spatial mosaic from images acquired on the same day. Satellite data products like Landsat or Sentinel-2 are organized in tiling schemes where image scenes can overlap, meaning the same measurement may appear in multiple images for a given date. Using TSMs helps streamline time series exports, reducing the need for post-processing and preventing bias or errors in statistical calculations—such as double-counting the same measurement in image metrics.

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| TSM                      | bool    | false        | true, false                     | Build one spatial mosaic per unique acquisition date in TSS or CIC. The Time Series Mosaic is closely related to the Time Series Stack, but converts spatially overlapping, individual images into mosaics containing only a single image for each unique acquisition date. |
| TSM_BASE_IMGCOL          | str     | TSS          | TSS, CIC                        | Source collection for mosaicking (raw TSS or custom CIC). |


## LEVEL-3

### TEMPORAL SUBWINDOWS | FOLDING
Temporal folding is a setting to be used for NVO, STM, and PBC. It allows to create temporal subwindows within the global time settings (SPACE AND TIME). All settings below will be combined into all possible unique combinations: For example, if the user specifies FOLD_YEAR=True and FOLD_MONTH=True and YEAR_MIN=2020, YEAR_MAX=2021, MONTH_MIN=1, MONTH_MAX=12, there are 2x12=24 subwindows (each month for each year). 

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| FOLD_YEAR                | bool    | false        | true, false                     | Partition time series into yearly subwindows for each unique year from YEAR_MIN to YEAR_MAX. |
| FOLD_MONTH               | bool    | false        | true, false                     | Partition into monthly subwindows for each unique month from MONTH_MIN to MONTH_MAX. (Jan..Dec across years). |
| FOLD_CUSTOM              | dict    | {year: null, month: null, doy: null, date: null} | year: [YYYY-YYYY, ...] / [YYYY+-OFFSET, ...]; month: [MM-MM, ...] / [MM+-OFFSET, ...]; doy: [DOY-DOY, ...] / [DOY+-OFFSET, ...]; date: [YYYYMMDD-YYYYMMDD, ...] / [YYYYMMDD+-OFFSET, ...]  | Custom windows via ranges or target±offset lists. For example, {year: [2020-2021], doy: [192+-30, 240+-30]}. |


### NUMBER OF VALID OBSERVATIONS (NVO)
Determines the number of valid (unmasked) observations per pixel within a specified temporal window, providing a measure of the reliability and robustness of resulting data products. Since clouds, snow, and shadows are masked at the pixel level, it is important to evaluate data availability for each study area and time period individually ([Lewińska et al., 2024](https://doi.org/10.1016/j.dib.2024.111054)). For instance, image features like STMs may not be reliable if they are based on only a small number of observations ([Frantz et al., 2023](https://doi.org/10.1016/j.rse.2023.113823)).

| Parameter     | Type | Default | Allowed Values / Format | Description |
|---------------|------|---------|-------------------------|-------------|
| NVO           | bool | false   | true, false             | Compute per-pixel count of valid (unmasked) observations from TSS. |
| NVO_FOLDING   | bool | false   | true, false             | If true, counts produced per temporal fold (year/month/custom). Usefull to assess the number of observations that were used to generate STMs, for example. |


### TIME SERIES INTERPOLATION (TSI)
Many applications require spatial-temporal continuity, equidistant observations, and cannot deal with missing pixel values. TSI addresses this by applying interpolation techniques to generate a gap-free, equidistant time series from the TSS, TSM, or CIC. Users define the desired time interval (e.g., every 10 days) and select an interpolation method with relevant parameters. GEEO supports two interpolation approaches: weighted linear interpolation (based on proximity to the nearest observation) and Radial Basis Function (RBF, or ‘Gaussian’) interpolation. See the Juypter Notebook 'geeo/docs/habitat-type-mapping_geeo.ipynb' for a worked example of TSI. 

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| TSI                      | str     | null         | null, WLIN, 1RBF, 2RBF, 3RBF    | Interpolation method: WLIN (weighted linear) or Radial-Basis-Function (RBF) using one to three Gaussian kernels; null disables. Up to three RBF kernels with different settings can be chosen, allowing the time series to be adaptively smoothed according to data availability ([Frantz, 2019](https://doi.org/10.3390/rs11091124)). |
| TSI_BASE_IMGCOL          | str     | TSS          | TSS, TSM, CIC                   | Collection to interpolate (raw, mosaicked, or custom). |
| INTERVAL                 | int     | 16           | days                            | Desired spacing between interpolated timestamps. |
| INTERVAL_UNIT            | str     | day          | day, month, year                | Unit used for INTERVAL spacing. |
| INIT_JANUARY_1ST         | bool    | false        | true, false                     | Force interpolation to start Jan 1 (of YEAR_MIN). |
| SIGMA1                   | int     | 16           | days                            | First RBF kernel width (days). |
| SIGMA2                   | int     | 32           | days                            | Second RBF kernel width (2RBF/3RBF only). |
| SIGMA3                   | int     | 64           | days                            | Third RBF kernel width (3RBF only). |
| WIN1                     | int     | 16           | days                            | +- of days (i.e. half-window size) maximum offset for observations in TSI_BASE_IMGCOL to be included for each INTERVAL for kernel 1.  |
| WIN2                     | int     | 32           | days                            | +- of days (i.e. half-window size) maximum offset for observations in TSI_BASE_IMGCOL to be included for each INTERVAL for kernel 2. |
| WIN3                     | int     | 64           | days                            | +- of days (i.e. half-window size) maximum offset for observations in TSI_BASE_IMGCOL to be included for each INTERVAL for kernel 3. |
| BW1                      | int     | 4            |                                 | Expected/ideal number of observations between two timesteps (e.g. within 16 days when INTERVAL=16) used to calculate the weight for 2nd kernel (2RBF/3RBF) over the first. If there are enough observations to satisfy the ideal setting for a single kernel, the second kernel weight is set to zero, i.e. obervations further away do not impact the interpolation result. |
| BW2                      | int     | 8            |                                 | Weight for 3rd kernel (3RBF), see BW1. |


### SPECTRAL-TEMPORAL-METRICS (STM)
STMs are a commonly applied dimensionality reduction in remote sensing, in which pixel-wise statistics across individual bands/features are calculated for a given set of imagery over time. They capture spectral information over time and can be retrieved for different temporal sub-windows (years, seasons, months, etc.). STMs are robust, spatially continuous, and can be efficiently calculated, and as such, underpin a huge amount of remote sensing products that are currently available.  

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| STM                      | list    | null         | mean, median, sum, min, max, stdDev, variance, p1, p5, p10, p15, p20, p25, p30, p35, p40, p45, p50, p55, p60, p65, p70, p75, p80, p85, p90, p95, p99, skew, kurtosis, count, first, last  | Metrics to calculate for FEATURES and TIME (possibly FOLDING if STM_FOLDING set to True); null skips STM calculation.  |
| STM_BASE_IMGCOL          | str     | TSS          | TSS, TSM, TSI, CIC              | Collection to use for STM calculation. |
| STM_FOLDING              | bool    | false        | true, false                     | Compute metrics for folds specified in FOLD_YEAR, FOLD_MONTH, or FOLD_CUSTOM. |
| STM_FOLDING_LIST_ITER    | bool    | false        | true, false                     | Alternate per-fold implementation (list iteration). Use default (False) first. For some large area applications with few temporal fold, setting this to True might perform better. The original implementation for folding uses joins, i.e. creates an ee.ImageCollection for each desired fold, joins the images matching the temporal filter to the collection and then maps over the collection to perform the operations. Setting this to true will inesatd create a list of temporal filters, map over the list and filter the base imgcol iteratively. Usually, for many applications joins are way more efficient than list iterations!. |


### PIXEL-BASED COMPOSITES (PBC)
PBC produces cloud-free, radiometrically, and potentially phenologically consistent imagery that is spatially continuous across large regions. Various parameters can be used to rank pixels by their suitability, guiding how the mosaic is assembled. A typical PBC method is maximum NDVI compositing, which selects imagery from periods of peak photosynthetic activity.

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| PBC                      | str     | null         | null, BAP, MAX-RNB, NLCD, FEATURE | Compositing methods are: Best-Available-Pixel compositing ([Griffiths et al. 2013](https://doi.org/10.1109/JSTARS.2012.2228167)) Overview on BAP compositing [here](https://eol.pages.cms.hu-berlin.de/gcg_eo/04_baps.html); MAX-RNB ([Qiu et al. 2023](https://doi.org/10.1016/j.rse.2022.113375)); FEATURE (e.g. NDVI) (max value if not PBC_INVERT_QUALITY_METRIC); NLCD (quality-flag logic) ([Jin et al. 2023](https://doi.org/10.34133/remotesensing.0022)); null disables. |
| PBC_BASE_IMGCOL          | str     | TSS          | TSS, TSM, TSI, CIC              | Source collection for compositing. |
| PBC_FOLDING              | bool    | false        | true, false                     | Output separate composites per temporal fold. |
| PBC_INVERT_QUALITY_METRIC| bool    | false        | true, false                     | For FEATURE composites: select minimum (invert) instead of maximum. |
| PBC_BAP_DOY_EQ_YEAR      | int     | 30           | days                            | Only BAP: DOY offset where seasonal (DOY) score weight equals YEAR score. |
| PBC_BAP_MAX_CLOUDDISTANCE| int     | 500          | meters                          | Only BAP: Cloud distance (m) giving CLOUD score = 1. |
| PBC_BAP_MIN_CLOUDDISTANCE| int     | 0            | meters                          | Only BAP: Cloud distance (m) giving CLOUD score = 0. |
| PBC_BAP_WEIGHT_DOY       | float   | 0.6          | 0-1                             | Only BAP: Weight for DOY (seasonal proximity) component. |
| PBC_BAP_WEIGHT_YEAR      | float   | 0.2          | 0-1                             | Only BAP: Weight for YEAR (temporal recency) component. |
| PBC_BAP_WEIGHT_CLOUD     | float   | 0.2          | 0-1                             | Only BAP: Weight for CLOUD (cloud distance) component. |


## LEVEL-4

### LAND SURFACE PHENOLOGY (LSP)
LSP metrics are intended to characterize seasonal events in vegetation life cycles. Typically, a vegetation index like NDVI is used to detect phenological stages - such as start-of-season or peak-season - by linking changes in the spectral curve over time to these stages, often through straightforward thresholding approaches.

| Parameter                 | Type  | Default | Allowed Values / Format | Description |
|---------------------------|-------|---------|-------------------------|-------------|
| LSP                       | str   | null    | null, POLAR             | Land Surface Phenology method. Currently, only POLAR is implemented. POLAR implements the polar-coordinates-based land surfave phenology retrieval developed by [Brooks et al. 2020](https://doi.org/10.3390/f11060606) and fine-tuned to possible seasonal adjustments by [Frantz et al. 2022](https://doi.org/10.3390/rs14030597). For a concise description the reader is referred to section *"3.2.2. Land Surface Phenology for Vegetation Dynamics 2.0"* in Frantz et al.'s paper. |
| LSP_BASE_IMGCOL           | str   | TSI     | TSI, CIC                | Source collection for phenology (TSI recommended for a gap-free and equidistant time series). |
| LSP_BAND                  | str   | NDVI    | Any feature band        | Band providing vegetation signal for phenometrics (e.g. NDVI). |
| LSP_YEAR_MIN              | int   | null    | year or null            | Override global YEAR_MIN for phenology (null uses global). |
| LSP_YEAR_MAX              | int   | null    | year or null            | Override global YEAR_MAX for phenology (null uses global). |
| LSP_ADJUST_SEASONAL       | bool  | false   | true, false             | Enable per-year seasonal start/end DOY adjustment as introduced by [Frantz et al. 2022](https://doi.org/10.3390/rs14030597). |
| LSP_ADJUST_SEASONAL_MAX_DAYS | int| 40      | days                    | Max ± days shift allowed when seasonal adjustment enabled. |

The output LSP metrics (bands) are currently: 

- The long-term (entire time series) average vector pointing to the DOY of peak season (`RAVG`), the long-term average vector pointing to the DOY of the off season trough (`THETA`)
- `t_start` and `t_end` (+ `t_start_adj`; `t_end_adj` for seasonal adjustment) providing the start and end point for each phenological year in days since 1970-01-01.
- The LSP metrics: start-of-season (SOS; Number of days (or radial angle) corresponding to 15% of cumulative annual LSP_BAND), start-of-peak-season (SOP; 25% cum.), mid-of-season (MOS; 50% cum.), end-of-peak season (EOP; 75% cum.), and end-of-season (EOS; 80% cum.)
- `valid_firstyear`: binary flag indicating validity of LSP metrics per pixel for first and last year. Set to 1 if first year has valid LSP metrics (then last year has not) for that pixel. 0 means that the valid LSP metrics start at y+1 and the last year has valid LSP metrics. For example: Consider a time series from 2020-2025. 0 means that valid LSP metrics were calculated for 2021-2025. 1 means that valid LSP metrics were calculated for 2020-2024. Varies on per-pixel basis.

## EXPORT
Image or table export of the above products to Google Drive or as Earth Engine Asset. Allows to specify, and takes care of, spatial and image metadata, including projection, resampling method, data type and scale.

### IMAGE SETTINGS
| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| PIX_RES                  | int     | 30           | meters                          | Output pixel size (meters). |
| CRS                      | str     | EPSG:4326    | (EPSG), (WKT-string), UTM; Mollweide; (GLANCE) AF, AS, NA, SA, OC, EU | Target projection (EPSG code, WKT string, UTM (automatically finds UTM zone), Mollweide, or [GLANCE](https://github.com/measures-glance/glance-grids) continent identifier (AF, AS, NA, SA, OC, EU)). |
| CRS_TRANSFORM            | list    | null         | [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation] | Explicit affine transform (required if using IMG_DIMENSIONS). |
| IMG_DIMENSIONS           | str     | null         | WidthxHeight, e.g. 1000x1000    | Fixed pixel dimensions (requires CRS_TRANSFORM). |
| RESAMPLING_METHOD        | str     | null         | null, bilinear, bicubic         | Optional resampling for all subsequent operations (null = nearest neighbour). |
| DATATYPE                 | str     | int16        | uint8, int8, uint16, int16, uint32, int32, float, double        | Export datatype after applying DATATYPE_SCALE. |
| DATATYPE_SCALE           | int     | 10000        |                                 | Multiplicative factor (e.g. reflectance * 10000 before casting). **Attention**! Must be compatible with datatype. |
| NODATA_VALUE             | int     | -9999        |                                 | Fill value for masked pixels in exports. |

If the ROI, CRS and PIX_RES facilitate exact matching of the pixel grid, the ouput image will perfectly align with the specified spatial extent. See the Jupyter Notebook 'geeo/docs/tutorial_1_spatial-tiling-and-metadata.ipynb' for an example and further descriptions.

### PRODUCTS TO EXPORT
| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| EXPORT_IMAGE             | bool    | false        | true, false                     | Export final image products in general. Still requires the desired products to be selected below. |
| EXPORT_TABLE             | bool    | false        | true, false                     | Export attribute/sample table (reduceRegions or reduceRegion). |
| EXPORT_TSS               | bool    | false        | true, false                     | Export preprocessed Time Series Stack (TSS). |
| EXPORT_CIC               | bool    | false        | true, false                     | Export custom ImageCollection (CIC). |
| EXPORT_TSM               | bool    | false        | true, false                     | Export Time Series Mosaic (TSM). |
| EXPORT_NVO               | bool    | false        | true, false                     | Export Number of Valid Observations (NVO). |
| EXPORT_TSI               | bool    | false        | true, false                     | Export interpolated time series (TSI). |
| EXPORT_STM               | bool    | false        | true, false                     | Export Spectral Temporal Metrics (STM). |
| EXPORT_PBC               | bool    | false        | true, false                     | Export pixel-based composites (PBC). |
| EXPORT_LSP               | bool    | false        | true, false                     | Export Land Surface Phenology (LSP). |

### GENERAL EXPORT SETTINGS
| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| EXPORT_LOCATION          | str     | Drive        | Drive, Asset                    | Destination: Google Drive folder or EE Asset collection. |
| EXPORT_DIRECTORY         | str     | null         |                                 | Drive subfolder name or full asset ID folder; null = default/root. |
| EXPORT_DESC              | str     | GEEO         |                                 | Prefix for export task & filenames. |
| EXPORT_DESC_DETAIL_TIME  | bool    | false        | true, false                     | Append detailed temporal window key to export names. |
| EXPORT_BANDNAMES_AS_CSV  | bool    | false        | true, false                     | Also export CSV listing band names (useful for +100 band images). |
| EXPORT_TABLE_FORMAT      | str     | CSV          | CSV, GeoJSON, KML, etc.         | Output file format for table export. |
| EXPORT_TABLE_METHOD      | str     | reduceRegions| reduceRegions, reduceRegion     | Multi-feature vs single-geometry reduction method. See [Earth Engine docs](https://developers.google.com/earth-engine/guides/best_practices#reduceregion_vs_reduceregions_vs_for-loop) for more info. |
| EXPORT_TABLE_TILE_SCALE  | float   | 1            | 0.1-16                          | Tile scale tuning: smaller tileScale -> larger tiles, poentially faster but may run out of memory. Larger tileScale = smaller tiles, may enable computations that run out of memory with the default. If after setting this to 16, the export still runs out of memory, consider a less complex (smaller) ROI, smaller extent, shorter time series, etc. I.e. run in chunks.|
| EXPORT_TABLE_BUFFER      | int     | null         | meters                          | Buffer distance applied to features before reduction. |
| EXPORT_TABLE_REDUCER     | str     | first        | first, ...                      | Reducer applied to intersecting pixels. |
| EXPORT_TABLE_DROP_NODATA | bool    | false        | true, false                     | Drop rows comprised entirely of nodata. |
| EXPORT_PER_FEATURE       | bool    | false        | true, false                     | Export separate image per feature / band (**Attention**: can result in many tasks). |
| EXPORT_PER_TIME          | bool    | false        | true, false                     | Export each timestamp image separately (**Attention**: can result in even more tasks). |

---
