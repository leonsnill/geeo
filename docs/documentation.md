# GEEO Parameter Reference

## SPACE AND TIME

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| YEAR_MIN                 | int     | 2023         | Any year                        | Minimum year for analysis.                       |
| YEAR_MAX                 | int     | 2023         | Any year                        | Maximum year for analysis.                       |
| MONTH_MIN                | int     | 1            | 1-12                            | Minimum month for analysis.                      |
| MONTH_MAX                | int     | 12           | 1-12                            | Maximum month for analysis.                      |
| DOY_MIN                  | int     | 1            | 1-366                           | Minimum day of year.                             |
| DOY_MAX                  | int     | 366          | 1-366                           | Maximum day of year.                             |
| DATE_MIN                 | str     | null         | YYYYMMDD                        | Alternative minimum date (overrides above).      |
| DATE_MAX                 | str     | null         | YYYYMMDD                        | Alternative maximum date (overrides above).      |
| ROI                      | list    | [12.9, 52.2, 13.9, 52.7] | See description               | Region of Interest: point, rectangle, polygon, or path. |
| ROI_SIMPLIFY_GEOM_TO_BBOX| bool    | false        | true, false                     | Simplify geometry to bounding box.               |

---

## SENSOR AND DATA QUALITY SETTINGS

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| SENSORS                  | list    | [L9, L8, L7, L5, L4] | L9, L8, L7, L5, L4, S2      | Sensors to use.                                  |
| MAX_CLOUD                | int     | 75           | 0-100                           | Maximum cloud cover percentage.                  |
| EXCLUDE_SLCOFF           | bool    | false        | true, false                     | Exclude Landsat-7 ETM+ SLC-off.                  |
| GCP_MIN_LANDSAT          | int     | 1            | >=1                             | Min Ground Control Points for Landsat.           |
| MASKS_LANDSAT            | list    | [cloud, cshadow, snow, fill, dilated] | See description | Landsat masks to apply.                          |
| MASKS_LANDSAT_CONF       | str     | Medium       | Medium, High                    | Confidence level for Landsat masks.              |
| MASKS_S2                 | str     | CPLUS        | null, CPLUS, PROB, SCL          | Sentinel-2 masking method.                       |
| MASKS_S2_CPLUS           | float   | 0.6          | 0-1                             | S2 CPLUS threshold.                              |
| MASKS_S2_PROB            | int     | 30           | 0-100                           | S2 PROB threshold.                               |
| MASKS_S2_NIR_THRESH_SHADOW| float  | 0.2          | 0-1                             | S2 NIR threshold for shadow mask.                |
| ERODE_DILATE             | bool    | false        | true, false                     | Erode and dilate mask.                           |
| ERODE_RADIUS             | int     | 60           | meters                          | Erosion radius.                                  |
| DILATE_RADIUS            | int     | 120          | meters                          | Dilation radius.                                 |
| ERODE_DILATE_SCALE       | int     | 90           | meters                          | Pixel scale for erode/dilate.                    |
| BLUE_MAX_MASKING         | float   | null         | 0-1                             | Max blue reflectance to mask.                    |

---

## BANDS | INDICES | FEATURES

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| FEATURES                 | list    | [BLU, GRN, RED, NIR, SW1, SW2] | See description | Selected bands, indices, and features.           |
| DEM                      | bool    | false        | true, false                     | Add Copernicus DEM (30m) to output.              |

---

## LINEAR UNMIXING

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| UMX                      | dict    | null         | See description                 | Endmember dictionary for unmixing.               |
| UMX_SUM_TO_ONE           | bool    | true         | true, false                     | Sum to one constraint.                           |
| UMX_NON_NEGATIVE         | bool    | true         | true, false                     | Non-negative constraint.                         |
| UMX_REMOVE_INPUT_FEATURES| bool    | true         | true, false                     | Keep only unmixing features.                     |

---

## TIME SERIES STACK (TSS) and MOSAIC (TSM)

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| TSM                      | bool    | false        | true, false                     | Enable Time Series Mosaic (TSM).                 |

---

## TEMPORAL SUBWINDOWS | FOLDING

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| FOLD_YEAR                | bool    | false        | true, false                     | Fold per year.                                   |
| FOLD_MONTH               | bool    | false        | true, false                     | Fold per month.                                  |
| FOLD_CUSTOM              | dict    | {year: null, month: null, doy: null, date: null} | See description | Custom folding (range or target+-offset).        |

---

## TIME SERIES INTERPOLATION (TSI)

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| TSI                      | str     | null         | null, WLIN, 1RBF, 2RBF, 3RBF    | Time Series Interpolation method.                |
| TSI_BASE_IMGCOL          | str     | TSS          | TSS, TSM                        | Base image collection for TSI.                   |
| INTERVAL                 | int     | 16           | days                            | Interpolation interval.                          |
| INTERVAL_UNIT            | str     | day          | day, month, year                | Interval unit.                                   |
| INIT_JANUARY_1ST         | bool    | false        | true, false                     | Initialize interval on Jan 1st.                  |
| SIGMA1                   | int     | 16           | days                            | Sigma1 for RBF.                                  |
| SIGMA2                   | int     | 32           | days                            | Sigma2 for RBF.                                  |
| SIGMA3                   | int     | 64           | days                            | Sigma3 for RBF.                                  |
| WIN1                     | int     | 16           | days                            | Window1 for RBF.                                 |
| WIN2                     | int     | 32           | days                            | Window2 for RBF.                                 |
| WIN3                     | int     | 64           | days                            | Window3 for RBF.                                 |
| BW1                      | int     | 4            |                                 | Weight for WIN1.                                 |
| BW2                      | int     | 8            |                                 | Weight for WIN2.                                 |

---

## SPECTRAL TEMPORAL METRICS (STM)

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| STM                      | list    | null         | min, p5, ..., max, mean, etc.   | Spectral-Temporal Metrics to compute.            |
| STM_BASE_IMGCOL          | str     | TSS          | TSS, TSM, TSI                   | Collection to derive STMs from.                  |
| STM_FOLDING              | bool    | false        | true, false                     | Use folding settings from above.                 |

---

## PIXEL-BASED COMPOSITING (PBC)

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| PBC                      | str     | null         | null, BAP, MAX-RNB, NLCD, FEATURE | Pixel-Based Compositing method.               |
| PBC_BASE_IMGCOL          | str     | TSS          | TSS, TSM, TSI                   | Base image collection for PBC.                   |
| PBC_INVERT_QUALITY_METRIC| bool    | false        | true, false                     | Invert if minima are better.                     |
| PBC_BAP_DOY_EQ_YEAR      | int     | 30           | days                            | DOY-offset for BAP.                              |
| PBC_BAP_MAX_CLOUDDISTANCE| int     | 500          | meters                          | Max cloud distance for BAP.                      |
| PBC_BAP_MIN_CLOUDDISTANCE| int     | 0            | meters                          | Min cloud distance for BAP.                      |
| PBC_BAP_WEIGHT_DOY       | float   | 0.6          | 0-1                             | Weight for DOY-score.                            |
| PBC_BAP_WEIGHT_YEAR      | float   | 0.2          | 0-1                             | Weight for YEAR-score.                           |
| PBC_BAP_WEIGHT_CLOUD     | float   | 0.2          | 0-1                             | Weight for CLOUD-score.                          |

---

## EXPORT

| Parameter                | Type    | Default      | Allowed Values / Format         | Description                                      |
|--------------------------|---------|--------------|---------------------------------|--------------------------------------------------|
| PIX_RES                  | int     | 30           | meters                          | Pixel resolution.                                |
| CRS                      | str     | EPSG:4326    | EPSG, WKT, UTM, Mollweide, etc. | Coordinate Reference System.                     |
| CRS_TRANSFORM            | list    | null         | [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation] | GeoTransform parameters for affine transformation of output. |
| IMG_DIMENSIONS           | str     | null         | WidthxHeight, e.g. 1000x1000    | The dimensions to use for the exported image in pixels.    |
| RESAMPLING_METHOD        | str     | null         | null, bilinear, bicubic         | Resampling method.                               |
| DATATYPE                 | str     | int16        | uint8, int8, uint16, ...        | Output data type.                                |
| DATATYPE_SCALE           | int     | 10000        |                                 | Scale factor.                                    |
| NODATA_VALUE             | int     | -9999        |                                 | Output no data value.                            |
| EXPORT_IMAGE             | bool    | false        | true, false                     | Export image.                                    |
| EXPORT_TABLE             | bool    | false        | true, false                     | Export table.                                    |
| EXPORT_TSS               | bool    | false        | true, false                     | Export TSS.                                      |
| EXPORT_TSM               | bool    | false        | true, false                     | Export TSM.                                      |
| EXPORT_TSI               | bool    | false        | true, false                     | Export TSI.                                      |
| EXPORT_STM               | bool    | false        | true, false                     | Export STM.                                      |
| EXPORT_PBC               | bool    | false        | true, false                     | Export PBC.                                      |
| EXPORT_TRD               | bool    | false        | true, false                     | Export TRD.                                      |
| EXPORT_LOCATION          | str     | Drive        | Drive, Asset                    | Export location.                                 |
| EXPORT_DIRECTORY         | str     | null         |                                 | Export directory or assetId.                     |
| EXPORT_DESC              | str     | GEEO         |                                 | Image description in filename.                   |
| EXPORT_DESC_DETAIL_TIME  | bool    | false        | true, false                     | Append detailed time description.                |
| EXPORT_BANDNAMES_AS_CSV  | bool    | false        | true, false                     | Bandnames as CSV.                                |
| EXPORT_TABLE_FORMAT      | str     | CSV          | CSV, GeoJSON, KML, etc.         | Table export format.                             |
| EXPORT_TABLE_METHOD      | str     | reduceRegions| reduceRegions, reduceRegion     | Table export method.                             |
| EXPORT_TABLE_TILE_SCALE  | float   | 1            | 0.1-16                          | Compute tile size.                               |
| EXPORT_TABLE_BUFFER      | int     | null         | meters                          | Buffer features before reduction.                |
| EXPORT_TABLE_REDUCER     | str     | first        | first, ...                      | Reducer for table export.                        |
| EXPORT_TABLE_DROP_NODATA | bool    | false        | true, false                     | Drop nodata rows.                                |
| EXPORT_PER_FEATURE       | bool    | false        | true, false                     | Export per feature/band.                         |
| EXPORT_PER_TIME          | bool    | false        | true, false                     | Export per time.                                 |
