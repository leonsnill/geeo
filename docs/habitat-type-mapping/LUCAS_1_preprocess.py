'''
Mapping pan-European Land Cover - Part 1
This file pre-processes the LUCAS dataset according to the methodology described by Pflugmacher et al. (2019) 
and uploads the final dataset to a publicly accessible Earth Engine Asset.

Harmonized LUCAS dataset available in EE:
https://www.nature.com/articles/s41597-020-00675-z
https://developers.google.com/earth-engine/datasets/catalog/JRC_LUCAS_HARMO_THLOC_V1#description

'''

# ---------------------------------------------------------------------------------------------------
# import packages
import ee
ee.Authenticate()
ee.Initialize(project='eexnill')
import eerepr
eerepr.initialize()
import geemap
import geeo


# ---------------------------------------------------------------------------------------------------
# LUCAS Harmonized (Theoretical Location, 2006-2018) V1
# we use the theoretical location (THLOC) dataset which provides the best possible location accuracy (see Weigand et al. 2020)
lucas = ee.FeatureCollection('JRC/LUCAS_HARMO/THLOC/V1')

# filter to 2015 and 2018
lucas = lucas.filter(ee.Filter.inList('year', [2015, 2018]))

# filter so that lc1_perc >= 50% (str), except for artificial land (lc1 = A10) where all samples are taken
lucas = lucas.filter(
    ee.Filter.Or(
        ee.Filter.Or(
            ee.Filter.inList('lc1', ['A00', 'A10', 'A11', 'A12']),
            ee.Filter.And(
                ee.Filter.neq('lc1', ''),
                ee.Filter.inList('lc1_perc', ['> 75 %', '', '50 - 75 %'])
            )
        )
    )
)

# LC label set?
lucas = lucas.filter(ee.Filter.neq('lc1', ''))

# plot area > 0.5ha, keep all A* (artificial) regardless of area
lucas = lucas.filter(
    ee.Filter.Or(
        ee.Filter.inList('lc1', ['A00', 'A10', 'A11', 'A12']),
        ee.Filter.inList('parcel_area_ha', ['1 - 10 ha', '> 10 ha', '0.5 - 1 ha', ''])
    )
)

# cols to keep
cols = ['id', 'year', 'lc1', 'lc1_label', 'lc1_perc', 'parcel_area_ha', 'revisit']
lucas = lucas.select(cols)

# add LC ID and name (to match Pflugmacher et al., 2019)
lc_id_dict = ee.Dictionary({
    # 0: artificial land
    'A00': 0, 'A10': 0, 'A11': 0, 'A12': 0,
    # 1: cropland, seasonal
    'B10': 1, 'B11': 1, 'B12': 1, 'B13': 1, 'B14': 1, 'B15': 1, 'B16': 1, 'B17': 1, 'B18': 1, 'B19': 1,
    'B20': 1, 'B21': 1, 'B22': 1, 'B23': 1,
    'B30': 1, 'B31': 1, 'B32': 1, 'B33': 1, 'B34': 1, 'B35': 1, 'B36': 1, 'B37': 1, 'B38': 1,
    'B40': 1, 'B41': 1, 'B42': 1, 'B43': 1, 'B44': 1, 'B45': 1,
    'B50': 1, 'B51': 1, 'B52': 1, 'B53': 1, 'B54': 1,
    # 2: cropland, perennial
    'B70': 2, 'B71': 2, 'B72': 2, 'B73': 2, 'B74': 2, 'B75': 2, 'B76': 2, 'B77': 2, 'B78': 2,
    'B80': 2, 'B81': 2, 'B82': 2, 'B83': 2, 'B84': 2, 'B85': 2,
    # 3: forest, broadleaved
    'C10': 3, 'C11': 3, 'C12': 3, 'C13': 3,
    # 4: forest, coniferous
    'C20': 4, 'C21': 4, 'C22': 4, 'C23': 4,
    # 5: forest, mixed 
    'C30': 5, 'C31': 5, 'C32': 5, 'C33': 5,
    # 6: shrubland
    'D00': 6, 'D10': 6, 'D20': 6,
    # 7: grassland
    'E00': 7, 'E10': 7, 'E20': 7,
    # 8: barren (F00)
    'F00': 8, 'F10': 8, 'F20': 8, 'F30': 8, 'F40': 8,
    # 9: wetland
    'H10': 9, 'H11': 9, 'H12': 9,
    'H20': 9, 'H21': 9, 'H22': 9, 'H23': 9,
    # 10: water
    'G00': 10, 'G10': 10, 'G20': 10, 'G11': 10, 'G12': 10, 'G21': 10, 'G22': 10, 'G30': 10, 'G40': 10,
    # 11: snow/ice
    'G50': 11
})

# ID to name mapping
lc_id_name = ee.Dictionary({
    0: 'Artificial land',
    1: 'Cropland, seasonal', 
    2: 'Cropland, perennial',
    3: 'Forest, broadleaved',
    4: 'Forest, coniferous', 
    5: 'Forest, mixed',
    6: 'Shrubland',
    7: 'Grassland',
    8: 'Barren',
    9: 'Wetland',
    10: 'Water',
    11: 'Snow/ice'
})

# first filter to known keys
lc_keys = ee.List(lc_id_dict.keys())

lucas = lucas.filter(
    ee.Filter.inList("lc1", lc_keys)
)

lucas = lucas.map(
    lambda f: f.set('LC_ID', lc_id_dict.get(f.get('lc1')))
)
lucas = lucas.map(
    lambda f: f.set('LC_NAME', lc_id_name.get(f.get('LC_ID')))
)

# filter out features where LC_ID is None/empty
lucas = lucas.filter(ee.Filter.notNull(['LC_ID']))

# -------------------------------------------------------------------------------------------------
# export ee.FeatureCollection to Asset
out_table = ee.batch.Export.table.toAsset(
    collection=lucas, 
    description='LUCAS_LC',
    assetId='projects/eexnill/assets/geeo_public/LUCAS_HARMO_V1_EO_LC'
    )
ee.batch.Task.start(out_table)


'''
REFERENCES

Pflugmacher, D. et al. (2019) ‘Mapping pan-European land cover using Landsat spectral-temporal metrics and the European LUCAS survey’, 
Remote Sensing of Environment, 221(December 2018), pp. 583–595. Available at: https://doi.org/10.1016/j.rse.2018.12.001.

Weigand, M. et al. (2020) ‘Spatial and semantic effects of LUCAS samples on fully automated land use/land cover classification in high-resolution Sentinel-2 data’, 
International Journal of Applied Earth Observation and Geoinformation, 88, p. 102065. Available at: https://doi.org/10.1016/j.jag.2020.102065.
'''
# EOF