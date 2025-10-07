'''
Mapping pan-European Land Cover - Part 3
This file merges the exprted Landsat-based Time Series Stack (TSS) table data inot a single FeatureCollection
and uploads the final datasets as Earth Engine Asset.
'''
# ------------------------------------------------------------------------------------------------------------
# import packages
import ee
ee.Authenticate()
ee.Initialize(project='eexnill')
import eerepr
eerepr.initialize()

def list_assets_in_folder(folder_path):
    try:
        assets = ee.data.listAssets({'parent': folder_path})
        asset_list = [asset['name'] for asset in assets.get('assets', [])]
        print(f"Assets in folder '{folder_path}':")
        for asset in asset_list:
            print(asset)
        return asset_list
    except Exception as e:
        print(f"Error listing assets in folder: {e}")
        return []

# merge collections into single collection
def merge_feature_collections(asset_ids):
    merged = ee.FeatureCollection([])
    for aid in asset_ids:
        merged = merged.merge(ee.FeatureCollection(aid))
    return merged


# load list of assets in folder
dir_asset = 'projects/eexnill/assets/geeo_public/landsat-tss'
l_assets = list_assets_in_folder(dir_asset)

# ------------------------------------------------------------------------------------------------------------
# CHELSA + DEM
l_assets = [a for a in l_assets if 'CHELSA_DEM' in a and 'LUCAS' in a]

# merge (server-side deferred)
lucas = merge_feature_collections(l_assets)

# export to collections
task = ee.batch.Export.table.toAsset(
    collection=lucas,
    description='LUCAS',
    assetId='projects/eexnill/assets/geeo_public/landsat-tss/LUCAS_HARMO_V1_EO_LC_CHELSA_DEM'
)
task.start()

# ------------------------------------------------------------------------------------------------------------
# TSS
l_assets = [a for a in l_assets if 'TSS' in a and 'LUCAS' in a]
lucas = merge_feature_collections(l_assets)
# export to collections
task = ee.batch.Export.table.toAsset(
    collection=lucas,
    description='TSS_LUCAS',
    assetId='projects/eexnill/assets/geeo_public/landsat-tss/TSS_LUCAS_HARMO_V1_EO_LC_EU'
)
task.start()

# ------------------------------------------------------------------------------------------------------------
# export to Drive for local analyses

# subfilter
lucas_aux = ee.FeatureCollection('projects/eexnill/assets/geeo_public/landsat-tss/LUCAS_HARMO_V1_EO_LC_CHELSA_DEM')
lucas_tss = ee.FeatureCollection('projects/eexnill/assets/geeo_public/landsat-tss/TSS_LUCAS_HARMO_V1_EO_LC_EU')

lucas_aux = lucas_aux.distinct(['YYYYMMDD', 'id'])
lucas_tss = lucas_tss.distinct(['YYYYMMDD', 'id'])

task = ee.batch.Export.table.toDrive(
    collection=lucas_aux,
    description='LUCAS_HARMO_V1_EO_LC_CHELSA_DEM'
)
task.start()
task = ee.batch.Export.table.toDrive(
    collection=lucas_tss,
    description='TSS_LUCAS_HARMO_V1_EO_LC_EU'
)
task.start()

# EOF