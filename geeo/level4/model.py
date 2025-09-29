import os
import sys
import time
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import is_classifier, is_regressor
import ee

# ==============================================================================
# GENERAL FUNCTIONS

# function to calculate the size of a DataFrame or list in MB
def get_size_in_mb(obj):
    """Calculate the size of a DataFrame or list in MB."""
    size_bytes = sys.getsizeof(obj)
    if isinstance(obj, pd.DataFrame):
        size_bytes += obj.memory_usage(deep=True).sum()
    elif isinstance(obj, list):
        size_bytes += sum(sys.getsizeof(item) for item in obj)
    size_mb = size_bytes / (1024 ** 2)  # Convert bytes to MB
    return size_mb

# function to check if a folder/asset exists in the Earth Engine asset
def asset_exists(folder_path):
    try:
        ee.data.getAsset(folder_path)
        return True
    except ee.EEException:
        return False

# function to create a folder in the Earth Engine asset
def create_folder(folder_path):
    ee.data.createAsset({'type': 'Folder'}, folder_path)

# function to ensure that each sublist ends with an empty string
def ensure_ends_with_empty_string(list_of_lists):
    for sublist in list_of_lists:
        if sublist[-1] != '':
            sublist.append('')
    return list_of_lists

# ==============================================================================
# DECISION TREES

# --------------------------------------------------------------------------
# the following function 'tree_to_string' was copied and adapted 
# from the geemap.ml module written by Kel Markert and Qiusheng Wu
# https://geemap.org/ml/#geemap.ml.tree_to_string
# Copyright (c) 2020-2024, Qiusheng Wu
# --------------------------------------------------------------------------

def tree_to_string(tree, features, model_type='classification'):

    # extract out the information need to build the tree string
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature_idx = tree.feature
    impurities = tree.impurity
    n_samples = tree.n_node_samples
    thresholds = tree.threshold
    features = [features[i] for i in feature_idx]

    raw_vals = np.squeeze(tree.value)

    if model_type == 'classification':
        # take argmax along class axis from values
        values = raw_vals.argmax(axis=-1)
        out_type = int

    elif model_type == 'regression':
        # take values and drop un needed axis
        values = np.around(raw_vals, decimals=4)
        out_type = float

    else:
        raise ValueError("model_type must be either 'classification' or 'regression'")
    
    # use iterative pre-order search to extract node depth and leaf information
    node_ids = np.zeros(shape=n_nodes, dtype=np.int64)
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        node_ids[node_id] = node_id

        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    # create a table of the initial structure
    # each row is a node or leaf
    df = pd.DataFrame(
        {
            "node_id": node_ids,
            "node_depth": node_depth,
            "is_leaf": is_leaves,
            "children_left": children_left,
            "children_right": children_right,
            "value": values,
            "criterion": impurities,
            "n_samples": n_samples,
            "threshold": thresholds,
            "feature_name": features,
            "sign": ["<="] * n_nodes,
        },
        dtype="object",
    )

    # get a dict of right node rows and assign key based on index where to insert
    inserts = {}
    for row in df.itertuples():
        child_r = row.children_right
        if child_r > row.Index:
            ordered_row = np.array(row)
            ordered_row[-1] = ">"
            inserts[child_r] = ordered_row[1:]  # drop index value
    # sort the inserts as to keep track of the additive indexing
    inserts_sorted = {k: inserts[k] for k in sorted(inserts.keys())}

    # loop through the row inserts and add to table (array)
    table_values = df.values
    for i, k in enumerate(inserts_sorted.keys()):
        table_values = np.insert(table_values, (k + i), inserts_sorted[k], axis=0)

    # make the ordered table array into a dataframe
    # note: df is dtype "object", need to cast later on
    ordered_df = pd.DataFrame(table_values, columns=df.columns)

    max_depth = np.max(ordered_df.node_depth.astype(int))
    tree_str = f"1) root {n_samples[0]} 9999 9999 ({impurities.sum()})\n"
    previous_depth = -1
    cnts = []
    # loop through the nodes and calculate the node number and values per node
    for row in ordered_df.itertuples():
        node_depth = int(row.node_depth)
        left = int(row.children_left)
        right = int(row.children_right)
        if left != right:
            if row.Index == 0:
                cnt = 2
            elif previous_depth > node_depth:
                depths = ordered_df.node_depth.values[: row.Index]
                idx = np.where(depths == node_depth)[0][-1]
                # cnt = (cnts[row.Index-1] // 2) + 1
                cnt = cnts[idx] + 1
            elif previous_depth < node_depth:
                cnt = cnts[row.Index - 1] * 2
            elif previous_depth == node_depth:
                cnt = cnts[row.Index - 1] + 1

            if node_depth == (max_depth - 1):
                value = out_type(ordered_df.iloc[row.Index + 1].value)
                samps = int(ordered_df.iloc[row.Index + 1].n_samples)
                criterion = float(ordered_df.iloc[row.Index + 1].criterion)
                tail = " *\n"
            else:
                if (
                    (bool(ordered_df.loc[ordered_df.node_id == left].iloc[0].is_leaf))
                    and (
                        bool(
                            int(row.Index)
                            < int(ordered_df.loc[ordered_df.node_id == left].index[0])
                        )
                    )
                    and (str(row.sign) == "<=")
                ):
                    rowx = ordered_df.loc[ordered_df.node_id == left].iloc[0]
                    tail = " *\n"
                    value = out_type(rowx.value)
                    samps = int(rowx.n_samples)
                    criterion = float(rowx.criterion)

                elif (
                    (bool(ordered_df.loc[ordered_df.node_id == right].iloc[0].is_leaf))
                    and (
                        bool(
                            int(row.Index)
                            < int(ordered_df.loc[ordered_df.node_id == right].index[0])
                        )
                    )
                    and (str(row.sign) == ">")
                ):
                    rowx = ordered_df.loc[ordered_df.node_id == right].iloc[0]
                    tail = " *\n"
                    value = out_type(rowx.value)
                    samps = int(rowx.n_samples)
                    criterion = float(rowx.criterion)

                else:
                    value = out_type(row.value)
                    samps = int(row.n_samples)
                    criterion = float(row.criterion)
                    tail = "\n"

            # extract out the information needed in each line
            spacing = (node_depth + 1) * "  "  # for pretty printing
            fname = str(row.feature_name)  # name of the feature (i.e. band name)
            tresh = float(row.threshold)  # threshold
            sign = str(row.sign)

            #tree_str += f"{spacing}{cnt}) {fname} {sign} {tresh:.4f} {0} {0} {value}{tail}"
            tree_str += f"{spacing}{cnt}) {fname} {sign} {tresh:.6f} {samps} {criterion:.4f} {value}{tail}"
            previous_depth = node_depth
        cnts.append(cnt)

    return tree_str

# -------------------------------------------------------------------------- 

# function to convert a list of tree strings to a pandas DataFrame
def dt_string_list_to_df(trees):
    # Split each tree string into rows
    tree_rows = [tree.split('\n') for tree in trees]
    # append another line break to the end of each tree string
    tree_rows= ensure_ends_with_empty_string(tree_rows)
    # determinne the maximum number of rows
    max_rows = max(len(rows) for rows in tree_rows)
    data = {}
    for i, rows in enumerate(tree_rows):
        col_name = f'tree{i+1}'
        data[col_name] = rows + ['NA'] * (max_rows - len(rows))
    # convert dictonary to a pandas DataFrame
    df = pd.DataFrame(data)
    
    # append a row of 'NA': Needed for ee.CLassifier.decisionTreeEnsemble correct formating 
    # 'Classifier.decisionTreeEnsemble: Error parsing line XXXX: unexpected end of input.'
    na_row = ['NA'] * len(df.columns)
    df.loc[len(df)] = na_row
    return df

# function to convert a DataFrame to an Earth Engine asset
def dt_df_to_asset(df, folder, name='model', num_chunks=None, overwrite_asset=False):
    if not asset_exists(folder):
        create_folder(folder)
        print(f'Folder {folder} created.')
    else:
        print(f'Folder {folder} exists.')

    # size of df
    total_size_mb = get_size_in_mb(df)
    # should not exceed 10MB, otherwise split
    if num_chunks is None:
        num_chunks = int(total_size_mb // 10) + 1
    # convert number of chunks to rows
    chunk_size = int(np.ceil(len(df) / num_chunks))
    df_chunks = [df[i:i+chunk_size] for i in range(0, df.shape[0], chunk_size)]
    
    asset_paths = []
    for i, chunk_df in enumerate(df_chunks):
        #chunk_df = df.iloc[i * chunk_size:(i + 1) * chunk_size]
        gee_features = [ee.Feature(ee.Geometry.Point([0, 0]), row.to_dict()) for _, row in chunk_df.iterrows()]
        feature_collection = ee.FeatureCollection(gee_features)

        if num_chunks == 1:
            asset_id = folder + '/' + name
        else:
            asset_id = folder + '/' + name + f'_chunk-{i + 1}'

        if asset_exists(asset_id):
            if overwrite_asset:
                ee.data.deleteAsset(asset_id)
                print(f'Asset {asset_id} already exists. Overwriting...')
            else:
                print(f'Asset {asset_id} already exists. Set overwrite_asset=True to upload.')
                break

        asset_paths.append(asset_id)
        task = ee.batch.Export.table.toAsset(
            collection=feature_collection,
            description=name+f'_chunk-{i + 1}',
            assetId=asset_id
        )
        task.start()
        print(f'Started upload number {i + 1} to {asset_id}.')

    # task status
    print("Waiting for uploads to complete...")
    while any(task.status()['state'] in ['READY', 'RUNNING'] for task in ee.batch.Task.list()[:num_chunks]):
        time.sleep(5)

    # if chunk size > 1, combine the assets
    if num_chunks > 1:
        uploaded_collections = [ee.FeatureCollection(path) for path in asset_paths]
        combined_feature_collection = ee.FeatureCollection(uploaded_collections).flatten()
        combined_asset_path = folder + '/' + name
        task = ee.batch.Export.table.toAsset(
            collection=combined_feature_collection,
            description=name,
            assetId=combined_asset_path
        )
        task.start()
        print(f'Started upload for combined asset to {combined_asset_path}.')
        print("Waiting for final upload to complete...")
        while ee.batch.Task.list()[0].status()['state'] in ['READY', 'RUNNING']:
            time.sleep(5)
        asset = combined_asset_path
        
        # delete the chunk assets
        for a in asset_paths:
            ee.data.deleteAsset(a)
    else:
        asset = asset_paths[0]

    print("All uploads completed.")
    return ee.FeatureCollection(asset)

# function to convert an Earth Engine FeatureCollection asset (df of string decision rules) to a decisionTreeEnsemble
def dt_table_asset_to_model(asset):
    feature_collection = ee.FeatureCollection(asset)
    tree_names = ee.List(list(feature_collection.first().getInfo()['properties'].keys()))

    def get_tree_string(tree):
        tree_string = feature_collection.aggregate_array(tree).filter(ee.Filter.neq('item', "NA")).join("\n")
        return tree_string

    tree_list = ee.List(tree_names.map(get_tree_string))
    model = ee.Classifier.decisionTreeEnsemble(tree_list)

    return model

# function to extract the trees from a model and convert them to a string
def extract_trees_to_string(trees, feature_names, model_type='classification', n_jobs=1):
    if n_jobs == 1:
        tree_strings = [tree_to_string(tree, feature_names, model_type) for tree in trees]
    elif n_jobs == -1:
        tree_strings = Parallel(n_jobs=os.cpu_count())(
            delayed(tree_to_string)(tree, feature_names, model_type) for tree in trees
        )
    else:
        tree_strings = Parallel(n_jobs=n_jobs)(
            delayed(tree_to_string)(tree, feature_names, model_type) for tree in trees
        )

    return tree_strings

# wrapper function to convert a list of tree strings to a decisionTreeEnsemble model
def dt_model_to_ee(tree_model, features=None, folder=None, upload_asset=True, name='model', overwrite_asset=False, n_jobs=1):
    
    # create list of trees for different inputs.
    if isinstance(tree_model, sklearn.ensemble._forest.RandomForestClassifier) \
        or isinstance(tree_model, sklearn.ensemble._forest.RandomForestRegressor):
        estimators = [est.tree_ for est in tree_model.estimators_]
    elif isinstance(tree_model, sklearn.tree._classes.DecisionTreeClassifier) \
        or isinstance(tree_model, sklearn.tree._classes.DecisionTreeRegressor):
        estimators = [tree_model.tree_]
    else:
        raise ValueError("tree_model must be a DecisionTreeClassifier, DecisionTreeRegressor, \
                         RandomForestClassifier, or RandomForestRegressor")
    
    # get feature names from model object
    # if features not provided, try to get from model
    if features is None:
        try:
            features = list(tree_model.feature_names_in_)
        except:
            raise ValueError("features must be provided if not present in the model object")

    # get model type
    if is_classifier(tree_model):
        model_type = 'classification'
    elif is_regressor(tree_model):
        model_type = 'regression'
    else:
        raise ValueError("tree_model must be a DecisionTreeClassifier, DecisionTreeRegressor, \
                         RandomForestClassifier, or RandomForestRegressor")

    # convert estimators to tree strings
    print('Converting tree estimators to strings ...')
    trees_string = extract_trees_to_string(estimators, features, model_type=model_type, n_jobs=n_jobs)

    # calculate the size of the tree string to determine if the model can be created directly
    tree_string_size = get_size_in_mb(trees_string)
    
    # check if the tree string size is smaller than 10 MB and if the user wants to upload the asset
    if tree_string_size <= 10 and not upload_asset:
        print('Uploading model ...')
        ee_strings = [ee.String(tree) for tree in trees_string]
        model = ee.Classifier.decisionTreeEnsemble(ee_strings)
    else:
        df = dt_string_list_to_df(trees_string)
        asset = dt_df_to_asset(df, folder, name, overwrite_asset=overwrite_asset)
        model = dt_table_asset_to_model(asset)
    return model
