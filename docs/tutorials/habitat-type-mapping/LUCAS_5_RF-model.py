'''
Mapping pan-European Land Cover - Part 5
This file creates RF models from the STM+CHELSA+DEM data and uploads the final model as Earth Engine Asset.
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV
from sklearn import metrics
import matplotlib.pyplot as plt
import dill

# ------------------------------------------------------------------------------------------------------------------
# input data
# LUCAS + predictor data
df = pd.read_csv('STM_LUCAS_HARMO_V1_EO_LC_EU_final.csv')

# y
target_column = 'LC_ID'
columns_not_for_model = ['id', 'year', '.geo']
# X
features = [col for col in df.columns if col != target_column and col not in columns_not_for_model]

# drop na in features or target
df = df.dropna(subset=features + [target_column])

# ------------------------------------------------------------------------------------------------------------------
# feature selection

# 1) correlation analysis to drop highly correlated features (>0.9)
corr_matrix = df[features].corr().abs()
to_drop = set()
while True:
    pairs = np.where(corr_matrix > 0.9)
    pairs = [(features[i], features[j]) for i, j in zip(*pairs) if i < j and features[i] not in to_drop and features[j] not in to_drop]
    if not pairs:
        break
    for f1, f2 in pairs:
        var1 = df[f1].var()
        var2 = df[f2].var()
        drop = f1 if var1 < var2 else f2
        to_drop.add(drop)
    corr_matrix = corr_matrix.drop(index=to_drop, columns=to_drop)

features = [f for f in features if f not in to_drop]

# 2) recursive feature elimination with cross-validation (RFECV)
X = df[features].values
y = df[target_column].values

# split in train and test
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=30
)

# RFECV using training data only
rf = RandomForestClassifier(
    n_estimators=250,
    max_samples=5000,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced_subsample',
    n_jobs=25,
    random_state=30
)

rfecv = RFECV(
    rf,
    step=0.05,
    cv=StratifiedKFold(5, shuffle=True, random_state=30),
    scoring='balanced_accuracy',
    n_jobs=5,
    verbose=1
)

rfecv.fit(X_train_full, y_train)
# get selected features
selected_features = np.array(features)[rfecv.support_]

# plt feature ranking by imp
importances = rfecv.estimator_.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(selected_features)), importances[indices], align="center")
plt.xticks(range(len(selected_features)), np.array(selected_features)[indices], rotation=90)
plt.tight_layout()
plt.show()

'''
Resulting selected features:

['summer_SW1_p95', 'summer_NIR_p95', 'summer_NBR_p5', 'summer_MDWI_p95', 'summer_NBR_stdDev', 'bio5', 'slope', 
'full_NIR_p25', 'full_NBR_p5', 'DEM', 'full_NIR_p95', 'autumn_MDWI_p75', 'full_NBR_stdDev', 'summer_RED_stdDev', 
'full_RED_p75', 'full_NBR_p75', 'full_NBR_p95', 'spring_NIR_p95', 'full_MDWI_p75', 'autumn_NBR_p25', 'bio9', 
'autumn_SW1_p95', 'full_NIR_stdDev', 'bio6', 'bio18', 'autumn_NDVI_p25', 'spring_NDVI_p75', 
'spring_MDWI_p25', 'full_NDVI_p25', 'full_SW2_stdDev', 'spring_MDWI_p50', 'spring_SW1_p95', 'spring_NDVI_p5']
'''

# ------------------------------------------------------------------------------------------------------------------
# evaluate model performance with selected features
# assess hyperparameters visualy to use small model that is more efficient in EE
X_train = X_train_full[:, [features.index(f) for f in selected_features]]
X_test = X_test_full[:, [features.index(f) for f in selected_features]]

# number of trees
n_estimators_range = [10, 50, 100, 250, 500, 1000, 2000]
scores = []
for n in n_estimators_range:
    rf_eval = RandomForestClassifier(
        n_estimators=n,
        n_jobs=25,
        random_state=30,
        max_samples=10000,
        class_weight='balanced_subsample'
    )
    rf_eval.fit(X_train, y_train)
    y_pred = rf_eval.predict(X_test)
    scores.append(metrics.balanced_accuracy_score(y_test, y_pred))

plt.plot(n_estimators_range, scores)
plt.xlabel('n_estimators')
plt.ylabel('Balanced Accuracy')
plt.title('Error Curve: Number of Trees')
plt.show()

# number of samples (per tree) / instead of using n samples eqivalent to input sample size
sample_sizes = [1000, 5000, 10000, 20000, 50000, 100000]
scores = []
rng = np.random.default_rng(30)
for s in sample_sizes:
    rf_lc = RandomForestClassifier(
        n_estimators=300,
        n_jobs=50,
        random_state=30,
        max_samples=s,
        min_samples_leaf=2,
        min_samples_split=5,
        max_depth=50,
        class_weight='balanced_subsample'
    )
    rf_lc.fit(X_train, y_train)
    y_pred = rf_lc.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

plt.plot([min(s, len(X_train)) for s in sample_sizes], scores)
plt.xlabel('Training Samples')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.show()

# ------------------------------------------------------------------------------------------------------------------
# final model(s)

# train and test
rf_final = RandomForestClassifier(
    n_estimators=50,
    n_jobs=50,
    random_state=30,
    min_samples_leaf=2,
    min_samples_split=3,
    max_samples=50000,
    max_depth=30,
    class_weight='balanced_subsample'
)
rf_final.fit(X_train, y_train)
y_pred = rf_final.predict(X_test)
print('Final Model Performance:')
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Balanced Accuracy:', metrics.balanced_accuracy_score(y_test, y_pred))
print('Classification Report:\n', metrics.classification_report(y_test, y_pred))

# all data for final model
rf_final = RandomForestClassifier(
    n_estimators=50,
    n_jobs=50,
    random_state=30,
    min_samples_leaf=2,
    min_samples_split=3,
    max_samples=50000,
    max_depth=50,
    class_weight='balanced_subsample'
)
rf_final.fit(df[selected_features].values, y)

# save model
with open('/data/Aldhani/users/nillleox/projects/geeo/habitat-mapping/RF-MODEL_LUCAS_NTREE-50.pkl', 'wb') as f:
    dill.dump(rf_final, f)

# ------------------------------------------------------------------------------------------------------------------
# upload to GEE
import ee
ee.Authenticate() 
ee.Initialize(project='eexnill')
from geeo.level4.model import dt_model_to_ee

'''
FEATURES (from above)

['summer_SW1_p95',
 'summer_NIR_p95',
 'summer_NBR_p5',
 'summer_MDWI_p95',
 'summer_NBR_stdDev',
 'bio5',
 'slope',
 'full_NIR_p25',
 'full_NBR_p5',
 'DEM',
 'full_NIR_p95',
 'autumn_MDWI_p75',
 'full_NBR_stdDev',
 'summer_RED_stdDev',
 'full_RED_p75',
 'full_NBR_p75',
 'full_NBR_p95',
 'spring_NIR_p95',
 'full_MDWI_p75',
 'autumn_NBR_p25',
 'bio9',
 'autumn_SW1_p95',
 'full_NIR_stdDev',
 'bio6',
 'bio18',
 'autumn_NDVI_p25',
 'spring_NDVI_p75',
 'spring_MDWI_p25',
 'full_NDVI_p25',
 'full_SW2_stdDev',
 'spring_MDWI_p50',
 'spring_SW1_p95',
 'spring_NDVI_p5']
'''

# upload decision tree as strings to ee.FeatureCollection 
my_model = dt_model_to_ee(tree_model=rf_final, features=selected_features, 
                          folder='projects/eexnill/assets/geeo_public', 
                          upload_asset=True, name='RF-MODEL_LUCAS_NTREE-50', 
                          overwrite_asset=True, n_jobs=50)

# test if everything uploaded correctly
# load table asset to model
model = 'projects/eexnill/assets/geeo_public/RF-MODEL_LUCAS_NTREE-50' 
# geeo has a function to load the table asset to a model
from geeo.level4.model import dt_table_asset_to_model
rf = dt_table_asset_to_model(model)
rf

# EOF