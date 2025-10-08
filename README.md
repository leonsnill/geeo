# Geographical and Ecological Earth Observation (geeo)
**GEEO** is a processing pipeline and collection of algorithms that uses the [Google Earth Engine (GEE) Python API](https://developers.google.com/earth-engine/guides/python_install) for creating Analysis-Ready-Data (ARD) from optical imagery, including the Landsat and Sentinel-2 archives. The package is structured along hierarchical levels, emphasizing a standardized, reproducible and efficient workflow, covering the suite from image preprocessing, harmonization, and spatial organisation, to advanced feature generation and time series analyses. 
Processing instructions are readily defined using a .yml-file or python dictionary, facilitating stand-alone use or interactive integration into processing workflows using GEE:

![sample SVG image](geeo/data/fig/geeo_workflow_manuscript.svg)

**GEEO** includes processing routines frequently applied in geographical and ecological studies that use satellite remote sensing. The selection of routines is primarily influenced by work conducted in the [Biogeography Lab](https://pages.cms.hu-berlin.de/biogeo/website/) and [Earth Observation Lab](https://eolab.geographie.hu-berlin.de/) at Humboldt University of Berlin. Inspiration for the modular structure and parameter file communication comes from David Frantz' [Framework for Operational Radiometric Correction for Environmental monitoring (FORCE)](https://force-eo.readthedocs.io/en/latest/index.html), a highly-advanced all-in-one processing engine for medium-resolution Earth Observation image archives for your computing infrastructure.

<br>

# Access and installation

You need to have [access to Google Earth Engine](https://developers.google.com/earth-engine/guides/access) and its [Python API](https://developers.google.com/earth-engine/guides/python_install). Read the following sections how to set up the latter for your local Python environment or by using a Jupyter Notebook hosted on Google Colab.  

## Local Python environment
Make sure you have a Python 3 distribution of our choice installed (e.g. Anaconda). Prior to installing **GEEO**, you preferably want to set up a new environment and also install the package's dependencies using `conda` (alternatively the dependencies are installed automatically when `pip` installing **GEEO**):

```bash
conda create -n geeo_env -c conda-forge python=3.12 ipykernel earthengine-api pyyaml pandas geopandas matplotlib tqdm ipyleaflet ipywidgets gdal scikit-learn eerepr geemap
conda activate geeo_env
```

Once created and activated, you can directly install the package from GitHub using `pip`:

```bash
pip install git+https://github.com/leonsnill/geeo.git
```

## Google Colab
You can also directly get started using [Google Colab](https://colab.research.google.com/) for hosting a Jupyter Notebook.

In a new .ipynb-notebook, simply install **GEEO** in the first code chunk like so:

```python
!pip install git+https://github.com/leonsnill/geeo.git
```

## Import after installation

Import, authenticate and initialize the Earth Engine python API, then import **GEEO**. Make sure to set the Earth Engine eligible Google Cloud project name when initializing.

```python
# Google cloud project name with Earth Engine API enabled
gcloud_project_name = ''

import ee
ee.Authenticate()
ee.Initialize(project=gcloud_project_name)
import geeo
```

<br>

# Documentation and examples

In the [docs folder](docs), you will find the **[documentation to the parameter settings](docs/documentation.md)**, the settings for instructing the core processing chain of geeo and their description. To become familiar with the module design and handling, you also find some example Jupyter Notebooks on how to instruct certain processing chains, visualize ouputs and request exports in the [docs folder](docs) - or access them using Colab:

- Introducing geeo: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonsnill/geeo/blob/master/docs/tutorial_0_introducing-geeo.ipynb)
- Worked example on habitat mapping: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonsnill/geeo/blob/master/docs/habitat-type-mapping_geeo.ipynb) 

Additional tutorials
- Spatial tiling and metadata: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonsnill/geeo/blob/master/docs/tutorial_1_spatial-tiling-and-metadata.ipynb)
- Auxiliary exports: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonsnill/geeo/blob/master/docs/tutorial_2_auxiliary-exports.ipynb)

More tutorials are in development and the list will be updated in the future.


### Quick use overview

Import, authenticate and initialize the Earth Engine python API, then import **GEEO**:

```python
gcloud_project_name = ''
import ee
ee.Authenticate()
ee.Initialize(project=gcloud_project_name)
import geeo
```

The settings for the main processing chain of geeo can be defined using either a .yml-file or python dictionary. 

#### Option 1) Parameter file
```python
# create new .yml file to set instructions
geeo.create_parameter_file('new_param_file')

# open parameter file in editor, set instructions, and safe.

# then simply run instructions:
run = geeo.run_param('new_param_file.yml')
```

The `run` variable (dictionary) will contain all parameter settings and newly calculated products. If exports are requested, they will be printed to console and run in the [Earth Engine Task Manager](https://code.earthengine.google.com/tasks). 

#### Option 2) Dictionary 
**GEEO** also allows for giving processing instructions using a python dictionary directly as input:

```python
# specify key-value pair matching parameter names
# all non-specified parameters will be set to default settings from parameter template
param_dict = {
    'YEAR_MIN': 1985,
    'YEAR_MAX': 1990,
    # ...
}
# run
run = geeo.run_param(param_dict)
```

---

Let`s use the dictionary approach to illustrate a basic workflow. Consider the simple task of calculating Spectral-Temporal-Metrics (STMs) of the NDVI using Landsat for three seasons for the area of greater Berlin in 2024. STMs are a commonly applied form of statistical reduction in remote sensing, in which pixel-wise statistics across individual bands/features (-> NDVI) are calculated for a given set of imagery (-> Landsat) over tim (-> Mar-May, Jun-Aug, Sep-Nov 2024).
We simply edit the parameters that we actually need for calculating the desired output:

```python
# -----------------------------------------------------------------
# Option 2) python dictionary of user-settings

param_dict = {
    
    'YEAR_MIN': 2024,
    'YEAR_MAX': 2024,
    'ROI': [13.07, 52.37, 13.78, 52.64],  # Berlin simplified to bounding box coordinates
    'SENSORS': ['L8', 'L9'],  # Landsat-8 and Landsat-9 Collection 2 Surface Reflectance Tier 1 data
    'FEATURES': ['NDVI'],  # we only want the NDVI
    
    'STM': ['p10', 'p50', 'p90', 'stdDev'],  # reducer metrics: 10%, 50% (median), and 90% percentiles percentile, standard deviation
    'FOLD_CUSTOM': {'month': ['3-5', '6-8', '9-11']},  # spring, summer, autumn sub-windows
    'STM_FOLDING': True,  # apply sub-windows to STM calculation
    
    'EXPORT_IMAGE': False,  # setting to export image products in general
    'EXPORT_STM': False  # setting to export STM products in particular (if EXPORT_IMAGE was True!)

}
# all remaining settings will be set to default values from blueprint!

# run the instructions
run = geeo.run_param(param_dict)

# get STM ee.ImageCollection (collection of ee.Image subwindows)
stm = run.get('STM')
print('STM bands: ', stm.first().bandNames().getInfo())
```

We left many parameters to default values for illustrational purposes. This includes export settings regarding resampling, projection and metadata, as well as quality masking settings. They are all set to reasonable, - and if possible - common/generic settings (e.g. cloud masking). Nevertheless, being explicit about these settings for your actual implementations is highly recommended!

What we now did in essence with just these few lines of code, is:

- Combine Landsat-8 and Landsat-9 Collection 2 Tier 1 Surface Reflectance collections into one collection, including scaling the band to actual reflectance values (0-100)
- Apply quality masks to 'invalid' pixels (cloud, cloud shadow, snow/ice, fill, dilated clouds)
- Calculate the NDVI 
- Calculate the percentiles and standard deviation for three temporal subwindows for 2024
- Export result to Drive/Asset (*in theory*, here we set it to False to not trigger unneccesary processing) 

---