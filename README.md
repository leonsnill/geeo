# Geographical and Ecological Earth Observation (geeo)
**geeo** is a processing pipeline and collection of algorithms that uses the [Google Earth Engine (GEE) Python API](https://developers.google.com/earth-engine/guides/python_install) for creating Analysis-Ready-Data (ARD) from optical imagery, including the Landsat and Sentinel-2 archives. The package is structured along hierarchical levels, emphasizing a standardized, reproducible and efficient workflow, covering the suite from image preprocessing, harmonization, and spatial organisation, to advanced feature generation and time series analyses. 
Processing instructions are readily defined using a .yml-file or python dictionary, facilitating stand-alone use or interactive integration into processing workflows using GEE:

![sample SVG image](geeo/data/fig/geeo_workflow_manuscript.svg)

**geeo** includes processing routines frequently applied in geographical and ecological studies that use satellite remote sensing. The selection of routines is primarily influenced by work conducted in the [Biogeography Lab](https://pages.cms.hu-berlin.de/biogeo/website/) and [Earth Observation Lab](https://eolab.geographie.hu-berlin.de/) at Humboldt University of Berlin. Inspiration for the modular structure and parameter file communication comes from David Frantz' [Framework for Operational Radiometric Correction for Environmental monitoring (FORCE)](https://force-eo.readthedocs.io/en/latest/index.html), a highly-advanced all-in-one processing engine for medium-resolution Earth Observation image archives for your computing infrastructure.

# Access and installation

You need to have [access to Google Earth Engine](https://developers.google.com/earth-engine/guides/access) and its [Python API](https://developers.google.com/earth-engine/guides/python_install). Read the following sections how to set up the latter for your local environment or by using a Jupyter Notebook hosted via Google Colab.  

### Local Python environment
Make sure you have a Python 3 distribution of our choice installed (e.g. Anaconda). Prior to installing **geeo**, you preferably want to set up a new environment and also install the package's dependencies using `conda` (alternatively the dependencies are installed automatically when `pip` installing **geeo**):

```bash
conda create -n geeo_env python=3.12 ipykernel earthengine-api pyyaml pandas geopandas matplotlib tqdm ipyleaflet ipywidgets gdal scikit-learn geemap
conda activate geeo_env
```

Once created and activated, you can directly install the package from GitHub using `pip`:

```bash
pip install git+https://github.com/leonsnill/geeo.git
```

### Google Colab
You can also directly get started using [Google Colab](https://colab.research.google.com/) for hosting a Jupyter Notebook.

In a new .ipynb-notebook, simply install **geeo** in the first code chunk like so:

```python
!pip install git+https://github.com/leonsnill/geeo.git
```

**Tutorial notebooks** 

Introducing geeo: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonsnill/geeo/blob/master/docs/tutorial_0_introducing-geeo.ipynb). 

Spatial tiling and metadata: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonsnill/geeo/blob/master/docs/tutorial_1_spatial-tiling-and-metadata.ipynb)

Auxiliary exports: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonsnill/geeo/blob/master/docs/tutorial_2_auxiliary-exports.ipynb)

Worked example of habitat mapping: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonsnill/geeo/blob/master/docs/habitat-type-mapping_geeo.ipynb) 

### Import

Import, authenticate and initialize the Earth Engine python API, then import **geeo**.

```python
# Google cloud project name with Earth Engine API enabled
my_project_name = ''

import ee
ee.Authenticate()
ee.Initialize(project=my_project_name)
import geeo
```


# Documentation and examples

In the [docs folder](docs), you will find the **[documentation to the parameter settings](docs/documentation.md)**, the settings for instructing the core processing chain of geeo. To become familiar with the module design and handling, you also find some example Jupyter Notebooks on how to instruct certain processing chains, visualize ouputs and request exports in the [docs folder](docs).

### Quick use overview

The settings for the main processing chain of geeo can be defined using either a .yml text file or python dictionary. 

```python
# Google cloud project name with Earth Engine API enabled
my_project_name = ''
# import required packages
import ee
ee.Authenticate()
ee.Initialize(project=my_project_name)
import geeo


# -----------------------------------------------------------------
# Option 1) Parameter .yml file
# create new .yml file to set instructions
geeo.create_parameter_file('new_param_file')
# open parameter file in editor and set instructions ...
# run instructions: level-2 -> level-3 -> level-4 -> export
run = geeo.run_param('new_param_file.yml')

# -----------------------------------------------------------------
# Option 2) python dictionary of user-settings
param_dict = {
    # your settings ...
}
run = geeo.run_param(param_dict)
```

Let`s use the dict approach to illustrate the basic workflow. Consider the simple task of calculating Spectral-Temporal-Metrics (STMs) of the NDVI using Landsat for three seasons for the area of greater Berlin in 2024.

```python
# -----------------------------------------------------------------
# Option 2) python dictionary of user-settings

param_dict = {
    
    'YEAR_MIN': 2024,
    'YEAR_MAX': 2024,
    'ROI': [13.07, 52.37, 13.78, 52.64],  # Berlin
    'SENSORS': ['L8', 'L9'],  # Landsat-8 and Landsat-9
    'FEATURES': ['NDVI'],
    
    'STM': ['p10', 'p50', 'p90', 'stdDev'],  # reducer metrics
    'FOLD_CUSTOM': {'month': ['3-5', '6-8', '9-11']},  # spring, summer, autumn sub-windows
    'STM_FOLDING': True,  # apply sub-windows to STM calculation
    
    'EXPORT_IMAGE': False,  # global setting to export any image
    'EXPORT_STM': False  # setting to export STMs if EXPORT_IMAGE was True

}
# all remaining settings will be set to default values from blueprint!

run = geeo.run_param(param_dict)

# get STM ee.ImageCollection (collection of ee.Image subwindows)
stm = run.get('STM')
print('STM bands: ', stm.first().bandNames().getInfo())
```

We leave many settings to default for illustrational purposes, including export settings regarding resampling, projection and metadata.
But in essence, that is all that would be needed to:

- Combine Landsat-8 and Landsat-9 Collection 2 Tier 1 Surface Reflectance collections
- Apply quality masks (cloud, c. shadow, etc.)
- Calculate the NDVI
- Construct temporal subwindows for filtering
- Retrieve STMs for subwindows
- Export result to Drive/Asset (*in theory*, here set to False) 
