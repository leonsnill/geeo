# Geographical and Ecological Earth Observation (GEEO)
GEEO is a processing pipeline and collection of algorithms for obtaining Analysis-Ready-Data (ARD) from Landsat and Sentinel-2 using the Google Earth Engine Python API.
The modules are organized along different hierarchical levels, and processing instructions are readily defined by the user via a parameter file (.yml) or Python dictionary:

![sample SVG image](geeo/data/fig/geeo_workflow_update.svg)

GEEO includes processing routines frequently applied in geographical and ecological studies that use satellite remote sensing. The selection of routines is primarily influenced by work conducted in the [Biogeography Lab](https://pages.cms.hu-berlin.de/biogeo/website/) and [Earth Observation Lab](https://eolab.geographie.hu-berlin.de/) at Humboldt-University of Berlin. Inspiration for structuring the module along a parameter file comes from David Frantz' [Framework for Operational Radiometric Correction for Environmental monitoring (FORCE)](https://force-eo.readthedocs.io/en/latest/index.html).

---

### Installation

Make sure you have a Python 3 distribution of our choice installed (e.g. Anaconda). Prior to installing `geeo`, you preferably want to set up a new virtual environment and also install the package's dependencies using `conda` (alternatively the dependencies are installed automatically when `pip` installing `geeo`):

```bash
conda create -n geeo_env python=3.11 ipykernel earthengine-api pyyaml pandas geopandas matplotlib tqdm ipyleaflet ipywidgets gdal scikit-learn
conda activate geeo_env
```

Once created and activated, you can directly install the package using `pip`:

```bash
pip install git+https://github.com/leonsnill/geeo.git
```

---

### Getting started

To become familiar with the module design and handling we have prepared short examples to instruct certain processing chains, visualize ouputs and request exports. You can find them in the folder [docs](docs). Here you also find the [documentation to the parameter file](docs/documentation.md).


