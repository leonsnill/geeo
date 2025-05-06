from setuptools import setup, find_packages

setup(
    name='geeo',
    url='https://github.com/leonsnill/geeo.git',
    author=['Leon Nill', 'Shawn Schneidereit', 'Matthias Baumann'],
    author_email='leon.nill@geo.hu-berlin.de',
    packages=find_packages(),
    package_data={
        'geeo': ['config/*.yml', 'data/GLANCE-tiles/*.gpkg', 'data/ne_10m_land.gpkg'],
    },
    install_requires=['ipykernel', 'earthengine-api', 'pyyaml', 
                      'pandas', 'geopandas', 'matplotlib', 'tqdm',
                      'ipyleaflet', 'ipywidgets', 'gdal', 'scikit-learn'],
    version='1.0',
    license='MIT',
    description='GEEO is a processing pipeline and collection of algorithms for obtaining Analysis-Ready-Data (ARD) from Landsat and Sentinel-2 using the Google Earth Engine Python API.',
    #long_description=open('README.md').read(),
)

# EOF