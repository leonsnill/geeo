import ee


# =================================================================================================
# FUNCTIONS

# Normalized Difference Vegetation Index (NDVI); Rouse et al. (1974)
def ndvi(img):
       ndvi = img.normalizedDifference(['NIR', 'RED']).rename('NDVI')
       return img.addBands(ndvi)

# kernalized Normalized Difference Vegetation Index (kNDVI); Camps-Valls et al. 2021
# https://www.science.org/doi/10.1126/sciadv.abc7447
def kndvi(img):
     kndvi = ee.Image(
          ((img.normalizedDifference(['NIR', 'RED'])).pow(2)).tanh()
     ).rename('KNDVI')
     return img.addBands(kndvi)

# Enhanced Vegetation Index (EVI); Huete et al. (2002)
def evi(gain=2.5, l=1, c1=6, c2=7.5):
    def wrap(img):
        evi = img.expression(
            'gain * ((nir - red) / (nir + c1 * red - c2 * blue + l))',
            {
                'gain': gain,
                'nir': img.select('NIR'),
                'red': img.select('RED'),
                'blue': img.select('BLU'),
                'c1': c1,
                'c2': c2,
                'l': l
            }
        ).rename('EVI')
        return img.addBands(evi)
    return wrap

# Normalized Difference Moisture Index (NDMI); Gao (1996)
def ndmi(img):
       ndmi = img.normalizedDifference(['NIR', 'SW1']).rename('NDMI')
       return img.addBands(ndmi)

# Normalized Difference Water Index (NDWI); McFeeters (1996)
def ndwi(img):
       ndwi = img.normalizedDifference(['GRN', 'NIR']).rename('NDWI')
       return img.addBands(ndwi)

# Modified Normalized Difference Water Index (MNDWI); Xu (2006)
def mndwi(img):
       mndwi = img.normalizedDifference(['GRN', 'SW1']).rename('MDWI')
       return img.addBands(mndwi)

# Zha et al. (2003): Use of Normalized Difference Built-Up Index in Automatically Mapping Urban Areas from TM Imagery
def ndbi(img):
    ndbi = img.normalizedDifference(['SW1', 'NIR']).rename('NDB')
    return img.addBands(ndbi)

# Normalized Burn Ration; Key and Benson (2006)
def nbr(img):
    nbr = img.normalizedDifference(['NIR', 'SW2']).rename('NBR')
    return img.addBands(nbr)

# RNB; Qiu et al. 2023
def rnb(img):
    rnb = img.select('NIR').divide(img.select('BLU')).rename('RNB')
    return img.addBands(rnb)



# Fractional Vegetation Cover (FVC); Jimenez-Munoz et al. (2009)
def fvc(ndvi_soil=0.15, ndvi_vegetation=0.9):
    """
    Derive fractional vegetation cover from linear relationship to NDVI
    Default valaues follow recommendations for higher resolution data according to Jimenez-Munoz et al. (2009):
    "Comparison Between Fractional Vegetation Cover Retrievals from Vegetation Indices and Spectral Mixture Analysis:
    Case Study of PROBA/CHRIS Data Over an Agricultural Area", Sensors, 9(2), 768–793.
    """
    def wrap(img):
        fvc = img.expression(
            '((NDVI-NDVI_s)/(NDVI_v-NDVI_s))**2',
            {
                'NDVI': img.select('NDV'),
                'NDVI_s': ndvi_soil,
                'NDVI_v': ndvi_vegetation
            }
        )
        fvc = fvc.where(fvc.expression('fvc > 1',
                                                   {'fvc': fvc}), 1)
        fvc = fvc.where(fvc.expression('fvc < 0',
                                                   {'fvc': fvc}), 0).rename('FVC')
        return img.addBands(fvc)
    return wrap

# SWIR ratio (endmember trangular space Kowalski et al. 2022)
def sw_ratio(img):
    sw_ratio = img.select('SW2').divide(img.select('SW1')).rename('SWR')
    return img.addBands(sw_ratio)


# =================================================================================================
# Tasseled Cap Transformation (brightness, greenness, wetness) based on Christ 1985

# changed the below to the expression version because a .multiply() call on a .select() ee.Image
# for some uknown reason mixes up the mask at the mask borders when using bilinear / cubic resampling
# Fix is to call resample after calculation of all bands or simply use the below version
'''
def tcg(img):
     coeffs = ee.Image([-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446])
     tcg = img.select(['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2']).multiply(coeffs).reduce(ee.Reducer.sum()).rename('TCG')
     return img.addBands(tcg)
def tcb(img):
     coeffs = ee.Image([0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303])
     tcb = img.select(['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2']).multiply(coeffs).reduce(ee.Reducer.sum()).rename('TCB')
     return img.addBands(tcb)
def tcw(img):
     coeffs = ee.Image([0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109])
     tcw = img.select(['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2']).multiply(coeffs).reduce(ee.Reducer.sum()).rename('TCW')
     return img.addBands(tcw)
'''
def tcg(img):
    tcg = img.expression(
                         'B*(-0.1603) + G*(-0.2819) + R*(-0.4934) + NIR*0.7940 + SWIR1*(-0.0002) + SWIR2*(-0.1446)',
                         {
                         'B': img.select(['BLU']),
                         'G': img.select(['GRN']),
                         'R': img.select(['RED']),
                         'NIR': img.select(['NIR']),
                         'SWIR1': img.select(['SW1']),
                         'SWIR2': img.select(['SW2'])
                         }).rename('TCG')
    return img.addBands(tcg)


def tcb(img):
    tcb = img.expression(
                         'B*0.2043 + G*0.4158 + R*0.5524 + NIR*0.5741 + SWIR1*0.3124 + SWIR2*0.2303',
                         {
                         'B': img.select(['BLU']),
                         'G': img.select(['GRN']),
                         'R': img.select(['RED']),
                         'NIR': img.select(['NIR']),
                         'SWIR1': img.select(['SW1']),
                         'SWIR2': img.select(['SW2'])
                         }).rename('TCB')
    return img.addBands(tcb)


def tcw(img):
       tcw = img.expression(
              'B*0.0315 + G*0.2021 + R*0.3102 + NIR*0.1594 + SWIR1*(-0.6806) + SWIR2*(-0.6109)',
              {
                     'B': img.select(['BLU']),
                     'G': img.select(['GRN']),
                     'R': img.select(['RED']),
                     'NIR': img.select(['NIR']),
                     'SWIR1': img.select(['SW1']),
                     'SWIR2': img.select(['SW2'])
              }).rename('TCW')
       return img.addBands(tcw)

# =================================================================================================
# LINEAR SPECRAL UNMIXING

def unmix(bands, endmember_values, endmember_names, sumToOne=False, nonNegative=False):
     def wrap(img):
        img_unx = ee.Image(img.select(bands).unmix(endmember_values, sumToOne=sumToOne, nonNegative=nonNegative)).rename(endmember_names)
        return img.addBands(img_unx)
     return wrap


# =================================================================================================
# INDEX DICT
dict_features = {
    'NDVI': ndvi,
    'KNDVI': kndvi, 
    'EVI': evi(gain=2.5, l=1, c1=6, c2=7.5),
    'NDWI': ndwi,
    'NDMI': ndmi,
    'MNDWI': mndwi,
    'NDBI': ndbi,
    'NBR': nbr,
    'TCG': tcg,
    'TCB': tcb,
    'TCW': tcw,
    'NDBI': ndbi,
    'MDWI': mndwi,
    'SWR': sw_ratio
}

# EOF