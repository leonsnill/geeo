import ee

def rename_bands_l4(img):
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL', 'QA_RADSAT']
    new_bands = ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2', 'QA_PIXEL', 'QA_RADSAT']
    return img.select(bands).rename(new_bands).set('satellite', 'L4')


def rename_bands_l5(img):
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'ST_B6', 'SR_B7', 'QA_PIXEL', 'QA_RADSAT']
    new_bands = ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'LST', 'SW2', 'QA_PIXEL', 'QA_RADSAT']
    return img.select(bands).rename(new_bands).set('satellite', 'L5')


def rename_bands_l7(img):
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'ST_B6', 'SR_B7', 'QA_PIXEL', 'QA_RADSAT']
    new_bands = ['BLU', 'GRN', 'RED', 'NIR', 'SW1', 'LST', 'SW2', 'QA_PIXEL', 'QA_RADSAT']
    return img.select(bands).rename(new_bands).set('satellite', 'L7')


def rename_bands_l8(img):
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 
    'ST_B10', 'QA_PIXEL', 'QA_RADSAT']
    new_bands = ['CLU', 'BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2',
    'LST', 'QA_PIXEL', 'QA_RADSAT']
    return img.select(bands).rename(new_bands).set('satellite', 'L8')


def rename_bands_l9(img):
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 
    'ST_B10', 'QA_PIXEL', 'QA_RADSAT']
    new_bands = ['CLU', 'BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2',
    'LST', 'QA_PIXEL', 'QA_RADSAT']
    return img.select(bands).rename(new_bands).set('satellite', 'L9')


def rename_bands_s2(img):
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9',
             'B11', 'B12', 'QA60', 'SCL']  
    new_bands = ['AER', 'BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'NIR', 'RE4', 'WV',
                 'SW1', 'SW2', 'QA', 'SCL']
    return img.select(bands).rename(new_bands).set('satellite', 'S2')


def rename_bands_hlss30(img):
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9',
             'B11', 'B12', 'Fmask']  
    new_bands = ['AER', 'BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'NIR', 'RE4', 'WV',
                 'SW1', 'SW2', 'Fmask']
    return img.select(bands).rename(new_bands).set('satellite', 'HLSS30')


def rename_bands_hlsl30(img):
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 
    'B10', 'Fmask']
    new_bands = ['CLU', 'BLU', 'GRN', 'RED', 'NIR', 'SW1', 'SW2',
    'LST', 'Fmask']
    return img.select(bands).rename(new_bands).set('satellite', 'HLSL30')


def rename_bands_s2_l1c(img):
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9',
             'B11', 'B12', 'QA60']  
    new_bands = ['AER', 'BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'NIR', 'RE4', 'WV',
                 'SW1', 'SW2', 'QA']
    return img.select(bands).rename(new_bands).set('satellite', 'S2')


def scale_bands(bands, scale=1e4, offset=0.0):
    def wrap(img):
        imgs = img.select(bands).multiply(scale).add(offset)
        return img.addBands(imgs, overwrite=True)
    return wrap


def scale_and_dtype(inp, scale=1, dtype=None, nodata=None):

    # if input is an image
    if isinstance(inp, ee.image.Image):
        # scale image
        if scale != 1:
            inp = inp.multiply(scale).set('system:index', inp.get('system:index'), 'system:time_start', inp.get('system:time_start'))
        # set nodata value
        if nodata:
            inp = inp.unmask(nodata).set('system:index', inp.get('system:index'), 'system:time_start', inp.get('system:time_start'))
        # set data type
        if dtype:
            if dtype == 'int8':
                inp = inp.int8()
            elif dtype == 'uint8':
                inp = inp.uint8()
            elif dtype == 'int16':
                inp = inp.int16()
            elif dtype == 'uint16':
                inp = inp.uint16()
            elif dtype == 'int32':
                inp = inp.int32()
            elif dtype == 'uint32':
                inp = inp.uint32()
            elif dtype == 'float':
                inp = inp.float()
            elif dtype == 'double':
                inp = inp.double()
            else:
                raise ValueError(f"Unknown data type: {dtype}")
            
    
    elif isinstance(inp, ee.imagecollection.ImageCollection):
        # scale image collection
        if scale:
            inp = inp.map(lambda img: img.multiply(scale).set('system:index', img.get('system:index'), 'system:time_start', img.get('system:time_start')))
        # set nodata value
        if nodata:
            inp = inp.map(lambda img: img.unmask(nodata).set('system:index', img.get('system:index'), 'system:time_start', img.get('system:time_start')))
        # set data type
        if dtype:
            if dtype == 'int8':
                inp = inp.map(lambda img: img.int8())
            elif dtype == 'uint8':
                inp = inp.map(lambda img: img.uint8())
            elif dtype == 'int16':
                inp = inp.map(lambda img: img.int16())
            elif dtype == 'uint16':
                inp = inp.map(lambda img: img.uint16())
            elif dtype == 'int32':
                inp = inp.map(lambda img: img.int32())
            elif dtype == 'uint32':
                inp = inp.map(lambda img: img.uint32())
            elif dtype == 'float':
                inp = inp.map(lambda img: img.float())
            elif dtype == 'double':
                inp = inp.map(lambda img: img.double())
            else:
                raise ValueError(f"Unknown data type: {dtype}")
    
    else:
        raise ValueError(f"Unknown input type: {type(inp)}")

    return inp



    