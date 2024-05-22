import zarr
import requests
import os
import io
import numpy as np
import tifffile
from skimage.measure import block_reduce
from numcodecs import Blosc
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import shuffle
Blosc.use_threads = True
USER = os.environ.get('SCROLLPRIZE_USER')
PASS = os.environ.get('SCROLLPRIZE_PASS')

def download(url):
    print(url)
    response = requests.get(url, auth=(USER, PASS))
    if response.status_code == 200:
        filedata = io.BytesIO(response.content)
        if url.endswith('.tif'):
            with tifffile.TiffFile(filedata) as tif:
                data = tif.asarray()
                if data.dtype == np.uint16:
                    return ((data >> 8) & 0xf0).astype(np.uint8)
                else:
                    raise
        elif url.endswith('.jpg'):
            data = np.array(Image.open(filedata))
            return data & 0xf0
        elif url.endswith('.png'):
            data = np.array(Image.open(filedata))
            return data
    else:
        raise Exception(f'Failed to download {url}. Status code: {response.status_code}')

class CVol:
    def __init__(self, path):
        os.makedirs(path, exist_ok=True)
        os.makedirs(f'{path}/PHerc0332/20231027191953',exist_ok=True)
        os.makedirs(f'{path}/PHerc0332/20231117143551',exist_ok=True)
        os.makedirs(f'{path}/PHerc0332/20231201141544',exist_ok=True)

        os.makedirs(f'{path}/PHerc1667/20231107190228',exist_ok=True)
        os.makedirs(f'{path}/PHerc1667/20231117161658',exist_ok=True)

        os.makedirs(f'{path}/Scroll1/20230205180739',exist_ok=True)

        os.makedirs(f'{path}/Scroll2/20230210143520',exist_ok=True)
        os.makedirs(f'{path}/Scroll2/20230212125146',exist_ok=True)

        pass


if __name__ == '__main__':
    vol = CVol("D:/vesuvius.cvol")