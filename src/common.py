import numpy as np
import requests
import tifffile
import io
import platform
import os
from PIL import Image
import time
from functools import wraps

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

if platform.system() == 'Windows':
  CACHEDIR = "D:/"
else:
  CACHEDIR = "/mnt/d/"


def timing_decorator(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"{func.__name__} executed in {elapsed_time:.6f} seconds")
    return result

  return wrapper


@timing_decorator
def download(url):
  path = url.replace("https://", "")
  path = os.path.join(CACHEDIR,path)
  if not os.path.exists(path):
    print(f"downloading {url}")
    response = requests.get(url)
    if response.status_code == 200:
      os.makedirs(os.path.dirname(path), exist_ok=True)
      with io.BytesIO(response.content) as filedata, tifffile.TiffFile(filedata) as tif:
        data = (tif.asarray() >> 8).astype(np.uint8) & 0xf0
        tifffile.imwrite(path, data)
    else:
      raise Exception(f'Cannot download {url}')

  if url.endswith('.tif'):
      tif = tifffile.imread(path)
      data = tif
      if data.dtype == np.uint16:
        return ((data >> 8) & 0xf0).astype(np.uint8)
      else:
        return data
  elif url.endswith('.jpg') or url.endswith('.png'):
    data = np.asarray(Image.open(path))
    return data & 0xf0