import numpy as np
import requests
import tifffile
import io
import platform
import os
from PIL import Image
import time
from functools import wraps
import glob

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

if platform.system() == 'Windows':
  CACHEDIR = "c:/vesuvius.cache"
else:
  CACHEDIR = "/mnt/c/"


def timing_decorator(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    start_time = time.perf_counter()
    print(f"executing {func.__name__}")
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
  else:
    print(f"getting {path} from cache")

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

@timing_decorator
def get_tiff_stack_chunk(scroll, volume, z, y, x):
  assert 1 <= scroll <= 4
  scrolltxt = ['',
               'Scroll1/PHercParis4.volpkg',
               'Scroll2/PHercParis3.volpkg',
               'Scroll3/PHerc332.volpkg',
               'Scroll4/PHerc1667.volpkg'][scroll]

  url = f"https://dl.ash2txt.org/full-scrolls/{scrolltxt}/volume_grids/{volume}/cell_yxz_{y:03d}_{x:03d}_{z:03d}.tif"
  chunk = download(url)
  return chunk


@timing_decorator
def get_chunk(scroll, volume, zstart, ystart, xstart, zsize, ysize, xsize):
  directory_path = r"D:\dl.ash2txt.org\full-scrolls\Scroll1\PHercParis4.volpkg\volumes_masked\20230205180739"
  # Get list of TIFF files and sort them
  tiff_files = sorted(glob.glob(os.path.join(directory_path, '*.tiff')))
  tiff_files.extend(sorted(glob.glob(os.path.join(directory_path, '*.tif'))))

  # Validate inputs
  if not tiff_files:
    raise ValueError("No TIFF files found in directory")

  # Check if we have enough files
  if zstart + zsize > len(tiff_files):
    raise ValueError(f"Not enough TIFF files for requested z-range. Found {len(tiff_files)} files.")

  # Check dimensions of first image to validate x,y coordinates
  sample_img = np.array(Image.open(tiff_files[0]))
  if (xstart + xsize > sample_img.shape[1] or
      ystart + ysize > sample_img.shape[0]):
    raise ValueError(f"Requested cube exceeds image dimensions {sample_img.shape}")

  # Initialize 3D array to store the cube
  cube = np.zeros((zsize, ysize, xsize), dtype=sample_img.dtype)

  # Extract the requested portion from each relevant TIFF file
  for z in range(zsize):
    current_file = tiff_files[z + zstart]
    img = np.array(Image.open(current_file))
    if img.dtype == np.uint16:
      img >>=8
      img = img.astype(np.uint8)
    cube[z, :, :] = img[ystart:ystart + ysize,
                    xstart:xstart + xsize]

  return cube

#def get_full_chunk(scroll,volume,start,size):
#  ret = []
#  for z in range(start,size[0]+start):
