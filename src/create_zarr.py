import os
import zarr
from numcodecs import Blosc
import skimage
import numpy as np

import common
import preprocessing
import numbamath

class ZVol:
  def __init__(self, basepath, create=False, writeable=False):
    if create:
      if os.path.exists(f"{basepath}/vesuvius.zvol"):
        raise ValueError("Cannot create a zvol if there is already a zvol there")
      synch = zarr.ProcessSynchronizer(f'{basepath}/vesuvius.sync')
      self.root = zarr.group(store=f"{basepath}/vesuvius.zvol", synchronizer=synch,)
      compressor = Blosc(cname='blosclz', clevel=9, shuffle=Blosc.BITSHUFFLE)
      #scroll 1
      self.root.zeros('20230205180739', shape=(14376,7888,8096), chunks=(256,256,256), dtype='u1', compressor=compressor)
      self.root.zeros('20230205180739_downloaded', shape=(14376,), dtype='u1')
      #self.root.zeros('20230206171837', shape=(10532,7812,8316), chunks=(256,256,256), dtype='u1', compressor=compressor)
      #self.root.zeros('20230206171837_downloaded', shape=(10532,), dtype='u1')


      #scroll 2
      #volumes = self.root.zeros('20230206082907', shape=(14376,7888,8096), chunks=(256,256,256), dtype='u1', compressor=compressor)
      #volumes = self.root.zeros('20230206171837', shape=(10532,7812,8316), chunks=(256,256,256), dtype='u1', compressor=compressor)
    else:
      self.root = zarr.open(f"{basepath}/vesuvius.zvol", mode="r+" if writeable else "r")

  def chunk(self, volume, start, size):
    if start[0] + size[0] < 0 or start[0] + size[0] >= self.root[volume].shape[0]:
      raise ValueError(f"{start} {size} out of dimension for {self.root[volume].shape}")
    if start[1] + size[1] < 0 or start[1] + size[1] >= self.root[volume].shape[1]:
      raise ValueError(f"{start} {size} out of dimension for {self.root[volume].shape}")
    if start[2] + size[2] < 0 or start[2] + size[2] >= self.root[volume].shape[2]:
      raise ValueError(f"{start} {size} out of dimension for {self.root[volume].shape}")

    for z in range(start[0],start[0]+size[0],1):
      if self.root[volume + '_downloaded'][z] == 0:
        data =  common.download(f"https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_masked/20230205180739/{z:05d}.tif")
        mask = data == 0
        #data = skimage.restoration.denoise_tv_chambolle(data, weight=.20)
        #data = preprocessing.global_local_contrast_3d(data)
        data = skimage.filters.gaussian(data, sigma=1)
        data = skimage.exposure.equalize_adapthist(data, nbins=16)
        data = data.astype(np.float32)
        data = numbamath.rescale_array(data)
        data *= 255.
        data = data.astype(np.uint8)
        data[mask] = 0


        self.root[volume][z,:,:] =data & 0xf0
        self.root[volume+'_downloaded'][z] = 1
        #download it and write it here


if __name__ == '__main__':
  zvol = ZVol('e:/', create=True)
  zvol.chunk('20230205180739',(1024,1024,1024),(128,128,128))