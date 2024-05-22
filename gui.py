import tifffile
import numpy as np

from PIL import Image

from volman import VolMan

vol = VolMan("D:/vesuvius.volman")
chunk = vol.chunk('PHerc1667','volumes','20231107190228',[0,1000,1000],[10000,4,4])
image = chunk[0,:,:]
# Save the compressed image as a new TIFF file
tifffile.imwrite('compressed_image.tiff', image, compression='lzw')
