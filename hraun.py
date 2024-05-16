import volman
import preprocessing

import pyvista as pv
from volman import the_index



def main(scroll, source, idnum, chunk_size, chunk_offset, outdir,
         inklabels_path=None,
         preprocess_=True,
         superpixel_=False,
         downsample=1,
         isolevel=128,
         postprocess_=True,
         colormap='viridis',
         quantize_=8):

    print("Stacking tiffs")
    vm = volman.VolMan('.')
    yoff, xoff, zoff = chunk_offset
    ysize, xsize, zsize = chunk_size
    chunk = vm.chunk(scroll, source, idnum, yoff, xoff, zoff, ysize, xsize, zsize)

    if inklabels_path:
        print("applying ink labels")
        chunk = preprocessing.project_mask_to_volume("20230929220926_inklabels.png", chunk, (yoff, xoff), 50)


    print("done visualizing")

