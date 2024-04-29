import os
import tifffile
from skimage import measure, exposure
from skimage.filters import gaussian
from skimage.measure import block_reduce
from skimage.filters import unsharp_mask

import volman
import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from scipy import ndimage
import cv2
import math
import snic






def preprocess(chunk, downsample):
    chunk = preprocessing.clip(chunk)
    chunk = chunk.astype(np.float32)
    chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
    chunk = block_reduce(chunk,(downsample,downsample,downsample),np.mean)
    chunk = preprocessing.global_local_contrast_3d(chunk)
    chunk = chunk.astype(np.float32)
    chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
    chunk = np.rot90(chunk,k=3)
    return chunk

def colorize(verts, faces, normals, values, colormap='viridis'):
    colors = matplotlib.cm.get_cmap(colormap)(values)
    colors = (colors * 256).astype(np.uint8)
    return colors

def postprocess(verts, faces, normals, values):
    values = (values - values.min()) / (values.max() - values.min())
    #values = exposure.equalize_adapthist(values)
    #normals = exposure.equalize_adapthist(normals)
    #values = gaussian(values,sigma=1)
    #normals = gaussian(normals,sigma=1)
    #p2, p98 = np.percentile(values, (3, 97))
    #values = exposure.rescale_intensity(values, in_range=(p2, p98))
    return verts, faces, normals, values

def main(scroll, source, idnum, chunk_size, chunk_offset, outdir,
         inklabels_path=None,
         preprocess_=True,
         superpixel_=False,
         downsample=1,
         isolevel=0.5,
         postprocess_=True,
         colormap='viridis'):
    print("Stacking tiffs")
    vm = volman.VolMan('.')
    yoff,xoff,zoff = chunk_offset
    ysize,xsize,zsize = chunk_size
    chunk = vm.chunk(scroll,source,idnum, yoff, xoff, zoff, ysize, xsize, zsize)

    if inklabels_path:
        print("applying ink labels")
        chunk = preprocessing.project_mask_to_volume("20230929220926_inklabels.png", chunk, (yoff,xoff), 0)

    if preprocess_:
        print("preprocessing")
        chunk = preprocess(chunk, downsample)

    if superpixel_:
        print("superpixeling")

        # Create a contiguous copy of combined_chunk
        contig_chunk = np.ascontiguousarray(chunk, dtype=np.float32)
        contig_chunk *=256
        contig_chunk = contig_chunk.astype(np.uint8)
        contig_chunk &= 0x80
        contig_chunk = contig_chunk.astype(np.float32)
        contig_chunk /=256.0
        neigh_overflow, labels, superpixels = snic.snic(contig_chunk, d_seed=5, compactness=10.0, lowmid=.25, midhig=.75)

        print("superpixel masking")

        #combined_chunk = do_mask(combined_chunk,labels, superpixels, .2)
        #combined_chunk = (combined_chunk - combined_chunk.min()) / (combined_chunk.max() - combined_chunk.min())

    print("marching cubes")
    verts, faces, normals, values = measure.marching_cubes(chunk, level=isolevel, allow_degenerate=False)

    if postprocess_:
        print("postprocessing")
        verts, faces, normals, values = postprocess(verts, faces, normals, values)
    print("colorizing")
    colors = colorize(verts, faces, normals, values, colormap)

    print("writing to file")
    ply_filename = f"chunk_{chunk_offset[0]}_{chunk_offset[1]}_{chunk_offset[2]}_downsample_{downsample}.ply"
    ply_path = os.path.join(outdir, ply_filename)
    verts = verts.astype(np.float16)
    normals = normals.astype(np.float16)
    num_verts = len(verts)
    num_faces = len(faces)

    with open(ply_path, 'wt', buffering=1024 * 1024 * 128) as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {num_verts}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write(f"element face {num_faces}\n")
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")

        vert_lines = [f"{v[0]} {v[1]} {v[2]} {n[0]} {n[1]} {n[2]} {c[0]} {c[1]} {c[2]}\n" for v, n, c in
                      zip(verts, normals, colors)]
        ply_file.writelines(vert_lines)

        face_lines = [f"3 {f[0]} {f[1]} {f[2]}\n" for f in faces]
        ply_file.writelines(face_lines)


if __name__ == '__main__':
    chunk_size = (500, 500, 65)
    chunk_offset = (4500, 1500, 0)
    outdir = "generated_ply"

    main('Scroll1', 'segments', '20230929220926', chunk_size, chunk_offset, outdir,colormap='gray')
