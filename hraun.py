import os
from skimage import measure, exposure
from skimage.filters import gaussian
from skimage.measure import block_reduce
import numpy as np
import matplotlib

import snic

import volman
import preprocessing

def quantize(chunk, level):
    if level == 7:
        chunk &= 0xfe
    if level == 6:
        chunk &= 0xfc
    if level == 5:
        chunk &= 0xf8
    if level == 4:
        chunk &= 0xf0
    if level == 3:
        chunk &= 0xe0
    if level == 2:
        chunk &= 0xc0
    if level == 1:
        chunk &= 0x80
    return chunk


def superpixel(chunk):
    contig_chunk = np.ascontiguousarray(chunk, dtype=np.float32)
    neigh_overflow, labels, superpixels = snic.snic(contig_chunk, d_seed=5, compactness=10.0, lowmid=.25, midhig=.75)

    print("superpixel masking")

    #combined_chunk = do_mask(chunk,labels, superpixels, .2)
    #combined_chunk = (combined_chunk - combined_chunk.min()) / (combined_chunk.max() - combined_chunk.min())
    return chunk

def preprocess(chunk, downsample):
    chunk = preprocessing.clip(chunk)
    chunk = chunk.astype(np.float32)
    chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
    chunk = block_reduce(chunk, (downsample,downsample,downsample), np.mean)
    chunk = preprocessing.global_local_contrast_3d(chunk)
    chunk = chunk.astype(np.float32)
    #chunk = preprocessing.avg_pool_3d(chunk, (2,2,2), 1)
    chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
    chunk = np.rot90(chunk,k=3)
    return chunk

def colorize(verts, faces, normals, values, colormap):
    colors = matplotlib.cm.get_cmap(colormap)(values)
    colors = (colors * 256).astype(np.uint8)
    return colors

def postprocess(verts, faces, normals, values):
    values = (values - values.min()) / (values.max() - values.min())
    values = exposure.equalize_adapthist(values)
    normals = exposure.equalize_adapthist(normals)
    values = gaussian(values,sigma=1)
    normals = gaussian(normals,sigma=1)
    p5, p95 = np.percentile(values, (5, 95))
    values = exposure.rescale_intensity(values, in_range=(p5, p95))
    return verts, faces, normals, values

def writeply(outdir, ply_filename, verts, normals, faces, colors):
    os.makedirs(outdir, exist_ok=True)
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

def main(scroll, source, idnum, chunk_size, chunk_offset, outdir,
         inklabels_path=None,
         preprocess_=True,
         superpixel_=False,
         downsample=1,
         isolevel=0.5,
         postprocess_=True,
         colormap='viridis',
         quantize_=8):

    print("Stacking tiffs")
    vm = volman.VolMan('.')
    yoff,xoff,zoff = chunk_offset
    ysize,xsize,zsize = chunk_size
    chunk = vm.chunk(scroll,source,idnum, yoff, xoff, zoff, ysize, xsize, zsize)

    if quantize_ < 8:
        print("quantizing")
        chunk = quantize(chunk, quantize_)

    if inklabels_path:
        print("applying ink labels")
        chunk = preprocessing.project_mask_to_volume("20230929220926_inklabels.png", chunk, (yoff,xoff), 50)

    if preprocess_:
        print("preprocessing")
        chunk = preprocess(chunk, downsample)

    if superpixel_:
        print("WARNING: superpixeling is not yet supported. skipping...")
        #superpixel(chunk)

    print("marching cubes")
    if chunk.dtype != np.float32:
        chunk = chunk.astype(np.float32)
        chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
    verts, faces, normals, values = measure.marching_cubes(chunk, level=isolevel, allow_degenerate=False)

    if postprocess_:
        print("postprocessing")
        verts, faces, normals, values = postprocess(verts, faces, normals, values)
    print("colorizing")
    colors = colorize(verts, faces, normals, values, colormap)

    print("writing to file")
    ply_filename = f"{scroll}_{source}_{idnum}_off_{yoff}x{xoff}x{zoff}_size_{ysize}x{xsize}x{zsize}_downsample_{downsample}_q{quantize_}.ply"
    writeply(outdir, ply_filename, verts, normals, faces, colors)


if __name__ == '__main__':
    chunk_size = (500, 500, 65)
    chunk_offset = (4500, 1500, 0)
    outdir = "generated_ply"
    #main('Scroll1', 'segments', '20230929220926', chunk_size, chunk_offset, outdir, quantize_=3,downsample=2)


    chunk_size = (500, 500, 500)
    chunk_offset = (1500, 1500, 1500)
    outdir = "generated_ply"
    #main('PHerc1667', 'volumes', '20231117161658', chunk_size, chunk_offset, outdir, quantize_=8,downsample=2)
    #main('PHerc1667', 'volumes', '20231117161658', chunk_size, chunk_offset, outdir, quantize_=4,downsample=2)
    main('PHerc1667', 'volumes', '20231117161658', chunk_size, chunk_offset, outdir, quantize_=3,downsample=2)
