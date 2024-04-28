import os
import tifffile
from skimage import measure, exposure
from skimage.filters import gaussian
from skimage.measure import block_reduce
from skimage.filters import unsharp_mask
from glcae import global_local_contrast_3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from scipy import ndimage
import snic

def clip(chunk):
    flat_arr = chunk.flatten()
    hist, bins = np.histogram(flat_arr, bins=256, range=(0, 256))
    cum_sum = np.cumsum(hist)
    total_pixels = flat_arr.shape[0]
    lower_idx = np.argmax(cum_sum >= 0.025 * total_pixels)
    upper_idx = np.argmax(cum_sum >= 0.975 * total_pixels)
    arr_capped = np.clip(chunk, bins[lower_idx], bins[upper_idx])
    arr_rescaled = ((arr_capped - bins[lower_idx]) * 255 / (bins[upper_idx] - bins[lower_idx])).astype(np.uint8)
    return arr_rescaled

def rescale(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def do_mask(combined_chunk, labels, superpixels, threshold):
    mask = np.zeros_like(combined_chunk, dtype=bool)

    for i, superpixel in enumerate(superpixels):
        if i == 0:
            continue  # Skip the background superpixel (label 0)

        if superpixel.c >= threshold:
            mask |= (labels == i)

    masked_chunk = np.zeros_like(combined_chunk)
    masked_chunk[mask] = combined_chunk[mask]
    return masked_chunk


def avg_pool_3d(input, pool_size, stride):
    input_depth, input_height, input_width = input.shape

    output = np.zeros_like(input)

    for d in range(0, input_depth, stride):
        for i in range(0, input_height, stride):
            for j in range(0, input_width, stride):
                start_d = d
                start_i = i
                start_j = j
                end_d = min(start_d + pool_size[0], input_depth)
                end_i = min(start_i + pool_size[1], input_height)
                end_j = min(start_j + pool_size[2], input_width)

                pool_region = input[start_d:end_d, start_i:end_i, start_j:end_j]
                output[start_d:end_d, start_i:end_i, start_j:end_j] = np.mean(pool_region)

    return output

def load_cropped_tiff_slices(tiff_directory, start_slice, end_slice, crop_start, crop_end):
    slices_data = []
    for slice_index in range(start_slice, end_slice):
        tiff_filename = f"{slice_index:02d}.tif"
        tiff_path = os.path.join(tiff_directory, tiff_filename)
        tiff_data = tifffile.memmap(tiff_path)
        if tiff_data.dtype != np.uint8:
            raise ValueError("invalid input dtype from tiff files, must be uint8")
        slices_data.append(tiff_data[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]])
    return slices_data


def preprocess(chunk, pool_size):
    chunk = clip(chunk)
    chunk = chunk.astype(np.float32)
    chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
    chunk = block_reduce(chunk,pool_size,np.mean)
    chunk = global_local_contrast_3d(chunk)
    chunk = chunk.astype(np.float32)
    chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
    chunk = np.rot90(chunk,k=3)
    return chunk

def process_chunk(tiff_directory, chunk_size, chunk_offset, pool_size):
    start_slice = chunk_offset[2]
    end_slice = start_slice + chunk_size[2]

    crop_start = (chunk_offset[0], chunk_offset[1])
    crop_end = (chunk_offset[0] + chunk_size[0], chunk_offset[1] + chunk_size[1])
    print("getting tiff memmaps")
    slices_data = load_cropped_tiff_slices(tiff_directory, start_slice, end_slice, crop_start, crop_end)
    print("stacking tiffs")
    combined_chunk = np.stack(slices_data, axis=-1)
    print("applying ink labels")
    combined_chunk = project_mask_to_volume("20230929220926_inklabels.png", combined_chunk, crop_start, 50)

    print("preprocessing")
    combined_chunk = preprocess(combined_chunk, pool_size)
    print("superpixeling")

    # Set SNIC parameters
    d_seed = 15
    compactness = 1.0
    lowmid = 0.5
    midhig = 0.75

    # Create a contiguous copy of combined_chunk
    #contig_chunk = np.ascontiguousarray(combined_chunk, dtype=np.float32)
    #contig_chunk *=256
    #contig_chunk = contig_chunk.astype(np.uint8)
    #contig_chunk *= 0xfe
    #contig_chunk = contig_chunk.astype(np.float32)
    #contig_chunk /=256.0
    #neigh_overflow, labels, superpixels = snic.snic(contig_chunk, d_seed, compactness, lowmid, midhig)

    print("masking")

    #combined_chunk = do_mask(combined_chunk,labels, superpixels, .2)
    #combined_chunk = (combined_chunk - combined_chunk.min()) / (combined_chunk.max() - combined_chunk.min())

    print("marching cubes")
    verts, faces, normals, values = measure.marching_cubes(combined_chunk, level=.5, allow_degenerate=False)

    print("normalizing values")
    values = (values - values.min()) / (values.max() - values.min())
    values = exposure.equalize_adapthist(values)
    normals = exposure.equalize_adapthist(normals)
    #values = gaussian(values,sigma=1)
    #normals = gaussian(normals,sigma=1)
    #p2, p98 = np.percentile(values, (3, 97))
    #values = exposure.rescale_intensity(values, in_range=(p2, p98))
    print("colorizing")
    colors = matplotlib.cm.get_cmap('viridis')(values)
    colors = (colors*256).astype(np.uint8)



    print("writing to file")
    ply_filename = f"chunk_{chunk_offset[0]}_{chunk_offset[1]}_{chunk_offset[2]}_pool_{pool_size[0]}_{pool_size[1]}_{pool_size[2]}.ply"
    ply_path = os.path.join(output_directory, ply_filename)
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

        #vert_lines = [f"{v[0]} {v[1]} {v[2]}  {c[0]} {c[1]} {c[2]}\n" for v,  c in
        #              zip(verts,  colors)]
        #ply_file.writelines(vert_lines)

        face_lines = [f"3 {f[0]} {f[1]} {f[2]}\n" for f in faces]
        ply_file.writelines(face_lines)


def project_mask_to_volume(mask_path, voxel_volume, crop_start, brightness_adjust):
    Image.MAX_IMAGE_PIXELS = None

    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask) > 0

    cropped_mask = mask[crop_start[0]:crop_start[0]+voxel_volume.shape[0],
                         crop_start[1]:crop_start[1]+voxel_volume.shape[1]]

    mask_3d = np.repeat(cropped_mask[:, :, np.newaxis], voxel_volume.shape[2], axis=2)

    modified_volume = np.copy(voxel_volume)

    modified_volume[mask_3d] = np.minimum(modified_volume[mask_3d] + brightness_adjust, 255)

    return modified_volume

def convert_to_8bit(tiff_directory):
    def process_tiff_stack(tiff_directory):
        tiff_files = [f for f in os.listdir(tiff_directory) if f.endswith('.tif')]

        for tiff_file in tiff_files:
            tiff_path = os.path.join(tiff_directory, tiff_file)
            tiff_data = tifffile.imread(tiff_path)
            if tiff_data.dtype == np.uint8:
                continue
            tiff_data //=256
            tiff_data = tiff_data.astype(np.uint8)
            tifffile.imwrite(tiff_path, tiff_data)

            print(f"Processed: {tiff_file}")
    process_tiff_stack(tiff_directory)

if __name__ == '__main__':
    #tiff_directory = r"C:\Users\forrest\dev\Hraun\dl.ash2txt.org\full-scrolls\PHerc1667.volpkg\volumes\20231117161658"
    #do_sobel("new-en.png")
    #do_clip("new-en.png")
    #exit(0)
    tiff_directory = r"C:\Users\forrest\dev\Hraun\dl.ash2txt.org\full-scrolls\Scroll1.volpkg\paths\20230929220926\layers"
    chunk_size = (1000, 1000, 65)
    chunk_offset = (1200, 4500, 0)
    pool_size = (1, 1, 1)


    #convert_to_8bit(tiff_directory)

    output_directory = "generated_ply"
    os.makedirs(output_directory, exist_ok=True)

    process_chunk(tiff_directory, chunk_size, chunk_offset, pool_size)
    print("Processing completed.")