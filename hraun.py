import os
import io
import numpy as np
import tifffile
from skimage import measure, exposure
from matplotlib import cm
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import gaussian
from skimage.measure import block_reduce
from scipy.ndimage import uniform_filter
from skimage import segmentation, filters
from skimage.filters import unsharp_mask
from glcae import global_local_contrast_3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import snic

def clip(chunk):
    # Flatten the array to 1D for histogram calculation
    flat_arr = chunk.flatten()

    # Calculate the histogram
    hist, bins = np.histogram(flat_arr, bins=256, range=(0, 256))

    # Calculate the cumulative sum of the histogram
    cum_sum = np.cumsum(hist)

    # Calculate the total number of pixels
    total_pixels = flat_arr.shape[0]

    # Calculate the indices corresponding to the middle 95% of values
    lower_idx = np.argmax(cum_sum >= 0.025 * total_pixels)
    upper_idx = np.argmax(cum_sum >= 0.975 * total_pixels)

    # Cap off the bottom and top 2.5% of values
    arr_capped = np.clip(chunk, bins[lower_idx], bins[upper_idx])

    # Rescale the capped array back to the full range of uint8
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
        tiff_filename = f"{slice_index:05d}.tif"
        tiff_path = os.path.join(tiff_directory, tiff_filename)
        tiff_data = tifffile.memmap(tiff_path)
        if tiff_data.dtype != np.uint8:
            raise ValueError("invalid input dtype from tiff files, must be uint8")
        slices_data.append(tiff_data[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]])
    return slices_data



def preprocess(chunk):
    chunk = clip(chunk)
    chunk = chunk.astype(np.float32)
    chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
    #threshold = np.mean(chunk)/1.5
    #mask = chunk < threshold
    #chunk[mask] = 0
    #chunk[~mask] -= threshold
    chunk = global_local_contrast_3d(chunk)
    #chunk = gaussian(chunk,sigma=1)
    chunk = chunk.astype(np.float32)
    chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
    chunk = np.rot90(chunk,k=3)
    return chunk

def process_chunk(tiff_directory, chunk_size, chunk_offset):
    start_slice = chunk_offset[2]
    end_slice = start_slice + chunk_size[2]

    crop_start = (chunk_offset[0], chunk_offset[1])
    crop_end = (chunk_offset[0] + chunk_size[0], chunk_offset[1] + chunk_size[1])
    print("getting tiff memmaps")
    slices_data = load_cropped_tiff_slices(tiff_directory, start_slice, end_slice, crop_start, crop_end)
    print("stacking tiffs")
    combined_chunk = np.stack(slices_data, axis=-1)
    del slices_data
    print("preprocessing")
    combined_chunk = preprocess(combined_chunk)
    print("superpixeling")


    # Set SNIC parameters
    d_seed = 20
    compactness = 50.0
    lowmid = 0.25
    midhig = 0.75

    #combined_chunk *=256.0

    # Call the SNIC function
    print("combined_chunk shape:", combined_chunk.shape)
    print("combined_chunk dtype:", combined_chunk.dtype)
    print("combined_chunk flags:", combined_chunk.flags)

    # Create a contiguous copy of combined_chunk
    contig_chunk = np.ascontiguousarray(combined_chunk, dtype=np.float32)

    print("combined_chunk shape after contiguous copy:", contig_chunk.shape)
    print("combined_chunk dtype after contiguous copy:", contig_chunk.dtype)
    print("combined_chunk flags after contiguous copy:", contig_chunk.flags)

    neigh_overflow, labels, superpixels = snic.snic(contig_chunk, d_seed, compactness, lowmid, midhig)

    print()

    #combined_chunk = do_mask(combined_chunk,labels, superpixels, .5)
    combined_chunk = (combined_chunk - combined_chunk.min()) / (combined_chunk.max() - combined_chunk.min())

    print("marching cubes")
    # verts, faces, normals, values = measure.marching_cubes(superpixel_values, level=.50, allow_degenerate=False)
    verts, faces, normals, values = measure.marching_cubes(combined_chunk, level=.40, allow_degenerate=False)

    print("normalizing values")
    values = (values - values.min()) / (values.max() - values.min())
    values = exposure.equalize_adapthist(values)
    normals = exposure.equalize_adapthist(normals)
    #values = gaussian(values,sigma=1)
    #normals = gaussian(normals)
    #p2, p98 = np.percentile(values, (3, 97))
    #values = exposure.rescale_intensity(values, in_range=(p2, p98))
    print("colorizing")
    colors = cm.get_cmap('viridis')(values)
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


def convert_to_8bit(tiff_directory):
    import os
    import tifffile
    from skimage import exposure
    from skimage.restoration import denoise_tv_chambolle

    def process_tiff_stack(tiff_directory):
        # Get a list of TIFF files in the directory
        tiff_files = [f for f in os.listdir(tiff_directory) if f.endswith('.tif')]

        for tiff_file in tiff_files:
            tiff_path = os.path.join(tiff_directory, tiff_file)

            # Read the 16-bit unsigned TIFF image
            tiff_data = tifffile.imread(tiff_path)
            if tiff_data.dtype == np.uint8:
                continue
            tiff_data //=256
            tiff_data = tiff_data.astype(np.uint8)
            tifffile.imwrite(tiff_path, tiff_data)

            print(f"Processed: {tiff_file}")

    # Process the TIFF stack
    process_tiff_stack(tiff_directory)

def do_sobel(path):

    # Open the TIFF image
    image = Image.open(path)

    # Convert the image to a numpy array
    image_array = np.array(image)

    #vals, counts = np.unique(image_array, return_counts=True, )
    #most_frequent = sorted(zip(vals, counts), key=lambda x: x[1], reverse=True)[0]
    #threshold = most_frequent[0]
    mask = image_array < 128
    image_array[mask] = 0
    mask = image_array >= 128
    image_array[mask] -= 128
    image_array *=2

    # Apply the Sobel filter
    image_array = gaussian(image_array, sigma=3)
    sobel_x = ndimage.sobel(image_array, axis=0, mode='constant')
    sobel_y = ndimage.sobel(image_array, axis=1, mode='constant')
    sobel = np.hypot(sobel_x, sobel_y)
    sobel = unsharp_mask(sobel, radius=2, amount=1)
    sobel = unsharp_mask(sobel, radius=10, amount=1)


    # Display the original and filtered images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(image, cmap='viridis')
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(sobel, cmap='viridis')
    ax2.set_title('Sobel Filtered Image')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def do_clip(path):
    import numpy as np
    import matplotlib.pyplot as plt

    # Open the TIFF image
    image = Image.open(path)

    # Convert the image to a numpy array
    arr = np.array(image)

    # Flatten the array to 1D for histogram calculation
    flat_arr = arr.flatten()

    # Calculate the histogram
    hist, bins = np.histogram(flat_arr, bins=256, range=(0, 256))

    # Calculate the cumulative sum of the histogram
    cum_sum = np.cumsum(hist)

    # Calculate the total number of pixels
    total_pixels = flat_arr.shape[0]

    # Calculate the indices corresponding to the middle 95% of values
    lower_idx = np.argmax(cum_sum >= 0.025 * total_pixels)
    upper_idx = np.argmax(cum_sum >= 0.975 * total_pixels)

    # Cap off the bottom and top 2.5% of values
    arr_capped = np.clip(arr, bins[lower_idx], bins[upper_idx])

    # Rescale the capped array back to the full range of uint8
    arr_rescaled = ((arr_capped - bins[lower_idx]) * 255 / (bins[upper_idx] - bins[lower_idx])).astype(np.uint8)

    # Display the original and rescaled images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(arr, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(arr_rescaled, cmap='gray')
    ax2.set_title('Rescaled Image')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    tiff_directory = r"C:\Users\forrest\dev\Hraun\dl.ash2txt.org\full-scrolls\PHerc1667.volpkg\volumes\20231117161658"
    #do_sobel("new-en.png")
    #do_clip("new-en.png")
    #exit(0)
    #tiff_directory = r"C:\Users\forrest\dev\Hraun\dl.ash2txt.org\full-scrolls\Scroll1.volpkg\paths\20230929220926\layers"
    chunk_size = (64, 64, 64)
    chunk_offset = (2000, 2000, 2000)
    pool_size = (1, 1, 1)


    #convert_to_8bit(tiff_directory)

    output_directory = "generated_ply"
    os.makedirs(output_directory, exist_ok=True)

    process_chunk(tiff_directory, chunk_size, chunk_offset)
    print("Processing completed.")