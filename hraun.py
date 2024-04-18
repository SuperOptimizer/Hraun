import os
import io
import numpy as np
import tifffile
from skimage import measure, exposure
from matplotlib import cm
from skimage.restoration import denoise_tv_chambolle
from skimage.measure import block_reduce
from scipy.ndimage import uniform_filter

def avg_pool_3d(arr, pool_size):
    return uniform_filter(arr, size=pool_size, mode='nearest')

def do_2d_slices():
    def load_cropped_tiff_slices(tiff_directory, start_slice, end_slice, crop_start, crop_end):
        slices_data = []
        for slice_index in range(start_slice, end_slice):
            tiff_filename = f"{slice_index:02d}.tif"
            print(f"loading {slice_index}")
            tiff_path = os.path.join(tiff_directory, tiff_filename)
            tiff_data = tifffile.memmap(tiff_path)
            slices_data.append(tiff_data[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]])
        return np.stack(slices_data, axis=-1)

    def process_chunk(tiff_directory, chunk_size, chunk_offset, pool_size):
        combined_chunk_size = chunk_size#(size // pool for size, pool in zip(chunk_size, pool_size))
        combined_chunk = np.zeros(combined_chunk_size, dtype=np.uint8)

        start_slice = chunk_offset[2]
        end_slice = start_slice + chunk_size[2]

        crop_start = (chunk_offset[0], chunk_offset[1])
        crop_end = (chunk_offset[0] + chunk_size[0], chunk_offset[1] + chunk_size[1])

        for z in range(start_slice, end_slice, pool_size[2]):
            slices_data = load_cropped_tiff_slices(tiff_directory, z, min(z + pool_size[2], end_slice), crop_start,crop_end)
            slices_data = slices_data.astype(np.float32)
            slices_data = (slices_data - slices_data.min()) / (slices_data.max() - slices_data.min())
            slices_data = (slices_data*256).astype(np.uint8)
            chunk_z = (z - start_slice) // pool_size[2]
            combined_chunk[:, :, chunk_z] = slices_data[:, :, 0]

        print("preprocessing")
        combined_chunk = combined_chunk.astype(np.float32)
        combined_chunk = (combined_chunk - combined_chunk.min()) / (combined_chunk.max() - combined_chunk.min())
        combined_chunk = exposure.equalize_adapthist(combined_chunk)
        combined_chunk = denoise_tv_chambolle(combined_chunk)
        combined_chunk = avg_pool_3d(combined_chunk,(8,8,8))
        print("marching cubes")
        verts, faces, normals, values = measure.marching_cubes(combined_chunk, level=.5, allow_degenerate=False)
        print("normalizing values")
        values = (values - values.min()) / (values.max() - values.min())
        values = exposure.equalize_adapthist(values)
        #p2, p98 = np.percentile(values, (5, 95))
        #values = exposure.rescale_intensity(values, in_range=(p2, p98))

        print("colorizing")
        colors = cm.get_cmap('viridis')(values)
        colors = (colors*256).astype(np.uint8)

        print("writing to file")
        ply_filename = f"chunk_{chunk_offset[0]}_{chunk_offset[1]}_{chunk_offset[2]}_pool_{pool_size[0]}_{pool_size[1]}_{pool_size[2]}.ply"
        ply_path = os.path.join(output_directory, ply_filename)
        num_verts = len(verts)
        num_faces = len(faces)
        with open(ply_path, 'w') as ply_file:
            ply_file.write("ply\n")
            ply_file.write("format ascii 1.0\n")
            ply_file.write(f"element vertex {num_verts}\n")
            ply_file.write("property float x\n")
            ply_file.write("property float y\n")
            ply_file.write("property float z\n")
            ply_file.write("property uchar red\n")
            ply_file.write("property uchar green\n")
            ply_file.write("property uchar blue\n")
            ply_file.write(f"element face {num_faces}\n")
            ply_file.write("property list uchar int vertex_index\n")
            ply_file.write("end_header\n")

            chunk_size = 1000000
            for i in range(0, num_verts, chunk_size):
                chunk_verts = verts[i:i + chunk_size]
                chunk_colors = colors[i:i + chunk_size]
                vert_lines = [f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n" for v, c in zip(chunk_verts, chunk_colors)]
                ply_file.write(''.join(vert_lines))

            for i in range(0, num_faces, chunk_size):
                chunk_faces = faces[i:i + chunk_size]
                face_lines = [f"3 {f[0]} {f[1]} {f[2]}\n" for f in chunk_faces]
                ply_file.write(''.join(face_lines))

    tiff_directory = r"C:\Users\forrest\dev\Hraun\dl.ash2txt.org\full-scrolls\PHerc1667.volpkg\paths\20240304144030\layers"
    chunk_size = (1000, 1000, 64)
    chunk_offset = (2000, 2000, 0)
    pool_size = (1, 1, 1)

    output_directory = "generated_ply"
    os.makedirs(output_directory, exist_ok=True)

    process_chunk(tiff_directory, chunk_size, chunk_offset, pool_size)
    print("Processing completed.")

def convert_to_8bit():
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

    # Specify the directory containing the TIFF stack
    tiff_directory = r'C:\Users\forrest\dev\Hraun\dl.ash2txt.org\full-scrolls\PHerc1667.volpkg\volumes\20231117161658'

    # Process the TIFF stack
    process_tiff_stack(tiff_directory)

#convert_to_8bit()

do_2d_slices()