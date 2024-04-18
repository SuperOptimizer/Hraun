import os
import requests
import numpy as np
import skimage.filters
import tifffile
from skimage import measure, exposure
from matplotlib import cm
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import gaussian
from skimage.measure import block_reduce
import io

def do_3d_chunks():

    username = os.environ['SCROLLPRIZE_USER']
    password = os.environ['SCROLLPRIZE_PASS']

    base_url = "http://dl.ash2txt.org/full-scrolls/PHerc1667.volpkg/volume_grids/20231107190228/"

    start_idx = (7, 7, 19)

    output_directory = "generated_ply"
    os.makedirs(output_directory, exist_ok=True)
    cache_directory = "tiff_cache"
    os.makedirs(cache_directory, exist_ok=True)

    combined_chunk = np.zeros((500, 500, 500), dtype=np.uint8)

    for i in range(4):
        for j in range(4):
            for k in range(4):
                idx = (start_idx[0] + j, start_idx[1] + i, start_idx[2] + k)
                tiff_filename = f"cell_yxz_{idx[0]:03d}_{idx[1]:03d}_{idx[2]:03d}.tif"
                tiff_url = base_url + tiff_filename
                cache_path = os.path.join(cache_directory, tiff_filename)

                if os.path.exists(cache_path):
                    tiff_data = tifffile.imread(cache_path)
                else:
                    print(f"downloading {tiff_url}")
                    response = requests.get(tiff_url, auth=(username, password))
                    if response.status_code == 200:
                        tiff_data = tifffile.imread(io.BytesIO(response.content))
                        tifffile.imwrite(cache_path, tiff_data)
                    else:
                        print(f"Failed to download TIFF file: {tiff_filename}")
                        exit(1)

                chunk = tiff_data // 256
                chunk = block_reduce(chunk, block_size=(4, 4, 4), func=np.sum)
                chunk //= 64
                chunk = chunk.astype(np.uint8)
                combined_chunk[i*125:(i+1)*125, j*125:(j+1)*125, k*125:(k+1)*125] = chunk

    combined_chunk = (exposure.equalize_adapthist(combined_chunk) * 255).astype(np.uint8)
    combined_chunk = (denoise_tv_chambolle(combined_chunk) * 255).astype(np.uint8)

    verts, faces, normals, values = measure.marching_cubes(combined_chunk, level=128, allow_degenerate=False)

    values_normalized = (values - values.min()) / (values.max() - values.min())
    values_normalized = exposure.equalize_adapthist(values_normalized)

    colors = cm.get_cmap('viridis')(values_normalized)
    colors = (colors * 255).astype(np.uint8)

    ply_filename = f"combined_chunk_{start_idx[0]}_{start_idx[1]}_{start_idx[2]}.ply"
    ply_path = os.path.join(output_directory, ply_filename)

    with open(ply_path, 'w') as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(verts)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write(f"element face {len(faces)}\n")
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")

        for vert, color in zip(verts, colors):
            ply_file.write(f"{vert[0]} {vert[1]} {vert[2]} {color[0]} {color[1]} {color[2]}\n")

        for face in faces:
            ply_file.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print("Processing completed.")

def do_2d_slices():
    import os
    import numpy as np
    import tifffile
    from skimage import measure, exposure
    from matplotlib import cm
    from skimage.restoration import denoise_tv_chambolle
    from skimage.measure import block_reduce

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
        combined_chunk_size = tuple(size // pool for size, pool in zip(chunk_size, pool_size))
        combined_chunk = np.zeros(combined_chunk_size, dtype=np.uint8)

        start_slice = chunk_offset[2]
        end_slice = start_slice + chunk_size[2]

        crop_start = (chunk_offset[0], chunk_offset[1])
        crop_end = (chunk_offset[0] + chunk_size[0], chunk_offset[1] + chunk_size[1])

        for z in range(start_slice, end_slice, pool_size[2]):
            slices_data = load_cropped_tiff_slices(tiff_directory, z, min(z + pool_size[2], end_slice), crop_start,crop_end)
            slices_data = slices_data.astype(np.uint32)

            slices_data = block_reduce(slices_data, block_size=pool_size, func=np.sum)
            slices_data //= np.prod(pool_size)
            slices_data = slices_data.astype(np.float32)
            slices_data = (slices_data - slices_data.min()) / (slices_data.max() - slices_data.min())
            #slices_data = gaussian(slices_data)
            slices_data = exposure.equalize_adapthist(slices_data)
            p2, p98 = np.percentile(slices_data, (5, 95))
            slices_data = exposure.rescale_intensity(slices_data, in_range=(p2, p98))
            slices_data = (slices_data*256).astype(np.uint8)

            chunk_z = (z - start_slice) // pool_size[2]
            combined_chunk[:, :, chunk_z] = slices_data[:, :, 0]

        print("marching cubes")
        combined_chunk = combined_chunk.astype(np.float32)
        combined_chunk = (combined_chunk - combined_chunk.min()) / (combined_chunk.max() - combined_chunk.min())
        verts, faces, normals, values = measure.marching_cubes(combined_chunk, level=.5, allow_degenerate=False)
        print("normalizing values")
        values = (values - values.min()) / (values.max() - values.min())
        values = exposure.equalize_adapthist(values)
        p2, p98 = np.percentile(values, (5, 95))
        values = exposure.rescale_intensity(values, in_range=(p2, p98))
        values = gaussian(values,sigma=1)

        #low,high = (.55,.65)

        #mask = values < low
        #values[mask] = .02
        #mask = values > high
        #values[mask] = .02
        #mask = values > .1
        #values[mask] = .98

        print("colorizing")
        colors = cm.get_cmap('viridis')(values)
        #mask = colors < low
        #colors[mask] = .02
        #mask = colors > high
        #colors[mask] = .02
        #mask = colors >= .1
        #colors[mask] = .98
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

            # Write vertices in chunks
            chunk_size = 1000000
            for i in range(0, num_verts, chunk_size):
                chunk_verts = verts[i:i + chunk_size]
                chunk_colors = colors[i:i + chunk_size]
                vert_lines = [f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n" for v, c in zip(chunk_verts, chunk_colors)]
                ply_file.write(''.join(vert_lines))

            # Write faces in chunks
            for i in range(0, num_faces, chunk_size):
                chunk_faces = faces[i:i + chunk_size]
                face_lines = [f"3 {f[0]} {f[1]} {f[2]}\n" for f in chunk_faces]
                ply_file.write(''.join(face_lines))

    tiff_directory = r"C:\Users\forrest\dev\Hraun\dl.ash2txt.org\full-scrolls\PHerc1667.volpkg\paths\20240304141530\layers"
    chunk_size = (12000, 12000, 64)
    chunk_offset = (000, 00, 0)
    pool_size = (8, 8, 8)

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