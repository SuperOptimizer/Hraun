import os
import requests
import numpy as np
import tifffile
from skimage import measure, exposure
from matplotlib import cm
from skimage.restoration import denoise_tv_chambolle
from skimage.measure import block_reduce
import io
from skimage.filters import gaussian
from skimage.filters import apply_hysteresis_threshold, meijering, gaussian, unsharp_mask, butterworth
import plotly.graph_objects as go

# Get the authentication credentials from environment variables
username = 'registeredusers'
password = 'only'

# Base URL for downloading the TIFF files
base_url = "http://dl.ash2txt.org/full-scrolls/PHerc1667.volpkg/volume_grids/20231107190228/"

# Define the starting index for the cube of chunks
start_idx = (8, 8, 20)

# Create a directory to store the generated PLY files
output_directory = "generated_ply"
os.makedirs(output_directory, exist_ok=True)

# Create a directory to cache the downloaded TIFF files
cache_directory = "tiff_cache"
os.makedirs(cache_directory, exist_ok=True)

# Initialize an empty array to store the combined chunks
combined_chunk = np.zeros((250, 250, 250), dtype=np.uint8)

# Iterate over the cube of chunks
for k in range(2):
    for i in range(2):
        for j in range(2):
            idx = (start_idx[0] + i, start_idx[1] + j, start_idx[2] + k)
            tiff_filename = f"cell_yxz_{idx[0]:03d}_{idx[1]:03d}_{idx[2]:03d}.tif"
            tiff_url = base_url + tiff_filename
            cache_path = os.path.join(cache_directory, tiff_filename)

            if os.path.exists(cache_path):
                # Read the TIFF file from the cache
                tiff_data = tifffile.imread(cache_path)
            else:
                response = requests.get(tiff_url, auth=(username, password))
                if response.status_code == 200:
                    tiff_data = tifffile.imread(io.BytesIO(response.content))
                    tifffile.imwrite(cache_path, tiff_data)  # Save the TIFF file to the cache
                else:
                    print(f"Failed to download TIFF file: {tiff_filename}")
                    exit(1)

            chunk = tiff_data // 256
            chunk = block_reduce(chunk, block_size=(4, 4, 4), func=np.sum)
            chunk //= 64
            chunk = chunk.astype(np.uint8)

            # Combine the pooled chunk into the combined_chunk array
            combined_chunk[i*125:(i+1)*125, j*125:(j+1)*125, k*125:(k+1)*125] = chunk

# Perform preprocessing on the combined chunk
combined_chunk = (exposure.equalize_adapthist(combined_chunk) * 255).astype(np.uint8)
combined_chunk = (denoise_tv_chambolle(combined_chunk) * 255).astype(np.uint8)

# Perform marching cubes on the combined chunk
verts, faces, normals, values = measure.marching_cubes(combined_chunk, level=100, allow_degenerate=False)

# Normalize the values from 0 to 1
values_normalized = (values - values.min()) / (values.max() - values.min())
values_normalized = exposure.equalize_adapthist(values_normalized)

# Map the normalized values to colors using the colormap
colors = cm.get_cmap('viridis')(values_normalized)
colors = (colors * 255).astype(np.uint8)

# Generate the PLY filename
ply_filename = f"combined_chunk_{start_idx[0]}_{start_idx[1]}_{start_idx[2]}.ply"
ply_path = os.path.join(output_directory, ply_filename)

# Open the PLY file for writing
with open(ply_path, 'w') as ply_file:
    # Write the PLY header
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

    # Write the vertex data
    for vert, color in zip(verts, colors):
        ply_file.write(f"{vert[0]} {vert[1]} {vert[2]} {color[0]} {color[1]} {color[2]}\n")

    # Write the face data
    for face in faces:
        ply_file.write(f"3 {face[0]} {face[1]} {face[2]}\n")

print("Processing completed.")