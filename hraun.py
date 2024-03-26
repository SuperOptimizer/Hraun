import os
import numpy as np
from skimage import io, measure, filters, exposure, util
import plotly.graph_objects as go
import requests
import matplotlib.pyplot as plt

base_url = 'http://dl.ash2txt.org/full-scrolls/PHerc0332.volpkg/volume_grids/20231027191953/'

local_directory = 'PHerc0332'

os.makedirs(local_directory, exist_ok=True)

x_index = 9
y_index = 10
z_index = 21

filename = f"cell_yxz_{x_index:03d}_{y_index:03d}_{z_index:03d}.tif"
url = base_url + filename
local_filename = os.path.join(local_directory, filename)

username = ''
password = ''

if not os.path.exists(local_filename):
    response = requests.get(url, auth=(username, password))

    with open(local_filename, 'wb') as file:
        file.write(response.content)

volume = io.imread(local_filename)
volume = volume[100:300,100:300,100:300]
volume //=256

volume = volume.astype(np.uint8)

volume >>= 4
volume <<= 4

volume = filters.gaussian(volume, sigma=1)
volume = filters.unsharp_mask(volume, radius=4, amount=1)
p10, p90 = np.percentile(volume, (10, 90))
volume = exposure.rescale_intensity(volume, in_range=(p10, p90))

volume *=256
volume = volume.astype(np.uint8)
verts, faces, normals, values = measure.marching_cubes(volume, level=128)

normalized_values = ((values - values.min()) / (values.max() - values.min()))

colormap = plt.cm.viridis(normalized_values)
colormap = colormap.astype(np.float32)
vertex_colors = np.zeros((colormap.shape[0], 4),dtype=np.float32)
vertex_colors[:, :3] = colormap[:, :3]
vertex_colors[:, 3] = 1.0

chunk_trace = go.Mesh3d(
    x=verts[:, 0],
    y=verts[:, 1],
    z=verts[:, 2],
    i=faces[:, 0],
    j=faces[:, 1],
    k=faces[:, 2],
    vertexcolor=vertex_colors,

)

chunk_layout = go.Layout(
    title=f'Marching Cubes Mesh of Chunk ({x_index}, {y_index}, {z_index})',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
)

chunk_fig = go.Figure(data=[chunk_trace], layout=chunk_layout)

chunk_fig.show()