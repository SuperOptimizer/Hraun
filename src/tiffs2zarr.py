import gc
import os
import numpy as np
import tifffile
import zarr
import numcodecs
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

mmaps = []

def process_chunk(zarr_array, chunk_info, input_folder, tiff_files):
    global mmaps
    start_z, end_z, start_y, end_y, start_x, end_x = chunk_info
    chunk = np.zeros((min(256, end_z - start_z), min(256, end_y - start_y), min(256, end_x - start_x)), dtype=np.uint8)

    for z in range(start_z, end_z):
        if z >= len(tiff_files):
            break

        chunk_z = z - start_z
        slice = mmaps[z][start_y:end_y, start_x:end_x]
        if not np.all(slice == 0):
            chunk[chunk_z, :end_y - start_y, :end_x - start_x] = slice
    if not np.all(chunk == 0):
        zarr_array[start_z:end_z, start_y:end_y, start_x:end_x] = chunk
    return chunk_info


def create_3d_array_from_tiffs(root, input_folder, output_file, z_size, y_size, x_size):
    global mmaps

    # Get list of TIFF files
    tiff_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.tif') or f.endswith('.tiff')])
    mmaps = [tifffile.memmap(os.path.join(input_folder, tiff_file)) for tiff_file in tiff_files]
    if len(tiff_files) != z_size:
        raise ValueError(f"Number of TIFF files ({len(tiff_files)}) does not match specified z_size ({z_size})")

    # Calculate the number of chunks in each dimension
    chunks_z = math.ceil(z_size / 256)
    chunks_y = math.ceil(y_size / 256)
    chunks_x = math.ceil(x_size / 256)

    # Create a list of chunk information
    chunk_infos = [
        (z * 256, min((z + 1) * 256, z_size),
         y * 256, min((y + 1) * 256, y_size),
         x * 256, min((x + 1) * 256, x_size))
        for z in range(chunks_z)
        for y in range(chunks_y)
        for x in range(chunks_x)
    ]

    # Process chunks in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, root[f'Scroll1/20230205180739'], chunk_info, input_folder, tiff_files) for chunk_info in chunk_infos]
        prev_z = 0
        for future in as_completed(futures):
            chunk_info = future.result()
            start_z, end_z, start_y, end_y, start_x, end_x = chunk_info
            #zarr_array[start_z:end_z, start_y:end_y, start_x:end_x] = chunk_data
            print(f"Processed and wrote chunk: z={start_z}-{end_z}, y={start_y}-{end_y}, x={start_x}-{end_x}")
            if start_z > prev_z:
                print("collecting garbage")
                for i in range(prev_z, start_z):
                    del mmaps[i]
                gc.collect()
                prev_z = start_z



    print(f"3D array created and saved to {output_file}")


def main():
    input_folder = r'D:\vesuvius.volman\Scroll1\volumes\20230205180739'
    output_file = r'c:\vesuvius.volman\20230205180739.zarr'
    z_size = 14376
    y_size = 7888
    x_size = 8096

    create_3d_array_from_tiffs(input_folder, output_file, z_size, y_size, x_size)


if __name__ == "__main__":
    main()