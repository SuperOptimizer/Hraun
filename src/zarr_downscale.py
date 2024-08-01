import zarr
import numpy as np
from skimage.measure import block_reduce
from concurrent.futures import ThreadPoolExecutor
import math
import numcodecs
from numcodecs import Blosc
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

compressor = numcodecs.Blosc(cname='blosclz', clevel=9, shuffle=Blosc.BITSHUFFLE)

def process_chunk(input_zarr, output_zarr, chunk_coords):
    z, y, x = chunk_coords
    #if z == 10 and y == 10 and x == 6:
    #    return chunk_coords
    #if z < 10:
    #    return chunk_coords
    try:
        # Read the chunk from the input Zarr
        chunk = input_zarr[z * 256:(z + 1) * 256, y * 256:(y + 1) * 256, x * 256:(x + 1) * 256]

        # Pad the chunk if it's smaller than 256x256x256
        pad_width = [(0, max(0, 256 - s)) for s in chunk.shape]
        if any(pw != (0, 0) for pw in pad_width):
            chunk = np.pad(chunk, pad_width, mode='constant', constant_values=0)

        # Perform 8x block reduction
        reduced_chunk = block_reduce(chunk, block_size=(8, 8, 8), func=np.max)

        # Calculate the actual size of the reduced chunk (without padding)
        actual_size = [min(32, math.ceil((input_zarr.shape[i] - chunk_coords[i] * 256) / 8)) for i in range(3)]

        # Trim the reduced chunk to the actual size
        reduced_chunk = reduced_chunk[:actual_size[0], :actual_size[1], :actual_size[2]]

        # Write the reduced chunk to the output Zarr
        output_chunk_z = z
        output_chunk_y = y
        output_chunk_x = x
        output_zarr[output_chunk_z * 32:(output_chunk_z * 32 + actual_size[0]),
                    output_chunk_y * 32:(output_chunk_y * 32 + actual_size[1]),
                    output_chunk_x * 32:(output_chunk_x * 32 + actual_size[2])] = reduced_chunk

        logging.info(f"Processed chunk: z={z}, y={y}, x={x}, shape={reduced_chunk.shape}")
    except Exception as e:
        logging.error(f"Error processing chunk at z={z}, y={y}, x={x}: {str(e)}")
        logging.error(traceback.format_exc())

    return chunk_coords

def downscale_zarr(input_path, output_path):
    try:
        # Open the input Zarr array
        input_zarr = zarr.open(input_path, mode='r')
        logging.info(f"Input Zarr shape: {input_zarr.shape}, chunks: {input_zarr.chunks}")

        # Calculate the dimensions of the output array
        output_shape = tuple(math.ceil(dim / 8) for dim in input_zarr.shape)
        logging.info(f"Output Zarr shape: {output_shape}")

        # Create the output Zarr array
        synchronizer = zarr.ProcessSynchronizer(output_path + '.sync')
        output_zarr = zarr.open(output_path, mode='w', shape=output_shape, chunks=(256, 256, 256),
                                dtype=np.uint8, synchronizer=synchronizer, compressor=compressor)

        # Calculate the number of chunks in each dimension
        chunks_z, chunks_y, chunks_x = (math.ceil(dim / 256) for dim in input_zarr.shape)
        logging.info(f"Number of chunks: z={chunks_z}, y={chunks_y}, x={chunks_x}")

        # Create a list of chunk coordinates
        chunk_coords = [(z, y, x)
                        for z in range(chunks_z)
                        for y in range(chunks_y)
                        for x in range(chunks_x)]

        # Process chunks in parallel
        with ThreadPoolExecutor(1) as executor:
            futures = [executor.submit(process_chunk, input_zarr, output_zarr, coords) for coords in chunk_coords]

            for future in futures:
                future.result()  # This will re-raise any exception that occurred in the thread

        logging.info(f"Downscaling complete. Output saved to {output_path}")
    except Exception as e:
        logging.error(f"Error in downscale_zarr: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    input_zarr_path = r"D:\dl.ash2txt.org\community-uploads\ryan\3d_predictions_scroll2.zarr"
    output_zarr_path = r"C:\vesuvius_scroll2_downscaled.zarr"
    downscale_zarr(input_zarr_path, output_zarr_path)