import numpy as np
from numba import jit
import requests
import tifffile
import io
import platform
import os
from PIL import Image
from enum import Enum
import random
import time
from functools import wraps
from snic import snic
import skimage


def timing_decorator(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"{func.__name__} executed in {elapsed_time:.6f} seconds")
    return result

  return wrapper


if platform.system() == 'Windows':
  CACHEDIR = "D:/"
else:
  CACHEDIR = "/mnt/d/"


@jit(nopython=True)
def rescale_array(arr):
  min_val = np.min(arr)
  max_val = np.max(arr)

  # Check if the array is constant
  if min_val == max_val:
    return np.zeros_like(arr)

  # Rescale to [0, 1]
  scaled = (arr - min_val) / (max_val - min_val)
  return scaled

@jit(nopython=True, inline='always')
def index_to_offset_3d(flat_index, kernel_size):
  kernel_d, kernel_h, kernel_w = kernel_size
  z_offset = flat_index // (kernel_h * kernel_w)
  y_offset = (flat_index % (kernel_h * kernel_w)) // kernel_w
  x_offset = flat_index % kernel_w

  z_offset = z_offset - kernel_d // 2
  y_offset = y_offset - kernel_h // 2
  x_offset = x_offset - kernel_w // 2

  return np.array([z_offset, y_offset, x_offset])


@jit(nopython=True, inline='always')
def calculate_padding(input_size, kernel_size, stride):
  output_size = (input_size + stride - 1) // stride
  padding_needed = max(0, (output_size - 1) * stride + kernel_size - input_size)
  padding_before = padding_needed // 2
  padding_after = padding_needed - padding_before
  return padding_before, padding_after


@jit(nopython=True, inline='always')
def calculate_distance_from_center(kd, kh, kw, kernel_d, kernel_h, kernel_w):
  center_d, center_h, center_w = kernel_d // 2, kernel_h // 2, kernel_w // 2
  return abs(kd - center_d) + abs(kh - center_h) + abs(kw - center_w)


@jit(nopython=True)
def avgpool(input_array, kernel_size, stride, dilation):
  depth, height, width = input_array.shape
  kernel_d, kernel_h, kernel_w = kernel_size
  stride_d, stride_h, stride_w = stride
  dilation_d, dilation_h, dilation_w = dilation

  pad_d_before, pad_d_after = calculate_padding(depth, kernel_d, stride_d)
  pad_h_before, pad_h_after = calculate_padding(height, kernel_h, stride_h)
  pad_w_before, pad_w_after = calculate_padding(width, kernel_w, stride_w)

  output_depth = (depth + stride_d - 1) // stride_d
  output_height = (height + stride_h - 1) // stride_h
  output_width = (width + stride_w - 1) // stride_w

  output = np.zeros((output_depth, output_height, output_width), dtype=np.float32)

  for d in range(output_depth):
    for h in range(output_height):
      for w in range(output_width):
        d_start = d * stride_d - pad_d_before
        h_start = h * stride_h - pad_h_before
        w_start = w * stride_w - pad_w_before
        pool_sum = 0.0
        pool_count = 0
        for kd in range(kernel_d):
          for kh in range(kernel_h):
            for kw in range(kernel_w):
              d_index = d_start + kd * dilation_d
              h_index = h_start + kh * dilation_h
              w_index = w_start + kw * dilation_w

              if (0 <= d_index < depth and
                  0 <= h_index < height and
                  0 <= w_index < width):
                value = input_array[d_index, h_index, w_index]
                pool_sum += value
                pool_count += 1
        output[d, h, w] = pool_sum / pool_count if pool_count > 0 else 0.0
  return output


@jit(nopython=True)
def maxpool(input_array, kernel_size, stride, dilation):
  depth, height, width = input_array.shape
  kernel_d, kernel_h, kernel_w = kernel_size
  stride_d, stride_h, stride_w = stride
  dilation_d, dilation_h, dilation_w = dilation

  pad_d_before, pad_d_after = calculate_padding(depth, kernel_d, stride_d)
  pad_h_before, pad_h_after = calculate_padding(height, kernel_h, stride_h)
  pad_w_before, pad_w_after = calculate_padding(width, kernel_w, stride_w)

  output_depth = (depth + stride_d - 1) // stride_d
  output_height = (height + stride_h - 1) // stride_h
  output_width = (width + stride_w - 1) // stride_w

  output = np.zeros((output_depth, output_height, output_width), dtype=np.float32)

  for d in range(output_depth):
    for h in range(output_height):
      for w in range(output_width):
        d_start = d * stride_d - pad_d_before
        h_start = h * stride_h - pad_h_before
        w_start = w * stride_w - pad_w_before
        pool_max = -np.inf
        for kd in range(kernel_d):
          for kh in range(kernel_h):
            for kw in range(kernel_w):
              d_index = d_start + kd * dilation_d
              h_index = h_start + kh * dilation_h
              w_index = w_start + kw * dilation_w

              if (0 <= d_index < depth and
                  0 <= h_index < height and
                  0 <= w_index < width):
                value = input_array[d_index, h_index, w_index]
                pool_max = max(pool_max, value)

        output[d, h, w] = pool_max if pool_max != -np.inf else 0.0
  return output


@jit(nopython=True)
def minpool(input_array, kernel_size, stride, dilation):
  depth, height, width = input_array.shape
  kernel_d, kernel_h, kernel_w = kernel_size
  stride_d, stride_h, stride_w = stride
  dilation_d, dilation_h, dilation_w = dilation

  pad_d_before, pad_d_after = calculate_padding(depth, kernel_d, stride_d)
  pad_h_before, pad_h_after = calculate_padding(height, kernel_h, stride_h)
  pad_w_before, pad_w_after = calculate_padding(width, kernel_w, stride_w)

  output_depth = (depth + stride_d - 1) // stride_d
  output_height = (height + stride_h - 1) // stride_h
  output_width = (width + stride_w - 1) // stride_w

  output = np.zeros((output_depth, output_height, output_width), dtype=np.float32)

  for d in range(output_depth):
    for h in range(output_height):
      for w in range(output_width):
        d_start = d * stride_d - pad_d_before
        h_start = h * stride_h - pad_h_before
        w_start = w * stride_w - pad_w_before
        pool_min = np.inf
        for kd in range(kernel_d):
          for kh in range(kernel_h):
            for kw in range(kernel_w):
              d_index = d_start + kd * dilation_d
              h_index = h_start + kh * dilation_h
              w_index = w_start + kw * dilation_w

              if (0 <= d_index < depth and
                  0 <= h_index < height and
                  0 <= w_index < width):
                value = input_array[d_index, h_index, w_index]
                pool_min = min(pool_min, value)
        output[d, h, w] = pool_min if pool_min != np.inf else 0.0

  return output


@jit(nopython=True)
def sumpool(input_array, kernel_size, stride, dilation):
  depth, height, width = input_array.shape
  kernel_d, kernel_h, kernel_w = kernel_size
  stride_d, stride_h, stride_w = stride
  dilation_d, dilation_h, dilation_w = dilation

  pad_d_before, pad_d_after = calculate_padding(depth, kernel_d, stride_d)
  pad_h_before, pad_h_after = calculate_padding(height, kernel_h, stride_h)
  pad_w_before, pad_w_after = calculate_padding(width, kernel_w, stride_w)

  output_depth = (depth + stride_d - 1) // stride_d
  output_height = (height + stride_h - 1) // stride_h
  output_width = (width + stride_w - 1) // stride_w

  output = np.zeros((output_depth, output_height, output_width), dtype=np.float32)

  for d in range(output_depth):
    for h in range(output_height):
      for w in range(output_width):
        d_start = d * stride_d - pad_d_before
        h_start = h * stride_h - pad_h_before
        w_start = w * stride_w - pad_w_before

        pool_sum = 0.0
        pool_count = 0

        for kd in range(kernel_d):
          for kh in range(kernel_h):
            for kw in range(kernel_w):
              d_index = d_start + kd * dilation_d
              h_index = h_start + kh * dilation_h
              w_index = w_start + kw * dilation_w

              if (0 <= d_index < depth and
                  0 <= h_index < height and
                  0 <= w_index < width):
                value = input_array[d_index, h_index, w_index]
                pool_sum += value
                pool_count += 1

        output[d, h, w] = pool_sum
  return output


@jit(nopython=True)
def argmaxpool(input_array, kernel_size, stride, dilation):
  depth, height, width = input_array.shape
  kernel_d, kernel_h, kernel_w = kernel_size
  stride_d, stride_h, stride_w = stride
  dilation_d, dilation_h, dilation_w = dilation

  pad_d_before, pad_d_after = calculate_padding(depth, kernel_d, stride_d)
  pad_h_before, pad_h_after = calculate_padding(height, kernel_h, stride_h)
  pad_w_before, pad_w_after = calculate_padding(width, kernel_w, stride_w)

  output_depth = (depth + stride_d - 1) // stride_d
  output_height = (height + stride_h - 1) // stride_h
  output_width = (width + stride_w - 1) // stride_w

  output = np.zeros((output_depth, output_height, output_width), dtype=np.uint16)

  for d in range(output_depth):
    for h in range(output_height):
      for w in range(output_width):
        d_start = d * stride_d - pad_d_before
        h_start = h * stride_h - pad_h_before
        w_start = w * stride_w - pad_w_before

        pool_max = -np.inf
        max_index = 0
        max_distance = np.inf

        for kd in range(kernel_d):
          for kh in range(kernel_h):
            for kw in range(kernel_w):
              d_index = d_start + kd * dilation_d
              h_index = h_start + kh * dilation_h
              w_index = w_start + kw * dilation_w

              if (0 <= d_index < depth and
                  0 <= h_index < height and
                  0 <= w_index < width):
                value = input_array[d_index, h_index, w_index]
                distance = calculate_distance_from_center(kd, kh, kw, kernel_d, kernel_h, kernel_w)
                if value > pool_max or (value == pool_max and distance < max_distance):
                  pool_max = value
                  max_index = kd * kernel_h * kernel_w + kh * kernel_w + kw
                  max_distance = distance
        output[d, h, w] = max_index
  return output


@jit(nopython=True)
def argminpool(input_array, kernel_size, stride, dilation):
  depth, height, width = input_array.shape
  kernel_d, kernel_h, kernel_w = kernel_size
  stride_d, stride_h, stride_w = stride
  dilation_d, dilation_h, dilation_w = dilation

  pad_d_before, pad_d_after = calculate_padding(depth, kernel_d, stride_d)
  pad_h_before, pad_h_after = calculate_padding(height, kernel_h, stride_h)
  pad_w_before, pad_w_after = calculate_padding(width, kernel_w, stride_w)

  output_depth = (depth + stride_d - 1) // stride_d
  output_height = (height + stride_h - 1) // stride_h
  output_width = (width + stride_w - 1) // stride_w

  output = np.zeros((output_depth, output_height, output_width), dtype=np.uint16)

  for d in range(output_depth):
    for h in range(output_height):
      for w in range(output_width):
        d_start = d * stride_d - pad_d_before
        h_start = h * stride_h - pad_h_before
        w_start = w * stride_w - pad_w_before
        pool_min = np.inf
        min_index = 0
        min_distance = np.inf

        for kd in range(kernel_d):
          for kh in range(kernel_h):
            for kw in range(kernel_w):
              d_index = d_start + kd * dilation_d
              h_index = h_start + kh * dilation_h
              w_index = w_start + kw * dilation_w

              if (0 <= d_index < depth and
                  0 <= h_index < height and
                  0 <= w_index < width):
                value = input_array[d_index, h_index, w_index]
                distance = calculate_distance_from_center(kd, kh, kw, kernel_d, kernel_h, kernel_w)
                if value < pool_min or (value == pool_min and distance < min_distance):
                  pool_min = value
                  min_index = kd * kernel_h * kernel_w + kh * kernel_w + kw
                  min_distance = distance
        output[d, h, w] = min_index
  return output


@jit(nopython=True)
def compute_histogram_3d(image, num_bins):
  hist = np.zeros((image.shape[2], num_bins), dtype=np.int32)
  bin_edges = np.linspace(0, 1, num_bins + 1)

  for c in range(image.shape[2]):
    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        for k in range(num_bins):
          if bin_edges[k] <= image[i, j, c] < bin_edges[k + 1]:
            hist[c, k] += 1
            break

  return hist


@jit(nopython=True)
def compute_cdf_3d(hist):
  cdf = np.zeros_like(hist, dtype=np.float32)
  for c in range(hist.shape[0]):
    cdf[c, 0] = hist[c, 0]
    for i in range(1, hist.shape[1]):
      cdf[c, i] = cdf[c, i - 1] + hist[c, i]
    cdf[c] /= cdf[c, -1]
  return cdf


@jit(nopython=True)
def apply_equalization_3d(image, cdf, num_bins):
  equalized = np.zeros_like(image)
  bin_edges = np.linspace(0, 1, num_bins + 1)

  for c in range(image.shape[2]):
    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        for k in range(num_bins):
          if bin_edges[k] <= image[i, j, c] < bin_edges[k + 1]:
            equalized[i, j, c] = cdf[c, k]
            break

  return equalized


@timing_decorator
def download(url):
  path = url.replace("https://", "")
  path = os.path.join(CACHEDIR,path)
  if not os.path.exists(path):
    print(f"downloading {url}")
    response = requests.get(url)
    if response.status_code == 200:
      os.makedirs(os.path.dirname(path), exist_ok=True)
      filedata = io.BytesIO(response.content)
      with tifffile.TiffFile(filedata) as tif:
        data = (tif.asarray() >> 8).astype(np.uint8) & 0xf0
        tifffile.imwrite(path, data)
    else:
      raise Exception(f'Cannot download {url}')

  if url.endswith('.tif'):
      tif = tifffile.imread(path)
      data = tif
      if data.dtype == np.uint16:
        return ((data >> 8) & 0xf0).astype(np.uint8)
      else:
        raise
  elif url.endswith('.jpg'):
    data = np.asarray(Image.open(filedata))
    return data & 0xf0
  elif url.endswith('.png'):
    data = np.asarray(Image.open(filedata))
    return data


@timing_decorator
@jit(nopython=True)
def find_seed_points(arr, nseeds):
  maxiter = 10

  maxes = argmaxpool(arr, (3, 3, 3), (1, 1, 1), (1, 1, 1))
  maxima = set()
  for seed in range(nseeds):
    z, y, x = random.randint(0, arr.shape[0] - 1), random.randint(0, arr.shape[1] - 1), random.randint(0,
                                                                                                       arr.shape[2] - 1)
    for i in range(maxiter):
      offset = index_to_offset_3d(maxes[z, y, x], (3, 3, 3))
      if offset[0] == 0 and offset[1] == 0 and offset[2] == 0:
        maxima.add((z, y, x))
        break
      z += offset[0]
      y += offset[1]
      x += offset[2]
    else:
      maxima.add((z, y, x))

  mins = argminpool(arr, (3, 3, 3), (1, 1, 1), (1, 1, 1))
  minima = set()
  for seed in range(nseeds):
    z, y, x = (random.randint(0, arr.shape[0] - 1),
               random.randint(0, arr.shape[1] - 1),
               random.randint(0, arr.shape[2] - 1))
    for i in range(maxiter):
      offset = index_to_offset_3d(mins[z, y, x], (3, 3, 3))
      if offset[0] == 0 and offset[1] == 0 and offset[2] == 0:
        minima.add((z, y, x))
        break
      z += offset[0]
      y += offset[1]
      x += offset[2]
    else:
      minima.add((z, y, x))
  return maxima, minima


@timing_decorator
@jit(nopython=True)
def walk(arr, seeds):
  labels = np.zeros(arr.shape, dtype=np.uint16)
  max_iter = 10
  for label, seed in enumerate(seeds):
    z, y, x = seed
    for i in range(max_iter):
      tries = 0
      nextval = 0.0
      candidates = []
      while tries < 10:
        tries += 1
        for zi in range(-1, 2):
          for yi in range(-1, 2):
            for xi in range(-1, 2):
              if zi == yi == xi == 0:
                continue
              if (z + zi < 0 or z + zi >= arr.shape[0] or
                  y + yi < 0 or y + yi >= arr.shape[1] or
                  x + xi < 0 or x + xi >= arr.shape[2]):
                continue
              if (arr[z + zi, y + yi, x + xi] > nextval and
                  labels[z + zi, y + yi, x + xi] == 0):
                candidates.insert(0,(z + zi, y + yi, x + xi))
                nextval = arr[z + zi, y + yi, x + xi]
        if len(candidates) == 0:
          print()

      assert len(candidates)> 0
      z, y, x = candidates[0]
      labels[z, y, x] = label
  return labels


@timing_decorator
@jit(nopython=True)
def get_neighbors(labels, num_labels):
  neighbors = np.zeros((num_labels, num_labels), dtype=np.uint16)
  num_neighbors = np.zeros((num_labels,), dtype=np.uint16)
  for z in range(labels.shape[0]):
    for y in range(labels.shape[1]):
      for x in range(labels.shape[2]):
        l = labels[z, y, x]
        if l == 0:
          continue
        for zi in range(-1, 2):
          for yi in range(-1, 2):
            for xi in range(-1, 2):
              if zi == 0 and yi == 0 and xi == 0:
                continue
              if (z + zi < 0 or z + zi >= labels.shape[0] or
                  y + yi < 0 or y + yi >= labels.shape[1] or
                  x + xi < 0 or x + xi >= labels.shape[2]):
                continue
              if (labels[z + zi, y + yi, x + xi] > 0 and
                  labels[z + zi, y + yi, x + xi] != l):
                neighbors[l][num_neighbors[l]] = labels[z + zi, y + yi, x + xi]
                num_neighbors[l] += 1
  return neighbors


@timing_decorator
def segment(arr):
  maxima, minima = find_seed_points(arr, 1024)
  labels = walk(arr, maxima)
  neighbors = get_neighbors(labels, 1024)
  print(neighbors)
  print(maxima)
  print(minima)


@timing_decorator
def main():
  data = download(
    "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volume_grids/20230205180739/cell_yxz_008_008_012.tif")
  data = sumpool(data, (4, 4, 4), (4, 4, 4), (1, 1, 1))
  data = data.astype(np.float32)
  data = rescale_array(data)
  data = skimage.exposure.equalize_adapthist(data)
  neigh_overflow, labels, superpixels = snic(data, 4, 5.0, 80, 160)
  segment(data)


if __name__ == "__main__":
  main()
