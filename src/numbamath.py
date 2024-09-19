from numba import njit, jit
import numpy as np


from common import timing_decorator

@timing_decorator
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