#pragma once

#include "common.h"
#include "chunk.h"
#include "cvol.h"

static u64 rdtscp(void)
{
  u32 hi, lo;
  asm volatile (
      "rdtscp"
      : "=d"(hi), "=a"(lo)
      :
      : "cx", "memory"
  );
  return (u64)hi<<32 | lo;
}


typedef struct bitmask3d {
  u8* data;
  s32 depth, height, width;
} bitmask3d;

static bitmask3d bitmask3d_new(s32 depth, s32 height, s32 width);
static void bitmask3d_free(bitmask3d* mask);
static bool bitmask3d_get(bitmask3d* mask, s32 z, s32 y, s32 x);
static void bitmask3d_set(bitmask3d* mask, s32 z, s32 y, s32 x, bool value);
static void bitmask3d_print(bitmask3d* mask);


static s32 get_index(s32 z, s32 y, s32 x) { return z / 2 * 4 + y / 2 * 2 + x / 2; }

static u8 get_bit_position(s32 z, s32 y, s32 x) { return z % 2 * 4 + y % 2 * 2 + x % 2; }

static bitmask3d bitmask3d_new(s32 depth, s32 height, s32 width) {
  s32 total_bytes = (depth + 1) / 2 * ((height + 1) / 2) * ((width + 1) / 2);
  return (bitmask3d){
    .data = calloc(total_bytes, sizeof(u8)),
    .depth = depth,
    .height = height,
    .width = width
  };
}

static void bitmask3d_free(bitmask3d* mask) {
  free(mask->data);
  mask->data = NULL;
  mask->depth = mask->height = mask->width = 0;
}

static bool bitmask3d_get(bitmask3d* mask, s32 z, s32 y, s32 x) {
  if (z < 0 || z >= mask->depth || y < 0 || y >= mask->height || x < 0 || x >= mask->width) {
    die("get: invalid index");
    return false;
  }

  s32 byte_index = get_index(z, y, x);
  u8 bit_position = get_bit_position(z, y, x);

  return (mask->data[byte_index] & (1 << bit_position)) != 0;
}

static void bitmask3d_set(bitmask3d* mask, s32 z, s32 y, s32 x, bool value) {
  if (z < 0 || z >= mask->depth || y < 0 || y >= mask->height || x < 0 || x >= mask->width) {
    die("set: invalid index");
    return;
  }

  s32 byte_index = get_index(z, y, x);
  u8 bit_position = get_bit_position(z, y, x);

  if (value) { mask->data[byte_index] |= (1 << bit_position); }
  else { mask->data[byte_index] &= ~(1 << bit_position); }
}

static void bitmask3d_print(bitmask3d* mask) {
  for (s32 z = 0; z < mask->depth; z++) {
    printf("Layer %d:\n", z);
    for (s32 y = 0; y < mask->height; y++) {
      for (s32 x = 0; x < mask->width; x++) { printf("%d", bitmask3d_get(mask, z, y, x) ? 1 : 0); }
      printf("\n");
    }
    printf("\n");
  }
}

static bitmask3d isomask(chunk* chunk, f32 iso) {
  auto ret = bitmask3d_new(chunk->depth, chunk->height, chunk->width);
  for (s32 z = 0; z < chunk->depth; z++)
    for (s32 y = 0; y < chunk->height; y++)
      for (s32 x = 0; x < chunk->width; x++) {
        u8 data = chunk_get_u8(chunk, z, y, x);
        bitmask3d_set(&ret, z, y, x, data > (u8)iso);
      }
  return ret;
}

static chunk maxpool(chunk* inchunk, s32 kernel, s32 stride) {
  chunk ret = chunk_new(inchunk->dtype, (inchunk->depth + stride - 1) / stride, (inchunk->height + stride - 1) / stride,
                        (inchunk->width + stride - 1) / stride);
  for (s32 z = 0; z < ret.depth; z++)
    for (s32 y = 0; y < ret.height; y++)
      for (s32 x = 0; x < ret.width; x++) {
        u8 max8 = 0;
        f32 max32 = -INFINITY;
        u8 val8;
        f32 val32;
        for (s32 zi = 0; zi < kernel; zi++)
          for (s32 yi = 0; yi < kernel; yi++)
            for (s32 xi = 0; xi < kernel; xi++) {
              if (z + zi > inchunk->depth || y + yi > inchunk->height || x + xi > inchunk->width) { continue; }
              if (inchunk->dtype == U8 && (val8 = chunk_get_u8(inchunk, z * stride + zi, y * stride + yi,
                                                               x * stride + xi)) > max8) { max8 = val8; }
              else if (inchunk->dtype == F32 && (val32 = chunk_get_f32(inchunk, z * stride + zi, y * stride + yi,
                                                                       x * stride + xi)) > max32) { max32 = val32; }
            }
        if (inchunk->dtype == U8) { chunk_set_u8(&ret, z, y, x, max8); }
        else if (inchunk->dtype == F32) { chunk_set_f32(&ret, z, y, x, max32); }
      }
  return ret;
}

static chunk avgpool(chunk* inchunk, s32 kernel, s32 stride) {
  chunk ret = chunk_new(inchunk->dtype, (inchunk->depth + stride - 1) / stride, (inchunk->height + stride - 1) / stride,
                        (inchunk->width + stride - 1) / stride);
  s32 len = kernel * kernel * kernel;
  s32 i = 0;
  void* data = malloc(len * inchunk->dtype == U8 ? 1 : 4);
  for (s32 z = 0; z < ret.depth; z++)
    for (s32 y = 0; y < ret.height; y++)
      for (s32 x = 0; x < ret.width; x++) {
        len = kernel * kernel * kernel;
        i = 0;
        for (s32 zi = 0; zi < kernel; zi++)
          for (s32 yi = 0; yi < kernel; yi++)
            for (s32 xi = 0; xi < kernel; xi++) {
              if (z + zi > inchunk->depth || y + yi > inchunk->height || x + xi > inchunk->width) {
                len--;
                continue;
              }
              if (inchunk->dtype == U8) {
                ((u8*)&data)[i++] = chunk_get_u8(inchunk, z * stride + zi, y * stride + yi, x * stride + xi);
              }
              if (inchunk->dtype == F32) {
                ((f32*)&data)[i++] = chunk_get_f32(inchunk, z * stride + zi, y * stride + yi, x * stride + xi);
              }
            }
        if (inchunk->dtype == U8) { chunk_set_u8(&ret, z, y, x, avgu8(data, len)); }
        else if (inchunk->dtype == F32) { chunk_set_f32(&ret, z, y, x, avgf32(data, len)); }
      }
  return ret;
}

static chunk create_box_kernel(s32 size) {
  chunk kernel = chunk_new(F32, size, size, size);
  float value = 1.0f / (size * size * size);
  for (s32 z = 0; z < size; z++) {
    for (s32 y = 0; y < size; y++) { for (s32 x = 0; x < size; x++) { chunk_set_f32(&kernel, z, y, x, value); } }
  }
  return kernel;
}

static chunk convolve3d(chunk* input, chunk* kernel) {
  chunk output = chunk_new(input->dtype, input->depth, input->height, input->width);
  s32 pad = kernel->depth / 2;

  for (s32 z = 0; z < input->depth; z++) {
    for (s32 y = 0; y < input->height; y++) {
      for (s32 x = 0; x < input->width; x++) {
        float sum = 0.0f;
        for (s32 kz = 0; kz < kernel->depth; kz++) {
          for (s32 ky = 0; ky < kernel->height; ky++) {
            for (s32 kx = 0; kx < kernel->width; kx++) {
              s32 iz = z + kz - pad;
              s32 iy = y + ky - pad;
              s32 ix = x + kx - pad;
              if (iz >= 0 && iz < input->depth && iy >= 0 && iy < input->height && ix >= 0 && ix < input->width) {
                float input_val = (input->dtype == U8)
                                    ? (float)chunk_get_u8(input, iz, iy, ix)
                                    : chunk_get_f32(input, iz, iy, ix);
                sum += input_val * chunk_get_f32(kernel, kz, ky, kx);
              }
            }
          }
        }
        if (input->dtype == U8) { chunk_set_u8(&output, z, y, x, (u8)fminf(fmaxf(sum, 0), 255)); }
        else { chunk_set_f32(&output, z, y, x, sum); }
      }
    }
  }
  return output;
}

static chunk unsharp_mask_3d(chunk* input, float amount, s32 kernel_size) {
  chunk kernel = create_box_kernel(kernel_size);
  chunk blurred = convolve3d(input, &kernel);
  chunk output = chunk_new(input->dtype, input->depth, input->height, input->width);

  for (s32 z = 0; z < input->depth; z++) {
    for (s32 y = 0; y < input->height; y++) {
      for (s32 x = 0; x < input->width; x++) {
        float original, blur;
        if (input->dtype == U8) {
          original = (float)chunk_get_u8(input, z, y, x);
          blur = (float)chunk_get_u8(&blurred, z, y, x);
        }
        else {
          original = chunk_get_f32(input, z, y, x);
          blur = chunk_get_f32(&blurred, z, y, x);
        }

        float sharpened = original + amount * (original - blur);

        if (input->dtype == U8) { chunk_set_u8(&output, z, y, x, (u8)fminf(fmaxf(sharpened, 0), 255)); }
        else { chunk_set_f32(&output, z, y, x, sharpened); }
      }
    }
  }

  chunk_free(&kernel);
  chunk_free(&blurred);

  return output;
}

