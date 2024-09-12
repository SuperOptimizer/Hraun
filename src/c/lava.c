#include <stdint.h>
#include <stdlib.h>

#include "common.h"
#include "lava.h"
#include "cvol.h"


static s32 get_index(s32 z, s32 y, s32 x) { return z / 2 * 4 + y / 2 * 2 + x / 2; }

static u8 get_bit_position(s32 z, s32 y, s32 x) { return z % 2 * 4 + y % 2 * 2 + x % 2; }

bitmask3d bitmask3d_new(s32 depth, s32 height, s32 width) {
  s32 total_bytes = (depth + 1) / 2 * ((height + 1) / 2) * ((width + 1) / 2);
  return (bitmask3d){
    .data = calloc(total_bytes, sizeof(u8)),
    .depth = depth,
    .height = height,
    .width = width
  };
}

void bitmask3d_free(bitmask3d* mask) {
  free(mask->data);
  mask->data = NULL;
  mask->depth = mask->height = mask->width = 0;
}

bool bitmask3d_get(bitmask3d* mask, s32 z, s32 y, s32 x) {
  if (z < 0 || z >= mask->depth || y < 0 || y >= mask->height || x < 0 || x >= mask->width) {
    die("get: invalid index");
    return false;
  }

  s32 byte_index = get_index(z, y, x);
  u8 bit_position = get_bit_position(z, y, x);

  return (mask->data[byte_index] & (1 << bit_position)) != 0;
}

void bitmask3d_set(bitmask3d* mask, s32 z, s32 y, s32 x, bool value) {
  if (z < 0 || z >= mask->depth || y < 0 || y >= mask->height || x < 0 || x >= mask->width) {
    die("set: invalid index");
    return;
  }

  s32 byte_index = get_index(z, y, x);
  u8 bit_position = get_bit_position(z, y, x);

  if (value) { mask->data[byte_index] |= (1 << bit_position); }
  else { mask->data[byte_index] &= ~(1 << bit_position); }
}

void bitmask3d_print(bitmask3d* mask) {
  for (s32 z = 0; z < mask->depth; z++) {
    printf("Layer %d:\n", z);
    for (s32 y = 0; y < mask->height; y++) {
      for (s32 x = 0; x < mask->width; x++) { printf("%d", bitmask3d_get(mask, z, y, x) ? 1 : 0); }
      printf("\n");
    }
    printf("\n");
  }
}

bitmask3d isomask(chunk* chunk, f32 iso) {
  auto ret = bitmask3d_new(chunk->depth, chunk->height, chunk->width);
  for (s32 z = 0; z < chunk->depth; z++)
    for (s32 y = 0; y < chunk->height; y++)
      for (s32 x = 0; x < chunk->width; x++) {
        u8 data = chunk_get_u8(chunk, z, y, x);
        bitmask3d_set(&ret, z, y, x, data > (u8)iso);
      }
  return ret;
}

chunk maxpool(chunk* inchunk, s32 kernel, s32 stride) {
  chunk ret = chunk_new(inchunk->dtype, (inchunk->depth + stride - 1)/stride, (inchunk->height + stride - 1)/stride, (inchunk->width + stride - 1)/stride);
  for(s32 z = 0; z < ret.depth; z++)
    for(s32 y = 0; y < ret.height; y++)
      for(s32 x = 0; x < ret.width; x++) {
        u8 max8 = 0; f32 max32 = -INFINITY;
        u8 val8; f32 val32;
        for(s32 zi = 0; zi < kernel; zi++)
          for(s32 yi = 0; yi < kernel; yi++)
            for(s32 xi = 0; xi < kernel; xi++) {
              if(z+zi > inchunk->depth || y+yi > inchunk->height || x+xi > inchunk->width) {
                printf("asdf\n");
                continue;
              }
              if(inchunk->dtype == U8 && (val8 = chunk_get_u8(inchunk, z*stride+zi, y*stride+yi, x*stride+xi)) > max8) {
                max8 = val8;
              }
              else if(inchunk->dtype == F32 && (val32 = chunk_get_f32(inchunk, z*stride+zi, y*stride+yi, x*stride+xi)) > max32) {
                max32 = val32;
              }
            }
        if(inchunk->dtype == U8) { chunk_set_u8(&ret,z,y,x,max8);}
        else if(inchunk->dtype == F32) { chunk_set_f32(&ret,z,y,x,max32);}
      }
  return ret;
}

chunk avgpool(chunk* inchunk, s32 kernel, s32 stride) {
  chunk ret = chunk_new(inchunk->dtype, (inchunk->depth + stride - 1)/stride, (inchunk->height + stride - 1)/stride, (inchunk->width + stride - 1)/stride);
  s32 len = kernel*kernel*kernel;
  void* data = malloc(len * inchunk->dtype == U8 ? 1 : 4);
  for(s32 z = 0; z < ret.depth; z++)
    for(s32 y = 0; y < ret.height; y++)
      for(s32 x = 0; x < ret.width; x++) {
        len = kernel*kernel*kernel;
        for(s32 zi = 0; zi < kernel; zi++)
          for(s32 yi = 0; yi < kernel; yi++)
            for(s32 xi = 0; xi < kernel; xi++) {
              if(z+zi > inchunk->depth || y+yi > inchunk->height || x+xi > inchunk->width) {len--; continue;}
              if(inchunk->dtype == U8) {*(u8*)&data[zi*kernel*kernel + yi*kernel + zi] = chunk_get_u8(inchunk, z*stride+zi, y*stride+yi, x*stride+xi);}
              if(inchunk->dtype == F32) {*(f32*)&data[zi*kernel*kernel + yi*kernel + zi] = chunk_get_f32(inchunk, z*stride+zi, y*stride+yi, x*stride+xi);}
            }
        if(inchunk->dtype == U8) { chunk_set_u8(&ret,z,y,x,avgu8(data,len));}
        else if(inchunk->dtype == F32) { chunk_set_f32(&ret,z,y,x,avgf32(data,len));}
      }
  return ret;
}