#pragma once

#include "common.h"

typedef struct chunk {
  dtype dtype;
  s32 depth, height, width;
  void* data;
} chunk;

chunk chunk_new(dtype dtype, s32 depth, s32 height, s32 width);
void chunk_free(chunk* chunk);
chunk chunk_cast(chunk* chunk, dtype dtype);

u8 chunk_get_u8(chunk* chunk, s32 z, s32 y, s32 x);
f32 chunk_get_f32(chunk* chunk, s32 z, s32 y, s32 x);
void chunk_set_u8(chunk* chunk, s32 z, s32 y, s32 x, u8 val);
void chunk_set_f32(chunk* chunk, s32 z, s32 y, s32 x, f32 val);

chunk maxpool(chunk* inchunk, s32 kernel, s32 stride);
chunk avgpool(chunk* inchunk, s32 kernel, s32 stride);