#pragma once

#include "common.h"

typedef struct chunk
{
  dtype dtype;
  s32 depth,height,width;
  void* data;
}chunk;

chunk chunk_new(dtype dtype, s32 depth, s32 height, s32 width);
void chunk_free(chunk* chunk);
void chunk_get(chunk* chunk, s32 z, s32 y, s32 x, void* val);
void chunk_set(chunk* chunk, s32 z, s32 y, s32 x, void* val);
chunk chunk_cast(chunk* chunk, dtype dtype);