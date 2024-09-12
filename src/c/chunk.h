#pragma once

#include "common.h"

typedef struct chunk {
  dtype dtype;
  s32 depth, height, width;
  void* data;
} chunk;

chunk chunk_new(dtype dtype, s32 depth, s32 height, s32 width);
void chunk_free(chunk* chunk);
void chunk_get(chunk* chunk, s32 z, s32 y, s32 x, void* val);
void chunk_set(chunk* chunk, s32 z, s32 y, s32 x, void* val);
chunk chunk_cast(chunk* chunk, dtype dtype);

u8 chunk_get_u8(chunk* chunk, s32 z, s32 y, s32 x);
s8 chunk_get_s8(chunk* chunk, s32 z, s32 y, s32 x);

u16 chunk_get_u16(chunk* chunk, s32 z, s32 y, s32 x);
s16 chunk_get_s16(chunk* chunk, s32 z, s32 y, s32 x);

u32 chunk_get_u32(chunk* chunk, s32 z, s32 y, s32 x);
s32 chunk_get_s32(chunk* chunk, s32 z, s32 y, s32 x);

u64 chunk_get_u64(chunk* chunk, s32 z, s32 y, s32 x);
s64 chunk_get_s64(chunk* chunk, s32 z, s32 y, s32 x);

f16 chunk_get_f16(chunk* chunk, s32 z, s32 y, s32 x);
f32 chunk_get_f32(chunk* chunk, s32 z, s32 y, s32 x);
f64 chunk_get_f64(chunk* chunk, s32 z, s32 y, s32 x);


void chunk_set_u8(chunk* chunk, s32 z, s32 y, s32 x, u8 val);
void chunk_set_s8(chunk* chunk, s32 z, s32 y, s32 x, s8 val);

void chunk_set_u16(chunk* chunk, s32 z, s32 y, s32 x, u16 val);
void chunk_set_s16(chunk* chunk, s32 z, s32 y, s32 x, s16 val);

void chunk_set_u32(chunk* chunk, s32 z, s32 y, s32 x, u32 val);
void chunk_set_s32(chunk* chunk, s32 z, s32 y, s32 x, s32 val);

void chunk_set_u64(chunk* chunk, s32 z, s32 y, s32 x, u64 val);
void chunk_set_s64(chunk* chunk, s32 z, s32 y, s32 x, s64 val);

void chunk_set_f16(chunk* chunk, s32 z, s32 y, s32 x, f16 val);
void chunk_set_f32(chunk* chunk, s32 z, s32 y, s32 x, f32 val);
void chunk_set_f64(chunk* chunk, s32 z, s32 y, s32 x, f64 val);
