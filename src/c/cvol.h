#pragma once

#include "common.h"
#include "lava.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>


#define SLICE_SIZE 8192

typedef struct
{
  u8* view;
  HANDLE file_handle;
  HANDLE mapping_handle;
} Mmap;
#else
typedef struct
{
  u8* view;
} Mmap;
#endif


// a cvol is a compressed volume of scroll data. we rely on filesystem level compression
// to compress chunks, and zero out an amount of low bits to reduce file size. chunks are 256 bytes in all directions
// a cvol can store up to 65535x65535x65535 voxels
typedef struct cvol
{
  s32 depth, height, width;
  Mmap* slices;
} cvol;

typedef struct chunk
{
  dtype dtype;
  s32 depth,height,width;
  void* data;
}chunk;


cvol cvol_new(char* path, s32 depth, s32 height, s32 width);
void cvol_del(cvol* cvol);
cvol cvol_open(char* cvoldir, s32 depth, s32 height, s32 width);
chunk cvol_chunk(cvol* cvol, s32 z, s32 y, s32 x, s32 zlen, s32 ylen, s32 xlen);

void chunk_free(chunk* chunk);
chunk chunk_cast(chunk* chunk, dtype dtype);
chunk chunk_new(dtype dtype, s32 depth, s32 height, s32 width);

u8 chunk_get_u8(chunk* chunk, s32 z, s32 y, s32 x);
void chunk_set_u8(chunk* chunk, s32 z, s32 y, s32 x, u8 val);
s8 chunk_get_s8(chunk* chunk, s32 z, s32 y, s32 x);
void chunk_set_s8(chunk* chunk, s32 z, s32 y, s32 x, s8 val);
u16 chunk_get_u16(chunk* chunk, s32 z, s32 y, s32 x);
void chunk_set_u16(chunk* chunk, s32 z, s32 y, s32 x, u16 val);
s16 chunk_get_s16(chunk* chunk, s32 z, s32 y, s32 x);
void chunk_set_s16(chunk* chunk, s32 z, s32 y, s32 x, s16 val);
u32 chunk_get_u32(chunk* chunk, s32 z, s32 y, s32 x);
void chunk_set_u32(chunk* chunk, s32 z, s32 y, s32 x, u32 val);
s32 chunk_get_s32(chunk* chunk, s32 z, s32 y, s32 x);
void chunk_set_s32(chunk* chunk, s32 z, s32 y, s32 x, s32 val);
u64 chunk_get_u64(chunk* chunk, s32 z, s32 y, s32 x);
void chunk_set_u64(chunk* chunk, s32 z, s32 y, s32 x, u64 val);
s64 chunk_get_s64(chunk* chunk, s32 z, s32 y, s32 x);
void chunk_set_s64(chunk* chunk, s32 z, s32 y, s32 x, s64 val);
f16 chunk_get_f16(chunk* chunk, s32 z, s32 y, s32 x);
void chunk_set_f16(chunk* chunk, s32 z, s32 y, s32 x, f16 val);
f32 chunk_get_f32(chunk* chunk, s32 z, s32 y, s32 x);
void chunk_set_f32(chunk* chunk, s32 z, s32 y, s32 x, f32 val);
f64 chunk_get_f64(chunk* chunk, s32 z, s32 y, s32 x);
void chunk_set_f64(chunk* chunk, s32 z, s32 y, s32 x, f64 val);