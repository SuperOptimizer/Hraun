#pragma once

#include "common.h"
#include "lava.h"
#include "chunk.h"

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
// to compress chunks, and zero out an amount of low bits to reduce file size.
typedef struct cvol
{
  s32 depth, height, width;
  Mmap* slices;
} cvol;


cvol cvol_new(char* path, s32 depth, s32 height, s32 width);
void cvol_del(cvol* cvol);
cvol cvol_open(char* cvoldir, s32 depth, s32 height, s32 width);
chunk cvol_chunk(cvol* cvol, s32 z, s32 y, s32 x, s32 zlen, s32 ylen, s32 xlen);

void chunk_free(chunk* chunk);
chunk chunk_cast(chunk* chunk, dtype dtype);
chunk chunk_new(dtype dtype, s32 depth, s32 height, s32 width);


void chunk_get(chunk* chunk, s32 z, s32 y, s32 x, void* val);
void chunk_set(chunk* chunk, s32 z, s32 y, s32 x, void* val);