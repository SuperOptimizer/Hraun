#pragma once

#include "common.h"
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


static  cvol cvol_new(char* path, s32 depth, s32 height, s32 width);
static  void cvol_del(cvol* cvol);
static  cvol cvol_open(char* cvoldir, s32 depth, s32 height, s32 width);
static  chunk cvol_chunk(cvol* cvol, s32 z, s32 y, s32 x, s32 zlen, s32 ylen, s32 xlen);



#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static  void mmap_(const char* filename, Mmap* out) {
  DWORD access = GENERIC_READ;
  DWORD protect = PAGE_READONLY;
  DWORD map_access = FILE_MAP_READ;
  DWORD creation_disposition = OPEN_EXISTING;

  out->file_handle = CreateFileA(filename, access, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                                 creation_disposition, FILE_ATTRIBUTE_NORMAL, NULL);
  if (out->file_handle == INVALID_HANDLE_VALUE) { die("failed to get file handle"); }
  DWORD file_size = GetFileSize(out->file_handle, NULL);
  DWORD last_error = GetLastError();

  if (file_size == INVALID_FILE_SIZE && last_error != NO_ERROR) { die("failed to get file size"); }

  die_if(file_size == 0, "failed to get file size");

  out->mapping_handle = CreateFileMappingA(out->file_handle, NULL, protect, 0, 0, NULL);
  if (!out->mapping_handle) { die("failed to get file mapping"); }

  out->view = MapViewOfFile(out->mapping_handle, map_access, 0, 0, 0);
  if (!out->view) { die("failed to get file view"); }
}

static  Mmap munmap_(Mmap mmap) {
  if (mmap.view) {
    UnmapViewOfFile(mmap.view);
    mmap.view = NULL;
  }
  if (mmap.mapping_handle) {
    CloseHandle(mmap.mapping_handle);
    mmap.mapping_handle = INVALID_HANDLE_VALUE;
  }
  if (mmap.file_handle) {
    CloseHandle(mmap.file_handle);
    mmap.file_handle = INVALID_HANDLE_VALUE;
  }
  return mmap;
}

#else

Mmap* mmap_(const char* filename)
{
  return
}

#endif


static  cvol cvol_new(char* path, s32 depth, s32 height, s32 width) {
  Mmap* slices = calloc(depth * sizeof(Mmap),1);
  static char fullpath[512] = {'\0'};
  for (int z = 0; z < depth; z++) {
    sprintf(fullpath, "%s/%05d.slice", path, z);
    mmap_(fullpath, &slices[z]);
  }
  printf("done mapping\n");
  return (cvol){.depth = depth, .height = height, .width = width, .slices = slices};
}


static  void cvol_del(cvol* cvol) {
  for (int z = 0; z < cvol->depth; z++)
    cvol->slices[z] = munmap_(cvol->slices[z]);
  free(cvol->slices);
}

static  u8 cvol_get(cvol* cvol, s32 z, s32 y, s32 x) { return cvol->slices[z].view[y * SLICE_SIZE + x]; }

static  void cvol_set(cvol* cvol, s32 z, s32 y, s32 x, u8 val) { cvol->slices[z].view[y * SLICE_SIZE + x] = val; }

static  cvol cvol_open(char* cvoldir, s32 depth, s32 height, s32 width) {
  cvol cvol = cvol_new(cvoldir, depth, height, width);

  return cvol;
}


static  chunk cvol_chunk(cvol* cvol, s32 z, s32 y, s32 x, s32 zlen, s32 ylen, s32 xlen) {
  assert(z + zlen-1 < cvol->depth);
  assert(y + ylen-1 < cvol->height);
  assert(x + xlen-1 < cvol->width);
  assert(zlen * ylen * xlen <= 1024*1024*1024);

  auto ret = chunk_new(U8, zlen, ylen, xlen);

  for (int zi = 0; zi < zlen; zi++) {
    for (int yi = 0; yi < ylen; yi++) {
      for (int xi = 0; xi < xlen; xi++) {
        u8 data = cvol_get(cvol, zi + z, yi + y, xi + x);
        chunk_set_u8(&ret, zi, yi, xi, data);
      }
    }
  }
  return ret;
}
