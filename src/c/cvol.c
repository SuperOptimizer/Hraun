#include <assert.h>

#include "common.h"
#include "cvol.h"


#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

Mmap mmap_(const char* filename)
{
    Mmap mmap = {0};

    DWORD access = GENERIC_READ;
    DWORD protect = PAGE_READONLY;
    DWORD map_access = FILE_MAP_READ;
    DWORD creation_disposition = OPEN_EXISTING;

    //if (rw)
    //{
    //    access |= GENERIC_WRITE;
    //    protect = PAGE_READWRITE;
    //    map_access |= FILE_MAP_WRITE;
    //    creation_disposition = OPEN_ALWAYS;
    //}

    mmap.file_handle = CreateFileA(filename, access, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                                   creation_disposition, FILE_ATTRIBUTE_NORMAL, NULL);
    if (mmap.file_handle == INVALID_HANDLE_VALUE)
    {
        die("failed to get file handle");
    }

    DWORD file_size = GetFileSize(mmap.file_handle, NULL);
    DWORD last_error = GetLastError();

    if (file_size == INVALID_FILE_SIZE && last_error != NO_ERROR)
    {
        die("failed to get file size");
    }

    if (file_size == 0)
    {
        LARGE_INTEGER li_file_size;
        li_file_size.QuadPart = SLICE_SIZE*SLICE_SIZE;

        if (!SetFilePointerEx(mmap.file_handle, li_file_size, NULL, FILE_BEGIN))
        {
            die("failed to set file pointer");
        }

        if (!SetEndOfFile(mmap.file_handle))
        {
            die("failed to set end of file");
        }

        DWORD bytes_written;
        char buffer[1024*1024] = {0};
        DWORD remaining = SLICE_SIZE*SLICE_SIZE;

        SetFilePointer(mmap.file_handle, 0, NULL, FILE_BEGIN);

        while (remaining > 0)
        {
            DWORD to_write = (remaining < 1024*1024) ? remaining : 1024*1024;
            if (!WriteFile(mmap.file_handle, buffer, to_write, &bytes_written, NULL) || bytes_written != to_write)
            {
                die("failed to initialize file with zeros");
            }
            remaining -= bytes_written;
        }

        printf("Created new slice file: %s\n", filename);
    }

    mmap.mapping_handle = CreateFileMappingA(mmap.file_handle, NULL, protect, 0, 0, NULL);
    if (!mmap.mapping_handle)
    {
        die("failed to get file mapping");
    }

    mmap.view = MapViewOfFile(mmap.mapping_handle, map_access, 0, 0, 0);
    if (!mmap.view)
    {
        die("failed to get file view");
    }

    return mmap;
}

Mmap munmap_(Mmap mmap)
{
  if(mmap.view) {UnmapViewOfFile(mmap.view); mmap.view = NULL;}
  if(mmap.mapping_handle) {CloseHandle(mmap.mapping_handle); mmap.mapping_handle = INVALID_HANDLE_VALUE;}
  if(mmap.file_handle) {CloseHandle(mmap.file_handle); mmap.file_handle = INVALID_HANDLE_VALUE;}
  return mmap;
}

#else

Mmap* mmap_(const char* filename)
{
  return
}

#endif


cvol cvol_new(char* path, s32 depth, s32 height, s32 width)
{
  Mmap* slices = malloc(depth * sizeof(Mmap));
  for (int z = 0; z < depth; z++){
    char fullpath[1024] = {'\0'};
    sprintf(fullpath, "%s/%05d.slice", path, z);
    slices[z] = mmap_(fullpath);
  }
  return (cvol){.depth = depth, .height = height, .width = width, .slices = slices};;
}


void cvol_del(cvol* cvol)
{
  for (int z = 0; z < cvol->depth; z++)
    cvol->slices[z] = munmap_(cvol->slices[z]);
  free(cvol->slices);
}

u8 cvol_get(cvol* cvol, s32 z, s32 y, s32 x)
{
  return cvol->slices[z].view[y * SLICE_SIZE + x];
}

void cvol_set(cvol* cvol, s32 z, s32 y, s32 x, u8 val)
{
  cvol->slices[z].view[y* SLICE_SIZE + x] = val;
}

cvol cvol_open(char* cvoldir, s32 depth, s32 height, s32 width)
{
  cvol cvol = cvol_new(cvoldir, depth, height, width);

  return cvol;
}


chunk cvol_chunk(cvol* cvol, s32 z, s32 y, s32 x, s32 zlen, s32 ylen, s32 xlen)
{
  assert(z + zlen-1 < cvol->depth);
  assert(y + ylen-1 < cvol->height);
  assert(x + xlen-1 < cvol->width);
  assert(zlen * ylen * xlen <= 1024*1024*1024);

  auto ret = chunk_new(U8,zlen,ylen,xlen);
  auto set = chunk_set_u8;

  for(int zi = 0; zi < zlen; zi++)
  {
    for(int yi = 0; yi < ylen; yi++)
    {
      for(int xi = 0; xi < xlen; xi++)
      {
        set(&ret, zi, yi, xi, cvol_get(cvol, zi + z, yi + y, xi + x));
      }
    }
  }
  return ret;
}


chunk chunk_new(dtype dtype, s32 depth, s32 height, s32 width)
{
  s32 datasize = 1;
  switch (dtype)
  {
  case U8: case S8: datasize = 1; break;
  case U16: case S16: case F16: datasize = 2; break;
  case U32: case S32: case F32: datasize = 4; break;
  case U64: case S64: case F64: datasize = 8; break;
  }
  uint8_t* data = malloc(depth * height * width * datasize);
  return (chunk){dtype, depth, height, width, data};
}

void chunk_free(chunk* chunk)
{
  free(chunk->data);
}

static void* chunk_getptr(chunk* chunk, s32 z, s32 y, s32 x)
{
  switch(chunk->dtype)
  {
  case U8: case S8:             return (u8*) &chunk->data[z * chunk->width * chunk->height + y * chunk->width + x]; break;
  case U16: case S16: case F16: return (u16*)&chunk->data[z * chunk->width * chunk->height + y * chunk->width + x]; break;
  case U32: case S32: case F32: return (u32*)&chunk->data[z * chunk->width * chunk->height + y * chunk->width + x]; break;
  case U64: case S64: case F64: return (u64*)&chunk->data[z * chunk->width * chunk->height + y * chunk->width + x]; break;
  }
  die("bad dtype");
}


void chunk_get(chunk* chunk, s32 z, s32 y, s32 x, void* val){
  switch(chunk->dtype)
  {
  case U8: case S8:             *(u8*)val = *(u8*)chunk_getptr(chunk, z, y, x); break;
  case U16: case S16: case F16: *(u16*)val = *(u16*)chunk_getptr(chunk, z, y, x); break;
  case U32: case S32: case F32: *(u32*)val = *(u32*)chunk_getptr(chunk, z, y, x); break;
  case U64: case S64: case F64: *(u64*)val = *(u64*)chunk_getptr(chunk, z, y, x); break;
  }
}

void chunk_set(chunk* chunk, s32 z, s32 y, s32 x, void* val){
  switch(chunk->dtype)
  {
  case U8: case S8:             *(u8*)chunk_getptr(chunk, z, y, x) = *(u8*)val; break;
  case U16: case S16: case F16: *(u16*)chunk_getptr(chunk, z, y, x) = *(u16*)val; break;
  case U32: case S32: case F32: *(u32*)chunk_getptr(chunk, z, y, x) = *(u32*)val; break;
  case U64: case S64: case F64: *(u64*)chunk_getptr(chunk, z, y, x) = *(u64*)val; break;
  }
}


chunk chunk_cast(chunk* chunk, dtype dtype){
  auto ret = chunk_new(dtype, chunk->depth, chunk->height, chunk->width);

  for(s32 z = 0; z < chunk->depth; z++)
    for(s32 y = 0; y < chunk->height; y++)
      for(s32 x = 0; x < chunk->width; x++)
      {
        void* dst = chunk_getptr(&ret, z, y, x);
        void* src = chunk_getptr(chunk, z, y, x);
        cast(dst, dtype, src, chunk->dtype);
      }
  return ret;
}