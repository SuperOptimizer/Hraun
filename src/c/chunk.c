
#include "common.h"
#include "chunk.h"

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