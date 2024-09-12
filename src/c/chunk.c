#include "common.h"
#include "chunk.h"

chunk chunk_new(dtype dtype, s32 depth, s32 height, s32 width) {
  s32 datasize = 1;
  switch (dtype) {
  case U8:datasize = 1; break;
  case F32: datasize = 4; break;
  }
  uint8_t* data = malloc(depth * height * width * datasize);
  return (chunk){dtype, depth, height, width, data};
}

void chunk_free(chunk* chunk) { free(chunk->data); }

static void* chunk_getptr(chunk* chunk, s32 z, s32 y, s32 x) {
  size_t index = z * chunk->width * chunk->height + y * chunk->width + x;
  size_t byte_offset;

  switch (chunk->dtype) {
  case U8: byte_offset = index * sizeof(u8); break;
  case F32: byte_offset = index * sizeof(f32); break;
  default: die("bad dtype");
  }

  return (u8*)chunk->data + byte_offset;
}

chunk chunk_cast(chunk* chunk, dtype dtype) {
  auto ret = chunk_new(dtype, chunk->depth, chunk->height, chunk->width);

  for (s32 z = 0; z < chunk->depth; z++)
    for (s32 y = 0; y < chunk->height; y++)
      for (s32 x = 0; x < chunk->width; x++) {
        if(chunk->dtype == U8) {
          if(dtype == U8) {
            chunk_set_u8(&ret,z,y,x,chunk_get_u8(chunk,z,y,x));
          } else {
            chunk_set_f32(&ret,z,y,x,chunk_get_u8(chunk,z,y,x));
          }
        } else {
          if(dtype == U8) {
            chunk_set_u8(&ret,z,y,x,(u8)chunk_get_f32(chunk,z,y,x));
          } else {
            chunk_set_f32(&ret,z,y,x,(u8)chunk_get_f32(chunk,z,y,x));
          }
        }
      }
  return ret;
}

u8 chunk_get_u8(chunk* chunk, s32 z, s32 y, s32 x) {
  assert(chunk->dtype == U8);
  return *(u8*)chunk_getptr(chunk, z, y, x);
}

f32 chunk_get_f32(chunk* chunk, s32 z, s32 y, s32 x) {
  assert(chunk->dtype == F32);
  return *(f32*)chunk_getptr(chunk, z, y, x);
}

void chunk_set_u8(chunk* chunk, s32 z, s32 y, s32 x, u8 val) {
  assert(chunk->dtype == U8);
  *(u8*)chunk_getptr(chunk, z, y, x) = val;
}

void chunk_set_f32(chunk* chunk, s32 z, s32 y, s32 x, f32 val) {
  assert(chunk->dtype == F32);
  *(f32*)chunk_getptr(chunk, z, y, x) = val;
}
