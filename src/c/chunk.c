
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
  size_t index = z * chunk->width * chunk->height + y * chunk->width + x;
  size_t byte_offset;

  switch(chunk->dtype)
  {
  case U8:  byte_offset = index * sizeof(u8);  break;
  case S8:  byte_offset = index * sizeof(s8);  break;
  case U16: byte_offset = index * sizeof(u16); break;
  case S16: byte_offset = index * sizeof(s16); break;
  case U32: byte_offset = index * sizeof(u32); break;
  case S32: byte_offset = index * sizeof(s32); break;
  case U64: byte_offset = index * sizeof(u64); break;
  case S64: byte_offset = index * sizeof(s64); break;
  case F16: byte_offset = index * sizeof(f16); break;
  case F32: byte_offset = index * sizeof(f32); break;
  case F64: byte_offset = index * sizeof(f64); break;
  default: die("bad dtype");
  }

  return (u8*)chunk->data + byte_offset;
}

void chunk_get(chunk* chunk, s32 z, s32 y, s32 x, void* val){
  switch(chunk->dtype)
  {
  case U8: *(u8*)val =  *(u8*)chunk_getptr(chunk, z, y, x); break;
  case S8: *(s8*)val =  *(s8*)chunk_getptr(chunk, z, y, x); break;
  case U16: *(u16*)val = *(u16*)chunk_getptr(chunk, z, y, x); break;
  case S16: *(s16*)val = *(s16*)chunk_getptr(chunk, z, y, x); break;
  case U32: *(u32*)val = *(u32*)chunk_getptr(chunk, z, y, x); break;
  case S32: *(s32*)val = *(s32*)chunk_getptr(chunk, z, y, x); break;
  case U64: *(u64*)val = *(u64*)chunk_getptr(chunk, z, y, x); break;
  case S64: *(s64*)val = *(s64*)chunk_getptr(chunk, z, y, x); break;
  case F16: *(f16*)val = *(f16*)chunk_getptr(chunk, z, y, x); break;
  case F32: *(f32*)val = *(f32*)chunk_getptr(chunk, z, y, x); break;
  case F64: *(f64*)val = *(f64*)chunk_getptr(chunk, z, y, x); break;
  }
}

void chunk_set(chunk* chunk, s32 z, s32 y, s32 x, void* val){
  switch(chunk->dtype)
  {
  case U8: *(u8*)chunk_getptr(chunk, z, y, x) = *(u8*)val; break;
  case S8: *(s8*)chunk_getptr(chunk, z, y, x) = *(s8*)val; break;
  case U16: *(u16*)chunk_getptr(chunk, z, y, x) = *(u16*)val; break;
  case S16: *(s16*)chunk_getptr(chunk, z, y, x) = *(s16*)val; break;
  case U32: *(u32*)chunk_getptr(chunk, z, y, x) = *(u32*)val; break;
  case S32: *(s32*)chunk_getptr(chunk, z, y, x) = *(s32*)val; break;
  case U64: *(u64*)chunk_getptr(chunk, z, y, x) = *(u64*)val; break;
  case S64: *(s64*)chunk_getptr(chunk, z, y, x) = *(s64*)val; break;
  case F16: *(f16*)chunk_getptr(chunk, z, y, x) = *(f16*)val; break;
  case F32: *(f32*)chunk_getptr(chunk, z, y, x) = *(f32*)val; break;
  case F64: *(f64*)chunk_getptr(chunk, z, y, x) = *(f64*)val; break;
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
        //printf("%d %f\n",*(u8*)src, *(f32*)dst);
      }
  return ret;
}


u8 chunk_get_u8(chunk* chunk, s32 z, s32 y, s32 x) {
  u8 ret;
  assert(chunk->dtype == U8);
  chunk_get(chunk,z,y,x,&ret);
  return ret;
}
s8 chunk_get_s8(chunk* chunk, s32 z, s32 y, s32 x) {
  s8 ret;
  assert(chunk->dtype == S8);
  chunk_get(chunk,z,y,x,&ret);
  return ret;
}

u16 chunk_get_u16(chunk* chunk, s32 z, s32 y, s32 x) {
  u16 ret;
  assert(chunk->dtype == U16);
  chunk_get(chunk,z,y,x,&ret);
  return ret;
}
s16 chunk_get_s16(chunk* chunk, s32 z, s32 y, s32 x) {
  s16 ret;
  assert(chunk->dtype == S16);
  chunk_get(chunk,z,y,x,&ret);
  return ret;
}

u32 chunk_get_u32(chunk* chunk, s32 z, s32 y, s32 x) {
  u32 ret;
  assert(chunk->dtype == U32);
  chunk_get(chunk,z,y,x,&ret);
  return ret;
}
s32 chunk_get_s32(chunk* chunk, s32 z, s32 y, s32 x) {
  s32 ret;
  assert(chunk->dtype == S32);
  chunk_get(chunk,z,y,x,&ret);
  return ret;
}

u64 chunk_get_u64(chunk* chunk, s32 z, s32 y, s32 x) {
  u64 ret;
  assert(chunk->dtype == U64);
  chunk_get(chunk,z,y,x,&ret);
  return ret;
}
s64 chunk_get_s64(chunk* chunk, s32 z, s32 y, s32 x) {
  s64 ret;
  assert(chunk->dtype == S64);
  chunk_get(chunk,z,y,x,&ret);
  return ret;
}

f16 chunk_get_f16(chunk* chunk, s32 z, s32 y, s32 x) {
  f16 ret;
  assert(chunk->dtype == F16);
  chunk_get(chunk,z,y,x,&ret);
  return ret;
}
f32 chunk_get_f32(chunk* chunk, s32 z, s32 y, s32 x) {
  f32 ret;
  assert(chunk->dtype == F32);
  chunk_get(chunk,z,y,x,&ret);
  return ret;
}
f64 chunk_get_f64(chunk* chunk, s32 z, s32 y, s32 x) {
  f64 ret;
  assert(chunk->dtype == F64);
  chunk_get(chunk,z,y,x,&ret);
  return ret;
}


void chunk_set_u8(chunk* chunk, s32 z, s32 y, s32 x, u8 val){
  assert(chunk->dtype == U8);
  chunk_set(chunk,z,y,x,&val);
}
void chunk_set_s8(chunk* chunk, s32 z, s32 y, s32 x, s8 val){
  assert(chunk->dtype == S8);
  chunk_set(chunk,z,y,x,&val);
}

void chunk_set_u16(chunk* chunk, s32 z, s32 y, s32 x, u16 val){
  assert(chunk->dtype == U16);
  chunk_set(chunk,z,y,x,&val);
}
void chunk_set_s16(chunk* chunk, s32 z, s32 y, s32 x, s16 val){
  assert(chunk->dtype == S16);
  chunk_set(chunk,z,y,x,&val);
}

void chunk_set_u32(chunk* chunk, s32 z, s32 y, s32 x, u32 val){
  assert(chunk->dtype == U32);
  chunk_set(chunk,z,y,x,&val);
}
void chunk_set_s32(chunk* chunk, s32 z, s32 y, s32 x, s32 val){
  assert(chunk->dtype == S32);
  chunk_set(chunk,z,y,x,&val);
}

void chunk_set_u64(chunk* chunk, s32 z, s32 y, s32 x, u64 val){
  assert(chunk->dtype == U64);
  chunk_set(chunk,z,y,x,&val);
}
void chunk_set_s64(chunk* chunk, s32 z, s32 y, s32 x, s64 val){
  assert(chunk->dtype == S64);
  chunk_set(chunk,z,y,x,&val);
}

void chunk_set_f16(chunk* chunk, s32 z, s32 y, s32 x, f16 val){
  assert(chunk->dtype == F16);
  chunk_set(chunk,z,y,x,&val);
}
void chunk_set_f32(chunk* chunk, s32 z, s32 y, s32 x, f32 val){
  assert(chunk->dtype == F32);
  chunk_set(chunk,z,y,x,&val);
}
void chunk_set_f64(chunk* chunk, s32 z, s32 y, s32 x, f64 val){
  assert(chunk->dtype == F64);
  chunk_set(chunk,z,y,x,&val);
}