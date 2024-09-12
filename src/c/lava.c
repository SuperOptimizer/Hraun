#include <stdint.h>
#include <stdlib.h>

#include "common.h"
#include "lava.h"
#include "cvol.h"


static s32 get_index(s32 z, s32 y, s32 x) { return z / 2 * 4 + y / 2 * 2 + x / 2; }

static u8 get_bit_position(s32 z, s32 y, s32 x) { return z % 2 * 4 + y % 2 * 2 + x % 2; }

bitmask3d bitmask3d_new(s32 depth, s32 height, s32 width) {
  s32 total_bytes = (depth + 1) / 2 * ((height + 1) / 2) * ((width + 1) / 2);
  return (bitmask3d){
    .data = calloc(total_bytes, sizeof(u8)),
    .depth = depth,
    .height = height,
    .width = width
  };
}

void bitmask3d_free(bitmask3d* mask) {
  free(mask->data);
  mask->data = NULL;
  mask->depth = mask->height = mask->width = 0;
}

bool bitmask3d_get(bitmask3d* mask, s32 z, s32 y, s32 x) {
  if (z < 0 || z >= mask->depth || y < 0 || y >= mask->height || x < 0 || x >= mask->width) {
    die("get: invalid index");
    return false;
  }

  s32 byte_index = get_index(z, y, x);
  u8 bit_position = get_bit_position(z, y, x);

  return (mask->data[byte_index] & (1 << bit_position)) != 0;
}

void bitmask3d_set(bitmask3d* mask, s32 z, s32 y, s32 x, bool value) {
  if (z < 0 || z >= mask->depth || y < 0 || y >= mask->height || x < 0 || x >= mask->width) {
    die("set: invalid index");
    return;
  }

  s32 byte_index = get_index(z, y, x);
  u8 bit_position = get_bit_position(z, y, x);

  if (value) { mask->data[byte_index] |= (1 << bit_position); }
  else { mask->data[byte_index] &= ~(1 << bit_position); }
}

void bitmask3d_print(bitmask3d* mask) {
  for (s32 z = 0; z < mask->depth; z++) {
    printf("Layer %d:\n", z);
    for (s32 y = 0; y < mask->height; y++) {
      for (s32 x = 0; x < mask->width; x++) { printf("%d", bitmask3d_get(mask, z, y, x) ? 1 : 0); }
      printf("\n");
    }
    printf("\n");
  }
}

bitmask3d isomask(chunk* chunk, u8 iso) {
  assert(chunk->dtype == U8);
  auto ret = bitmask3d_new(chunk->depth, chunk->height, chunk->width);
  for (s32 z = 0; z < chunk->depth; z++)
    for (s32 y = 0; y < chunk->height; y++)
      for (s32 x = 0; x < chunk->width; x++) {
        u8 data;
        chunk_get(chunk, z, y, x, &data);
        bitmask3d_set(&ret, z, y, x, data > iso);
      }
  return ret;
}
