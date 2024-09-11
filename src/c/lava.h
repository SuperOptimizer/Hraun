#pragma once

#include "common.h"


typedef struct bitmask3d
{
  u8* data;
  s32 depth, height, width;
} bitmask3d;

bitmask3d bitmask3d_new(s32 depth, s32 height, s32 width);
void bitmask3d_free(bitmask3d* mask);
bool bitmask3d_get(bitmask3d* mask, s32 z, s32 y, s32 x);
void bitmask3d_set(bitmask3d* mask, s32 z, s32 y, s32 x, bool value);
void bitmask3d_print(bitmask3d* mask);