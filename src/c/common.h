#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define overload __attribute__((overloadable))
#define auto __auto_type

// For variadic macro support
#define EXPAND(x) x

// Helper macro to count the number of arguments
#define COUNT_ARGS(...) EXPAND(COUNT_ARGS_HELPER(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define COUNT_ARGS_HELPER(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N

// Main die macro
#define die(...) EXPAND(CHOOSE_DIE_MACRO(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__))

#define CHOOSE_DIE_MACRO(N, ...) \
EXPAND(CHOOSE_DIE_MACRO_(N, __VA_ARGS__))
#define CHOOSE_DIE_MACRO_(N, ...) \
EXPAND(DIE_MACRO_##N(__VA_ARGS__))

#define DIE_MACRO_0() do { \
fprintf(stdout, "%s:%s:%d: Error\n", __FILE__, __func__, __LINE__); \
exit(1); \
} while(0)

#define DIE_MACRO_1(msg) do { \
fprintf(stdout, "%s:%s:%d: %s\n", __FILE__, __func__, __LINE__, msg); \
exit(1); \
} while(0)

#define DIE_MACRO_2(msg, a1) do { \
fprintf(stdout, "%s:%s:%d: ", __FILE__, __func__, __LINE__); \
fprintf(stdout, msg, a1); \
fprintf(stdout, "\n"); \
exit(1); \
} while(0)

#define DIE_MACRO_3(msg, a1, a2) do { \
fprintf(stdout, "%s:%s:%d: ", __FILE__, __func__, __LINE__); \
fprintf(stdout, msg, a1, a2); \
fprintf(stdout, "\n"); \
exit(1); \
} while(0)

#define DIE_MACRO_4(msg, a1, a2, a3) do { \
fprintf(stdout, "%s:%s:%d: ", __FILE__, __func__, __LINE__); \
fprintf(stdout, msg, a1, a2, a3); \
fprintf(stdout, "\n"); \
exit(1); \
} while(0)

#define DIE_MACRO_5(msg, a1, a2, a3, a4) do { \
fprintf(stdout, "%s:%s:%d: ", __FILE__, __func__, __LINE__); \
fprintf(stdout, msg, a1, a2, a3, a4); \
fprintf(stdout, "\n"); \
exit(1); \
} while(0)

#define DIE_MACRO_6(msg, a1, a2, a3, a4, a5) do { \
fprintf(stdout, "%s:%s:%d: ", __FILE__, __func__, __LINE__); \
fprintf(stdout, msg, a1, a2, a3, a4, a5); \
fprintf(stdout, "\n"); \
exit(1); \
} while(0)

#define DIE_MACRO_7(msg, a1, a2, a3, a4, a5, a6) do { \
fprintf(stdout, "%s:%s:%d: ", __FILE__, __func__, __LINE__); \
fprintf(stdout, msg, a1, a2, a3, a4, a5, a6); \
fprintf(stdout, "\n"); \
exit(1); \
} while(0)

#define DIE_MACRO_8(msg, a1, a2, a3, a4, a5, a6, a7) do { \
fprintf(stdout, "%s:%s:%d: ", __FILE__, __func__, __LINE__); \
fprintf(stdout, msg, a1, a2, a3, a4, a5, a6, a7); \
fprintf(stdout, "\n"); \
exit(1); \
} while(0)

#define DIE_MACRO_9(msg, a1, a2, a3, a4, a5, a6, a7, a8) do { \
fprintf(stdout, "%s:%s:%d: ", __FILE__, __func__, __LINE__); \
fprintf(stdout, msg, a1, a2, a3, a4, a5, a6, a7, a8); \
fprintf(stdout, "\n"); \
exit(1); \
} while(0)

#define DIE_MACRO_10(msg, a1, a2, a3, a4, a5, a6, a7, a8, a9) do { \
fprintf(stdout, "%s:%s:%d: ", __FILE__, __func__, __LINE__); \
fprintf(stdout, msg, a1, a2, a3, a4, a5, a6, a7, a8, a9); \
fprintf(stdout, "\n"); \
exit(1); \
} while(0)
#define DIE_MACRO_10(msg, a1, a2, a3, a4, a5, a6, a7, a8, a9) do { \
fprintf(stdout, "%s:%s:%d: ", __FILE__, __func__, __LINE__); \
fprintf(stdout, msg, a1, a2, a3, a4, a5, a6, a7, a8, a9); \
fprintf(stdout, "\n"); \
exit(1); \
} while(0)

// die_if macro
#define die_if(cond, ...) \
do { \
if (cond) { \
die(__VA_ARGS__); \
} \
} while(0)


typedef uint8_t u8;
typedef int8_t s8;
typedef uint16_t u16;
typedef int16_t s16;
typedef uint32_t u32;
typedef int32_t s32;
typedef uint64_t u64;
typedef int64_t s64;
typedef _Float16 f16;
typedef float f32;
typedef double f64;

typedef enum dtype { U8, U16, U32, U64, S8, S16, S32, S64, F16, F32, F64 } dtype;


static inline void cast(void* dst, dtype dst_dtype, void* src, dtype src_dtype) {
  switch (dst_dtype) {
  case U8: switch (src_dtype) {
    case U8: *(u8*)dst = *(u8*)src;
      break;
    case S8: *(u8*)dst = *(s8*)src;
      break;

    case U16: *(u8*)dst = *(u16*)src;
      break;
    case S16: *(u8*)dst = *(s16*)src;
      break;

    case U32: *(u8*)dst = *(u32*)src;
      break;
    case S32: *(u8*)dst = *(s32*)src;
      break;

    case U64: *(u8*)dst = *(u64*)src;
      break;
    case S64: *(u8*)dst = *(s64*)src;
      break;

    case F16: *(u8*)dst = *(f16*)src;
      break;
    case F32: *(u8*)dst = *(f32*)src;
      break;
    case F64: *(u8*)dst = *(f64*)src;
      break;
    default: die("invalid dtype");
    }
    break;

  case S8: switch (src_dtype) {
    case U8: *(s8*)dst = *(u8*)src;
      break;
    case S8: *(s8*)dst = *(s8*)src;
      break;

    case U16: *(s8*)dst = *(u16*)src;
      break;
    case S16: *(s8*)dst = *(s16*)src;
      break;

    case U32: *(s8*)dst = *(u32*)src;
      break;
    case S32: *(s8*)dst = *(s32*)src;
      break;

    case U64: *(s8*)dst = *(u64*)src;
      break;
    case S64: *(s8*)dst = *(s64*)src;
      break;

    case F16: *(s8*)dst = *(f16*)src;
      break;
    case F32: *(s8*)dst = *(f32*)src;
      break;
    case F64: *(s8*)dst = *(f64*)src;
      break;
    default: die("invalid dtype");
    }
    break;

  case U16: switch (src_dtype) {
    case U8: *(u16*)dst = *(u8*)src;
      break;
    case S8: *(u16*)dst = *(s8*)src;
      break;

    case U16: *(u16*)dst = *(u16*)src;
      break;
    case S16: *(u16*)dst = *(s16*)src;
      break;

    case U32: *(u16*)dst = *(u32*)src;
      break;
    case S32: *(u16*)dst = *(s32*)src;
      break;

    case U64: *(u16*)dst = *(u64*)src;
      break;
    case S64: *(u16*)dst = *(s64*)src;
      break;

    case F16: *(u16*)dst = *(f16*)src;
      break;
    case F32: *(u16*)dst = *(f32*)src;
      break;
    case F64: *(u16*)dst = *(f64*)src;
      break;
    default: die("invalid dtype");
    }
    break;

  case S16: switch (src_dtype) {
    case U8: *(s16*)dst = *(u8*)src;
      break;
    case S8: *(s16*)dst = *(s8*)src;
      break;

    case U16: *(s16*)dst = *(u16*)src;
      break;
    case S16: *(s16*)dst = *(s16*)src;
      break;

    case U32: *(s16*)dst = *(u32*)src;
      break;
    case S32: *(s16*)dst = *(s32*)src;
      break;

    case U64: *(s16*)dst = *(u64*)src;
      break;
    case S64: *(s16*)dst = *(s64*)src;
      break;

    case F16: *(s16*)dst = *(f16*)src;
      break;
    case F32: *(s16*)dst = *(f32*)src;
      break;
    case F64: *(s16*)dst = *(f64*)src;
      break;
    default: die("invalid dtype");
    }
    break;


  case U32: switch (src_dtype) {
    case U8: *(u32*)dst = *(u8*)src;
      break;
    case S8: *(u32*)dst = *(s8*)src;
      break;

    case U16: *(u32*)dst = *(u16*)src;
      break;
    case S16: *(u32*)dst = *(s16*)src;
      break;

    case U32: *(u32*)dst = *(u32*)src;
      break;
    case S32: *(u32*)dst = *(s32*)src;
      break;

    case U64: *(u32*)dst = *(u64*)src;
      break;
    case S64: *(u32*)dst = *(s64*)src;
      break;

    case F16: *(u32*)dst = *(f16*)src;
      break;
    case F32: *(u32*)dst = *(f32*)src;
      break;
    case F64: *(u32*)dst = *(f64*)src;
      break;
    default: die("invalid dtype");
    }
    break;

  case S32: switch (src_dtype) {
    case U8: *(s32*)dst = *(u8*)src;
      break;
    case S8: *(s32*)dst = *(s8*)src;
      break;

    case U16: *(s32*)dst = *(u16*)src;
      break;
    case S16: *(s32*)dst = *(s16*)src;
      break;

    case U32: *(s32*)dst = *(u32*)src;
      break;
    case S32: *(s32*)dst = *(s32*)src;
      break;

    case U64: *(s32*)dst = *(u64*)src;
      break;
    case S64: *(s32*)dst = *(s64*)src;
      break;

    case F16: *(s32*)dst = *(f16*)src;
      break;
    case F32: *(s32*)dst = *(f32*)src;
      break;
    case F64: *(s32*)dst = *(f64*)src;
      break;
    default: die("invalid dtype");
    }
    break;


  case U64: switch (src_dtype) {
    case U8: *(u64*)dst = *(u8*)src;
      break;
    case S8: *(u64*)dst = *(s8*)src;
      break;

    case U16: *(u64*)dst = *(u16*)src;
      break;
    case S16: *(u64*)dst = *(s16*)src;
      break;

    case U32: *(u64*)dst = *(u32*)src;
      break;
    case S32: *(u64*)dst = *(s32*)src;
      break;

    case U64: *(u64*)dst = *(u64*)src;
      break;
    case S64: *(u64*)dst = *(s64*)src;
      break;

    case F16: *(u64*)dst = *(f16*)src;
      break;
    case F32: *(u64*)dst = *(f32*)src;
      break;
    case F64: *(u64*)dst = *(f64*)src;
      break;
    default: die("invalid dtype");
    }
    break;

  case S64: switch (src_dtype) {
    case U8: *(s64*)dst = *(u8*)src;
      break;
    case S8: *(s64*)dst = *(s8*)src;
      break;

    case U16: *(s64*)dst = *(u16*)src;
      break;
    case S16: *(s64*)dst = *(s16*)src;
      break;

    case U32: *(s64*)dst = *(u32*)src;
      break;
    case S32: *(s64*)dst = *(s32*)src;
      break;

    case U64: *(s64*)dst = *(u64*)src;
      break;
    case S64: *(s64*)dst = *(s64*)src;
      break;

    case F16: *(s64*)dst = *(f16*)src;
      break;
    case F32: *(s64*)dst = *(f32*)src;
      break;
    case F64: *(s64*)dst = *(f64*)src;
      break;
    default: die("invalid dtype");
    }
    break;

  case F16: switch (src_dtype) {
    case U8: *(f16*)dst = *(u8*)src;
      break;
    case S8: *(f16*)dst = *(s8*)src;
      break;

    case U16: *(f16*)dst = *(u16*)src;
      break;
    case S16: *(f16*)dst = *(s16*)src;
      break;

    case U32: *(f16*)dst = *(u32*)src;
      break;
    case S32: *(f16*)dst = *(s32*)src;
      break;

    case U64: *(f16*)dst = *(u64*)src;
      break;
    case S64: *(f16*)dst = *(s64*)src;
      break;

    case F16: *(f16*)dst = *(f16*)src;
      break;
    case F32: *(f16*)dst = *(f32*)src;
      break;
    case F64: *(f16*)dst = *(f64*)src;
      break;
    default: die("invalid dtype");
    }
    break;

  case F32: switch (src_dtype) {
    case U8: *(f32*)dst = *(u8*)src;
      break;
    case S8: *(f32*)dst = *(s8*)src;
      break;

    case U16: *(f32*)dst = *(u16*)src;
      break;
    case S16: *(f32*)dst = *(s16*)src;
      break;

    case U32: *(f32*)dst = *(u32*)src;
      break;
    case S32: *(f32*)dst = *(s32*)src;
      break;

    case U64: *(f32*)dst = *(u64*)src;
      break;
    case S64: *(f32*)dst = *(s64*)src;
      break;

    case F16: *(f32*)dst = *(f16*)src;
      break;
    case F32: *(f32*)dst = *(f32*)src;
      break;
    case F64: *(f32*)dst = *(f64*)src;
      break;
    default: die("invalid dtype");
    }
    break;

  case F64: switch (src_dtype) {
    case U8: *(f64*)dst = *(u8*)src;
      break;
    case S8: *(f64*)dst = *(s8*)src;
      break;

    case U16: *(f64*)dst = *(u16*)src;
      break;
    case S16: *(f64*)dst = *(s16*)src;
      break;

    case U32: *(f64*)dst = *(u32*)src;
      break;
    case S32: *(f64*)dst = *(s32*)src;
      break;

    case U64: *(f64*)dst = *(u64*)src;
      break;
    case S64: *(f64*)dst = *(s64*)src;
      break;

    case F16: *(f64*)dst = *(f16*)src;
      break;
    case F32: *(f64*)dst = *(f32*)src;
      break;
    case F64: *(f64*)dst = *(f64*)src;
      break;
    default: die("invalid dtype");
    }
    break;
  }
  //printf("asdf\n");
}
