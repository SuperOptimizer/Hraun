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

typedef enum dtype { U8, F32 } dtype;

static inline u8 maxu8(u8 a, u8 b) { return a > b ? a : b; }
static inline f32 maxf32(f32 a, f32 b) { return a > b ? a : b; }

static inline u8 minu8(u8 a, u8 b) { return a < b ? a : b; }
static inline f32 minf32(f32 a, f32 b) { return a < b ? a : b; }

static inline u8  avgu8(u8* data, s32 len) { u32 sum = 0; for(int i = 0; i < len; i++)sum+= data[i]; return sum / len; }
static inline f32 avgf32(u8* data, s32 len) { f64 sum = 0; for(int i = 0; i < len; i++)sum+= data[i]; return sum / len; }

