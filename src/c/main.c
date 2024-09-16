#include "lava.h"

#include <stdio.h>

#include "lava.h"

#ifdef  __linux__
#define CVOLPATH "/mnt/d/vesuvius.cvol"
#else
#define CVOLPATH "D:/vesuvius.cvol/"
#endif

int main(int argc, char** argv) {
  cvol cvol = cvol_open(CVOLPATH, 13091, 7888, 8096);
  chunk asdf = cvol_chunk(&cvol, 2048, 2048, 2048, 64, 64, 64);
  chunk f32asdf = chunk_cast(&asdf, F32);
  printf("%d\n", chunk_get_u8(&asdf, 0, 0, 0));
  printf("%f\n", chunk_get_f32(&f32asdf, 0, 0, 0));
  printf("Hello World!\n");
  return 0;
}

