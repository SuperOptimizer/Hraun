#include "lava.h"

#include <stdio.h>

#include "common.h"
#include "cvol.h"

#define TIFFPATH "D:/dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_masked/20230205180739/"
#define CVOLPATH "D:/vesuvius.cvol/"

int main(int argc, char** argv) {
  cvol cvol = cvol_open(CVOLPATH, 13091, 7888, 8096);
  chunk asdf = cvol_chunk(&cvol, 2048, 2048, 2048, 256, 256, 256);
  chunk f32asdf = chunk_cast(&asdf, F32);
  printf("%d\n", chunk_get_u8(&asdf, 0, 0, 0));
  printf("%f\n", chunk_get_f32(&f32asdf, 0, 0, 0));
  printf("Hello World!\n");
  return 0;
}

