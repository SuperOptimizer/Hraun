
#include "lava.h"

#include <stdio.h>

#include "common.h"
#include "cvol.h"

#define TIFFPATH "D:/dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_masked/20230205180739/"
#define CVOLPATH "D:/vesuvius.cvol/"

int main(int argc, char** argv)
{
  cvol cvol = cvol_open(CVOLPATH,13091,7888,8096);
  chunk asdf = cvol_chunk(&cvol,2048,2048,2048,256,256,256);
  bitmask3d bitmask3d = bitmask3d_new(256,256,256);
  bitmask3d_get(&bitmask3d, 128,128,128);
  printf("%d\n",chunk_get(&asdf,0,0,0));
  printf("Hello World!\n");
  return 0;
}
