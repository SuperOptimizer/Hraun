#include "common.h"
#include "chunk.h"
#include "cvol.h"
#include "lava.h"

#define TIFFPATH "D:/dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_masked/20230205180739/"
#define CVOLPATH "D:/vesuvius.cvol/"

int main(int argc, char** argv) {
  cvol cvol = cvol_open(CVOLPATH, 13091, 7888, 8096);
  chunk asdf = cvol_chunk(&cvol, 2048, 2048, 2048, 128, 128, 128);
  chunk pooled = maxpool(&asdf,2,2);
  u8 data[8];
  data[0] = chunk_get_u8(&asdf,0,0,0);
  data[1] = chunk_get_u8(&asdf,0,0,1);
  data[2] = chunk_get_u8(&asdf,0,1,0);
  data[3] = chunk_get_u8(&asdf,0,1,1);
  data[4] = chunk_get_u8(&asdf,1,0,0);
  data[5] = chunk_get_u8(&asdf,1,0,1);
  data[6] = chunk_get_u8(&asdf,1,1,0);
  data[7] = chunk_get_u8(&asdf,1,1,1);
  printf("%u %u %u %u %u %u %u %u\n",data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7]);
  printf("%u\n",chunk_get_u8(&pooled,0,0,0));
  chunk_free(&pooled);

  pooled = avgpool(&asdf,2,2);
  data[0] = chunk_get_u8(&asdf,0,0,0);
  data[1] = chunk_get_u8(&asdf,0,0,1);
  data[2] = chunk_get_u8(&asdf,0,1,0);
  data[3] = chunk_get_u8(&asdf,0,1,1);
  data[4] = chunk_get_u8(&asdf,1,0,0);
  data[5] = chunk_get_u8(&asdf,1,0,1);
  data[6] = chunk_get_u8(&asdf,1,1,0);
  data[7] = chunk_get_u8(&asdf,1,1,1);
  printf("%u %u %u %u %u %u %u %u\n",data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7]);
  printf("%u\n",chunk_get_u8(&pooled,0,0,0));

  chunk_free(&pooled);

  return 0;
}

