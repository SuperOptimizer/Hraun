#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <tiffio.h>

#define CHUNK_SIZE 512
#define CHUNK_VOLUME (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)

void create_chunk_file(const char* filename, size_t size) {
    int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if (fd == -1) {
        perror("Error opening file for writing");
        exit(EXIT_FAILURE);
    }

    if (lseek(fd, size - 1, SEEK_SET) == -1) {
        close(fd);
        perror("Error calling lseek() to 'stretch' the file");
        exit(EXIT_FAILURE);
    }

    if (write(fd, "", 1) == -1) {
        close(fd);
        perror("Error writing last byte of the file");
        exit(EXIT_FAILURE);
    }

    close(fd);
}

void convert_tiff_to_chunks(const char** tiff_files, int num_files, int volume_z, int volume_y, int volume_x) {
    int padded_z = ((volume_z + CHUNK_SIZE - 1) / CHUNK_SIZE) * CHUNK_SIZE;
    int padded_y = ((volume_y + CHUNK_SIZE - 1) / CHUNK_SIZE) * CHUNK_SIZE;
    int padded_x = ((volume_x + CHUNK_SIZE - 1) / CHUNK_SIZE) * CHUNK_SIZE;

    int num_chunks_z = padded_z / CHUNK_SIZE;
    int num_chunks_y = padded_y / CHUNK_SIZE;
    int num_chunks_x = padded_x / CHUNK_SIZE;

    for (int cz = 0; cz < num_chunks_z; cz++) {
        for (int cy = 0; cy < num_chunks_y; cy++) {
            for (int cx = 0; cx < num_chunks_x; cx++) {
                char chunk_filename[256];
                snprintf(chunk_filename, sizeof(chunk_filename), "chunk_%d_%d_%d.chunk", cz, cy, cx);
                create_chunk_file(chunk_filename, CHUNK_VOLUME);

                int fd = open(chunk_filename, O_RDWR);
                if (fd == -1) {
                    perror("Error opening chunk file");
                    exit(EXIT_FAILURE);
                }

                uint8_t* chunk_data = mmap(0, CHUNK_VOLUME, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                if (chunk_data == MAP_FAILED) {
                    close(fd);
                    perror("Error mmapping the chunk file");
                    exit(EXIT_FAILURE);
                }

                memset(chunk_data, 0, CHUNK_VOLUME);  // Initialize with zeros

                for (int z = 0; z < CHUNK_SIZE; z++) {
                    int global_z = cz * CHUNK_SIZE + z;
                    if (global_z >= volume_z) break;

                    TIFF* tif = TIFFOpen(tiff_files[global_z], "r");
                    if (tif == NULL) {
                        fprintf(stderr, "Error opening TIFF file: %s\n", tiff_files[global_z]);
                        continue;
                    }

                    uint32_t width, height;
                    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
                    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

                    uint8_t* buffer = (uint8_t*)_TIFFmalloc(width * height);
                    if (!buffer) {
                        fprintf(stderr, "Error allocating memory for TIFF buffer\n");
                        TIFFClose(tif);
                        continue;
                    }

                    if (!TIFFReadRawStrip(tif, 0, buffer, width * height)) {
                        fprintf(stderr, "Error reading TIFF data\n");
                        _TIFFfree(buffer);
                        TIFFClose(tif);
                        continue;
                    }

                    for (int y = 0; y < CHUNK_SIZE; y++) {
                        int global_y = cy * CHUNK_SIZE + y;
                        if (global_y >= volume_y) break;

                        for (int x = 0; x < CHUNK_SIZE; x++) {
                            int global_x = cx * CHUNK_SIZE + x;
                            if (global_x >= volume_x) break;

                            if (global_x < width && global_y < height) {
                                chunk_data[z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x] = buffer[global_y * width + global_x];
                            }
                        }
                    }

                    _TIFFfree(buffer);
                    TIFFClose(tif);
                }

                if (munmap(chunk_data, CHUNK_VOLUME) == -1) {
                    perror("Error un-mmapping the chunk file");
                }
                close(fd);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <volume_z> <volume_y> <volume_x> <tiff_file1> [tiff_file2] ...\n", argv[0]);
        return 1;
    }

    int volume_z = atoi(argv[1]);
    int volume_y = atoi(argv[2]);
    int volume_x = atoi(argv[3]);

    int num_files = argc - 4;
    const char** tiff_files = (const char**)&argv[4];

    convert_tiff_to_chunks(tiff_files, num_files, volume_z, volume_y, volume_x);

    printf("Conversion completed successfully.\n");
    return 0;
}