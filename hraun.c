#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <math.h>


#define TIFF_LITTLE_ENDIAN 0x4949
#define TIFF_BIG_ENDIAN    0x4D4D

typedef struct {
    uint16_t tag;
    uint16_t type;
    uint32_t count;
    uint32_t value_offset;
} TIFFTag;

typedef struct {
    uint16_t byte_order;
    uint16_t version;
    uint32_t ifd_offset;
} TIFFHeader;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint16_t bits_per_sample;
    uint16_t compression;
    uint16_t photometric_interpretation;
    uint16_t samples_per_pixel;
    uint32_t rows_per_strip;
    uint16_t planar_configuration;
    char* image_description;
    char* software;
    uint32_t* strip_offsets;
    uint32_t strip_offsets_count;
} TIFFInfo;

int is_little_endian() {
    uint16_t endian_check = 0x0001;
    return (*(char*)&endian_check == 0x01);
}

uint16_t swap_bytes_16(uint16_t value) {
    return ((value >> 8) & 0x00FF) | ((value << 8) & 0xFF00);
}

uint32_t swap_bytes_32(uint32_t value) {
    return ((value >> 24) & 0x000000FF) |
           ((value >>  8) & 0x0000FF00) |
           ((value <<  8) & 0x00FF0000) |
           ((value << 24) & 0xFF000000);
}

void free_tiff_info(TIFFInfo* info) {
    free(info->image_description);
    free(info->software);
    free(info->strip_offsets);
}

int parse_tiff_header(unsigned char* tiff_data, TIFFInfo* info) {
    TIFFHeader header;
    memcpy(&header, tiff_data, sizeof(TIFFHeader));

    int little_endian = is_little_endian();
    int tiff_little_endian = (header.byte_order == TIFF_LITTLE_ENDIAN);

    if (little_endian != tiff_little_endian) {
        header.version = swap_bytes_16(header.version);
        header.ifd_offset = swap_bytes_32(header.ifd_offset);
    }

    if (header.version != 42) {
        fprintf(stderr, "Unsupported TIFF version: %d\n", header.version);
        return 0;
    }

    uint32_t ifd_offset = header.ifd_offset;
    while (ifd_offset != 0) {
        uint16_t num_entries;
        memcpy(&num_entries, tiff_data + ifd_offset, sizeof(uint16_t));
        if (little_endian != tiff_little_endian) {
            num_entries = swap_bytes_16(num_entries);
        }

        ifd_offset += sizeof(uint16_t);

        for (int i = 0; i < num_entries; i++) {
            TIFFTag tag;
            memcpy(&tag, tiff_data + ifd_offset, sizeof(TIFFTag));
            if (little_endian != tiff_little_endian) {
                tag.tag = swap_bytes_16(tag.tag);
                tag.type = swap_bytes_16(tag.type);
                tag.count = swap_bytes_32(tag.count);
                tag.value_offset = swap_bytes_32(tag.value_offset);
            }

            switch (tag.tag) {
                case 256: // ImageWidth
                    info->width = tag.value_offset;
                    break;
                case 257: // ImageLength
                    info->height = tag.value_offset;
                    break;
                case 258: // BitsPerSample
                    info->bits_per_sample = tag.value_offset;
                    break;
                case 259: // Compression
                    info->compression = tag.value_offset;
                    break;
                case 262: // PhotometricInterpretation
                    info->photometric_interpretation = tag.value_offset;
                    break;
                case 273: // StripOffsets
                    info->strip_offsets_count = tag.count;
                    info->strip_offsets = (uint32_t*)malloc(tag.count * sizeof(uint32_t));
                    if (tag.count == 1) {
                        info->strip_offsets[0] = tag.value_offset;
                    } else {
                        memcpy(info->strip_offsets, tiff_data + tag.value_offset, tag.count * sizeof(uint32_t));
                        if (little_endian != tiff_little_endian) {
                            for (uint32_t j = 0; j < tag.count; j++) {
                                info->strip_offsets[j] = swap_bytes_32(info->strip_offsets[j]);
                            }
                        }
                    }
                    break;
                case 277: // SamplesPerPixel
                    info->samples_per_pixel = tag.value_offset;
                    break;
                case 278: // RowsPerStrip
                    info->rows_per_strip = tag.value_offset;
                    break;
                case 284: // PlanarConfiguration
                    info->planar_configuration = tag.value_offset;
                    break;
                case 270: // ImageDescription
                    info->image_description = (char*)malloc(tag.count * sizeof(char));
                    memcpy(info->image_description, tiff_data + tag.value_offset, tag.count);
                    break;
                case 305: // Software
                    info->software = (char*)malloc(tag.count * sizeof(char));
                    memcpy(info->software, tiff_data + tag.value_offset, tag.count);
                    break;
                default:
                    // Unknown tag, skip it
                    break;
            }

            ifd_offset += sizeof(TIFFTag);
        }

        memcpy(&ifd_offset, tiff_data + ifd_offset, sizeof(uint32_t));
        if (little_endian != tiff_little_endian) {
            ifd_offset = swap_bytes_32(ifd_offset);
        }
    }

    return 1;
}

typedef struct {
    unsigned char** data;
    uint32_t* data_offsets;
    int width;
    int height;
    int depth;
} Volume;

Volume* create_volume(const char* directory) {
    Volume* volume = (Volume*)malloc(sizeof(Volume));
    if (!volume) {
        fprintf(stderr, "Failed to allocate memory for volume\n");
        exit(1);
    }

    DIR* dir = opendir(directory);
    if (!dir) {
        fprintf(stderr, "Failed to open directory: %s\n", directory);
        exit(1);
    }

    struct dirent* entry;
    int num_files = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG && strstr(entry->d_name, ".tif") != NULL) {
            num_files++;
        }
    }

    rewinddir(dir);

    volume->depth = num_files;
    volume->data = (unsigned char**)malloc(num_files * sizeof(unsigned char*));
    volume->data_offsets = (uint32_t*)malloc(num_files * sizeof(uint32_t));
    if (!volume->data || !volume->data_offsets) {
        fprintf(stderr, "Failed to allocate memory for volume data\n");
        exit(1);
    }

    int file_index = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG && strstr(entry->d_name, ".tif") != NULL) {
            char file_path[1024];
            snprintf(file_path, sizeof(file_path), "%s/%s", directory, entry->d_name);
            printf("loading %s\n",file_path);
            if(file_index == 1000)break;
            int fd = open(file_path, O_RDONLY);
            if (fd == -1) {
                fprintf(stderr, "Failed to open TIFF file: %s\n", file_path);
                exit(1);
            }

            struct stat sb;
            if (fstat(fd, &sb) == -1) {
                fprintf(stderr, "Failed to get file size: %s\n", file_path);
                exit(1);
            }

            unsigned char* tiff_data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (tiff_data == MAP_FAILED) {
                fprintf(stderr, "Failed to memory map TIFF file: %s\n", file_path);
                exit(1);
            }

            TIFFInfo info;
            if (parse_tiff_header(tiff_data, &info)) {
                volume->data[file_index] = tiff_data;
                volume->data_offsets[file_index] = info.strip_offsets[0];
                volume->width = info.width;
                volume->height = info.height;
                file_index++;
            } else {
                fprintf(stderr, "Failed to parse TIFF header: %s\n", file_path);
            }

            free_tiff_info(&info);
            close(fd);
        }
    }

    closedir(dir);

    return volume;
}

unsigned char volume_at(Volume* volume, int x, int y, int z) {
    if (x < 0 || x >= volume->width || y < 0 || y >= volume->height || z < 0 || z >= volume->depth) {
        fprintf(stderr, "Invalid coordinates: (%d, %d, %d)\n", x, y, z);
        exit(1);
    }

    unsigned char* tiff_data = volume->data[z];
    uint32_t data_offset = volume->data_offsets[z];
    int index = data_offset + y * volume->width + x;

    return tiff_data[index];
}

unsigned char*** volume_chunk(Volume* volume, int x_offset, int y_offset, int z_offset,
                              int x_size, int y_size, int z_size) {
    if (x_offset < 0 || x_offset + x_size > volume->width ||
        y_offset < 0 || y_offset + y_size > volume->height ||
        z_offset < 0 || z_offset + z_size > volume->depth) {
        fprintf(stderr, "Invalid chunk dimensions\n");
        exit(1);
    }

    unsigned char*** chunk = (unsigned char***)malloc(z_size * sizeof(unsigned char**));
    if (!chunk) {
        fprintf(stderr, "Failed to allocate memory for chunk\n");
        exit(1);
    }

    for (int z = 0; z < z_size; z++) {
        chunk[z] = (unsigned char**)malloc(y_size * sizeof(unsigned char*));
        if (!chunk[z]) {
            fprintf(stderr, "Failed to allocate memory for chunk slice\n");
            exit(1);
        }

        for (int y = 0; y < y_size; y++) {
            chunk[z][y] = (unsigned char*)malloc(x_size * sizeof(unsigned char));
            if (!chunk[z][y]) {
                fprintf(stderr, "Failed to allocate memory for chunk row\n");
                exit(1);
            }

            for (int x = 0; x < x_size; x++) {
                chunk[z][y][x] = volume_at(volume, x_offset + x, y_offset + y, z_offset + z);
            }
        }
    }

    return chunk;
}

void free_volume(Volume* volume) {
    for (int i = 0; i < volume->depth; i++) {
        munmap(volume->data[i], volume->width * volume->height * sizeof(unsigned char));
    }

    free(volume->data);
    free(volume->data_offsets);
    free(volume);
}

void free_chunk(unsigned char*** chunk, int z_size, int y_size) {
    for (int z = 0; z < z_size; z++) {
        for (int y = 0; y < y_size; y++) {
            free(chunk[z][y]);
        }
        free(chunk[z]);
    }
    free(chunk);
}


static const int edgeTable[256]={
        0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
        0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
        0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
        0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
        0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
        0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
        0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
        0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
        0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
        0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
        0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
        0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
        0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
        0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
        0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
        0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
        0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
        0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
        0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
        0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
        0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
        0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
        0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
        0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
        0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
        0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
        0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
        0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
        0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
        0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
        0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
        0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };
static const int triTable[256][16] =
        {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
         {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
         {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
         {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
         {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
         {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
         {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
         {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
         {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
         {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
         {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
         {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
         {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
         {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
         {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
         {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
         {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
         {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
         {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
         {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
         {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
         {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
         {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
         {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
         {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
         {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
         {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
         {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
         {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
         {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
         {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
         {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
         {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
         {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
         {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
         {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
         {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
         {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
         {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
         {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
         {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
         {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
         {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
         {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
         {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
         {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
         {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
         {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
         {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
         {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
         {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
         {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
         {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
         {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
         {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
         {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
         {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
         {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
         {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
         {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
         {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
         {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
         {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
         {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
         {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
         {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
         {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
         {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
         {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
         {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
         {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
         {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
         {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
         {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
         {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
         {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
         {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
         {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
         {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
         {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
         {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
         {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
         {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
         {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
         {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
         {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
         {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
         {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
         {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
         {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
         {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
         {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
         {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
         {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
         {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
         {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
         {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
         {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
         {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
         {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
         {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
         {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
         {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
         {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
         {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
         {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
         {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
         {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
         {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
         {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
         {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
         {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
         {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
         {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
         {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
         {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
         {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
         {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
         {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
         {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
         {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
         {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
         {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
         {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
         {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
         {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
         {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
         {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
         {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
         {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
         {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
         {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
         {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
         {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
         {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
         {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
         {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
         {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
         {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
         {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
         {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
         {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
         {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
         {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
         {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
         {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
         {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
         {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
         {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
         {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
         {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
         {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
         {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
         {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
         {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
         {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
         {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
         {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
         {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
         {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
         {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
         {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
         {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
         {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
         {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
         {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
         {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
         {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
         {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
         {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
         {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
         {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
         {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
         {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
         {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
         {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
         {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
         {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
         {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
         {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
         {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};


typedef struct {
    float x, y, z;
} Vector3;

typedef struct {
    Vector3 p[3];
} Triangle;

typedef struct {
    float x, y, z;
    float r, g, b;
} ColoredVector3;

typedef struct {
    ColoredVector3 p[3];
} ColoredTriangle;

Vector3 interpolate(float isolevel, Vector3 p1, Vector3 p2, float v1, float v2) {
    if (fabs(v1 - v2) < 1e-6) {
        return p1;
    }

    float mu = (isolevel - v1) / (v2 - v1);
    Vector3 p = {
            p1.x + mu * (p2.x - p1.x),
            p1.y + mu * (p2.y - p1.y),
            p1.z + mu * (p2.z - p1.z)
    };
    return p;
}

int march_cube(float* cube, float isolevel, Triangle* triangles) {
    int cubeindex = 0;
    Vector3 vertlist[12];

    if (cube[0] < isolevel) cubeindex |= 1;
    if (cube[1] < isolevel) cubeindex |= 2;
    if (cube[2] < isolevel) cubeindex |= 4;
    if (cube[3] < isolevel) cubeindex |= 8;
    if (cube[4] < isolevel) cubeindex |= 16;
    if (cube[5] < isolevel) cubeindex |= 32;
    if (cube[6] < isolevel) cubeindex |= 64;
    if (cube[7] < isolevel) cubeindex |= 128;

    if (edgeTable[cubeindex] == 0)
        return 0;

    int i, j, vertexIndex;
    Vector3 vertices[8] = {
            {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
    };

    if (edgeTable[cubeindex] & 1)
        vertlist[0] = interpolate(isolevel, vertices[0], vertices[1], cube[0], cube[1]);
    if (edgeTable[cubeindex] & 2)
        vertlist[1] = interpolate(isolevel, vertices[1], vertices[2], cube[1], cube[2]);
    if (edgeTable[cubeindex] & 4)
        vertlist[2] = interpolate(isolevel, vertices[2], vertices[3], cube[2], cube[3]);
    if (edgeTable[cubeindex] & 8)
        vertlist[3] = interpolate(isolevel, vertices[3], vertices[0], cube[3], cube[0]);
    if (edgeTable[cubeindex] & 16)
        vertlist[4] = interpolate(isolevel, vertices[4], vertices[5], cube[4], cube[5]);
    if (edgeTable[cubeindex] & 32)
        vertlist[5] = interpolate(isolevel, vertices[5], vertices[6], cube[5], cube[6]);
    if (edgeTable[cubeindex] & 64)
        vertlist[6] = interpolate(isolevel, vertices[6], vertices[7], cube[6], cube[7]);
    if (edgeTable[cubeindex] & 128)
        vertlist[7] = interpolate(isolevel, vertices[7], vertices[4], cube[7], cube[4]);
    if (edgeTable[cubeindex] & 256)
        vertlist[8] = interpolate(isolevel, vertices[0], vertices[4], cube[0], cube[4]);
    if (edgeTable[cubeindex] & 512)
        vertlist[9] = interpolate(isolevel, vertices[1], vertices[5], cube[1], cube[5]);
    if (edgeTable[cubeindex] & 1024)
        vertlist[10] = interpolate(isolevel, vertices[2], vertices[6], cube[2], cube[6]);
    if (edgeTable[cubeindex] & 2048)
        vertlist[11] = interpolate(isolevel, vertices[3], vertices[7], cube[3], cube[7]);

    int num_triangles = 0;
    for (i = 0; triTable[cubeindex][i] != -1; i += 3) {
        for (j = 0; j < 3; j++) {
            vertexIndex = triTable[cubeindex][i + j];
            triangles[num_triangles].p[j] = vertlist[vertexIndex];
        }
        num_triangles++;
    }

    return num_triangles;
}

void marching_cubes_chunk(float*** chunk, int chunk_size, float isolevel, Triangle** triangles, int* num_triangles) {
    int x, y, z;
    float cube[8];
    Triangle cube_triangles[5];

    *num_triangles = 0;
    *triangles = NULL;

    for (z = 0; z < chunk_size - 1; z++) {
        for (y = 0; y < chunk_size - 1; y++) {
            for (x = 0; x < chunk_size - 1; x++) {
                cube[0] = chunk[z][y][x];
                cube[1] = chunk[z][y][x + 1];
                cube[2] = chunk[z][y + 1][x + 1];
                cube[3] = chunk[z][y + 1][x];
                cube[4] = chunk[z + 1][y][x];
                cube[5] = chunk[z + 1][y][x + 1];
                cube[6] = chunk[z + 1][y + 1][x + 1];
                cube[7] = chunk[z + 1][y + 1][x];

                int num_cube_triangles = march_cube(cube, isolevel, cube_triangles);

                if (num_cube_triangles > 0) {
                    *triangles = (Triangle*)realloc(*triangles, (*num_triangles + num_cube_triangles) * sizeof(Triangle));
                    for (int i = 0; i < num_cube_triangles; i++) {
                        (*triangles)[*num_triangles + i] = cube_triangles[i];
                        (*triangles)[*num_triangles + i].p[0].x += x;
                        (*triangles)[*num_triangles + i].p[0].y += y;
                        (*triangles)[*num_triangles + i].p[0].z += z;
                        (*triangles)[*num_triangles + i].p[1].x += x;
                        (*triangles)[*num_triangles + i].p[1].y += y;
                        (*triangles)[*num_triangles + i].p[1].z += z;
                        (*triangles)[*num_triangles + i].p[2].x += x;
                        (*triangles)[*num_triangles + i].p[2].y += y;
                        (*triangles)[*num_triangles + i].p[2].z += z;
                    }
                    *num_triangles += num_cube_triangles;
                }
            }
        }
    }
}

void write_ply(const char* filename, Triangle* triangles, int num_triangles) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Failed to open PLY file for writing: %s\n", filename);
        exit(1);
    }

    fprintf(file, "ply\n");
    fprintf(file, "format ascii 1.0\n");
    fprintf(file, "element vertex %d\n", num_triangles * 3);
    fprintf(file, "property float x\n");
    fprintf(file, "property float y\n");
    fprintf(file, "property float z\n");
    fprintf(file, "element face %d\n", num_triangles);
    fprintf(file, "property list uchar int vertex_index\n");
    fprintf(file, "end_header\n");

    for (int i = 0; i < num_triangles; i++) {
        for (int j = 0; j < 3; j++) {
            fprintf(file, "%f %f %f\n", triangles[i].p[j].x, triangles[i].p[j].y, triangles[i].p[j].z);
        }
    }

    for (int i = 0; i < num_triangles; i++) {
        fprintf(file, "3 %d %d %d\n", i * 3, i * 3 + 1, i * 3 + 2);
    }

    fclose(file);
}

float*** rescale_chunk(unsigned char*** chunk, int chunk_size) {
    int x, y, z;
    float*** rescaled_chunk = (float***)malloc(chunk_size * sizeof(float**));
    for (z = 0; z < chunk_size; z++) {
        rescaled_chunk[z] = (float**)malloc(chunk_size * sizeof(float*));
        for (y = 0; y < chunk_size; y++) {
            rescaled_chunk[z][y] = (float*)malloc(chunk_size * sizeof(float));
            for (x = 0; x < chunk_size; x++) {
                rescaled_chunk[z][y][x] = (float)chunk[z][y][x] / 255.0f;
            }
        }
    }
    return rescaled_chunk;
}

void free_rescaled_chunk(float*** rescaled_chunk, int chunk_size) {
    int z, y;
    for (z = 0; z < chunk_size; z++) {
        for (y = 0; y < chunk_size; y++) {
            free(rescaled_chunk[z][y]);
        }
        free(rescaled_chunk[z]);
    }
    free(rescaled_chunk);
}

int main() {
    const char* directory = "/mnt/c/Users/forrest/dev/Hraun/dl.ash2txt.org/full-scrolls/PHerc1667.volpkg/volumes/20231117161658";
    Volume* volume = create_volume(directory);

    printf("Volume dimensions: %d x %d x %d\n", volume->width, volume->height, volume->depth);

    int chunk_size = 100;
    int chunk_offset_x = 1000;
    int chunk_offset_y = 1000;
    int chunk_offset_z = 500;

    unsigned char*** chunk = volume_chunk(volume, chunk_offset_x, chunk_offset_y, chunk_offset_z, chunk_size, chunk_size, chunk_size);

    float*** rescaled_chunk = rescale_chunk(chunk, chunk_size);

    float isolevel = 0.5f;
    Triangle* triangles;
    int num_triangles;

    marching_cubes_chunk(rescaled_chunk, chunk_size, isolevel, &triangles, &num_triangles);

    printf("Generated %d triangles\n", num_triangles);

    write_ply("output.ply", triangles, num_triangles);

    free(triangles);
    free_rescaled_chunk(rescaled_chunk, chunk_size);
    free_chunk(chunk, chunk_size, chunk_size);
    free_volume(volume);

    return 0;
}