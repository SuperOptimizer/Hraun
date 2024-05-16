#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

//#define USE_OPENMP

#ifdef USE_OPENMP
#include <omp.h>
#endif

#define SECTOR_SIZE 256
#define BLOCK_SIZE 4

typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} SectorCoord;

typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} BlockCoord;

typedef struct {
    uint8_t data[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];
} Block;

typedef struct {
    Block blocks[SECTOR_SIZE / BLOCK_SIZE][SECTOR_SIZE / BLOCK_SIZE][SECTOR_SIZE / BLOCK_SIZE];
} Sector;

typedef struct {
    float x;
    float y;
    float z;
} Vector3;

typedef struct {
    float m[3][3];
} Matrix3x3;

SectorCoord calculateSectorCoord(uint32_t x, uint32_t y, uint32_t z) {
    SectorCoord coord;
    coord.x = x / SECTOR_SIZE;
    coord.y = y / SECTOR_SIZE;
    coord.z = z / SECTOR_SIZE;
    return coord;
}

BlockCoord calculateBlockCoord(uint32_t x, uint32_t y, uint32_t z) {
    BlockCoord coord;
    coord.x = (x % SECTOR_SIZE) / BLOCK_SIZE;
    coord.y = (y % SECTOR_SIZE) / BLOCK_SIZE;
    coord.z = (z % SECTOR_SIZE) / BLOCK_SIZE;
    return coord;
}

Sector* mapSector(const char* volumePath, SectorCoord sectorCoord) {
    char sectorFileName[256];
    snprintf(sectorFileName, sizeof(sectorFileName), "%s/%03u_%03u_%03u.sector", volumePath, sectorCoord.x, sectorCoord.y, sectorCoord.z);
    int sectorFile = open(sectorFileName, O_RDONLY);
    if (sectorFile == -1) {
        return NULL;
    }
    Sector* sector = (Sector*)mmap(NULL, sizeof(Sector), PROT_READ, MAP_PRIVATE, sectorFile, 0);
    close(sectorFile);
    return sector;
}

uint8_t getVoxel(Sector* sector, uint32_t x, uint32_t y, uint32_t z) {
    BlockCoord blockCoord = calculateBlockCoord(x, y, z);
    Block block = sector->blocks[blockCoord.x][blockCoord.y][blockCoord.z];
    uint8_t voxelValue = block.data[x % BLOCK_SIZE][y % BLOCK_SIZE][z % BLOCK_SIZE];
    return voxelValue;
}

Matrix3x3 calculateRotationMatrix(Vector3 start, Vector3 end) {
    Vector3 direction = {end.x - start.x, end.y - start.y, end.z - start.z};
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    direction.x /= length;
    direction.y /= length;
    direction.z /= length;

    Vector3 xAxis = {1.0f, 0.0f, 0.0f};
    Vector3 rotationAxis = {direction.y * xAxis.z - direction.z * xAxis.y,
                            direction.z * xAxis.x - direction.x * xAxis.z,
                            direction.x * xAxis.y - direction.y * xAxis.x};
    float rotationAngle = acosf(direction.x);

    float cosAngle = cosf(rotationAngle);
    float sinAngle = sinf(rotationAngle);
    float oneMinusCosAngle = 1.0f - cosAngle;

    Matrix3x3 rotationMatrix = {
        {cosAngle + rotationAxis.x * rotationAxis.x * oneMinusCosAngle,
         rotationAxis.x * rotationAxis.y * oneMinusCosAngle - rotationAxis.z * sinAngle,
         rotationAxis.x * rotationAxis.z * oneMinusCosAngle + rotationAxis.y * sinAngle},
        {rotationAxis.y * rotationAxis.x * oneMinusCosAngle + rotationAxis.z * sinAngle,
         cosAngle + rotationAxis.y * rotationAxis.y * oneMinusCosAngle,
         rotationAxis.y * rotationAxis.z * oneMinusCosAngle - rotationAxis.x * sinAngle},
        {rotationAxis.z * rotationAxis.x * oneMinusCosAngle - rotationAxis.y * sinAngle,
         rotationAxis.z * rotationAxis.y * oneMinusCosAngle + rotationAxis.x * sinAngle,
         cosAngle + rotationAxis.z * rotationAxis.z * oneMinusCosAngle}
    };

    return rotationMatrix;
}

void readChunk(Sector** sectors, uint32_t numSectors, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ,
               uint32_t chunkWidth, uint32_t chunkHeight, uint32_t chunkDepth,
               uint8_t* chunkData, const char* axisOrder) {
    uint32_t axisMap[3];
    for (int i = 0; i < 3; i++) {
        switch (axisOrder[i]) {
            case 'x':
            case 'X':
                axisMap[i] = 0;
                break;
            case 'y':
            case 'Y':
                axisMap[i] = 1;
                break;
            case 'z':
            case 'Z':
                axisMap[i] = 2;
                break;
        }
    }

    #ifdef USE_OPENMP
    #pragma omp parallel for collapse(3)
    #endif
    for (uint32_t z = 0; z < chunkDepth; z++) {
        for (uint32_t y = 0; y < chunkHeight; y++) {
            for (uint32_t x = 0; x < chunkWidth; x++) {
                uint32_t logicalX = offsetX + x;
                uint32_t logicalY = offsetY + y;
                uint32_t logicalZ = offsetZ + z;

                uint32_t physicalCoords[3] = {logicalX, logicalY, logicalZ};
                SectorCoord sectorCoord = calculateSectorCoord(physicalCoords[axisMap[0]], physicalCoords[axisMap[1]], physicalCoords[axisMap[2]]);
                uint32_t sectorIndex = (sectorCoord.z * numSectors * numSectors) + (sectorCoord.y * numSectors) + sectorCoord.x;
                Sector* sector = sectors[sectorIndex];

                uint32_t chunkIndex = (z * chunkHeight + y) * chunkWidth + x;
                chunkData[chunkIndex] = getVoxel(sector, physicalCoords[axisMap[0]], physicalCoords[axisMap[1]], physicalCoords[axisMap[2]]);
            }
        }
    }
}

void readObliqueChunk(Sector** sectors, uint32_t numSectors, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ,
               uint32_t chunkWidth, uint32_t chunkHeight, uint32_t chunkDepth,
               uint8_t* chunkData, const char* axisOrder, Vector3 obliqueStart, Vector3 obliqueEnd) {
    uint32_t axisMap[3];
    for (int i = 0; i < 3; i++) {
        switch (axisOrder[i]) {
            case 'x':
            case 'X':
                axisMap[i] = 0;
                break;
            case 'y':
            case 'Y':
                axisMap[i] = 1;
                break;
            case 'z':
            case 'Z':
                axisMap[i] = 2;
                break;
        }
    }

    Matrix3x3 rotationMatrix = calculateRotationMatrix(obliqueStart, obliqueEnd);

    #ifdef USE_OPENMP
    #pragma omp parallel for collapse(3)
    #endif
    for (uint32_t z = 0; z < chunkDepth; z++) {
        for (uint32_t y = 0; y < chunkHeight; y++) {
            for (uint32_t x = 0; x < chunkWidth; x++) {
                uint32_t logicalX = offsetX + x;
                uint32_t logicalY = offsetY + y;
                uint32_t logicalZ = offsetZ + z;

                float rotatedX = rotationMatrix.m[0][0] * logicalX + rotationMatrix.m[0][1] * logicalY + rotationMatrix.m[0][2] * logicalZ;
                float rotatedY = rotationMatrix.m[1][0] * logicalX + rotationMatrix.m[1][1] * logicalY + rotationMatrix.m[1][2] * logicalZ;
                float rotatedZ = rotationMatrix.m[2][0] * logicalX + rotationMatrix.m[2][1] * logicalY + rotationMatrix.m[2][2] * logicalZ;

                uint32_t physicalCoords[3] = {(uint32_t)rotatedX, (uint32_t)rotatedY, (uint32_t)rotatedZ};
                SectorCoord sectorCoord = calculateSectorCoord(physicalCoords[axisMap[0]], physicalCoords[axisMap[1]], physicalCoords[axisMap[2]]);
                uint32_t sectorIndex = (sectorCoord.z * numSectors * numSectors) + (sectorCoord.y * numSectors) + sectorCoord.x;
                Sector* sector = sectors[sectorIndex];

                uint32_t chunkIndex = (z * chunkHeight + y) * chunkWidth + x;
                chunkData[chunkIndex] = getVoxel(sector, physicalCoords[axisMap[0]], physicalCoords[axisMap[1]], physicalCoords[axisMap[2]]);
            }
        }
    }
}

void unmapSectors(Sector** sectors, uint32_t numSectors) {
    for (uint32_t i = 0; i < numSectors * numSectors * numSectors; i++) {
        if (sectors[i] != NULL) {
            munmap(sectors[i], sizeof(Sector));
        }
    }
    free(sectors);
}

int main() {
    const char* volumePath = "path/to/volume/folder";

    uint32_t offsetX = 5000;
    uint32_t offsetY = 2000;
    uint32_t offsetZ = 3500;
    uint32_t chunkWidth = 500;
    uint32_t chunkHeight = 500;
    uint32_t chunkDepth = 500;
    uint8_t* chunkData = (uint8_t*)malloc(chunkWidth * chunkHeight * chunkDepth * sizeof(uint8_t));
    const char* axisOrder = "zyx";

    Vector3 obliqueStart = {0.0f, 0.0f, 0.0f};
    Vector3 obliqueEnd = {1.0f, 1.0f, 1.0f};

    uint32_t numSectors = ceil(10000.0 / SECTOR_SIZE);
    Sector** sectors = (Sector**)malloc(numSectors * numSectors * numSectors * sizeof(Sector*));
    memset(sectors, 0, numSectors * numSectors * numSectors * sizeof(Sector*));

    #ifdef USE_OPENMP
    int numProcs = omp_get_num_procs();
    int numThreads = numProcs / 2;
    omp_set_num_threads(numThreads);
    #endif

    for (uint32_t z = 0; z < numSectors; z++) {
        for (uint32_t y = 0; y < numSectors; y++) {
            for (uint32_t x = 0; x < numSectors; x++) {
                SectorCoord sectorCoord = {x, y, z};
                uint32_t sectorIndex = (z * numSectors * numSectors) + (y * numSectors) + x;
                sectors[sectorIndex] = mapSector(volumePath, sectorCoord);
            }
        }
    }

    readChunk(sectors, numSectors, offsetX, offsetY, offsetZ, chunkWidth, chunkHeight, chunkDepth, chunkData, axisOrder, obliqueStart, obliqueEnd);

    // Process the chunk data as needed
    // ...

    free(chunkData);
    unmapSectors(sectors, numSectors);

    return 0;
}