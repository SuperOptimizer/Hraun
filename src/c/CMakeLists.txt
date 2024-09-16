cmake_minimum_required(VERSION 3.10)
project(lava C)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


# Add compiler flags based on build type
set(CMAKE_C_FLAGS " -march=native -m64 -Weverything ")
set(CMAKE_C_FLAGS_DEBUG "-g -Og -Wall -Wextra -Wpedantic -DDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG ")



add_executable(lava main.c chunk.h cvol.h lava.h snic.h)
add_executable(lavatest test.c chunk.h cvol.h lava.h snic.h)

#target_link_libraries(lava PUBLIC "libcmt.lib")

set_property(TARGET lavatest PROPERTY C_STANDARD 23)
set_property(TARGET lava PROPERTY C_STANDARD 23)

# Print the build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")