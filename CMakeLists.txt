cmake_minimum_required(VERSION 3.10)
project(SIMD_Inference_Engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES "x86_64")
# Hardware specific flags for AVX2 and FMA support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma -Wall -Wextra")

# Source files
set(SOURCES
    main.cpp
    simd_engine.cpp
)

add_executable(simd_engine ${SOURCES})
