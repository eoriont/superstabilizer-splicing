cmake_minimum_required(VERSION 3.16)
project(surface_general_defect)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to your local Stim repo (already defined elsewhere, but safe to include here)
set(STIM_ROOT "/Users/eli/dev/Stim")  # absolute path, no ~

# Build the single executable
add_executable(surface_general_defect surface_general_defect.cpp)

# Include Stim headers and link its static library
target_include_directories(surface_general_defect PRIVATE "${STIM_ROOT}/src")
target_link_libraries(surface_general_defect PRIVATE "${STIM_ROOT}/build/out/libstim.a")

message(STATUS "Building surface_general_defect with Stim from ${STIM_ROOT}")
