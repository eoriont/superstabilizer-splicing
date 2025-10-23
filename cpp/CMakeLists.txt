cmake_minimum_required(VERSION 3.16)
project(stim_cpp_demo)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(STIM_ROOT "/Users/eli/dev/Stim")  # no ~, no quotes around ~
add_executable(superstabilizer_splicing gen_entangled.cpp)
target_include_directories(superstabilizer_splicing PRIVATE "${STIM_ROOT}/src")
target_link_libraries(superstabilizer_splicing PRIVATE "${STIM_ROOT}/build/out/libstim.a")
