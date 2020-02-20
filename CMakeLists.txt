cmake_minimum_required(VERSION 3.16)
project(lptml_cuda)

include(CheckLanguage)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")


add_library(mosek64 SHARED IMPORTED)
set_property(TARGET mosek64 PROPERTY IMPORTED_LOCATION "/home/francesco/CLionProjects/lptml_cuda/lib/mosek/9.1/tools/platform/linux64x86/bin/libmosek64.so")

add_library(fusion64 SHARED IMPORTED)
set_property(TARGET fusion64 PROPERTY IMPORTED_LOCATION "/home/francesco/CLionProjects/lptml_cuda/lib/mosek/9.1/tools/platform/linux64x86/bin/libfusion64.so")


set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address -lineinfo")
set (CMAKE_EXE_LINKER_FLAGS} "${CMAKE_EXE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address -lineinfo")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/home/francesco/CLionProjects/lptml_cuda/lib/mosek/9.1/tools/platform/linux64x86/h -L/home/francesco/CLionProjects/lptml_cuda/lib/mosek/9.1/tools/platform/linux64x86/bin/ -lmosek64 -lfusion64 -Wl, rpath-link,/home/francesco/CLionProjects/lptml_cuda/lib/mosek/9.1/tools/platform/linux64x86/bin -Wl,-rpath=/home/francesco/CLionProjects/lptml_cuda/lib/mosek/9.1/tools/platform/linux64x86/bin -Wall -O3")

add_executable(lptml_cuda main.cu lib/csv.h lib/lptml.cuh lib/utils.h lib/lptml.h lib/types.h lib/cuda_wrapper.cuh)
target_link_libraries(lptml_cuda mosek64 fusion64)