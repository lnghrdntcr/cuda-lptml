cmake_minimum_required(VERSION 3.16)
project(lptml_cuda)

include(CheckLanguage)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
#[[if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")]]
# -I<HEADERDIR> -L<LIBDIR>
set(CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
#[[

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -I/home/francesco/CLionProjects/lptml_cuda/lib/mosek/9.1/tools/platform/linux64x86/h -L/home/francesco/CLionProjects/lptml_cuda/lib/mosek/9.1/tools/platform/linux64x86/bin/libmosek64.so /home/francesco/CLionProjects/lptml_cuda/lib/mosek/9.1/tools/platform/linux64x86/bin/ -Wl, rpath-link,/home/francesco/CLionProjects/lptml_cuda/lib/mosek/9.1/tools/platform/linux64x86/bin -Wl,-rpath=/home/francesco/CLionProjects/lptml_cuda/lib/mosek/9.1/tools/platform/linux64x86/bin")
]]
#[["lib/mosek/9.1/tools/platform/linux64x86/h/*.h"]]
file(GLOB mosek_src
        "")
add_executable(lptml_cuda main.cu ${mosek_src} lib/csv.h lib/lptml.cuh lib/utils.h lib/lptml.h lib/types.h lib/cuda_wrapper.cuh)