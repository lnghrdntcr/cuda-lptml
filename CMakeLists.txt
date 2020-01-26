cmake_minimum_required(VERSION 3.16)
project(lptml_cuda)

include(CheckLanguage)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 17)

add_executable(lptml_cuda main.cu lib/csv.h lib/lptml.cuh lib/utils.h lib/lptml.h lib/types.h lib/cuda_wrapper.cuh)