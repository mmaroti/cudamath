cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

add_executable(linapprox
  src/linapprox.cu)

target_compile_features(linapprox PUBLIC cxx_std_11)
