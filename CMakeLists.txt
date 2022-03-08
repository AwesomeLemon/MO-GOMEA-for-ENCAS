cmake_minimum_required(VERSION 3.17)
project(MO_GOMEA C)

set(CMAKE_C_STANDARD 11)

add_executable(MO_GOMEA MO_GOMEA.c)
target_compile_options(MO_GOMEA PUBLIC -w -pipe -pthread -Wall -W -O3 -march=native)
#target_include_directories(MO_GOMEA PUBLIC /export/scratch1/home/aleksand/miniconda3/envs/supernetevo/include/python3.8)
#target_include_directories(MO_GOMEA PUBLIC /home/chebykin/.conda/envs/supernetevo/include/python3.8)
target_include_directories(MO_GOMEA PUBLIC /home/chebykin/.conda/envs/env_encas/include/python3.8)

target_link_libraries(MO_GOMEA m)
target_link_libraries(MO_GOMEA python3.8)