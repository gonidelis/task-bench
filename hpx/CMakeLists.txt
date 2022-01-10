# Copyright (c) 2021 Nanmiao Wu
# Copyright (c) 2021 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


cmake_minimum_required(VERSION 3.17)

project(TaskBench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

# Setting up dependencies
find_package(HPX REQUIRED)

link_directories(${CMAKE_SOURCE_DIR}/../core)

add_executable(async_mpi_par async_mpi_par.cc)
target_link_libraries(async_mpi_par PUBLIC HPX::hpx HPX::wrap_main core_s)

add_executable(async_mpi_executor async_mpi_executor.cc)
target_link_libraries(async_mpi_executor PUBLIC HPX::hpx HPX::wrap_main core_s)

add_executable(channel_executor channel_executor.cc)
target_link_libraries(channel_executor PUBLIC HPX::hpx HPX::wrap_main core_s)

add_executable(channel_par channel_par.cc)
target_link_libraries(channel_par PUBLIC HPX::hpx HPX::wrap_main core_s)

add_executable(mpi_executor mpi_executor.cc)
target_link_libraries(mpi_executor PUBLIC HPX::hpx HPX::wrap_main core_s)

add_executable(mpi_par mpi_par.cc)
target_link_libraries(mpi_par PUBLIC HPX::hpx HPX::wrap_main core_s)

add_executable(hpx_local_executor hpx_local_executor.cc)
target_link_libraries(hpx_local_executor PUBLIC HPX::hpx HPX::wrap_main core_s)

add_executable(hpx_local_par hpx_local_par.cc)
target_link_libraries(hpx_local_par PUBLIC HPX::hpx HPX::wrap_main core_s)

add_executable(new_hpx_local new_hpx_local.cc)
target_link_libraries(new_hpx_local PUBLIC HPX::hpx HPX::wrap_main core_s)

