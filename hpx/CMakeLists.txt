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

#add_executable(main main.cc)
#target_link_libraries(main PUBLIC HPX::hpx HPX::wrap_main core_s)

add_executable(test_comm_ring test_comm_ring.cc)
target_link_libraries(test_comm_ring PUBLIC HPX::hpx HPX::wrap_main)

add_executable(test_comm_int test_comm_int.cc)
target_link_libraries(test_comm_int PUBLIC HPX::hpx HPX::wrap_main)