
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

add_executable(local_async local_async.cpp)
target_link_libraries(local_async PUBLIC HPX::hpx HPX::wrap_main core_s)

add_executable(local_par_for local_par_for.cpp)
target_link_libraries(local_par_for PUBLIC HPX::hpx HPX::wrap_main core_s)