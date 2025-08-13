cmake_minimum_required(VERSION 3.14)
project(simd_helper_test)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")

find_package(Eigen3 REQUIRED)
find_package(simd_helper REQUIRED)

add_executable(simd_helper_test main.cc)
target_include_directories(simd_helper_test PUBLIC ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(simd_helper_test PUBLIC simd_helper::simd_helper)