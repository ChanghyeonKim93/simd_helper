cmake_minimum_required(VERSION 3.14)
project(simd_helper VERSION 0.1.0)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")

find_package(Eigen3 REQUIRED)

add_library(simd_helper INTERFACE)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  message("-- Detected architecture: x86_64")
  target_compile_options(simd_helper INTERFACE "-std=c++17" "-O2" "-march=native" "-msse" "-mavx" "-mavx2")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("-- Detected architecture: ARM64 (aarch64)")
  target_compile_options(simd_helper INTERFACE "-std=c++17" "-O2" "-ffast-math" "-ftree-vectorize" "-march=armv8-a+simd" "-mtune=cortex-a72")
else()
  message("-- Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

target_include_directories(simd_helper
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(simd_helper INTERFACE Eigen3::Eigen)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(FILES
  simd_helper.h
  simd_matrix.h
  simd_scalar.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/simd_helper
)

install(TARGETS simd_helper
  EXPORT simd_helper-targets 
)

install(EXPORT simd_helper-targets
  FILE      simd_helper-config.cmake
  NAMESPACE simd_helper::
  DESTINATION lib/cmake/simd_helper
)

# ===================================================================
# Test codes
# ===================================================================
find_package(GTest REQUIRED)

add_executable(simd_helper_test simd_helper_test.cc)

target_link_libraries(
  simd_helper_test
  PUBLIC GTest::gtest_main
         simd_helper
)