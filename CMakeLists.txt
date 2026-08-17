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
  target_compile_options(simd_helper INTERFACE "-mavx" "-mavx2")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("-- Detected architecture: ARM64 (aarch64)")
  target_compile_options(simd_helper INTERFACE "-march=armv8-a+simd")
else()
  message("-- Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

target_compile_features(simd_helper INTERFACE cxx_std_17)

target_include_directories(
  simd_helper INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                        $<INSTALL_INTERFACE:include>)
target_link_libraries(simd_helper INTERFACE Eigen3::Eigen)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(FILES simd_helper/simd_helper.h simd_helper/simd_matrix.h
              simd_helper/simd_scalar.h simd_helper/soa_container.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/simd_helper)

install(TARGETS simd_helper EXPORT simd_helper-targets)

install(
  EXPORT simd_helper-targets
  FILE simd_helper-targets.cmake
  NAMESPACE simd_helper::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simd_helper)

configure_package_config_file(
  cmake/simd_helper-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/simd_helper-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simd_helper)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/simd_helper-config-version.cmake
  COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/simd_helper-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/simd_helper-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simd_helper)

# ===================================================================
# Test codes
# ===================================================================
option(BUILD_TESTING "Build unit tests (requires GTest)" ON)

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  enable_testing()

  add_executable(${PROJECT_NAME}_calculation_test test/calculation_test.cc)
  target_link_libraries(${PROJECT_NAME}_calculation_test
                        PUBLIC GTest::gtest_main simd_helper)
  add_test(NAME calculation_test COMMAND ${PROJECT_NAME}_calculation_test)
endif()
