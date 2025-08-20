#ifndef SIMD_HELPER_SIMD_HELPER_H_
#define SIMD_HELPER_SIMD_HELPER_H_

#include "simd_matrix.h"
#include "simd_scalar.h"

#define ALIGN_BYTES 32
// AVX2 (512 bits = 64 Bytes), AVX (256 bits ), SSE4.2 (128 bits )
/** \internal Like malloc, but the returned pointer is guaranteed to be 32-byte
 * aligned. Fast, but wastes 32 additional bytes of memory. Does not throw any
 * exception.
 *
 * (256 bits) two LSB addresses of 32 bytes-aligned : 00, 20, 40, 60, 80, A0,
 * C0, E0
 * (128 bits) two LSB addresses of 16 bytes-aligned : 00, 10, 20, 30, 40, 50,
 * 60, 70, 80, 90, A0, B0, C0, D0, E0, F0
 */

namespace simd {

/// @brief Allocates aligned memory for a given number of data elements.
/// @tparam DataType The type of data to allocate memory for.
/// @param num_data The number of data elements to allocate memory for.
/// @return Pointer to the allocated aligned memory.
template <typename DataType>
inline DataType* GetAlignedMemory(const size_t num_data) {
  return reinterpret_cast<DataType*>(
      std::aligned_alloc(ALIGN_BYTES, num_data * sizeof(DataType)));
}

/// @brief Frees aligned memory allocated with GetAlignedMemory.
/// @tparam DataType  The type of data for which the memory was allocated.
/// @param ptr Pointer to the aligned memory to be freed.
template <typename DataType>
inline void FreeAlignedMemory(DataType* ptr) {
  if (ptr != nullptr) std::free(reinterpret_cast<void*>(ptr));
}

}  // namespace simd

#endif  // SIMD_HELPER_SIMD_HELPER_H_