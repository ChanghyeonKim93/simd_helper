#ifndef SIMD_HELPER_SOA_CONTAINER_H_
#define SIMD_HELPER_SOA_CONTAINER_H_

#include "Eigen/Dense"

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

/// @brief Structure of Arrays (SOA) container for storing data in a
/// column-major format.
/// @tparam kRow The number of rows in the container.
/// @tparam kCol The number of columns in the container.
template <int kRow, int kCol>
class SOAContainer {
 public:
  SOAContainer() {}

  SOAContainer(const int num_data) : capacity_(num_data) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] = simd::GetAlignedMemory<float>(num_data);
  }

  ~SOAContainer() {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        simd::FreeAlignedMemory<float>(data_[row][col]);
  }

  void Append(const Eigen::Matrix<float, kRow, kCol>& value) {
    if (index_ >= capacity_ - 1) return;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col][index_] = value(row, col);
    ++index_;
  }

  void Reserve(const int new_size) {
    if (new_size <= capacity_) {
      capacity_ = new_size;
      index_ = 0;
    } else {
      capacity_ = new_size;
      index_ = 0;
      for (int row = 0; row < kRow; ++row) {
        for (int col = 0; col < kCol; ++col) {
          simd::FreeAlignedMemory<float>(data_[row][col]);
          data_[row][col] = simd::GetAlignedMemory<float>(new_size);
        }
      }
    }
  }

  void Clear() { index_ = 0; }

  int GetSize() const { return index_ + 1; }

  int GetCapacity() const { return capacity_; }

  float* GetElementPtr(const int row, const int col) const {
    if (row < 0 || row >= kRow || col < 0 || col >= kCol) return nullptr;
    return data_[row][col];
  }

 private:
  float* data_[kRow][kCol] = {nullptr};
  int index_{0};
  int capacity_{-1};
};

}  // namespace simd

#endif  // SIMD_HELPER_SOA_CONTAINER_H_