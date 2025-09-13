#ifndef SIMD_HELPER_SIMD_SCALAR_H_
#define SIMD_HELPER_SIMD_SCALAR_H_

#include <iostream>
#include <limits>

#include "simd_helper/soa_container.h"

#if defined(__amd64__) || defined(__x86_64__)
#define CPU_ARCH_AMD64 1
#elif defined(__arm64__) || defined(__aarch64__)
#define CPU_ARCH_ARM 1
#endif

#if CPU_ARCH_AMD64

#include <immintrin.h>

#define SIMD_FLOAT_PACK_LANES 8
#define SIMD_INT32_PACK_LANES 8
#define SIMD_INT16_PACK_LANES 16
#define SIMD_INT8_PACK_LANES 32

using PackedFloat32 = __m256;
using PackedInt8 = __m256i;    // 8bits integer x 32
using PackedInt16 = __m256i;   // 16bits integer x 16
using PackedInt32 = __m256i;   // 32bits integer x 8
using PackedUInt8 = __m256i;   // 8bits unsigned integer x 32
using PackedUInt16 = __m256i;  // 16bits unsigned integer x 16
using PackedUInt32 = __m256i;  // 32bits unsigned integer x 8

#elif CPU_ARCH_ARM

#include <arm_neon.h>

#define SIMD_FLOAT_PACK_LANES 4
#define SIMD_INT32_PACK_LANES 4
#define SIMD_INT16_PACK_LANES 8
#define SIMD_INT8_PACK_LANES 16

using PackedFloat32 = float32x4_t;
using PackedInt8 = int8x16_t;     // 8bits integer x 16
using PackedInt16 = int16x8_t;    // 16bits integer x 8
using PackedInt32 = int32x4_t;    // 32bits integer x 4
using PackedUInt8 = uint8x16_t;   // 8bits unsigned integer x 16
using PackedUInt16 = uint16x8_t;  // 16bits unsigned integer x 8
using PackedUInt32 = uint32x4_t;  // 32bits unsigned integer x 4

#else
#error "Unsupported architecture."
#endif

namespace simd {

#if defined(CPU_ARCH_AMD64)
/// @brief Set the SIMD register with 8 float values
/// @param n1 to n8 The values to set in the SIMD register.
inline PackedFloat32 Set(float n1, float n2, float n3, float n4, float n5,
                         float n6, float n7, float n8) {
  // Note: The order of parameters is reversed to match the AMD64 intrinsic
  // function's requirement.
  return _mm256_set_ps(n8, n7, n6, n5, n4, n3, n2, n1);
}
#elif defined(CPU_ARCH_ARM)
/// @brief Set the SIMD register with 4 float values.
/// @param n1 to n4 The values to set in the SIMD register.
inline SimdFloat Set(float n1, float n2, float n3, float n4) {
  const float temp[] = {n1, n2, n3, n4};
  return vld1q_f32(temp);
}
#endif

/// @brief Broadcast a single float value to all elements of the SIMD register
/// @param input The float value to broadcast
/// @return The SIMD register with all elements set to the input value
inline PackedFloat32 Broadcast(float input) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_set1_ps(input);
#elif defined(CPU_ARCH_ARM)
  return vdupq_n_f32(input);
#endif
}

/// @brief Load float values from memory into a SIMD register
/// @param input Pointer to the float values in memory
/// @return The SIMD register containing the loaded float values
inline PackedFloat32 Load(const float* input) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_load_ps(input);
#elif defined(CPU_ARCH_ARM)
  return vld1q_f32(input);
#endif
}

/// @brief Store float values from a SIMD register into memory
/// @param output Pointer to the memory location to store the float values
/// @param input The SIMD register containing the float values to store
inline void Store(const PackedFloat32& input, float* output) {
#if defined(CPU_ARCH_AMD64)
  _mm256_store_ps(output, input);
#elif defined(CPU_ARCH_ARM)
  vst1q_f32(output, input);
#endif
}

/// @brief Add two SIMD registers element-wise
/// @param a The first SIMD register
/// @param b The second SIMD register
/// @return The result of the element-wise addition
inline PackedFloat32 Add(const PackedFloat32& a, const PackedFloat32& b) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_add_ps(a, b);
#elif defined(CPU_ARCH_ARM)
  return vaddq_f32(a, b);
#endif
}

/// @brief Subtract two SIMD registers element-wise
/// @param a The first SIMD register
/// @param b The second SIMD register
/// @return The result of the element-wise subtraction
inline PackedFloat32 Subtract(const PackedFloat32& a, const PackedFloat32& b) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_sub_ps(a, b);
#elif defined(CPU_ARCH_ARM)
  return vsubq_f32(a, b);
#endif
}

/// @brief Multiply two SIMD registers element-wise
/// @param a The first SIMD register
/// @param b The second SIMD registerSSS
inline PackedFloat32 Multiply(const PackedFloat32& a, const PackedFloat32& b) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_mul_ps(a, b);
#elif defined(CPU_ARCH_ARM)
  return vmulq_f32(a, b);
#endif
}

/// @brief Divide two SIMD registers element-wise
/// @param a The numerator SIMD register
/// @param b The denominator SIMD register
/// @return The result of the element-wise division
inline PackedFloat32 Divide(const PackedFloat32& a, const PackedFloat32& b) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_div_ps(a, b);
#elif defined(CPU_ARCH_ARM)
  return vdivq_f32(a, b);
#endif
}

/// @brief Get the reciprocal of each element in the SIMD register
/// @param input The SIMD register to compute the reciprocal for
/// @return The SIMD register containing the reciprocals of the input elements
inline PackedFloat32 GetReciprocalNumber(const PackedFloat32& input) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_rcp_ps(input);
#elif defined(CPU_ARCH_ARM)
  return vrecpeq_f32(input);
#endif
}

/// @brief Round each element in the SIMD register to the nearest integer
/// @param input The SIMD register to round
/// @return The SIMD register containing the rounded values
inline PackedFloat32 Round(const PackedFloat32& input) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_round_ps(input, _MM_FROUND_NINT);
#elif defined(CPU_ARCH_ARM)
  return vrndaq_f32(input);
#endif
}

/// @brief Compute the ceiling of each element in the SIMD register
/// @param input The SIMD register to compute the ceiling for
/// @return The SIMD register containing the ceiling values
inline PackedFloat32 Ceil(const PackedFloat32& input) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_ceil_ps(input);
#elif defined(CPU_ARCH_ARM)
  return vrndpq_f32(input);
#endif
}

/// @brief Compute the floor of each element in the SIMD register
/// @param input The SIMD register to compute the floor for
/// @return The SIMD register containing the floor values
inline PackedFloat32 Floor(const PackedFloat32& input) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_floor_ps(input);
#elif defined(CPU_ARCH_ARM)
  return vrndmq_f32(input);
#endif
}

/// @brief Compute the square root of each element in the SIMD register
/// @param input The SIMD register to compute the square root for
/// @return The SIMD register containing the square root values
inline PackedFloat32 Sqrt(const PackedFloat32& input) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_sqrt_ps(input);
#elif defined(CPU_ARCH_ARM)
  return vsqrtq_f32(input);
#endif
}

/// @brief Compute the exponential of each element in the SIMD register
/// @param a The SIMD register to compute the exponential for
/// @param b The SIMD register to compute the exponential for
/// @return The SIMD register containing the exponential values
inline PackedFloat32 Min(const PackedFloat32& a, const PackedFloat32& b) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_min_ps(a, b);
#elif defined(CPU_ARCH_ARM)
  return vminq_f32(a, b);
#endif
}

/// @brief Compute the maximum of each element in the SIMD register
/// @param a The first SIMD register
/// @param b The second SIMD register
/// @return The SIMD register containing the maximum values
inline PackedFloat32 Max(const PackedFloat32& a, const PackedFloat32& b) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_max_ps(a, b);
#elif defined(CPU_ARCH_ARM)
  return vmaxq_f32(a, b);
#endif
}

/// @brief Compute the absolute value of each element in the SIMD register
/// @param input The SIMD register to compute the absolute value for
/// @param threshold The threshold value to compare against
/// @return The SIMD register containing the absolute values
inline PackedFloat32 IsGreaterThan(const PackedFloat32& input,
                                   const PackedFloat32& threshold) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_cmp_ps(input, threshold, _CMP_GT_OS);
#elif defined(CPU_ARCH_ARM)
  return vreinterpretq_f32_u32(vcgtq_f32(input, threshold));
#endif
}

/// @brief Check if each element in the SIMD register is less than the threshold
/// @param input The SIMD register to compare
/// @param threshold The threshold value to compare against
/// @return The SIMD register containing the comparison results
inline PackedFloat32 IsLessThan(const PackedFloat32& input,
                                const PackedFloat32& threshold) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_cmp_ps(input, threshold, _CMP_LT_OS);
#elif defined(CPU_ARCH_ARM)
  return vreinterpretq_f32_u32(vcltq_f32(input, threshold));
#endif
}

/// @brief Check if each element in the SIMD register is greater than or equal
/// to the
/// @param input The SIMD register to compare
/// @param threshold The threshold value to compare against
/// @return The SIMD register containing the comparison results
inline PackedFloat32 IsGreaterThanOrEqual(const PackedFloat32& input,
                                          const PackedFloat32& threshold) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_cmp_ps(input, threshold, _CMP_GE_OS);
#elif defined(CPU_ARCH_ARM)
  return vreinterpretq_f32_u32(vcgeq_f32(input, threshold));
#endif
}

/// @brief Check if each element in the SIMD register is less than or equal to
/// the
/// @param input The SIMD register to compare
/// @param threshold The threshold value to compare against
/// @return The SIMD register containing the comparison results
inline PackedFloat32 IsLessThanOrEqual(const PackedFloat32& input,
                                       const PackedFloat32& threshold) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_cmp_ps(input, threshold, _CMP_LE_OS);
#elif defined(CPU_ARCH_ARM)
  return vreinterpretq_f32_u32(vcleq_f32(input, threshold));
#endif
}

/// @brief Select elements from two SIMD registers based on a mask
/// @param mask The mask SIMD register
/// @param value_for_true The SIMD register to select from when the mask is true
/// @param value_for_false The SIMD register to select from when the mask is
/// false
/// @return The SIMD register containing the selected values
inline PackedFloat32 Select(const PackedFloat32& mask,
                            const PackedFloat32& value_for_true,
                            const PackedFloat32& value_for_false) {
#if defined(CPU_ARCH_AMD64)
  return _mm256_blendv_ps(value_for_false, value_for_true, mask);
#elif defined(CPU_ARCH_ARM)
  return vbslq_f32(vreinterpretq_u32_f32(mask), value_for_true,
                   value_for_false);
#endif
}

PackedFloat32 __one{Broadcast(1.0f)};
PackedFloat32 __minus_one{Broadcast(-1.0f)};
PackedFloat32 __zero{Broadcast(0.0f)};

template <int kRow, int kCol>
class Matrix;

using Scalar = Matrix<1, 1>;

template <>
class Matrix<1, 1> {
 public:
  // static member methods
  static const size_t data_stride{SIMD_FLOAT_PACK_LANES};

  static inline Matrix<1, 1> Zeros() { return Matrix<1, 1>(0.0f); }

  static inline Matrix<1, 1> Ones() { return Matrix<1, 1>(1.0f); }

 public:
  // Initialization & Assignment operations

  /// @brief Default constructor initializes all elements to zero.
  Matrix<1, 1>() { data_ = __zero; }

  /// @brief Constructor initializes all elements to the given input value.
  /// @param input The value to initialize all elements of the matrix.
  explicit Matrix<1, 1>(const float input) { data_ = Broadcast(input); }

#if defined(CPU_ARCH_AMD64)
  /// @brief Constructor initializes the matrix with 8 float values.
  /// @param n1_to_n8 The values to initialize the matrix.
  Matrix<1, 1>(const float n1, const float n2, const float n3, const float n4,
               const float n5, const float n6, const float n7, const float n8) {
    data_ = Set(n8, n7, n6, n5, n4, n3, n2, n1);
  }
#elif defined(CPU_ARCH_ARM)
  /// @brief Constructor initializes the matrix with 4 float values.
  /// @param n1_to_n4 The values to initialize the matrix.
  Matrix<1, 1>(const float n1, const float n2, const float n3, const float n4) {
    const float temp[] = {n1, n2, n3, n4};
    data_ = Load(temp);
  }
#endif

  /// @brief Copy constructor initializes the matrix with another matrix.
  /// @param rhs The matrix to copy from
  Matrix<1, 1>(const Matrix<1, 1>& rhs) { data_ = rhs.data_; }

  /// @brief Move constructor initializes the matrix with another matrix.
  /// @param rhs The matrix to move from
  Matrix<1, 1>(const Matrix<1, 1>&& rhs) { data_ = std::move(rhs.data_); }

  /// @brief Constructor initializes the matrix with a pointer to float data.
  /// @param rhs Pointer to float data to initialize the matrix.
  explicit Matrix<1, 1>(const float* rhs) { data_ = Load(rhs); }

  /// @brief Constructor initializes the matrix with a SIMD data type.
  /// @param rhs The SIMD data type to initialize the matrix.
  explicit Matrix<1, 1>(const PackedFloat32& rhs) { data_ = rhs; }

  Matrix(const SOAContainer<1, 1>& soa_container, const int start_index) {
    data_ = Load(soa_container.GetElementPtr(0, 0) + start_index);
  }

  /// @brief Assignment operator to set the matrix to a scalar value.
  /// @param rhs The scalar value to assign to the matrix.
  /// @return Reference to the current matrix.
  Matrix<1, 1>& operator=(const float rhs) {
    data_ = Broadcast(rhs);
    return *this;
  }

  Matrix<1, 1>& operator=(const Matrix<1, 1>& rhs) {
    data_ = rhs.data_;
    return *this;
  }

  Matrix<1, 1>& operator=(const Matrix<1, 1>&& rhs) {
    data_ = std::move(rhs.data_);
    return *this;
  }

  // Comparison operations

  Matrix<1, 1> operator<(const float scalar) const {
    return Matrix<1, 1>(
        Select(IsLessThan(data_, Broadcast(scalar)), __one, __zero));
  }

  Matrix<1, 1> operator<=(const float scalar) const {
    return Matrix<1, 1>(
        Select(IsLessThanOrEqual(data_, Broadcast(scalar)), __one, __zero));
  }

  Matrix<1, 1> operator>(const float scalar) const {
    return Matrix<1, 1>(
        Select(IsGreaterThan(data_, Broadcast(scalar)), __one, __zero));
  }

  Matrix<1, 1> operator>=(const float scalar) const {
    return Matrix<1, 1>(
        Select(IsGreaterThanOrEqual(data_, Broadcast(scalar)), __one, __zero));
  }

  Matrix<1, 1> operator<(const Matrix<1, 1>& rhs) const {
    return Matrix<1, 1>(Select(IsLessThan(data_, rhs.data_), __one, __zero));
  }

  Matrix<1, 1> operator<=(const Matrix<1, 1>& rhs) const {
    return Matrix<1, 1>(
        Select(IsLessThanOrEqual(data_, rhs.data_), __one, __zero));
  }

  Matrix<1, 1> operator>(const Matrix<1, 1>& rhs) const {
    return Matrix<1, 1>(Select(IsGreaterThan(data_, rhs.data_), __one, __zero));
  }

  Matrix<1, 1> operator>=(const Matrix<1, 1>& rhs) const {
    return Matrix<1, 1>(
        Select(IsGreaterThanOrEqual(data_, rhs.data_), __one, __zero));
  }

  // Arithmetic operations
  Matrix<1, 1> operator+() const { return Matrix<1, 1>(*this); }

  Matrix<1, 1> operator-() const {
    return Matrix<1, 1>(Subtract(__zero, data_));
  }

  Matrix<1, 1> operator+(const float rhs) const {
    return Matrix<1, 1>(Add(data_, Broadcast(rhs)));
  }

  Matrix<1, 1> operator-(const float rhs) const {
    return Matrix<1, 1>(Subtract(data_, Broadcast(rhs)));
  }

  Matrix<1, 1> operator*(const float rhs) const {
    return Matrix<1, 1>(Multiply(data_, Broadcast(rhs)));
  }

  Matrix<1, 1> operator/(const float rhs) const {
    return Matrix<1, 1>(Divide(data_, Broadcast(rhs)));
  }

  friend Matrix<1, 1> operator+(const float lhs, const Matrix<1, 1>& rhs) {
    return Matrix<1, 1>(Add(Broadcast(lhs), rhs.data_));
  }

  friend Matrix<1, 1> operator-(const float lhs, const Matrix<1, 1>& rhs) {
    return Matrix<1, 1>(Subtract(Broadcast(lhs), rhs.data_));
  }

  friend Matrix<1, 1> operator*(const float lhs, const Matrix<1, 1>& rhs) {
    return Matrix<1, 1>(Multiply(Broadcast(lhs), rhs.data_));
  }

  friend Matrix<1, 1> operator/(const float lhs, const Matrix<1, 1>& rhs) {
    return Matrix<1, 1>(Divide(Broadcast(lhs), rhs.data_));
  }

  Matrix<1, 1> operator+(const Matrix<1, 1>& rhs) const {
    return Matrix<1, 1>(Add(data_, rhs.data_));
  }

  Matrix<1, 1> operator-(const Matrix<1, 1>& rhs) const {
    return Matrix<1, 1>(Subtract(data_, rhs.data_));
  }

  Matrix<1, 1> operator*(const Matrix<1, 1>& rhs) const {
    return Matrix<1, 1>(Multiply(data_, rhs.data_));
  }

  Matrix<1, 1> operator/(const Matrix<1, 1>& rhs) const {
    return Matrix<1, 1>(Divide(data_, rhs.data_));
  }

  // Compound assignment operations
  Matrix<1, 1>& operator+=(const Matrix<1, 1>& rhs) {
    data_ = Add(data_, rhs.data_);
    return *this;
  }

  Matrix<1, 1>& operator-=(const Matrix<1, 1>& rhs) {
    data_ = Subtract(data_, rhs.data_);
    return *this;
  }

  Matrix<1, 1>& operator*=(const Matrix<1, 1>& rhs) {
    data_ = Multiply(data_, rhs.data_);
    return *this;
  }

  /// @brief Returns the square root of the matrix elements.
  /// @return Matrix<1, 1> with square root of the elements.
  Matrix<1, 1> sqrt() const { return Matrix<1, 1>(Sqrt(data_)); }

  /// @brief Returns the sign of the matrix elements.
  /// @return Matrix<1, 1> with elements set to 1.0f for positive elements and
  /// -1.0f for negative elements.
  Matrix<1, 1> sign() const {
    const PackedFloat32 positive_mask = IsGreaterThanOrEqual(data_, __zero);
    return Matrix<1, 1>(Select(positive_mask, __one, __minus_one));
  }

  /// @brief Returns the absolute value of the matrix elements.
  /// @return Matrix<1, 1> with absolute values of the elements.
  Matrix<1, 1> abs() const {
    // Use bitwise AND to clear the sign bit
#if defined(CPU_ARCH_AMD64)
    const PackedFloat32 minus_zero__ = Broadcast(-0.0f);
    const PackedFloat32 result = _mm256_andnot_ps(minus_zero__, data_);
#elif defined(CPU_ARCH_ARM)
    uint32x4_t sign_mask = vdupq_n_u32(0x7FFFFFFF);
    uint32x4_t data_as_int = vreinterpretq_u32_f32(data_);
    uint32x4_t abs_as_int = vandq_u32(data_as_int, sign_mask);
    SimdFloat result = vreinterpretq_f32_u32(abs_as_int);
#endif
    return Matrix<1, 1>(result);
  }

  /// @brief Compute exponential of each element. In this implementation, we use
  /// range reduction method.  e^x = 2^n * e^r where n = floor(x / ln(2)),
  /// r = x - n * ln(2)
  Matrix<1, 1> exp() const {
    constexpr float kMinValidInput = -87.0f;
    constexpr float kMaxValidInput = 88.0f;
    const PackedFloat32 ln2 = Broadcast(0.693147180559945f);     // ln(2)
    const PackedFloat32 inv_ln2 = Broadcast(1.44269504088896f);  // 1/ln(2)

    const PackedFloat32 min_valid_input = Broadcast(kMinValidInput);
    const PackedFloat32 max_valid_input = Broadcast(kMaxValidInput);
    const PackedFloat32 clamped_x =
        Min(Max(data_, min_valid_input), max_valid_input);

    const auto& x = clamped_x;

    // Calculate n = floor(x/ln(2))
    const PackedFloat32 n_float =
        Floor(Multiply(x, inv_ln2));  // n = floor(x/ln(2))

    // Calculate r = x - n*ln(2)
    const PackedFloat32 r__ = Subtract(x, Multiply(n_float, ln2));

    // e^r ≈ 1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5! + r^6/6! + r^7/7!
    PackedFloat32 exp_r__ = Broadcast(1.0f / 5040.0f);
    exp_r__ = Add(Multiply(exp_r__, r__), Broadcast(1.0f / 720.0f));
    exp_r__ = Add(Multiply(exp_r__, r__), Broadcast(1.0f / 120.0f));
    exp_r__ = Add(Multiply(exp_r__, r__), Broadcast(1.0f / 24.0f));
    exp_r__ = Add(Multiply(exp_r__, r__), Broadcast(1.0f / 6.0f));
    exp_r__ = Add(Multiply(exp_r__, r__), Broadcast(1.0f / 2.0f));
    exp_r__ = Add(Multiply(exp_r__, r__), Broadcast(1.0f / 1.0f));
    exp_r__ = Add(Multiply(exp_r__, r__), Broadcast(1.0f));

    // Calculate 2^n using bit manipulation. Convert n to integer type for bit
    // shifting
#if defined(CPU_ARCH_AMD64)
    PackedInt32 n_int = _mm256_cvtps_epi32(n_float);
    // 2^n is computed by adding n to the exponent field of 1.0
    // Shift n left by 23 bits (exponent position in IEEE-754)
    __m256i exponent_bits = _mm256_slli_epi32(n_int, 23);

    // For positive n, 2^n = (1.0 * 2^n)
    // For negative n, 2^n = (1.0 * 2^n) = 1.0 / 2^(-n)
    PackedFloat32 pow2n = _mm256_castsi256_ps(
        _mm256_add_epi32(exponent_bits, _mm256_castps_si256(Broadcast(1.0f))));
#elif defined(CPU_ARCH_ARM)
    PackedInt32 n_int = vcvtq_s32_f32(n_float);

    // 2^n calculation
    PackedInt32 exponent_bits = vshlq_n_s32(n_int, 23);
    SimdFloat pow2n = vreinterpretq_f32_s32(
        vaddq_s32(exponent_bits, vreinterpretq_s32_f32(Broadcast(1.0f))));
#endif
    PackedFloat32 result = Multiply(pow2n, exp_r__);

    // Handle overflow and underflow
#if defined(CPU_ARCH_AMD64)
    PackedFloat32 is_too_small = IsLessThan(data_, min_valid_input);
    result = _mm256_andnot_ps(is_too_small, result);
    PackedFloat32 is_too_large = IsGreaterThan(data_, max_valid_input);
    PackedFloat32 inf_val =
        _mm256_set1_ps(std::numeric_limits<float>::infinity());
    result = _mm256_blendv_ps(result, inf_val, is_too_large);
#elif defined(CPU_ARCH_ARM)
    PackedUInt32 is_too_small = vcltq_f32(data_, min_valid_input);
    result = vreinterpretq_f32_u32(
        vbicq_u32(vreinterpretq_u32_f32(result), is_too_small));
    PackedUInt32 is_too_large = vcgtq_f32(data_, max_valid_input);
    SimdFloat inf_val = Broadcast(std::numeric_limits<float>::infinity());
    result = vbslq_f32(is_too_large, inf_val, result);
#endif

    // e^x = 2^n * e^r
    return Matrix<1, 1>(result);
  }

  /// @brief Stores the SIMD data to normal memory.
  /// @param normal_memory Pointer to the normal memory where the data will be
  /// stored.
  void StoreData(float* normal_memory) const { Store(data_, normal_memory); }

  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const Matrix<1, 1>& scalar) {
    float multi_scalars[Scalar::data_stride];
    scalar.StoreData(multi_scalars);
    std::cout << "[";
    for (int i = 0; i < Scalar::data_stride; ++i) {
      std::cout << "[" << multi_scalars[i] << "]";
      if (i != Scalar::data_stride - 1) std::cout << ",\n";
    }
    std::cout << "]" << std::endl;
    return outputStream;
  }

 private:
  PackedFloat32 data_;
};

inline Scalar abs(const Scalar& input) { return input.abs(); }

inline Scalar sqrt(const Scalar& input) { return input.sqrt(); }

inline Scalar exp(const Scalar& input) { return input.exp(); }

}  // namespace simd

#endif  // SIMD_HELPER_SIMD_SCALAR_H_
