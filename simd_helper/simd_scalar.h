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

#if defined(CPU_ARCH_AMD64)
#include <immintrin.h>

#define __SIMD_DATA_STRIDE 8

using _s_data = __m256;

#define _s_set _mm256_set_ps
#define _s_set1 _mm256_set1_ps
#define _s_load _mm256_load_ps
#define _s_add _mm256_add_ps
#define _s_sub _mm256_sub_ps
#define _s_mul _mm256_mul_ps
#define _s_div _mm256_div_ps
#define _s_rcp _mm256_rcp_ps
#define _s_store _mm256_store_ps
#define _s_round(input) _mm256_round_ps((input), _MM_FROUND_NINT)
#define _s_ceil _mm256_ceil_ps
#define _s_floor _mm256_floor_ps
#define _s_sqrt _mm256_sqrt_ps
#define _s_min _mm256_min_ps
#define _s_max _mm256_max_ps

#elif defined(CPU_ARCH_ARM)
#include <arm_neon.h>

#define __SIMD_DATA_STRIDE 4

using _s_data = float32x4_t;

#define _s_set vsetq_f32
#define _s_set1 vdupq_n_f32
#define _s_load vld1q_f32
#define _s_add vaddq_f32
#define _s_sub vsubq_f32
#define _s_mul vmulq_f32
#define _s_div vdivq_f32
#define _s_rcp vrecpeq_f32
#define _s_store vst1q_f32
#define _s_round vrndaq_f32
#define _s_ceil vrndpq_f32
#define _s_floor vrndmq_f32
#define _s_sqrt vsqrtq_f32
#define _s_min vminq_f32
#define _s_max vmaxq_f32

#else
#error "Unsupported architecture."
#endif

_s_data __one{_s_set1(1.0f)};
_s_data __minus_one{_s_set1(-1.0f)};
_s_data __zero{_s_set1(0.0f)};

namespace simd {

template <int kRow, int kCol>
class Matrix;

using Scalar = Matrix<1, 1>;

template <>
class Matrix<1, 1> {
 public:
  // static member methods
  static const size_t data_stride{__SIMD_DATA_STRIDE};

  static inline Matrix<1, 1> Zeros() { return Matrix<1, 1>(0.0f); }

  static inline Matrix<1, 1> Ones() { return Matrix<1, 1>(1.0f); }

 public:
  // Initialization & Assignment operations

  /// @brief Default constructor initializes all elements to zero.
  Matrix<1, 1>() { data_ = __zero; }

  /// @brief Constructor initializes all elements to the given input value.
  /// @param input The value to initialize all elements of the matrix.
  explicit Matrix<1, 1>(const float input) { data_ = _s_set1(input); }

#if defined(CPU_ARCH_AMD64)
  /// @brief Constructor initializes the matrix with 8 float values.
  /// @param n1_to_n8 The values to initialize the matrix.
  Matrix<1, 1>(const float n1, const float n2, const float n3, const float n4,
               const float n5, const float n6, const float n7, const float n8) {
    data_ = _s_set(n8, n7, n6, n5, n4, n3, n2, n1);
  }
#elif defined(CPU_ARCH_ARM)
  /// @brief Constructor initializes the matrix with 4 float values.
  /// @param n1_to_n4 The values to initialize the matrix.
  Matrix<1, 1>(const float n1, const float n2, const float n3, const float n4) {
    const float temp[] = {n1, n2, n3, n4};
    data_ = vld1q_f32(temp);
  }
#endif

  /// @brief Copy constructor initializes the matrix with another matrix.
  /// @param rhs The matrix to copy from.
  Matrix<1, 1>(const Matrix<1, 1>& rhs) { data_ = rhs.data_; }

  /// @brief Constructor initializes the matrix with a pointer to float data.
  /// @param rhs Pointer to float data to initialize the matrix.
  explicit Matrix<1, 1>(const float* rhs) { data_ = _s_load(rhs); }

  /// @brief Constructor initializes the matrix with a SIMD data type.
  /// @param rhs The SIMD data type to initialize the matrix.
  explicit Matrix<1, 1>(const _s_data& rhs) { data_ = rhs; }

  Matrix(const SOAContainer<1, 1>& soa_container, const int start_index) {
    data_ = _s_load(soa_container.GetElementPtr(0, 0) + start_index);
  }

  /// @brief Assignment operator to set the matrix to a scalar value.
  /// @param rhs The scalar value to assign to the matrix.
  /// @return Reference to the current matrix.
  Matrix<1, 1>& operator=(const float rhs) {
    data_ = _s_set1(rhs);
    return *this;
  }

  Matrix<1, 1>& operator=(const Matrix<1, 1>& rhs) {
    data_ = rhs.data_;
    return *this;
  }

  // Comparison operations

  Matrix<1, 1> operator<(const float scalar) const {
#if defined(CPU_ARCH_AMD64)
    return Matrix<1, 1>(_mm256_and_ps(
        _mm256_cmp_ps(data_, _s_set1(scalar), _CMP_LT_OS), __one));
#elif defined(CPU_ARCH_ARM)
    return Matrix<1, 1>(
        vbslq_f32(vcltq_f32(data_, vdupq_n_f32(scalar)), __one, __zero));
#endif
  }

  Matrix<1, 1> operator<=(const float scalar) const {
#if defined(CPU_ARCH_AMD64)
    return Matrix<1, 1>(_mm256_and_ps(
        _mm256_cmp_ps(data_, _s_set1(scalar), _CMP_LE_OS), __one));
#elif defined(CPU_ARCH_ARM)
    return Matrix<1, 1>(
        vbslq_f32(vcleq_f32(data_, vdupq_n_f32(scalar)), __one, __zero));
#endif
  }

  Matrix<1, 1> operator>(const float scalar) const {
#if defined(CPU_ARCH_AMD64)
    return Matrix<1, 1>(_mm256_and_ps(
        _mm256_cmp_ps(data_, _s_set1(scalar), _CMP_GT_OS), __one));
#elif defined(CPU_ARCH_ARM)
    return Matrix<1, 1>(
        vbslq_f32(vcgtq_f32(data_, vdupq_n_f32(scalar)), __one, __zero));
#endif
  }

  Matrix<1, 1> operator>=(const float scalar) const {
#if defined(CPU_ARCH_AMD64)
    return Matrix<1, 1>(_mm256_and_ps(
        _mm256_cmp_ps(data_, _s_set1(scalar), _CMP_GE_OS), __one));
#elif defined(CPU_ARCH_ARM)
    return Matrix<1, 1>(Matrix<1, 1>(
        vbslq_f32(vcgeq_f32(data_, vdupq_n_f32(scalar)), __one, __zero)));
#endif
  }

  Matrix<1, 1> operator<(const Matrix<1, 1>& rhs) const {
#if defined(CPU_ARCH_AMD64)
    return Matrix<1, 1>(
        _mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_LT_OS), __one));
#elif defined(CPU_ARCH_ARM)
    return Matrix<1, 1>(vbslq_f32(vcltq_f32(data_, rhs.data_), __one, __zero));
#endif
  }

  Matrix<1, 1> operator<=(const Matrix<1, 1>& rhs) const {
#if defined(CPU_ARCH_AMD64)
    return Matrix<1, 1>(
        _mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_LE_OS), __one));
#elif defined(CPU_ARCH_ARM)
    return Matrix<1, 1>(vbslq_f32(vcleq_f32(data_, rhs.data_), __one, __zero));
#endif
  }

  Matrix<1, 1> operator>(const Matrix<1, 1>& rhs) const {
#if defined(CPU_ARCH_AMD64)
    return Matrix<1, 1>(
        _mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_GT_OS), __one));
#elif defined(CPU_ARCH_ARM)
    return Matrix<1, 1>(vbslq_f32(vcgtq_f32(data_, rhs.data_), __one, __zero));
#endif
  }

  Matrix<1, 1> operator>=(const Matrix<1, 1>& rhs) const {
#if defined(CPU_ARCH_AMD64)
    return Matrix<1, 1>(
        _mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_GE_OS), __one));
#elif defined(CPU_ARCH_ARM)
    return Matrix<1, 1>(vbslq_f32(vcgeq_f32(data_, rhs.data_), __one, __zero));
#endif
  }

  // Arithmetic operations
  Matrix<1, 1> operator+() const { return Matrix<1, 1>(*this); }

  Matrix<1, 1> operator-() const { return Matrix<1, 1>(_s_sub(__zero, data_)); }

  Matrix<1, 1> operator+(const float rhs) const {
    return Matrix<1, 1>(_s_add(data_, _s_set1(rhs)));
  }

  Matrix<1, 1> operator-(const float rhs) const {
    return Matrix<1, 1>(_s_sub(data_, _s_set1(rhs)));
  }

  Matrix<1, 1> operator*(const float rhs) const {
    return Matrix<1, 1>(_s_mul(data_, _s_set1(rhs)));
  }

  Matrix<1, 1> operator/(const float rhs) const {
    return Matrix<1, 1>(_s_div(data_, _s_set1(rhs)));
  }

  friend Matrix<1, 1> operator+(const float lhs, const Matrix<1, 1>& rhs) {
    return Matrix<1, 1>(_s_add(_s_set1(lhs), rhs.data_));
  }

  friend Matrix<1, 1> operator-(const float lhs, const Matrix<1, 1>& rhs) {
    return Matrix<1, 1>(_s_sub(_s_set1(lhs), rhs.data_));
  }

  friend Matrix<1, 1> operator*(const float lhs, const Matrix<1, 1>& rhs) {
    return Matrix<1, 1>(_s_mul(_s_set1(lhs), rhs.data_));
  }

  friend Matrix<1, 1> operator/(const float lhs, const Matrix<1, 1>& rhs) {
    return Matrix<1, 1>(_s_div(_s_set1(lhs), rhs.data_));
  }

  Matrix<1, 1> operator+(const Matrix<1, 1>& rhs) const {
    return Matrix<1, 1>(_s_add(data_, rhs.data_));
  }

  Matrix<1, 1> operator-(const Matrix<1, 1>& rhs) const {
    return Matrix<1, 1>(_s_sub(data_, rhs.data_));
  }

  Matrix<1, 1> operator*(const Matrix<1, 1>& rhs) const {
    return Matrix<1, 1>(_s_mul(data_, rhs.data_));
  }

  Matrix<1, 1> operator/(const Matrix<1, 1>& rhs) const {
    return Matrix<1, 1>(_s_div(data_, rhs.data_));
  }

  // Compound assignment operations
  Matrix<1, 1>& operator+=(const Matrix<1, 1>& rhs) {
    data_ = _s_add(data_, rhs.data_);
    return *this;
  }

  Matrix<1, 1>& operator-=(const Matrix<1, 1>& rhs) {
    data_ = _s_sub(data_, rhs.data_);
    return *this;
  }

  Matrix<1, 1>& operator*=(const Matrix<1, 1>& rhs) {
    data_ = _s_mul(data_, rhs.data_);
    return *this;
  }

  /// @brief Returns the square root of the matrix elements.
  /// @return Matrix<1, 1> with square root of the elements.
  Matrix<1, 1> sqrt() const { return Matrix<1, 1>(_s_sqrt(data_)); }

  /// @brief Returns the sign of the matrix elements.
  /// @return Matrix<1, 1> with elements set to 1.0f for positive elements and
  /// -1.0f for negative elements.
  Matrix<1, 1> sign() const {
#if defined(CPU_ARCH_AMD64)
    __m256 is_positive =
        _mm256_cmp_ps(data_, __zero, _CMP_GE_OS);  // data_ >= 0.0
    __m256 result = _mm256_blendv_ps(__minus_one, __one, is_positive);
#elif defined(CPU_ARCH_ARM)
    // Compare data_ >= 0.0
    uint32x4_t is_positive = vcgeq_f32(data_, __zero);  // data_ >= 0.0
    float32x4_t result = vbslq_f32(is_positive, __one, __minus_one);
#endif
    return Matrix<1, 1>(result);
  }

  /// @brief Returns the absolute value of the matrix elements.
  /// @return Matrix<1, 1> with absolute values of the elements.
  Matrix<1, 1> abs() const {
    // Use bitwise AND to clear the sign bit
#if defined(CPU_ARCH_AMD64)
    const _s_data minus_zero__ = _s_set1(-0.0f);
    const _s_data result = _mm256_andnot_ps(minus_zero__, data_);
#elif defined(CPU_ARCH_ARM)
    uint32x4_t sign_mask = vdupq_n_u32(0x7FFFFFFF);
    uint32x4_t data_as_int = vreinterpretq_u32_f32(data_);
    uint32x4_t abs_as_int = vandq_u32(data_as_int, sign_mask);
    _s_data result = vreinterpretq_f32_u32(abs_as_int);
#endif
    return Matrix<1, 1>(result);
  }

  /// @brief Compute exponential of each element. In this implementation, we use
  /// range reduction method.  e^x = 2^n * e^r where n = floor(x / ln(2)),
  /// r = x - n * ln(2)
  Matrix<1, 1> exp() const {
    const float min_input = -87.0f;
    const float max_input = 88.0f;

    _s_data min_val = _s_set1(min_input);
    _s_data max_val = _s_set1(max_input);
    _s_data clamped_x = _s_min(_s_max(data_, min_val), max_val);

    // Precalculate constants
    const _s_data ln2 = _s_set1(0.693147180559945f);     // ln(2)
    const _s_data inv_ln2 = _s_set1(1.44269504088896f);  // 1/ln(2)

    const auto& x = clamped_x;

    // Calculate n = floor(x/ln(2))
    _s_data n_float = _s_floor(_s_mul(x, inv_ln2));  // n = floor(x/ln(2))

    // Calculate r = x - n*ln(2)
    _s_data r__ = _s_sub(x, _s_mul(n_float, ln2));

    // e^r ≈ 1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5! + r^6/6! + r^7/7!
    _s_data exp_r__ = _s_set1(1.0f / 5040.0f);
    exp_r__ = _s_add(_s_mul(exp_r__, r__), _s_set1(1.0f / 720.0f));
    exp_r__ = _s_add(_s_mul(exp_r__, r__), _s_set1(1.0f / 120.0f));
    exp_r__ = _s_add(_s_mul(exp_r__, r__), _s_set1(1.0f / 24.0f));
    exp_r__ = _s_add(_s_mul(exp_r__, r__), _s_set1(1.0f / 6.0f));
    exp_r__ = _s_add(_s_mul(exp_r__, r__), _s_set1(1.0f / 2.0f));
    exp_r__ = _s_add(_s_mul(exp_r__, r__), _s_set1(1.0f / 1.0f));
    exp_r__ = _s_add(_s_mul(exp_r__, r__), _s_set1(1.0f));

    // Calculate 2^n using bit manipulation. Convert n to integer type for bit
    // shifting
#if defined(CPU_ARCH_AMD64)
    __m256i n_int = _mm256_cvtps_epi32(n_float);
    // 2^n is computed by adding n to the exponent field of 1.0
    // Shift n left by 23 bits (exponent position in IEEE-754)
    __m256i exponent_bits = _mm256_slli_epi32(n_int, 23);

    // For positive n, 2^n = (1.0 * 2^n)
    // For negative n, 2^n = (1.0 * 2^n) = 1.0 / 2^(-n)
    _s_data pow2n = _mm256_castsi256_ps(
        _mm256_add_epi32(exponent_bits, _mm256_castps_si256(_s_set1(1.0f))));
#elif defined(CPU_ARCH_ARM)
    int32x4_t n_int = vcvtq_s32_f32(n_float);

    // 2^n calculation
    int32x4_t exponent_bits = vshlq_n_s32(n_int, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(
        vaddq_s32(exponent_bits, vreinterpretq_s32_f32(vdupq_n_f32(1.0f))));
#endif
    _s_data result = _s_mul(pow2n, exp_r__);

    // Handle overflow and underflow
#if defined(CPU_ARCH_AMD64)
    _s_data is_too_small = _mm256_cmp_ps(data_, min_val, _CMP_LT_OS);
    result = _mm256_andnot_ps(is_too_small, result);
    _s_data is_too_large = _mm256_cmp_ps(data_, max_val, _CMP_GT_OS);
    _s_data inf_val = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    result = _mm256_blendv_ps(result, inf_val, is_too_large);
#elif defined(CPU_ARCH_ARM)
    uint32x4_t is_too_small = vcltq_f32(data_, min_val);
    result = vreinterpretq_f32_u32(
        vbicq_u32(vreinterpretq_u32_f32(result), is_too_small));
    uint32x4_t is_too_large = vcgtq_f32(data_, max_val);
    float32x4_t inf_val = vdupq_n_f32(std::numeric_limits<float>::infinity());
    result = vbslq_f32(is_too_large, inf_val, result);
#endif

    // e^x = 2^n * e^r
    return Matrix<1, 1>(result);
  }

  /// @brief Stores the SIMD data to normal memory.
  /// @param normal_memory Pointer to the normal memory where the data will be
  /// stored.
  void StoreData(float* normal_memory) const { _s_store(normal_memory, data_); }

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
  _s_data data_;
};

inline Scalar abs(const Scalar& input) { return input.abs(); }

inline Scalar sqrt(const Scalar& input) { return input.sqrt(); }

inline Scalar exp(const Scalar& input) { return input.exp(); }

}  // namespace simd

#endif  // SIMD_HELPER_SIMD_SCALAR_H_
