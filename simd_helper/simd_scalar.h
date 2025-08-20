#ifndef SIMD_HELPER_SIMD_SCALAR_H_
#define SIMD_HELPER_SIMD_SCALAR_H_

#include <iostream>

#if defined(__amd64__) || defined(__x86_64__)
#define CPU_ARCH_AMD64 1
#elif defined(__arm64__) || defined(__aarch64__)
#define CPU_ARCH_ARM 1
#endif

namespace {

#if defined(CPU_ARCH_AMD64)
#include "immintrin.h"

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

#elif defined(CPU_ARCH_ARM)
#include "arm_neon.h"

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

#else
#error \
    "Unsupported architecture. Please define either __amd64__, __x86_64__, __arm64__, or __aarch64__."
#endif

_s_data __one{_s_set1(1.0f)};
_s_data __minus_one{_s_set1(-1.0f)};
_s_data __zero{_s_set1(0.0f)};

}  // namespace

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
  Matrix<1, 1>(const float input) { data_ = _s_set1(input); }

#if defined(CPU_ARCH_AMD64)
  Matrix<1, 1>(const float n1, const float n2, const float n3, const float n4,
               const float n5, const float n6, const float n7, const float n8) {
    data_ = _s_set(n8, n7, n6, n5, n4, n3, n2, n1);
  }
#elif defined(CPU_ARCH_ARM)
  MatrixBase<1, 1>(const float n1, const float n2, const float n3,
                   const float n4) {
    const float temp[] = {n1, n2, n3, n4};
    data_ = vld1q_f32(temp);
  }
#endif

  Matrix<1, 1>(const Matrix<1, 1>& rhs) { data_ = rhs.data_; }

  Matrix<1, 1>(const float* rhs) { data_ = _s_load(rhs); }

  Matrix<1, 1>(const _s_data& rhs) { data_ = rhs; }

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
    return _mm256_and_ps(_mm256_cmp_ps(data_, _s_set1(scalar), _CMP_LT_OS),
                         __one);
#elif defined(CPU_ARCH_ARM)
    return vbslq_f32(vcltq_f32(data_, vdupq_n_f32(scalar)), __one, __zero);
#endif
  }

  Matrix<1, 1> operator<=(const float scalar) const {
#if defined(CPU_ARCH_AMD64)
    return _mm256_and_ps(_mm256_cmp_ps(data_, _s_set1(scalar), _CMP_LE_OS),
                         __one);
#elif defined(CPU_ARCH_ARM)
    return vbslq_f32(vcleq_f32(data_, vdupq_n_f32(scalar)), __one, __zero);
#endif
  }

  Matrix<1, 1> operator>(const float scalar) const {
#if defined(CPU_ARCH_AMD64)
    return _mm256_and_ps(_mm256_cmp_ps(data_, _s_set1(scalar), _CMP_GT_OS),
                         __one);
#elif defined(CPU_ARCH_ARM)
    return vbslq_f32(vcgtq_f32(data_, vdupq_n_f32(scalar)), __one, __zero);
#endif
  }

  Matrix<1, 1> operator>=(const float scalar) const {
#if defined(CPU_ARCH_AMD64)
    return _mm256_and_ps(_mm256_cmp_ps(data_, _s_set1(scalar), _CMP_GE_OS),
                         __one);
#elif defined(CPU_ARCH_ARM)
    return vbslq_f32(vcgeq_f32(data_, vdupq_n_f32(scalar)), __one, __zero);
#endif
  }

  Matrix<1, 1> operator<(const Matrix<1, 1>& rhs) const {
#if defined(CPU_ARCH_AMD64)
    return _mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_LT_OS), __one);
#elif defined(CPU_ARCH_ARM)
    return vbslq_f32(vcltq_f32(data_, rhs.data_), __one, __zero);
#endif
  }

  Matrix<1, 1> operator<=(const Matrix<1, 1>& rhs) const {
#if defined(CPU_ARCH_AMD64)
    return _mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_LE_OS), __one);
#elif defined(CPU_ARCH_ARM)
    return vbslq_f32(vcleq_f32(data_, rhs.data_), __one, __zero);
#endif
  }

  Matrix<1, 1> operator>(const Matrix<1, 1>& rhs) const {
#if defined(CPU_ARCH_AMD64)
    return _mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_GT_OS), __one);
#elif defined(CPU_ARCH_ARM)
    return vbslq_f32(vcgtq_f32(data_, rhs.data_), __one, __zero);
#endif
  }

  Matrix<1, 1> operator>=(const Matrix<1, 1>& rhs) const {
#if defined(CPU_ARCH_AMD64)
    return _mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_GE_OS), __one);
#elif defined(CPU_ARCH_ARM)
    return vbslq_f32(vcgeq_f32(data_, rhs.data_), __one, __zero);
#endif
  }

  // Arithmetic operations
  Matrix<1, 1> operator+() const { return *this; }

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

  // Some useful operations
  Matrix<1, 1> sqrt() const { return Matrix<1, 1>(_s_sqrt(data_)); }

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

  Matrix<1, 1> abs() const {
    // Use bitwise AND to clear the sign bit
#if defined(CPU_ARCH_AMD64)
    __m256 result = _mm256_andnot_ps(_s_set1(-0.0f), data_);
#elif defined(CPU_ARCH_ARM)
    uint32x4_t sign_mask = vdupq_n_u32(0x7FFFFFFF);
    uint32x4_t data_as_int = vreinterpretq_u32_f32(data_);
    uint32x4_t abs_as_int = vandq_u32(data_as_int, sign_mask);
    float32x4_t result = vreinterpretq_f32_u32(abs_as_int);
#endif
    return Matrix<1, 1>(result);
  }

  /// @brief Approximated MatrixBase<1, 1> exp. by using Taylor series expansion
  /// up to 8-th order.
  ///
  /// e^x = 1+x+x^2/2!+x^3/3!+x^4/4!+x^5/5!+x^6/6!+x^7/7!+x^8/8!
  Matrix<1, 1> exp() const {
    const auto& x = data_;
    const _s_data x = data_;
    _s_data res = _s_set1(1.0f / 40320.0f);                 // 1/8!
    res = _s_add(_s_mul(res, x), _s_set1(1.0f / 5040.0f));  // + 1/7!
    res = _s_add(_s_mul(res, x), _s_set1(1.0f / 720.0f));   // + 1/6!
    res = _s_add(_s_mul(res, x), _s_set1(1.0f / 120.0f));   // + 1/5!
    res = _s_add(_s_mul(res, x), _s_set1(1.0f / 24.0f));    // + 1/4!
    res = _s_add(_s_mul(res, x), _s_set1(1.0f / 6.0f));     // + 1/3!
    res = _s_add(_s_mul(res, x), _s_set1(1.0f / 2.0f));     // + 1/2!
    res = _s_add(_s_mul(res, x), _s_set1(1.0f));            // + 1/1!
    res = _s_add(_s_mul(res, x), _s_set1(1.0f));            // + 1
    return Matrix<1, 1>(res);
  }

  // Store SIMD data to normal memory
  void StoreData(float* normal_memory) const { _s_store(normal_memory, data_); }

  // Debug functions
  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const Matrix<1, 1>& scalar) {
    float multi_scalars[__SIMD_DATA_STRIDE];
    scalar.StoreData(multi_scalars);
    std::cout << "[";
    for (int i = 0; i < __SIMD_DATA_STRIDE; ++i) {
      std::cout << "[" << multi_scalars[i] << "]";
      if (i != __SIMD_DATA_STRIDE - 1) std::cout << ",\n";
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

#endif  // NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_SCALAR_H_