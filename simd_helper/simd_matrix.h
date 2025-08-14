#ifndef SIMD_HELPER_SIMD_MATRIX_H_
#define SIMD_HELPER_SIMD_MATRIX_H_

#include "simd_scalar.h"

#include "Eigen/Dense"

namespace simd {

template <int kRow, int kCol>
class MatrixBase {
 protected:
  using EigenMatrix = Eigen::Matrix<float, kRow, kCol>;

 public:
  // static members
  static const size_t data_stride{__SIMD_DATA_STRIDE};

  static inline MatrixBase Zeros() { return MatrixBase(0.0f); }

  static inline MatrixBase Ones() { return MatrixBase(1.0f); }

  static inline MatrixBase Identity() {
    return MatrixBase(EigenMatrix::Identity());
  }

 public:
  // Initialization & Assignment operations
  MatrixBase() {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(0.0f);
  }

  MatrixBase(const float input) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(input);
  }

  MatrixBase(const MatrixBase& rhs) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = rhs.data_[r][c];
  }

  MatrixBase(const EigenMatrix& matrix) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(matrix(r, c));
  }

  MatrixBase(const std::vector<EigenMatrix>& matrices) {
    if (matrices.size() != data_stride)
      throw std::runtime_error("Wrong number of data.");

    float buf[data_stride];
    for (int r = 0; r < kRow; ++r) {
      for (int c = 0; c < kCol; ++c) {
        for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
          buf[k] = matrices[k](r, c);
        data_[r][c] = Scalar(buf);
      }
    }
  }

  MatrixBase(const std::vector<float*>& multi_elements) {
    if (multi_elements.size() != kRow * kCol)
      throw std::runtime_error("Wrong number of data");
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c)
        data_[r][c] = Scalar(multi_elements.at(r * kCol + c));
  }

  MatrixBase& operator=(const EigenMatrix& rhs) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(rhs(r, c));
    return *this;
  }

  MatrixBase& operator=(const MatrixBase& rhs) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = rhs.data_[r][c];
    return *this;
  }

  // Accessor methods
  Scalar& operator()(const int r, const int c) { return data_[r][c]; }

  const Scalar& operator()(const int r, const int c) const {
    return data_[r][c];
  }

  // Arithmetic operations
  MatrixBase operator+() const { return *this; }

  MatrixBase operator-() const { return MatrixBase(_s_sub(__zero, data_)); }

  // Arithmetic operations: element-wise operations
  MatrixBase operator+(const float rhs) const {
    MatrixBase res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] += rhs;
    return res;
  }

  MatrixBase operator-(const float rhs) const {
    MatrixBase res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] -= rhs;
    return res;
  }

  MatrixBase operator*(const float rhs) const {
    MatrixBase res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] *= rhs;
    return res;
  }

  MatrixBase operator/(const float rhs) const {
    MatrixBase res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] /= rhs;
    return res;
  }

  friend MatrixBase operator*(const float lhs, const MatrixBase& rhs) {
    return MatrixBase(_s_mul(rhs.data_, _s_set1(lhs)));
  }

  // Arithmetic operations: matrix-matrix operations
  MatrixBase operator+(const MatrixBase& rhs) const {
    MatrixBase res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] += rhs.data_[r][c];
    return res;
  }

  MatrixBase operator-(const MatrixBase& rhs) const {
    MatrixBase res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] -= rhs.data_[r][c];
    return res;
  }

  MatrixBase& operator+=(const MatrixBase& rhs) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] += rhs.data_[r][c];
    return *this;
  }

  MatrixBase& operator-=(const MatrixBase& rhs) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] -= rhs.data_[r][c];
    return *this;
  }

  template <int kRhsCol>
  inline MatrixBase<kRow, kRhsCol> operator*(
      const MatrixBase<kCol, kRhsCol>& rhs) const {
    MatrixBase<kRow, kRhsCol> res;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kRhsCol; ++c)
        for (int k = 0; k < kCol; ++k) res(r, c) += data_[r][k] * rhs(k, c);

    return res;
  }

  // Some matrix operations
  inline MatrixBase<kCol, kRow> transpose() const {
    MatrixBase<kCol, kRow> res;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res(c, r) = data_[r][c];
    return res;
  }

  inline Scalar squaredNorm() const {
    Scalar res(0.0f);
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res += data_[r][c] * data_[r][c];
    return res;
  }

  inline Scalar norm() const { return this->squaredNorm().sqrt(); }

  void setZero() {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(0.0f);
  }

  MatrixBase cwiseSqrt() const {
    MatrixBase res{MatrixBase::Zeros()};
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res(r, c) = data_[r][c].sqrt();
  }

  MatrixBase cwiseSign() const {
    MatrixBase res{MatrixBase::Zeros()};
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res(r, c) = data_[r][c].sign();
  }

  MatrixBase cwiseAbs() const {
    MatrixBase res{MatrixBase::Zeros()};
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res(r, c) = data_[r][c].abs();
    return res;
  }

  // Store SIMD data to normal memory
  void StoreData(std::vector<EigenMatrix>* multi_matrices) const {
    if (multi_matrices->size() != data_stride)
      multi_matrices->resize(data_stride);
    float buf[data_stride];
    for (int r = 0; r < kRow; ++r) {
      for (int c = 0; c < kCol; ++c) {
        data_[r][c].StoreData(buf);
        for (size_t k = 0; k < data_stride; ++k)
          multi_matrices->at(k)(r, c) = buf[k];
      }
    }
  }

  // Debug functions
  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const MatrixBase& simd_mat) {
    static std::stringstream ss;
    ss.str("");
    std::vector<EigenMatrix> multi_matrices;
    simd_mat.StoreData(&multi_matrices);
    ss << "{";
    for (int i = 0; i < MatrixBase::data_stride; ++i)
      ss << "[" << multi_matrices[i] << "]\n";
    ss << "}" << std::endl;
    std::cerr << ss.str();
    return outputStream;
  }

 protected:
  Scalar data_[kRow][kCol];
};

template <int kRow, int kCol>
using Matrix = MatrixBase<kRow, kCol>;

// Specialization for Nx1 matrix (== vector)
template <int kDim>
class Vector : public MatrixBase<kDim, 1> {
  using Base = MatrixBase<kDim, 1>;
  using EigenMatrix = typename MatrixBase<kDim, 1>::EigenMatrix;

 public:
  using MatrixBase<kDim, 1>::MatrixBase;

  // Accessor methods
  Scalar& operator()(const int r) {
    return this->MatrixBase<kDim, 1>::operator()(r, 0);
  }

  const Scalar& operator()(const int r) const {
    return this->MatrixBase<kDim, 1>::operator()(r, 0);
  }

  Scalar dot(const MatrixBase<kDim, 1>& rhs) const {
    Scalar res(0.0f);
    for (int i = 0; i < kDim; ++i) res += (*this)(i, 0) * rhs(i, 0);
    return res;
  }

  Scalar dot(const MatrixBase<1, kDim>& rhs) const {
    Scalar res(0.0f);
    for (int i = 0; i < kDim; ++i) res += (*this)(i, 0) * *rhs(0, i);
    return res;
  }

  Scalar dot(const Vector& rhs) const {
    Scalar res(0.0f);
    for (int i = 0; i < kDim; ++i) res += (*this)(i, 0) * rhs.data_(i);
    return res;
  }

  // Specialized operations for 3D vector
  template <int D = kDim>
  typename std::enable_if<D == 3, Vector>::type cross(const Vector& rhs) const {
    Vector result;
    result(0) = (*this)(1) * rhs(2) - (*this)(2) * rhs(1);
    result(1) = (*this)(2) * rhs(0) - (*this)(0) * rhs(2);
    result(2) = (*this)(0) * rhs(1) - (*this)(1) * rhs(0);
    return result;
  }

  template <int D = kDim>
  typename std::enable_if<D == 3, Vector>::type toSkewSymmetricMatrix(
      const Vector& rhs) const {
    Matrix<3, 3> result(0.0f);
    const Scalar& x = (*this)(0);
    const Scalar& y = (*this)(1);
    const Scalar& z = (*this)(2);
    result(0, 1) = -z;
    result(0, 2) = y;
    result(1, 0) = z;
    result(1, 2) = -x;
    result(2, 0) = -y;
    result(2, 1) = x;
    return result;
  }
};

}  // namespace simd

#endif  // SIMD_HELPER_SIMD_MATRIX_H_