#ifndef SIMD_HELPER_SIMD_MATRIX_H_
#define SIMD_HELPER_SIMD_MATRIX_H_

#include "simd_scalar.h"

#include "Eigen/Dense"

namespace simd {

template <int kRow, int kCol>
class Matrix {
 private:
  /// @brief Proxy class for block operations
  /// @tparam kBlockRow the number of rows in the block
  /// @tparam kBlockCol the number of columns in the block
  template <int kBlockRow, int kBlockCol>
  class BlockMatrix {
   public:
    BlockMatrix(Matrix* original_matrix, const int start_row,
                const int start_col)
        : original_matrix_(original_matrix),
          start_row_(start_row),
          start_col_(start_col) {}

    const Scalar& operator()(const int r, const int c) const {
      return (*original_matrix_)(start_row_ + r, start_col_ + c);
    }

    Scalar& operator()(const int r, const int c) {
      return (*original_matrix_)(start_row_ + r, start_col_ + c);
    }

    template <int C = kBlockCol>
    typename std::enable_if<C == 1, const Scalar&>::type operator()(
        const int r) const {
      return (*original_matrix_)(start_row_ + r, start_col_);
    }

    template <int C = kBlockCol>
    typename std::enable_if<C == 1, Scalar&>::type operator()(const int r) {
      return (*original_matrix_)(start_row_ + r, start_col_);
    }

    BlockMatrix& operator=(const Matrix<kBlockRow, kBlockCol>& rhs) {
      for (int r = 0; r < kBlockRow; ++r)
        for (int c = 0; c < kBlockCol; ++c)
          (*original_matrix_)(r + start_row_, c + start_col_) = rhs(r, c);
      return *this;
    }

    // Conversion operator to MatrixBase
    operator Matrix<kBlockRow, kBlockCol>() const {
      Matrix<kBlockRow, kBlockCol> result;
      for (int r = 0; r < kBlockRow; ++r)
        for (int c = 0; c < kBlockCol; ++c) result(r, c) = (*this)(r, c);
      return result;
    }

   private:
    Matrix* original_matrix_{nullptr};
    int start_row_{-1};
    int start_col_{-1};
  };

 protected:
  using EigenMatrix = Eigen::Matrix<float, kRow, kCol>;

 public:
  // static members
  static const size_t data_stride{__SIMD_DATA_STRIDE};

  static inline Matrix Zeros() { return Matrix(0.0f); }

  static inline Matrix Ones() { return Matrix(1.0f); }

  static inline Matrix Identity() { return Matrix(EigenMatrix::Identity()); }

 public:
  // Initialization & Assignment operations
  Matrix() {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(0.0f);
  }

  Matrix(const float input) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(input);
  }

  Matrix(const Matrix& rhs) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = rhs.data_[r][c];
  }

  Matrix(const EigenMatrix& matrix) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(matrix(r, c));
  }

  Matrix(const std::vector<EigenMatrix>& matrices) {
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

  Matrix(const std::vector<float*>& multi_elements) {
    if (multi_elements.size() != kRow * kCol)
      throw std::runtime_error("Wrong number of data");
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c)
        data_[r][c] = Scalar(multi_elements.at(r * kCol + c));
  }

  Matrix& operator=(const EigenMatrix& rhs) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(rhs(r, c));
    return *this;
  }

  Matrix& operator=(const Matrix& rhs) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = rhs.data_[r][c];
    return *this;
  }

  // Accessor methods
  Scalar& operator()(const int r, const int c) { return data_[r][c]; }

  const Scalar& operator()(const int r, const int c) const {
    return data_[r][c];
  }

  template <int kBlockRow, int kBlockCol>
  inline BlockMatrix<kBlockRow, kBlockCol> block(const int start_row,
                                                 const int start_col) {
    return BlockMatrix<kBlockRow, kBlockCol>(this, start_row, start_col);
  }

  template <int kRhsRow, int kRhsCol>
  inline Matrix<kRhsRow, kRhsCol> block(const int start_row,
                                        const int start_col) const {
    Matrix<kRhsRow, kRhsCol> res;
    for (int r = 0; r < kRhsRow; ++r)
      for (int c = 0; c < kRhsCol; ++c)
        res(r, c) = data_[start_row + r][start_col + c];
    return res;
  }

  // Arithmetic operations
  Matrix operator+() const { return *this; }

  Matrix operator-() const { return Matrix(_s_sub(__zero, data_)); }

  // Arithmetic operations: element-wise operations
  Matrix operator+(const float rhs) const {
    Matrix res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] += rhs;
    return res;
  }

  Matrix operator-(const float rhs) const {
    Matrix res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] -= rhs;
    return res;
  }

  Matrix operator*(const float rhs) const {
    Matrix res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] *= rhs;
    return res;
  }

  Matrix operator/(const float rhs) const {
    Matrix res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] /= rhs;
    return res;
  }

  friend Matrix operator*(const float lhs, const Matrix& rhs) {
    return MatrixBase(_s_mul(rhs.data_, _s_set1(lhs)));
  }

  // Arithmetic operations: matrix-matrix operations
  Matrix operator+(const Matrix& rhs) const {
    Matrix res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] += rhs.data_[r][c];
    return res;
  }

  Matrix operator-(const Matrix& rhs) const {
    Matrix res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] -= rhs.data_[r][c];
    return res;
  }

  Matrix& operator+=(const Matrix& rhs) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] += rhs.data_[r][c];
    return *this;
  }

  Matrix& operator-=(const Matrix& rhs) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] -= rhs.data_[r][c];
    return *this;
  }

  template <int kRhsCol>
  inline Matrix<kRow, kRhsCol> operator*(
      const Matrix<kCol, kRhsCol>& rhs) const {
    Matrix<kRow, kRhsCol> res;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kRhsCol; ++c)
        for (int k = 0; k < kCol; ++k) res(r, c) += data_[r][k] * rhs(k, c);

    return res;
  }

  inline Matrix<kRow, kCol> operator*(const Matrix<1, 1>& rhs) const {
    Matrix<kRow, kCol> res(*this);
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res(r, c) *= rhs;
    return res;
  }

  // Some matrix operations
  inline Matrix<kCol, kRow> transpose() const {
    Matrix<kCol, kRow> res;
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

  Matrix cwiseSqrt() const {
    Matrix res{Matrix::Zeros()};
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res(r, c) = data_[r][c].sqrt();
  }

  Matrix cwiseSign() const {
    Matrix res{Matrix::Zeros()};
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res(r, c) = data_[r][c].sign();
  }

  Matrix cwiseAbs() const {
    Matrix res{Matrix::Zeros()};
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
                                  const Matrix& simd_mat) {
    static std::stringstream ss;
    ss.str("");
    std::vector<EigenMatrix> multi_matrices;
    simd_mat.StoreData(&multi_matrices);
    ss << "{";
    for (int i = 0; i < Matrix::data_stride; ++i)
      ss << "[" << multi_matrices[i] << "]\n";
    ss << "}" << std::endl;
    std::cerr << ss.str();
    return outputStream;
  }

 protected:
  Scalar data_[kRow][kCol];
};

// Specialization for Nx1 matrix (== vector)
template <int kDim>
class Vector : public Matrix<kDim, 1> {
  using EigenMatrix = typename Matrix<kDim, 1>::EigenMatrix;

 public:
  using Matrix<kDim, 1>::Matrix;

  Vector(const Matrix<kDim, 1>& other) : Matrix<kDim, 1>(other) {}

  // Accessor methods
  Scalar& operator()(const int r) {
    return this->Matrix<kDim, 1>::operator()(r, 0);
  }

  const Scalar& operator()(const int r) const {
    return this->Matrix<kDim, 1>::operator()(r, 0);
  }

  Scalar dot(const Matrix<kDim, 1>& rhs) const {
    Scalar res(0.0f);
    for (int i = 0; i < kDim; ++i) res += (*this)(i, 0) * rhs(i, 0);
    return res;
  }

  Scalar dot(const Matrix<1, kDim>& rhs) const {
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