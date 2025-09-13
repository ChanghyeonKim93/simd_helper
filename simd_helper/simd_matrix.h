#ifndef SIMD_HELPER_SIMD_MATRIX_H_
#define SIMD_HELPER_SIMD_MATRIX_H_

#include <vector>

#include "simd_helper/simd_scalar.h"
#include "simd_helper/soa_container.h"

#include "Eigen/Dense"

namespace simd {

template <int kRow, int kCol>
class Matrix;

/// @brief Vector is a special case of Matrix with one column.
template <int kDim>
using Vector = Matrix<kDim, 1>;

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

    template <int BR = kBlockRow, int BC = kBlockCol>
    typename std::enable_if<!(BR == 1 && BC == 1), BlockMatrix&>::type
    operator=(const Matrix<kBlockRow, kBlockCol>& rhs) {
      for (int r = 0; r < kBlockRow; ++r)
        for (int c = 0; c < kBlockCol; ++c)
          (*original_matrix_)(r + start_row_, c + start_col_) = rhs(r, c);
      return *this;
    }

    template <int BR = kBlockRow, int BC = kBlockCol>
    typename std::enable_if<BR == 1 && BC == 1, BlockMatrix&>::type operator=(
        const Scalar& rhs) {
      (*original_matrix_)(start_row_, start_col_) = rhs;
      return *this;
    }

    // Conversion operator to Matrix
    operator Matrix<kBlockRow, kBlockCol>() const {
      Matrix<kBlockRow, kBlockCol> result;
      for (int r = 0; r < kBlockRow; ++r)
        for (int c = 0; c < kBlockCol; ++c) result(r, c) = (*this)(r, c);
      return result;
    }

    template <int BR = kBlockRow, int BC = kBlockCol>
    operator typename std::enable_if<BR == 1 && BC == 1, Scalar>::type() const {
      return (*original_matrix_)(start_row_, start_col_);
    }

   private:
    Matrix* original_matrix_{nullptr};
    int start_row_{-1};
    int start_col_{-1};
  };

 protected:
  using EigenMatrix = Eigen::Matrix<float, kRow, kCol>;

 public:
  // Static members

  static const size_t data_stride{SIMD_FLOAT_PACK_LANES};

  static inline Matrix Zeros() { return Matrix(0.0f); }

  static inline Matrix Ones() { return Matrix(1.0f); }

  static inline Matrix Identity() { return Matrix(EigenMatrix::Identity()); }

 public:
  // Initialization & Assignment operations

  Matrix() {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(0.0f);
  }

  explicit Matrix(const float input) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(input);
  }

  Matrix(const Matrix& rhs) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = rhs.data_[r][c];
  }

  template <int OtherRow, int OtherCol>
  explicit Matrix(
      const typename Matrix<OtherRow, OtherCol>::template BlockMatrix<
          kRow, kCol>& block) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) (*this)(r, c) = block(r, c);
  }

  explicit Matrix(const EigenMatrix& matrix) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(matrix(r, c));
  }

  explicit Matrix(const std::vector<EigenMatrix>& matrices) {
    if (matrices.size() != Matrix::data_stride)
      throw std::runtime_error("Wrong number of data.");

    float buf[Matrix::data_stride];
    for (int r = 0; r < kRow; ++r) {
      for (int c = 0; c < kCol; ++c) {
        for (size_t k = 0; k < Matrix::data_stride; ++k)
          buf[k] = matrices[k](r, c);
        data_[r][c] = Scalar(buf);
      }
    }
  }

  /// @brief Constructor initializes the matrix with multiple float pointers.
  /// Each float pointer directs to a multiple elements of the specific position
  /// of the matrix. For example, the first pointer points to multiple elements
  /// from the (0,0) position of the multiple matrices.
  /// @param multi_elements Vector of pointers to multiple float data to
  /// initialize the matrix.
  explicit Matrix(const std::vector<float*>& multi_elements) {
    if (multi_elements.size() != kRow * kCol)
      throw std::runtime_error("Wrong number of data");
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c)
        data_[r][c] = Scalar(multi_elements.at(r * kCol + c));
  }

  Matrix(const SOAContainer<kRow, kCol>& soa_container, const int start_index) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c)
        data_[r][c] = Scalar(soa_container.GetElementPtr(r, c) + start_index);
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

  /// @brief Returns a block of the matrix starting from the specified row and
  /// column
  /// @tparam kBlockRow The number of rows in the block.
  /// @tparam kBlockCol The number of columns in the block.
  /// @param start_row The starting row index of block on the original matrix
  /// @param start_col The starting column index of block on the original
  /// matrix
  /// @return BlockMatrix<kBlockRow, kBlockCol> representing the block of the
  /// matrix
  template <int kBlockRow, int kBlockCol>
  inline BlockMatrix<kBlockRow, kBlockCol> block(const int start_row,
                                                 const int start_col) {
    return BlockMatrix<kBlockRow, kBlockCol>(this, start_row, start_col);
  }

  template <int kBlockRow, int kBlockCol>
  inline Matrix<kBlockRow, kBlockCol> block(const int start_row,
                                            const int start_col) const {
    Matrix<kBlockRow, kBlockCol> res;
    for (int r = 0; r < kBlockRow; ++r)
      for (int c = 0; c < kBlockCol; ++c)
        res(r, c) = data_[start_row + r][start_col + c];
    return res;
  }

  // Arithmetic operations

  Matrix operator+() const { return *this; }

  Matrix operator-() const {
    Matrix res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] = -res.data_[r][c];
    return res;
  }

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

  Matrix operator*(const Scalar& rhs) const {
    Matrix res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] *= rhs;
    return res;
  }

  friend Matrix operator*(const float lhs, const Matrix& rhs) {
    Matrix res = rhs;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] *= Scalar(lhs);
    return res;
  }

  friend Matrix operator*(const Scalar& lhs, const Matrix& rhs) {
    Matrix res = rhs;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] *= lhs;
    return res;
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

  template <int R = kRow>
  typename std::enable_if<R == 1, Scalar>::type operator*(
      const Matrix<kCol, 1>& rhs) const {
    Scalar res(0.0f);
    for (int c = 0; c < kCol; ++c) res += data_[0][c] * rhs(c);
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

  // Vector specialized operations

  template <int C = kCol>
  typename std::enable_if<C == 1, Scalar&>::type operator()(const int r) {
    return data_[r][0];
  }

  template <int C = kCol>
  typename std::enable_if<C == 1, const Scalar&>::type operator()(
      const int r) const {
    return data_[r][0];
  }

  template <int C = kCol>
  typename std::enable_if<C == 1, Scalar>::type dot(
      const Matrix<kRow, 1>& rhs) const {
    return this->transpose() * rhs;
  }

  template <int R = kRow>
  typename std::enable_if<R == 1, Scalar>::type dot(
      const Matrix<1, kCol>& rhs) const {
    return (*this) * rhs.transpose();
  }

  template <int R = kRow, int C = kCol>
  typename std::enable_if<R == 3 && C == 1, Matrix<3, 1>>::type cross(
      const Matrix<3, 1>& rhs) const {
    Matrix<3, 1> result;
    result(0, 0) = (*this)(1, 0) * rhs(2, 0) - (*this)(2, 0) * rhs(1, 0);
    result(1, 0) = (*this)(2, 0) * rhs(0, 0) - (*this)(0, 0) * rhs(2, 0);
    result(2, 0) = (*this)(0, 0) * rhs(1, 0) - (*this)(1, 0) * rhs(0, 0);
    return result;
  }

  template <int R = kRow, int C = kCol>
  typename std::enable_if<R == 3 && C == 1, Matrix<3, 3>>::type hat() const {
    Matrix<3, 3> result;
    const Scalar zero(0.0f);
    const Scalar& x = data_[0][0];
    const Scalar& y = data_[1][0];
    const Scalar& z = data_[2][0];

    result(0, 0) = zero;
    result(0, 1) = -z;
    result(0, 2) = y;
    result(1, 0) = z;
    result(1, 1) = zero;
    result(1, 2) = -x;
    result(2, 0) = -y;
    result(2, 1) = x;
    result(2, 2) = zero;
    return result;
  }

  /// @brief Stores the SIMD data to normal memory.
  /// @param multi_matrices  Vector of Eigen matrices where the data will be
  /// stored.
  void StoreData(std::vector<EigenMatrix>* multi_matrices) const {
    if (multi_matrices->size() != Matrix::data_stride)
      multi_matrices->resize(Matrix::data_stride);
    float buf[Matrix::data_stride];
    for (int r = 0; r < kRow; ++r) {
      for (int c = 0; c < kCol; ++c) {
        data_[r][c].StoreData(buf);
        for (size_t k = 0; k < Matrix::data_stride; ++k)
          multi_matrices->at(k)(r, c) = buf[k];
      }
    }
  }

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

}  // namespace simd

#endif  // SIMD_HELPER_SIMD_MATRIX_H_
