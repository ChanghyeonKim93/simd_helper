#include <cassert>
#include <memory>

#include "gtest/gtest.h"

#include "Eigen/Dense"
#include "simd_helper/simd_helper.h"

using EigenVec3 = Eigen::Vector3f;
using EigenMat3x3 = Eigen::Matrix3f;

simd::Vector<3> GenerateVectors() {
  EigenVec3 v1(4, 6, 8);
  EigenVec3 v2(-3, 1, 6);
  EigenVec3 v3(9, 3, -5);
  EigenVec3 v4(1, 6, -10);
#if CPU_ARCH_AMD64
  simd::Vector<3> v__({v1, v2, v3, v4, v4, v3, v2, v1});
#elif CPU_ARCH_ARM
  simd::Vector<3> v__({v1, v2, v3, v4});
#endif
  return v__;
}

simd::Matrix<3, 3> GenerateMatrices() {
  EigenMat3x3 M1;
  M1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  EigenMat3x3 M2;
  M2 << 1, -2, 3, -4, 5, -6, 7, -8, 9;
  EigenMat3x3 M3;
  M3 << -1, 2, -3, 4, -5, 6, -7, 8, -9;
  EigenMat3x3 M4;
  M4 << 9, -8, 7, -6, 5, -4, 3, -1, 2;
#if CPU_ARCH_AMD64
  simd::Matrix<3, 3> M__({M1, M2, M3, M4, M1, M2, M3, M4});
#elif CPU_ARCH_ARM
  simd::Matrix<3, 3> M__({M1, M2, M3, M4});
#endif
  return M__;
}

std::vector<EigenVec3> GenerateMultiplyResults() {
  EigenMat3x3 M1;
  M1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  EigenMat3x3 M2;
  M2 << 1, -2, 3, -4, 5, -6, 7, -8, 9;
  EigenMat3x3 M3;
  M3 << -1, 2, -3, 4, -5, 6, -7, 8, -9;
  EigenMat3x3 M4;
  M4 << 9, -8, 7, -6, 5, -4, 3, -1, 2;
  EigenVec3 v1(4, 6, 8);
  EigenVec3 v2(-3, 1, 6);
  EigenVec3 v3(9, 3, -5);
  EigenVec3 v4(1, 6, -10);

  std::vector<EigenVec3> res_true;
  res_true.push_back(M1 * v1);
  res_true.push_back(M2 * v2);
  res_true.push_back(M3 * v3);
  res_true.push_back(M4 * v4);
  res_true.push_back(M1 * v4);
  res_true.push_back(M2 * v3);
  res_true.push_back(M3 * v2);
  res_true.push_back(M4 * v1);
  return res_true;
}

namespace simd_helper {

class CalculationTest : public ::testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(CalculationTest, SimdDataLoadAndSaveTest) {
  // SIMD scalar data
  float value[8] = {1.23456,     -0.45678,  4.16789, -1.42536,
                    14234.42252, 11.023334, 0.23133, -9.41111};
  simd::Scalar v__(value);
  float buf[8];
  v__.StoreData(buf);
  for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
    EXPECT_FLOAT_EQ(value[k], buf[k]);
}

TEST_F(CalculationTest, MatrixVectorMultiplyTest) {
  simd::Matrix<3, 3> M__ = GenerateMatrices();
  simd::Vector<3> v__ = GenerateVectors();
  const auto res_true = GenerateMultiplyResults();

  const auto Mv__ = M__ * v__;
  std::vector<EigenVec3> res;
  Mv__.StoreData(&res);
  for (int i = 0; i < 3; ++i)
    for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
      EXPECT_FLOAT_EQ(res_true[k](i), res[k](i));
}

TEST_F(CalculationTest, VectorAddSubTest) {
  EigenVec3 v1(4, 6, 8);
  EigenVec3 v2(-3, 1, 6);
  EigenVec3 v3(9, 3, -5);
  EigenVec3 v4(1, 6, -10);

  std::vector<EigenVec3> add_res_true;
  add_res_true.push_back(v1 + v2);
  add_res_true.push_back(v3 + v4);
  add_res_true.push_back(v1 + v4);
  add_res_true.push_back(v3 + v2);
  add_res_true.push_back(v3 + v2);
  add_res_true.push_back(v1 + v4);
  add_res_true.push_back(v3 + v4);
  add_res_true.push_back(v1 + v2);

  std::vector<EigenVec3> sub_res_true;
  sub_res_true.push_back(v1 - v2);
  sub_res_true.push_back(v3 - v4);
  sub_res_true.push_back(v1 - v4);
  sub_res_true.push_back(v3 - v2);
  sub_res_true.push_back(v3 - v2);
  sub_res_true.push_back(v1 - v4);
  sub_res_true.push_back(v3 - v4);
  sub_res_true.push_back(v1 - v2);

#if CPU_ARCH_AMD64
  simd::Vector<3> va__({v1, v3, v1, v3, v3, v1, v3, v1});
  simd::Vector<3> vb__({v2, v4, v4, v2, v2, v4, v4, v2});
#elif CPU_ARCH_ARM
  simd::Vector<3> va__({v1, v3, v1, v3});
  simd::Vector<3> vb__({v2, v4, v4, v2});
#endif
  std::vector<EigenVec3> res;
  const auto add__ = va__ + vb__;
  add__.StoreData(&res);
  for (int i = 0; i < 3; ++i)
    for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
      EXPECT_FLOAT_EQ(add_res_true[k](i), res[k](i));

  const auto sub__ = va__ - vb__;
  sub__.StoreData(&res);
  for (int i = 0; i < 3; ++i)
    for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
      EXPECT_FLOAT_EQ(sub_res_true[k](i), res[k](i));
}

TEST_F(CalculationTest, MatrixAddSubTest) {
  EigenMat3x3 M1;
  M1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  EigenMat3x3 M2;
  M2 << 1, -2, 3, -4, 5, -6, 7, -8, 9;
  EigenMat3x3 M3;
  M3 << -1, 2, -3, 4, -5, 6, -7, 8, -9;
  EigenMat3x3 M4;
  M4 << 9, -8, 7, -6, 5, -4, 3, -1, 2;

  std::vector<EigenMat3x3> add_res_true;
  add_res_true.push_back(M1 + M2);
  add_res_true.push_back(M3 + M4);
  add_res_true.push_back(M1 + M4);
  add_res_true.push_back(M3 + M2);
  add_res_true.push_back(M3 + M2);
  add_res_true.push_back(M2 + M4);
  add_res_true.push_back(M1 + M4);
  add_res_true.push_back(M4 + M2);

  std::vector<EigenMat3x3> sub_res_true;
  sub_res_true.push_back(M1 - M2);
  sub_res_true.push_back(M3 - M4);
  sub_res_true.push_back(M1 - M4);
  sub_res_true.push_back(M3 - M2);
  sub_res_true.push_back(M3 - M2);
  sub_res_true.push_back(M2 - M4);
  sub_res_true.push_back(M1 - M4);
  sub_res_true.push_back(M4 - M2);

#if CPU_ARCH_AMD64
  simd::Matrix<3, 3> Ma__({M1, M3, M1, M3, M3, M2, M1, M4});
  simd::Matrix<3, 3> Mb__({M2, M4, M4, M2, M2, M4, M4, M2});
#elif CPU_ARCH_ARM
  simd::Matrix<3, 3> Ma__({M1, M3, M1, M3});
  simd::Matrix<3, 3> Mb__({M2, M4, M4, M2});
#endif

  std::vector<EigenMat3x3> res;
  const auto add__ = Ma__ + Mb__;
  add__.StoreData(&res);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
        EXPECT_FLOAT_EQ(add_res_true[k](i, j), res[k](i, j));

  const auto sub__ = Ma__ - Mb__;
  sub__.StoreData(&res);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
        EXPECT_FLOAT_EQ(sub_res_true[k](i, j), res[k](i, j));
}

TEST_F(CalculationTest, VectorCompareTest) {
  float v1 = 4;
  float v2 = -3;
  float v3 = 9;
  float v4 = 1;

  float data1[8] = {v1, v3, v1, v3, v4, v1, v2, v3};
  float data2[8] = {v2, v4, v4, v2, v3, v2, v1, v4};

  std::vector<float> gt_true;
  gt_true.push_back(data1[0] > data2[0]);
  gt_true.push_back(data1[1] > data2[1]);
  gt_true.push_back(data1[2] > data2[2]);
  gt_true.push_back(data1[3] > data2[3]);
  gt_true.push_back(data1[4] > data2[4]);
  gt_true.push_back(data1[5] > data2[5]);
  gt_true.push_back(data1[6] > data2[6]);
  gt_true.push_back(data1[7] > data2[7]);

  std::vector<float> lt_true;
  lt_true.push_back(data1[0] < data2[0]);
  lt_true.push_back(data1[1] < data2[1]);
  lt_true.push_back(data1[2] < data2[2]);
  lt_true.push_back(data1[3] < data2[3]);
  lt_true.push_back(data1[4] < data2[4]);
  lt_true.push_back(data1[5] < data2[5]);
  lt_true.push_back(data1[6] < data2[6]);
  lt_true.push_back(data1[7] < data2[7]);

  simd::Scalar va__(data1);
  simd::Scalar vb__(data2);

  float res[8];
  const auto gt__ = va__ > vb__;
  gt__.StoreData(res);
  for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
    EXPECT_FLOAT_EQ(gt_true[k], res[k]);

  const auto lt__ = va__ < vb__;
  lt__.StoreData(res);
  for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
    EXPECT_FLOAT_EQ(lt_true[k], res[k]);
}

TEST_F(CalculationTest, VectorSignTest) {
  float v1 = 4;
  float v2 = -3;
  float v3 = 9;
  float v4 = 1;

  float data[8] = {v1, v3, v1, v3, v4, v1, v2, v3};

  std::vector<float> sign_true;
  sign_true.push_back(v1 > 0 ? 1.0f : -1.0f);
  sign_true.push_back(v3 > 0 ? 1.0f : -1.0f);
  sign_true.push_back(v1 > 0 ? 1.0f : -1.0f);
  sign_true.push_back(v3 > 0 ? 1.0f : -1.0f);
  sign_true.push_back(v4 > 0 ? 1.0f : -1.0f);
  sign_true.push_back(v1 > 0 ? 1.0f : -1.0f);
  sign_true.push_back(v2 > 0 ? 1.0f : -1.0f);
  sign_true.push_back(v3 > 0 ? 1.0f : -1.0f);

  simd::Scalar va__(data);

  float res[8];
  const auto sign__ = va__.sign();
  sign__.StoreData(res);

  for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
    EXPECT_FLOAT_EQ(sign_true[k], res[k]);
}

TEST_F(CalculationTest, VectorAbsTest) {
  float v1 = 4;
  float v2 = -3;
  float v3 = 9;
  float v4 = -1;

  float data[8] = {v1, v3, v1, v3, v4, v1, v2, v3};

  std::vector<float> abs_true;
  abs_true.push_back(std::abs(v1));
  abs_true.push_back(std::abs(v3));
  abs_true.push_back(std::abs(v1));
  abs_true.push_back(std::abs(v3));
  abs_true.push_back(std::abs(v4));
  abs_true.push_back(std::abs(v1));
  abs_true.push_back(std::abs(v2));
  abs_true.push_back(std::abs(v3));

  simd::Scalar va__(data);

  float res[8];
  const auto abs__ = va__.abs();
  abs__.StoreData(res);

  for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
    EXPECT_FLOAT_EQ(abs_true[k], res[k]);
}

TEST_F(CalculationTest, VectorSqrtTest) {
  float v1 = 4;
  float v2 = 9;
  float v3 = 16;
  float v4 = 25;

  float data[8] = {v1, v2, v3, v4, v1, v2, v3, v4};

  std::vector<float> sqrt_true;
  sqrt_true.push_back(std::sqrt(v1));
  sqrt_true.push_back(std::sqrt(v2));
  sqrt_true.push_back(std::sqrt(v3));
  sqrt_true.push_back(std::sqrt(v4));
  sqrt_true.push_back(std::sqrt(v1));
  sqrt_true.push_back(std::sqrt(v2));
  sqrt_true.push_back(std::sqrt(v3));
  sqrt_true.push_back(std::sqrt(v4));

  simd::Scalar va__(data);

  float res[8];
  const auto sqrt__ = va__.sqrt();
  sqrt__.StoreData(res);

  for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
    EXPECT_FLOAT_EQ(sqrt_true[k], res[k]);
}

TEST_F(CalculationTest, MemoryAlignmentTest) {
  const size_t alignment = 32;
  const size_t num_data = 10000;

  float* aligned_float = nullptr;
  aligned_float = simd::GetAlignedMemory<float>(num_data);
  EXPECT_NE(aligned_float, nullptr);
  bool is_aligned =
      (reinterpret_cast<std::uintptr_t>(aligned_float) % alignment == 0);
  EXPECT_TRUE(is_aligned);

  double* aligned_double = nullptr;
  aligned_double = simd::GetAlignedMemory<double>(num_data);
  EXPECT_NE(aligned_double, nullptr);
  is_aligned =
      (reinterpret_cast<std::uintptr_t>(aligned_double) % alignment == 0);
  EXPECT_TRUE(is_aligned);

  int* aligned_int = nullptr;
  aligned_int = simd::GetAlignedMemory<int>(num_data);
  EXPECT_NE(aligned_int, nullptr);
  is_aligned = (reinterpret_cast<std::uintptr_t>(aligned_int) % alignment == 0);
  EXPECT_TRUE(is_aligned);
}

TEST_F(CalculationTest, ExponentialTest) {
  float v1 = 0.25f;
  float v2 = 0.5f;
  float v3 = 1.0f;
  float v4 = 4.0f;

  float data[8] = {v1, v2, v3, v4, -v1, -v2, -v3, -v4};

  std::vector<float> exp_true;
  exp_true.push_back(std::exp(data[0]));
  exp_true.push_back(std::exp(data[1]));
  exp_true.push_back(std::exp(data[2]));
  exp_true.push_back(std::exp(data[3]));
  exp_true.push_back(std::exp(data[4]));
  exp_true.push_back(std::exp(data[5]));
  exp_true.push_back(std::exp(data[6]));
  exp_true.push_back(std::exp(data[7]));

  simd::Scalar va__(data);

  float res[8];
  const auto exp__ = va__.exp();
  exp__.StoreData(res);

  for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
    EXPECT_FLOAT_EQ(exp_true[k], res[k]);
}

TEST_F(CalculationTest, SqrtTest) {
  float v1 = 0.25f;
  float v2 = 0.5f;
  float v3 = 1.0f;
  float v4 = 4.0f;

  float data[8] = {v1, v2, v3, v4, 2.0f * v1, 2.0f * v2, 2.0f * v3, 2.0f * v4};

  std::vector<float> exp_true;
  exp_true.push_back(std::sqrt(data[0]));
  exp_true.push_back(std::sqrt(data[1]));
  exp_true.push_back(std::sqrt(data[2]));
  exp_true.push_back(std::sqrt(data[3]));
  exp_true.push_back(std::sqrt(data[4]));
  exp_true.push_back(std::sqrt(data[5]));
  exp_true.push_back(std::sqrt(data[6]));
  exp_true.push_back(std::sqrt(data[7]));

  simd::Scalar va__(data);

  float res[8];
  const auto exp__ = va__.sqrt();
  exp__.StoreData(res);

  for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
    EXPECT_FLOAT_EQ(exp_true[k], res[k]);
}

TEST_F(CalculationTest, SignTest) {
  float v1 = -0.25f;
  float v2 = 0.5f;
  float v3 = -1.0f;
  float v4 = 4.0f;

  float data[8] = {v1, v2, v3, v4, -v1, -v2, -v3, -v4};

  std::vector<float> sign_true;
  sign_true.push_back(data[0] > 0 ? 1.0f : -1.0f);
  sign_true.push_back(data[1] > 0 ? 1.0f : -1.0f);
  sign_true.push_back(data[2] > 0 ? 1.0f : -1.0f);
  sign_true.push_back(data[3] > 0 ? 1.0f : -1.0f);
  sign_true.push_back(data[4] > 0 ? 1.0f : -1.0f);
  sign_true.push_back(data[5] > 0 ? 1.0f : -1.0f);
  sign_true.push_back(data[6] > 0 ? 1.0f : -1.0f);
  sign_true.push_back(data[7] > 0 ? 1.0f : -1.0f);

  simd::Scalar va__(data);

  float res[8];
  const auto sign__ = va__.sign();
  sign__.StoreData(res);

  for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
    EXPECT_FLOAT_EQ(sign_true[k], res[k]);
}

TEST_F(CalculationTest, AbsTest) {
  float v1 = -0.25f;
  float v2 = 0.5f;
  float v3 = -1.0f;
  float v4 = 4.0f;

  float data[8] = {v1, v2, v3, v4, -v1, -v2, -v3, -v4};

  std::vector<float> abs_true;
  abs_true.push_back(std::abs(data[0]));
  abs_true.push_back(std::abs(data[1]));
  abs_true.push_back(std::abs(data[2]));
  abs_true.push_back(std::abs(data[3]));
  abs_true.push_back(std::abs(data[4]));
  abs_true.push_back(std::abs(data[5]));
  abs_true.push_back(std::abs(data[6]));
  abs_true.push_back(std::abs(data[7]));

  simd::Scalar va__(data);

  float res[8];
  const auto abs__ = va__.abs();
  abs__.StoreData(res);

  for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
    EXPECT_FLOAT_EQ(abs_true[k], res[k]);
}

TEST_F(CalculationTest, LossTest) {
  constexpr float c1 = 1.0f;
  constexpr float c2 = 1.0f;

  float v1 = 0.25f;
  float v2 = 0.5f;
  float v3 = 1.0f;
  float v4 = 400.0f;
  float data[8] = {v1, v2, v3, v4, -v1, -v2, -v3, -v4};

  std::vector<float> exp_true;
  exp_true.push_back(std::exp(-c2 * data[0]));
  exp_true.push_back(std::exp(-c2 * data[1]));
  exp_true.push_back(std::exp(-c2 * data[2]));
  exp_true.push_back(std::exp(-c2 * data[3]));
  exp_true.push_back(std::exp(-c2 * data[4]));
  exp_true.push_back(std::exp(-c2 * data[5]));
  exp_true.push_back(std::exp(-c2 * data[6]));
  exp_true.push_back(std::exp(-c2 * data[7]));

  std::vector<float> loss_true;
  loss_true.push_back(c1 - c1 * exp_true[0]);
  loss_true.push_back(c1 - c1 * exp_true[1]);
  loss_true.push_back(c1 - c1 * exp_true[2]);
  loss_true.push_back(c1 - c1 * exp_true[3]);
  loss_true.push_back(c1 - c1 * exp_true[4]);
  loss_true.push_back(c1 - c1 * exp_true[5]);
  loss_true.push_back(c1 - c1 * exp_true[6]);
  loss_true.push_back(c1 - c1 * exp_true[7]);

  std::vector<float> dloss_true;
  dloss_true.push_back(2.0 * c1 * c2 * exp_true[0]);
  dloss_true.push_back(2.0 * c1 * c2 * exp_true[1]);
  dloss_true.push_back(2.0 * c1 * c2 * exp_true[2]);
  dloss_true.push_back(2.0 * c1 * c2 * exp_true[3]);
  dloss_true.push_back(2.0 * c1 * c2 * exp_true[4]);
  dloss_true.push_back(2.0 * c1 * c2 * exp_true[5]);
  dloss_true.push_back(2.0 * c1 * c2 * exp_true[6]);
  dloss_true.push_back(2.0 * c1 * c2 * exp_true[7]);

  std::vector<float> d2loss_true;
  d2loss_true.push_back(-2.0 * c2 * dloss_true[0]);
  d2loss_true.push_back(-2.0 * c2 * dloss_true[1]);
  d2loss_true.push_back(-2.0 * c2 * dloss_true[2]);
  d2loss_true.push_back(-2.0 * c2 * dloss_true[3]);
  d2loss_true.push_back(-2.0 * c2 * dloss_true[4]);
  d2loss_true.push_back(-2.0 * c2 * dloss_true[5]);
  d2loss_true.push_back(-2.0 * c2 * dloss_true[6]);
  d2loss_true.push_back(-2.0 * c2 * dloss_true[7]);

  simd::Scalar sq_res__(data);
  const simd::Scalar exp_term__ = simd::exp((-c2) * sq_res__);
  auto loss__ = c1 - c1 * exp_term__;
  auto dloss__ = 2.0 * c1 * c2 * exp_term__;
  auto d2loss__ = (-2.0 * c2) * dloss__;

  float simd_result[8];
  exp_term__.StoreData(simd_result);
  for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
    EXPECT_FLOAT_EQ(exp_true[k], simd_result[k]);
  loss__.StoreData(simd_result);
  for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
    EXPECT_FLOAT_EQ(loss_true[k], simd_result[k]);
  dloss__.StoreData(simd_result);
  for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
    EXPECT_FLOAT_EQ(dloss_true[k], simd_result[k]);
  d2loss__.StoreData(simd_result);
  for (size_t k = 0; k < simd::Scalar::data_stride; ++k)
    EXPECT_FLOAT_EQ(d2loss_true[k], simd_result[k]);
}

}  // namespace simd_helper