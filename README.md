# SIMD Helper
Simd helper is a lightweight header-only C++ library wrapping SIMD operations for matrices and vectors, supporting both **Intel AVX** and **ARM Neon**.

#### Eigen library의 SIMD 병렬화와의 차이점
예시 상황)
`n x k` 크기의 행렬 A0, A1, ..., Am과 `k x l` 크기의 행렬 B0, B1, ..., Bm 가 있을 때, 각 행렬 곱 A0 * B0, A1 * B1, ..., Am * Bm을 구하는 상황.

* Eigen
  * 행렬 곱 A * B 내부 계산을 SIMD를 활용하여 병렬화 할 뿐, A0 * B0, A1 * B1 을 sequantial하게 풀어야 함.
* SIMD Helper
  * 행렬 곱 A0 * B0, A1 * B0, ..., Am * Bm 을 한꺼번에 풀도록 SIMD 병렬화 함.

## How to use
### Dependencies
* [Eigen](https://eigen.tuxfamily.org)
* C++ version > C++11

### Installation
1. Clone repository 
```
cd ${YOUR_WORKSPACE}
git clone https://github.com/changhyeonkim93/simd_helper
```

2. Build and install
```
cd ${YOUR_WORKSPACE}/simd_helper
mkdir build
cd build
sudo make -j${nproc} install
```

### Integration `simd_helper` to project

```cmake
...
# Add below line in CMakeLists.txt in your project
find_package(simd_helper REQUIRED)

...
# Link `simd_helper` for your library or executable
target_link_libraries(${YOUR_EXECUTABLE_NAME} PUBLIC simd_helper::simd_helper)
...
```

### Applications
1. [Nonlinear Optimization for SLAM](https://github.com/ChanghyeonKim93/nonlinear_optimizer_for_slam)
