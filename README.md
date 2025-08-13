# SIMD Helper
Intel AVX, ARM Neon을 이용한 행렬 병렬계산을 쉽게 할 수 있게 도와주는 라이브러리.

### Eigen library의 SIMD 병렬화와의 차이점
예시 상황)
`n x k` 크기의 행렬 A0, A1, ..., Am과 `k x l` 크기의 행렬 B0, B1, ..., Bm 가 있을 때, 각 행렬 곱 A0 * B0, A1 * B1, ..., Am * Bm을 구하는 상황.

* Eigen
  * 행렬 곱 A * B 내부 계산을 SIMD를 활용하여 병렬화 할 뿐, A0 * B0, A1 * B1 을 sequantial하게 풀어야 함.
* SIMD Helper
  * 행렬 곱 A0 * B0, A1 * B0, ..., Am * Bm 을 한꺼번에 풀도록 SIMD 병렬화 함.

