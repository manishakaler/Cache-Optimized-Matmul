#include <immintrin.h>
#include <algorithm>
#include <iostream>
#include <chrono>
#include "matrix.h"

using std::size_t;
using std::min;

// Inner kernel: update C(i, jj..j_end) += A(i,k) * B(k, jj..j_end)
// Processes 4 doubles at a time via AVX2 FMA
static inline void avx_update_row(const Matrix<double>& A,
                                   const Matrix<double>& B,
                                   Matrix<double>& C,
                                   size_t i, size_t k,
                                   size_t jj, size_t j_end) {
    double a_scalar  = A(i, k);
    __m256d a_vec    = _mm256_set1_pd(a_scalar);  // broadcast scalar to all 4 lanes

    size_t j = jj;
    // Vectorized loop: 4 doubles per iteration (256-bit load/store)
    for (; j + 3 < j_end; j += 4) {
        __m256d c_vec = _mm256_loadu_pd(&C(i, j));        // load 4 C values
        __m256d b_vec = _mm256_loadu_pd(&B(k, j));        // load 4 B values
        c_vec         = _mm256_fmadd_pd(a_vec, b_vec, c_vec); // FMA: c += a * b
        _mm256_storeu_pd(&C(i, j), c_vec);                // store back
    }
    // Scalar cleanup for remainder (n % 4 != 0)
    for (; j < j_end; ++j)
        C(i, j) += a_scalar * B(k, j);
}

Matrix<double> matmul_avx(const Matrix<double>& A,
                           const Matrix<double>& B,
                           int BS = 64) {
    size_t n = A.size();
    Matrix<double> C(n);

    for (size_t ii = 0; ii < n; ii += BS) {
        for (size_t jj = 0; jj < n; jj += BS) {
            for (size_t kk = 0; kk < n; kk += BS) {
                size_t i_end = min(ii + (size_t)BS, n);
                size_t j_end = min(jj + (size_t)BS, n);
                size_t k_end = min(kk + (size_t)BS, n);

                for (size_t i = ii; i < i_end; ++i)
                    for (size_t k = kk; k < k_end; ++k)
                        avx_update_row(A, B, C, i, k, jj, j_end);
            }
        }
    }
    return C;
}

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " N I [B]\n"
                  << "  N: matrix dimension\n  I: iterations\n  B: block size (default 64)\n";
        return 1;
    }
    size_t n = std::stoull(argv[1]);
    int iters = std::stoi(argv[2]);
    int BS    = (argc == 4) ? std::stoi(argv[3]) : 64;

    auto A = matmul_generate(n);
    auto B = matmul_generate(n);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) volatile auto C = matmul_avx(A, B, BS);
    auto t1 = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "avx_vectorized N=" << n << " B=" << BS << " iters=" << iters
              << " time=" << secs << "s\n";
    return 0;
}
