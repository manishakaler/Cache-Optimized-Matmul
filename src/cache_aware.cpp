#include <immintrin.h>
#include <algorithm>
#include <iostream>
#include <chrono>
#include "matrix.h"

using std::size_t;
using std::min;

// Cache-level tile parameters
static constexpr size_t Mc = 64;   // row-block (fits A panel in L2)
static constexpr size_t Kc = 128;  // depth-block
static constexpr size_t Nc = 64;   // col-block (fits B panel in L2)
// Micro-kernel dimensions
static constexpr size_t Mr = 1;    // rows per micro-kernel
static constexpr size_t Nr = 4;    // cols per micro-kernel (1 AVX register)

Matrix<double> matmul_cache(const Matrix<double>& A, const Matrix<double>& B) {
    size_t n = A.size();
    Matrix<double> C(n);

    // Three-level tiling: outer (col) → mid (k) → inner (row)
    for (size_t col_block = 0; col_block < n; col_block += Nc) {
        size_t col_end = min(col_block + Nc, n);

        for (size_t k_block = 0; k_block < n; k_block += Kc) {
            size_t k_end = min(k_block + Kc, n);

            for (size_t row_block = 0; row_block < n; row_block += Mc) {
                size_t row_end = min(row_block + Mc, n);

                // AVX2 micro-kernel: processes Nr=4 columns at a time
                for (size_t i = row_block; i < row_end; ++i) {
                    for (size_t k = k_block; k < k_end; ++k) {
                        double    a_scalar = A(i, k);
                        __m256d   a_vec    = _mm256_set1_pd(a_scalar);

                        size_t j = col_block;
                        for (; j + 3 < col_end; j += Nr) {
                            __m256d c_vec = _mm256_loadu_pd(&C(i, j));
                            __m256d b_vec = _mm256_loadu_pd(&B(k, j));
                            c_vec         = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                            _mm256_storeu_pd(&C(i, j), c_vec);
                        }
                        for (; j < col_end; ++j)
                            C(i, j) += a_scalar * B(k, j);
                    }
                }
            }
        }
    }
    return C;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " N I\n";
        return 1;
    }
    size_t n  = std::stoull(argv[1]);
    int iters = std::stoi(argv[2]);

    auto A = matmul_generate(n);
    auto B = matmul_generate(n);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) volatile auto C = matmul_cache(A, B);
    auto t1 = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "cache_aware N=" << n << " iters=" << iters
              << " time=" << secs << "s\n";
    return 0;
}
