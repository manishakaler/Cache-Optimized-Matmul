#include <immintrin.h>
#include <algorithm>
#include <iostream>
#include <chrono>
#include "matrix.h"

using std::size_t;
using std::min;

static constexpr size_t BLOCK_ROWS = 64;
static constexpr size_t BLOCK_K    = 128;
static constexpr size_t BLOCK_COLS = 64;
static constexpr size_t MR = 2;   // micro-rows  (2 C rows processed at once)
static constexpr size_t NR = 4;   // micro-cols  (4 doubles = 1 AVX register)

// 2×4 micro-kernel: accumulates k_begin..k_end contribution into
// C(row0..row0+1, col0..col0+3).
// B panel is loaded once and reused for both rows → 2x B-load reduction.
static inline void microkernel_2x4(const Matrix<double>& A,
                                    const Matrix<double>& B,
                                    Matrix<double>& C,
                                    size_t row0, size_t col0,
                                    size_t k_begin, size_t k_end) {
    __m256d c0 = _mm256_setzero_pd();   // accumulator: C row0
    __m256d c1 = _mm256_setzero_pd();   // accumulator: C row1

    for (size_t k = k_begin; k < k_end; ++k) {
        __m256d b   = _mm256_loadu_pd(&B(k, col0));          // load B(k, col0..col0+3)
        __m256d a0  = _mm256_set1_pd(A(row0,     k));        // broadcast A(row0, k)
        __m256d a1  = _mm256_set1_pd(A(row0 + 1, k));        // broadcast A(row1, k)
        c0 = _mm256_fmadd_pd(a0, b, c0);                     // c_row0 += a0 * b
        c1 = _mm256_fmadd_pd(a1, b, c1);                     // c_row1 += a1 * b  (b reused!)
    }

    // Store accumulators back to C (accumulate, not overwrite)
    double t0[4], t1[4];
    _mm256_storeu_pd(t0, c0);
    _mm256_storeu_pd(t1, c1);
    for (int off = 0; off < 4; ++off) {
        C(row0,     col0 + off) += t0[off];
        C(row0 + 1, col0 + off) += t1[off];
    }
}

Matrix<double> matmul_reg(const Matrix<double>& A, const Matrix<double>& B) {
    size_t n = A.size();
    Matrix<double> C(n);

    for (size_t col_block = 0; col_block < n; col_block += BLOCK_COLS) {
        size_t col_end = min(col_block + BLOCK_COLS, n);

        for (size_t k_block = 0; k_block < n; k_block += BLOCK_K) {
            size_t k_end = min(k_block + BLOCK_K, n);

            for (size_t row_block = 0; row_block < n; row_block += BLOCK_ROWS) {
                size_t row_end = min(row_block + BLOCK_ROWS, n);

                // Largest row / col indices that fit the 2×4 micro-kernel exactly
                size_t row_micro_end = row_end - (row_end - row_block) % MR;
                size_t col_micro_end = col_end - (col_end - col_block) % NR;

                // Main micro-kernel region
                for (size_t i = row_block; i < row_micro_end; i += MR) {
                    for (size_t j = col_block; j < col_micro_end; j += NR)
                        microkernel_2x4(A, B, C, i, j, k_block, k_end);

                    // Column cleanup (< NR columns remaining)
                    for (size_t j = col_micro_end; j < col_end; ++j)
                        for (size_t k = k_block; k < k_end; ++k) {
                            C(i,     j) += A(i,     k) * B(k, j);
                            C(i + 1, j) += A(i + 1, k) * B(k, j);
                        }
                }

                // Row cleanup (< MR rows remaining)
                for (size_t i = row_micro_end; i < row_end; ++i)
                    for (size_t k = k_block; k < k_end; ++k) {
                        double a_ik = A(i, k);
                        for (size_t j = col_block; j < col_end; ++j)
                            C(i, j) += a_ik * B(k, j);
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
    for (int i = 0; i < iters; ++i) volatile auto C = matmul_reg(A, B);
    auto t1 = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "register_kernel N=" << n << " iters=" << iters
              << " time=" << secs << "s\n";
    return 0;
}
