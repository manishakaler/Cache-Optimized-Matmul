#include <algorithm>
#include <iostream>
#include <chrono>
#include "matrix.h"

Matrix<double> matmul_blocked(const Matrix<double>& A,
                               const Matrix<double>& B,
                               std::size_t BS) {
    std::size_t n = A.size();
    Matrix<double> C(n);

    for (std::size_t bi = 0; bi < n; bi += BS) {
        for (std::size_t bk = 0; bk < n; bk += BS) {
            for (std::size_t bj = 0; bj < n; bj += BS) {
                std::size_t i_end = std::min(bi + BS, n);
                std::size_t k_end = std::min(bk + BS, n);
                std::size_t j_end = std::min(bj + BS, n);

                for (std::size_t i = bi; i < i_end; ++i) {
                    for (std::size_t k = bk; k < k_end; ++k) {
                        double a_ik = A(i, k);
                        std::size_t j = bj;
                        // 4-way unrolled inner loop
                        for (; j + 3 < j_end; j += 4) {
                            C(i, j)     += a_ik * B(k, j);
                            C(i, j + 1) += a_ik * B(k, j + 1);
                            C(i, j + 2) += a_ik * B(k, j + 2);
                            C(i, j + 3) += a_ik * B(k, j + 3);
                        }
                        for (; j < j_end; ++j)
                            C(i, j) += a_ik * B(k, j);
                    }
                }
            }
        }
    }
    return C;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " N I B\n"
                  << "  N: matrix dimension\n  I: iterations\n  B: block size\n";
        return 1;
    }
    std::size_t n  = std::stoull(argv[1]);
    int iters      = std::stoi(argv[2]);
    std::size_t BS = std::stoull(argv[3]);

    auto A = matmul_generate(n);
    auto B = matmul_generate(n);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) volatile auto C = matmul_blocked(A, B, BS);
    auto t1 = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "blocked N=" << n << " B=" << BS << " iters=" << iters
              << " time=" << secs << "s\n";
    return 0;
}
