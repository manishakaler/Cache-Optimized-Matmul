#include <iostream>
#include <chrono>
#include "matrix.h"

Matrix<double> matmul_unrolled(const Matrix<double>& A, const Matrix<double>& B) {
    std::size_t n = A.size();
    Matrix<double> C(n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            double a_ik = A(i, k);
            std::size_t j = 0;
            // Unrolled body: 4 independent MACs in flight
            for (; j + 3 < n; j += 4) {
                C(i, j)     += a_ik * B(k, j);
                C(i, j + 1) += a_ik * B(k, j + 1);
                C(i, j + 2) += a_ik * B(k, j + 2);
                C(i, j + 3) += a_ik * B(k, j + 3);
            }
            // Scalar cleanup for remainder
            for (; j < n; ++j)
                C(i, j) += a_ik * B(k, j);
        }
    }
    return C;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " N I\n";
        return 1;
    }
    std::size_t n = std::stoull(argv[1]);
    int iters     = std::stoi(argv[2]);

    auto A = matmul_generate(n);
    auto B = matmul_generate(n);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) volatile auto C = matmul_unrolled(A, B);
    auto t1 = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "unrolled N=" << n << " iters=" << iters
              << " time=" << secs << "s\n";
    return 0;
}
