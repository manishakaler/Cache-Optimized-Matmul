#include <iostream>
#include <chrono>
#include "matrix.h"

Matrix<double> matmul_naive(const Matrix<double>& A, const Matrix<double>& B) {
    std::size_t n = A.size();
    Matrix<double> C(n);

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < n; ++k)
                C(i, j) += A(i, k) * B(k, j);  // B access: column-stride → cache-hostile
    return C;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " N I\n"
                  << "  N: matrix dimension\n  I: iterations\n";
        return 1;
    }
    std::size_t n = std::stoull(argv[1]);
    int iters     = std::stoi(argv[2]);

    auto A = matmul_generate(n);
    auto B = matmul_generate(n);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) volatile auto C = matmul_naive(A, B);
    auto t1 = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "naive N=" << n << " iters=" << iters
              << " time=" << secs << "s\n";
    return 0;
}
