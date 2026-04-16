#pragma once
// ============================================================================
// matrix.h — Row-major NxN matrix stored in a flat vector
// ============================================================================

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <random>

template <typename T>
class Matrix {
public:
    Matrix() : n_(0) {}

    explicit Matrix(std::size_t n) : n_(n), data_(n * n, T{}) {}

    inline T& operator()(std::size_t i, std::size_t j) {
        return data_[i * n_ + j];
    }
    inline const T& operator()(std::size_t i, std::size_t j) const {
        return data_[i * n_ + j];
    }

    inline T* row_ptr(std::size_t i) { return data_.data() + i * n_; }
    inline const T* row_ptr(std::size_t i) const { return data_.data() + i * n_; }

    std::size_t size() const { return n_; }

    // Verify result against a reference (for correctness checking)
    bool approx_equal(const Matrix<T>& other, double tol = 1e-6) const {
        if (n_ != other.n_) return false;
        for (std::size_t i = 0; i < n_ * n_; ++i) {
            if (std::abs(data_[i] - other.data_[i]) > tol) return false;
        }
        return true;
    }

private:
    std::size_t n_;
    std::vector<T> data_;
};

// Generate a random NxN matrix with values in [0, 1)
inline Matrix<double> matmul_generate(std::size_t n) {
    Matrix<double> mat(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            mat(i, j) = dis(gen);
    return mat;
}
