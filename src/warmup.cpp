#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>
#include <chrono>

using namespace std;

long sequential_access(long* arr, size_t size, int iters) {
    long sum = 0;
    for (int it = 0; it < iters; ++it)
        for (size_t i = 0; i < size; ++i)
            sum += arr[i];
    return sum;
}

long strided_access(long* arr, size_t size, size_t stride, int iters) {
    long sum = 0;
    for (int it = 0; it < iters; ++it)
        for (size_t i = 0; i < size; i += stride)
            sum += arr[i];
    return sum;
}

long random_access(long* arr, size_t size, int iters) {
    vector<size_t> idx(size);
    iota(idx.begin(), idx.end(), 0);
    mt19937_64 rng(42);
    shuffle(idx.begin(), idx.end(), rng);

    long sum = 0;
    for (int it = 0; it < iters; ++it)
        for (size_t i = 0; i < size; ++i)
            sum += arr[idx[i]];
    return sum;
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        cerr << "Usage: " << argv[0] << " <size> <pattern> <iterations> [stride]\n";
        cerr << "  pattern: sequential | strided | random\n";
        return 1;
    }
    size_t array_size = atoll(argv[1]);
    string pattern    = argv[2];
    int    iters      = atoi(argv[3]);
    size_t stride     = (argc == 5) ? atoll(argv[4]) : 16;

    long* arr = new long[array_size];
    memset(arr, 1, array_size * sizeof(long));

    auto t0 = chrono::high_resolution_clock::now();
    long result = 0;
    if      (pattern == "sequential") result = sequential_access(arr, array_size, iters);
    else if (pattern == "strided")    result = strided_access(arr, array_size, stride, iters);
    else if (pattern == "random")     result = random_access(arr, array_size, iters);
    else { cerr << "Unknown pattern: " << pattern << "\n"; delete[] arr; return 1; }
    auto t1 = chrono::high_resolution_clock::now();

    double secs = chrono::duration<double>(t1 - t0).count();
    cout << "pattern=" << pattern
         << " size=" << array_size
         << " stride=" << stride
         << " iters=" << iters
         << " time=" << secs << "s"
         << " sum=" << result << "\n";

    delete[] arr;
    return 0;
}
