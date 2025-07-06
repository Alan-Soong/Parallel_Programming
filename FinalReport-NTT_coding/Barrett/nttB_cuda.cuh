#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>

#define BLOCK_SIZE 256

__device__ __forceinline__ int barrett_reduce(uint64_t a, uint64_t m, int mod) {
    uint64_t q = __umul64hi(a, m);
    int r = a - q * mod;
    return (r < mod) ? ((r < 0) ? r + mod : r) : r - mod;
}

__device__ int mod_mul(int a, int b, uint64_t m, int mod) {
    return barrett_reduce((uint64_t)a * b, m, mod);
}

int mod_pow(int base, int exp, int mod) {
    int64_t res = 1, b = base;
    while (exp) {
        if (exp & 1) res = res * b % mod;
        b = b * b % mod;
        exp >>= 1;
    }
    return int(res);
}

void bit_reverse(std::vector<int>& a) {
    int n = a.size();
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

__global__ void butterfly_kernel(int* a, int len, int wlen, int n, int mod, uint64_t m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half = len >> 1;
    int pos = (tid / half) * len + (tid % half);
    if (pos + half >= n) return;

    int w = 1;
    for (int i = 0; i < (tid % half); ++i)
        w = mod_mul(w, wlen, m, mod);

    int u = a[pos];
    int v = mod_mul(a[pos + half], w, m, mod);
    a[pos] = (u + v >= mod) ? u + v - mod : u + v;
    a[pos + half] = (u - v < 0) ? u - v + mod : u - v;
}

void cuda_ntt(std::vector<int>& a, bool invert, int root, int mod) {
    int n = a.size();
    bit_reverse(a);
    
    uint64_t m = ((uint64_t(1) << 63) << 1) / mod;
    int *d_a;
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMemcpy(d_a, a.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    for (int len = 2; len <= n; len <<= 1) {
        int wlen = mod_pow(root, (mod - 1) / len, mod);
        if (invert) wlen = mod_pow(wlen, mod - 2, mod);

        int half = len >> 1;
        int threads = (n / len) * half;
        int grid = (threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        butterfly_kernel<<<grid, BLOCK_SIZE>>>(d_a, len, wlen, n, mod, m);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(a.data(), d_a, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);

    if (invert) {
        int inv_n = mod_pow(n, mod - 2, mod);
        for (int& x : a) x = (int)((int64_t)x * inv_n % mod);
    }
}

__global__ void pointwise_mul_kernel(int *A, int *B, int *C, int n, uint64_t m, int mod) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = mod_mul(A[i], B[i], m, mod);
}

void cuda_pointwise_multiply(std::vector<int>& A, std::vector<int>& B, std::vector<int>& C, int mod) {
    int n = A.size();
    uint64_t m = ((uint64_t(1) << 63) << 1) / mod;
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMalloc(&d_C, n * sizeof(int));

    cudaMemcpy(d_A, A.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pointwise_mul_kernel<<<threads, BLOCK_SIZE>>>(d_A, d_B, d_C, n, m, mod);
    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), d_C, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int get_primitive_root(int p) {
    return 3; // for 7340033
}

void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    int lim = 1;
    while (lim < 2 * n - 1) lim <<= 1;

    std::vector<int> A(lim), B(lim), C(lim);
    for (int i = 0; i < n; ++i) {
        A[i] = a[i] % p;
        B[i] = b[i] % p;
    }

    int root = get_primitive_root(p);
    cuda_ntt(A, false, root, p);
    cuda_ntt(B, false, root, p);
    cuda_pointwise_multiply(A, B, C, p);
    cuda_ntt(C, true, root, p);

    for (int i = 0; i < 2 * n - 1; ++i)
        ab[i] = C[i];
}