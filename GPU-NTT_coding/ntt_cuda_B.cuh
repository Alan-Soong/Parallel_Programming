#include <cuda.h>
#include <curand.h>
#include <cufft.h>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <iostream>
#include <cmath>
#include <vector>

#define CHECK_CUDA_ERROR(err) (checkCudaError(err, __FILE__, __LINE__))

void checkCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Barrett规约结构体
struct Barrett {
    int p;
    uint64_t mu;
    
    __host__ __device__ __forceinline__ Barrett(int mod) : p(mod) {
        mu = (~0ULL) / p;
        if ((~0ULL) - (mu * p) < p) mu++;
    }
};

// 设备端Barrett规约
__device__ __forceinline__ int barrett_reduce_device(int64_t x, const Barrett &b) {
    uint64_t q = __umul64hi(b.mu, (uint64_t)x);
    int64_t r = x - (int64_t)q * b.p;
    if (r >= b.p) r -= b.p;
    if (r < 0) r += b.p;
    return (int)r;
}

// 主机端Barrett规约
__host__ __forceinline__ int barrett_reduce_host(int64_t x, const Barrett &b) {
    return x % b.p;
}

// 模幂运算
__host__ int mod_pow_host(int base, int exp, int mod) {
    Barrett bt(mod);
    int64_t res = 1;
    base = barrett_reduce_host(base, bt);
    
    while (exp > 0) {
        if (exp & 1) {
            res = barrett_reduce_host(res * base, bt);
        }
        base = barrett_reduce_host((int64_t)base * base, bt);
        exp >>= 1;
    }
    return (int)res;
}

// 模逆元计算
int mod_inv(int x, int p) {
    return mod_pow_host(x, p - 2, p);
}

// CUDA核函数：位反转
__global__ void bitReverse(int *data, int n, int log2n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        unsigned int rev = __brev(idx) >> (32 - log2n);
        if (idx < rev) {
            int temp = data[idx];
            data[idx] = data[rev];
            data[rev] = temp;
        }
    }
}

// CUDA核函数：蝶形变换
__global__ void butterflyTransform(int *data, int n, int len, int root, const Barrett bt) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int pairs_per_group = len >> 1;
    int num_groups = n / len;
    int total_pairs = num_groups * pairs_per_group;
    if (tid >= total_pairs) return;
    int group_id = tid / pairs_per_group;
    int pair_id = tid % pairs_per_group;
    int idx1 = group_id * len + pair_id;
    int idx2 = idx1 + pairs_per_group;
    int w = 1;
    int power = (n / len) * pair_id;
    int base = root;
    while (power) {
        if (power & 1) w = barrett_reduce_device((int64_t)w * base, bt);
        base = barrett_reduce_device((int64_t)base * base, bt);
        power >>= 1;
    }
    int u = data[idx1];
    int v = barrett_reduce_device((int64_t)data[idx2] * w, bt);
    int sum = u + v;
    if (sum >= bt.p) sum -= bt.p;
    int diff = u - v;
    if (diff < 0) diff += bt.p;
    data[idx1] = sum;
    data[idx2] = diff;
}

// CUDA核函数：点乘
__global__ void pointwiseMultiplyKernel(int *a, int *b, int *ab, int m, const Barrett bt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < m) {
        ab[i] = barrett_reduce_device((int64_t)a[i] * b[i], bt);
    }
}

// CUDA核函数：缩放结果
__global__ void scaleResultKernel(int *data, int size, int inv_size, const Barrett bt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        data[i] = barrett_reduce_device((int64_t)data[i] * inv_size, bt);
    }
}

// 设备端NTT实现
void ntt_cuda_device(int *d_data, int size, int root, const Barrett &bt) {
    int log2n = (int)log2f((float)size);
    bitReverse<<<(size + 255) / 256, 256>>>(d_data, size, log2n);
    cudaDeviceSynchronize();
    for (int len = 2; len <= size; len <<= 1) {
        int pairs_per_group = len >> 1;
        int num_groups = size / len;
        int total_pairs = num_groups * pairs_per_group;
        butterflyTransform<<<(total_pairs + 255) / 256, 256>>>(d_data, size, len, root, bt);
        cudaDeviceSynchronize();
    }
}

// 主机端NTT实现（只操作主机内存，搬运到设备调用 ntt_cuda_device）
void ntt(int *data, int size, int root, int prime, const Barrett &bt) {
    int *d_data;
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice);
    ntt_cuda_device(d_data, size, root, bt);
    cudaMemcpy(data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

// 多项式乘法主函数
void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    if (n <= 0 || p <= 0) {
        std::cerr << "错误：n和p必须为正整数" << std::endl;
        return;
    }
    int m = 1;
    while (m < 2 * n - 1) m <<= 1;
    Barrett barrett(p);
    int *d_a, *d_b, *d_ab;
    cudaMalloc(&d_a, m * sizeof(int));
    cudaMalloc(&d_b, m * sizeof(int));
    cudaMalloc(&d_ab, m * sizeof(int));
    std::vector<int> h_a(m, 0), h_b(m, 0);
    for(int i = 0; i < n; i++) {
        h_a[i] = ((a[i] % p) + p) % p;
        h_b[i] = ((b[i] % p) + p) % p;
    }
    cudaMemcpy(d_a, h_a.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    int generator = 3;
    int root = mod_pow_host(generator, (p-1)/m, p);
    int inv_root = mod_inv(root, p);
    int inv_m = mod_inv(m, p);
    ntt_cuda_device(d_a, m, root, barrett);
    ntt_cuda_device(d_b, m, root, barrett);
    pointwiseMultiplyKernel<<<(m + 255) / 256, 256>>>(d_a, d_b, d_ab, m, barrett);
    cudaDeviceSynchronize();
    ntt_cuda_device(d_ab, m, inv_root, barrett);
    scaleResultKernel<<<(m + 255) / 256, 256>>>(d_ab, m, inv_m, barrett);
    cudaDeviceSynchronize();
    cudaMemcpy(ab, d_ab, m * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ab);
    for (int i = 0; i < m; ++i) {
        ab[i] = (ab[i] % p + p) % p;
    }
}
