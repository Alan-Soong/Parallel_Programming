#include <cuda.h>
#include <curand.h>
#include <cufft.h>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CHECK_CUDA_ERROR(err) (checkCudaError(err, __FILE__, __LINE__))

void checkCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 主机端模幂运算
int mod_pow(int base, int exp, int mod) {
    int64_t res = 1;
    base = base % mod;
    while (exp) {
        if (exp & 1)
            res = (res * base) % mod;
        base = (int)(((int64_t)base * base) % mod);
        exp >>= 1;
    }
    return (int)res;
}

// Montgomery结构体：包含模数、基数、基数平方和基数逆元
struct Montgomery {
    int p;                // 模数
    int r;                // 基数
    int r_squared;        // r^2 mod p
    int inv_r;            // r^{-1} mod p
    
    __host__ Montgomery(int mod, int r = 1 << 16) : p(mod), r(r) {
        r_squared = mod_pow(r, 2, p);
        inv_r = mod_pow(r, p - 2, p);
    }
    // 设备端用默认构造
    __device__ Montgomery() {}
};

// 快速幂求模逆 (CUDA设备函数)
__device__ int mod_pow_cuda(int base, int exp, int mod) {
    int result = 1;
    while (exp) {
        if (exp & 1) result = (long long)result * base % mod;
        base = (long long)base * base % mod;
        exp >>= 1;
    }
    return result;
}

// 转换到Montgomery形式的主机端函数
int to_montgomery_host(int x, const Montgomery &m) {
    return (long long)x * m.r_squared % m.p;
}

// 转换回标准形式的主机端函数
int from_montgomery_host(int x, const Montgomery &m) {
    return (long long)x * m.inv_r % m.p;
}

// 使用Montgomery规约的设备端模幂运算
__device__ int device_mod_pow(int base, int exp, const Montgomery &m) {
    int result = 1;
    while (exp) {
        if (exp & 1)
            result = (long long)result * base % m.p;
        base = (long long)base * base % m.p;
        exp >>= 1;
    }
    return result;
}

// 转换到Montgomery形式的设备端核函数
__global__ void to_montgomery_kernel(int *a, int n, const Montgomery &m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] = (long long)a[idx] * m.r_squared % m.p;
}

// 转换回标准形式的设备端核函数
__global__ void from_montgomery_kernel(int *a, int n, const Montgomery &m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] = (long long)a[idx] * m.inv_r % m.p;
}

// 位反转核函数：与之前相同
__global__ void bit_reverse_kernel(int *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // 计算位反转的索引
    int j = 0;
    int temp_idx = idx;
    int log_n = 0;
    int temp_n = n;
    while (temp_n > 1) {
        log_n++;
        temp_n >>= 1;
    }
    
    for (int k = 0; k < log_n; k++) {
        j = (j << 1) | (temp_idx & 1);
        temp_idx >>= 1;
    }
    
    // 只有当idx < j时才交换，避免重复交换
    if (idx < j) {
        int temp = a[idx];
        a[idx] = a[j];
        a[j] = temp;
    }
}

// 使用Montgomery规约的蝶形变换核函数
__global__ void ntt_butterfly_kernel(int *a, int n, int len, int wlen, const Montgomery &m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = len >> 1;
    int stride = n / len;
    
    if (idx >= stride * half) return;
    
    int group = idx / half;
    int pos = idx % half;
    int base = group * len;
    
    // 计算当前线程对应的单位根幂次
    int w_now = 1;
    if (pos > 0) {
        int temp_wlen = wlen;
        int temp_pos = pos;
        while (temp_pos) {
            if (temp_pos & 1) w_now = (long long)w_now * temp_wlen % m.p;
            temp_wlen = (long long)temp_wlen * temp_wlen % m.p;
            temp_pos >>= 1;
        }
    }
    
    int u_idx = base + pos;
    int v_idx = base + pos + half;
    
    int u = a[u_idx];
    int v = (long long)a[v_idx] * w_now % m.p;
    
    a[u_idx] = (u + v) % m.p;
    a[v_idx] = (u - v + m.p) % m.p;
}

// 点乘核函数：保持与之前相同
__global__ void pointwise_multiply_kernel(int *a, int *b, int n, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    a[idx] = (long long)a[idx] * b[idx] % p;
}

// 标准化核函数：添加Montgomery还原
__global__ void normalize_kernel(int *a, int n, int inv_n, const Montgomery &m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    a[idx] = (long long)a[idx] * inv_n % m.p;
}

// GPU NTT函数（支持Montgomery规约）
void ntt_montgomery(int *d_a, int n, int w, const Montgomery &m) {
    int blockSize = 1024;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // 转换到Montgomery形式
    to_montgomery_kernel<<<gridSize, blockSize>>>(d_a, n, m);
    cudaDeviceSynchronize();
    
    // 位反转置换
    bit_reverse_kernel<<<gridSize, blockSize>>>(d_a, n);
    cudaDeviceSynchronize();
    
    // 蝶形变换
    for (int len = 2; len <= n; len <<= 1) {
        int wlen = device_mod_pow(w, (m.p - 1) / len, m);
        int half = len >> 1;
        int stride = n / len;
        int total_threads = stride * half;
        int butterGridSize = (total_threads + blockSize - 1) / blockSize;
        
        ntt_butterfly_kernel<<<butterGridSize, blockSize>>>(d_a, n, len, wlen, m);
        cudaDeviceSynchronize();
    }
}

// 主机端NTT实现（支持Montgomery规约）
void ntt(int *data, int size, int root, const Montgomery &m) {
    int *d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice));
    
    ntt_montgomery(d_data, size, root, m);
    
    CHECK_CUDA_ERROR(cudaMemcpy(data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
}

// 标准化结果函数（支持Montgomery规约）
void normalize_result(int *d_data, int size, const Montgomery &m) {
    int blockSize = 512;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // 计算逆变换系数
    int inv_n = mod_pow_cuda(size, m.p - 2, m.p);
    normalize_kernel<<<gridSize, blockSize>>>(d_data, size, inv_n, m);
    
    // 转换回标准形式
    from_montgomery_kernel<<<gridSize, blockSize>>>(d_data, size, m);
    cudaDeviceSynchronize();
}

// CUDA多项式乘法（支持Montgomery规约）
void poly_multiply_cuda(int *a, int *b, int *ab, int n, const Montgomery &m) {
    int m_size = 1;
    while (m_size < 2 * n - 1) m_size <<= 1;

    // 分配主机内存
    int *ta = (int*)calloc(m_size, sizeof(int));
    int *tb = (int*)calloc(m_size, sizeof(int));
    memcpy(ta, a, n * sizeof(int));
    memcpy(tb, b, n * sizeof(int));
    
    // 转换到Montgomery形式
    for (int i = 0; i < n; ++i) {
        ta[i] = to_montgomery_host(ta[i], m);
        tb[i] = to_montgomery_host(tb[i], m);
    }

    // 分配GPU内存
    int *d_ta, *d_tb;
    size_t size = m_size * sizeof(int);
    cudaMalloc(&d_ta, size);
    cudaMalloc(&d_tb, size);
    
    // 数据传输到GPU
    cudaMemcpy(d_ta, ta, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tb, tb, size, cudaMemcpyHostToDevice);
    
    // 执行正向NTT
    ntt_montgomery(d_ta, m_size, 3, m);
    ntt_montgomery(d_tb, m_size, 3, m);
    
    // 点乘
    int blockSize = 1024;
    int gridSize = (m_size + blockSize - 1) / blockSize;
    pointwise_multiply_kernel<<<gridSize, blockSize>>>(d_ta, d_tb, m_size, m.p);
    cudaDeviceSynchronize();
    
    // 逆向NTT
    int inv_root = mod_pow(3, m.p - 2, m.p);
    ntt_montgomery(d_ta, m_size, inv_root, m);
    
    // 标准化并转换回标准形式
    normalize_result(d_ta, m_size, m);
    cudaDeviceSynchronize();
    
    // 数据传输回主机
    cudaMemcpy(ta, d_ta, size, cudaMemcpyDeviceToHost);
    
    // 复制最终结果
    for (int i = 0; i < 2 * n - 1; ++i)
        ab[i] = ta[i];

    // 清理资源
    free(ta);
    free(tb);
    cudaFree(d_ta);
    cudaFree(d_tb);
}

// 保持兼容性的函数
void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    Montgomery m(p);
    poly_multiply_cuda(a, b, ab, n, m);
}