#include <cuda.h>
#include <curand.h>
#include <cufft.h>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <iostream>
#include <cmath>

#define CHECK_CUDA_ERROR(err) (checkCudaError(err, __FILE__, __LINE__))

void checkCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
                  << cudaGetErrorString(err) << std::endl;
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

// Montgomery结构体
struct Montgomery {
    int p;                // 模数
    int r;                // 基数
    int r_squared;        // r^2 mod p
    int inv_r;            // r^{-1} mod p
    
    __host__ Montgomery(int mod, int r = 1 << 16) : p(mod), r(r) {
        r_squared = (long long)r * r % p;
        inv_r = mod_pow(r, p - 2, p);
    }
    // 设备端用默认构造
    __device__ Montgomery() {}
};

// 计算模逆元
int mod_inv(int x, int p) {
    return mod_pow(x, p - 2, p);
}

// 计算Montgomery形式
int to_montgomery_host(int x, const Montgomery &m) {
    return (long long)x * m.r_squared % m.p;
}

// 转换回标准形式
int from_montgomery_host(int x, const Montgomery &m) {
    return (long long)x * m.inv_r % m.p;
}

// 使用Montgomery规约的设备端乘法
__device__ int montgomery_multiply_device(int a, int b, const Montgomery &m) {
    int64_t product = (long long)a * b;
    int64_t q = product * m.inv_r;
    int64_t r = product - q * m.p;
    return (int)(r + m.p) % m.p;
}

// 使用Montgomery规约的设备端模幂运算
__device__ int device_mod_pow(int base, int exp, const Montgomery &m) {
    int result = 1;
    while (exp) {
        if (exp & 1)
            result = montgomery_multiply_device(result, base, m);
        base = montgomery_multiply_device(base, base, m);
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

// 位反转核函数
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

// 点乘核函数
__global__ void pointwise_multiply_kernel(int *a, int *b, int n, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    a[idx] = (long long)a[idx] * b[idx] % p;
}

// 蝶形变换核函数（使用Montgomery规约）
__global__ void ntt_butterfly_kernel(int *a, int n, int len, int wlen, const Montgomery &m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = len >> 1;
    int stride = n / len;
    
    if (idx >= stride * half) return;
    
    int group = idx / half;
    int pos = idx % half;
    int base = group * len;
    
    int w_now = 1;
    if (pos > 0) {
        int temp_wlen = wlen;
        int temp_pos = pos;
        while (temp_pos) {
            if (temp_pos & 1) w_now = montgomery_multiply_device(w_now, temp_wlen, m);
            temp_wlen = montgomery_multiply_device(temp_wlen, temp_wlen, m);
            temp_pos >>= 1;
        }
    }
    
    int u_idx = base + pos;
    int v_idx = base + pos + half;
    
    int u = a[u_idx];
    int v = montgomery_multiply_device(a[v_idx], w_now, m);
    
    a[u_idx] = (u + v) % m.p;
    a[v_idx] = (u - v + m.p) % m.p;
}

// GPU NTT函数（使用Montgomery规约）
void ntt_cuda_device(int *d_a, int n, int w, const Montgomery &m) {
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
        int wlen = mod_pow(w, (m.p - 1) / len, m.p);
        int half = len >> 1;
        int stride = n / len;
        int total_threads = stride * half;
        int butterGridSize = (total_threads + blockSize - 1) / blockSize;
        
        ntt_butterfly_kernel<<<butterGridSize, blockSize>>>(d_a, n, len, wlen, m);
        cudaDeviceSynchronize();
    }
}

// 主机端NTT实现（使用Montgomery规约）
void ntt(int *data, int size, int root, const Montgomery &m) {
    int *d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice));
    
    ntt_cuda_device(d_data, size, root, m);
    
    CHECK_CUDA_ERROR(cudaMemcpy(data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
}

// 转换回标准形式的全局函数
void normalize_result(int *data, int size, const Montgomery &m) {
    int blockSize = 512;
    int gridSize = (size + blockSize - 1) / blockSize;
    from_montgomery_kernel<<<gridSize, blockSize>>>(data, size, m);
    cudaDeviceSynchronize();
}

// 优化版CUDA多项式乘法（使用Montgomery规约）
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
        ta[i] = (long long)ta[i] * m.r_squared % m.p;
        tb[i] = (long long)tb[i] * m.r_squared % m.p;
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
    ntt_cuda_device(d_ta, m_size, 3, m);
    ntt_cuda_device(d_tb, m_size, 3, m);
    
    // 点乘
    int blockSize = 512;
    int gridSize = (m_size + blockSize - 1) / blockSize;
    pointwise_multiply_kernel<<<gridSize, blockSize>>>(d_ta, d_tb, m_size, m.p);
    cudaDeviceSynchronize();
    
    // 逆向NTT
    ntt_cuda_device(d_ta, m_size, mod_inv(3, m.p), m);
    
    // 标准化
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