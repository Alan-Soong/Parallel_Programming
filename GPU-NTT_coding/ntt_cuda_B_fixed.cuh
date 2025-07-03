#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
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
__device__ __forceinline__ int barrett_reduce(int64_t x, const Barrett &bt) {
    uint64_t q = __umul64hi(bt.mu, (uint64_t)x);
    int64_t r = x - (int64_t)q * bt.p;
    if (r >= bt.p) r -= bt.p;
    if (r < 0) r += bt.p;
    return (int)r;
}

// 主机端Barrett规约
__host__ __forceinline__ int barrett_reduce_host(int64_t x, const Barrett &bt) {
    return x % bt.p;
}

// 设备端模幂运算
__device__ int mod_pow_device(int base, int exp, const Barrett &bt) {
    int result = 1;
    base = barrett_reduce(base, bt);
    
    while (exp > 0) {
        if (exp & 1) {
            result = barrett_reduce((int64_t)result * base, bt);
        }
        base = barrett_reduce((int64_t)base * base, bt);
        exp >>= 1;
    }
    return result;
}

// 主机端模幂运算
__host__ int mod_pow_host(int base, int exp, int mod) {
    Barrett bt(mod);
    int result = 1;
    base = barrett_reduce_host(base, bt);
    
    while (exp > 0) {
        if (exp & 1) {
            result = barrett_reduce_host((int64_t)result * base, bt);
        }
        base = barrett_reduce_host((int64_t)base * base, bt);
        exp >>= 1;
    }
    return result;
}

// 模逆元计算
__host__ int mod_inv(int x, int p) {
    return mod_pow_host(x, p - 2, p);
}

// 优化的位反转核函数
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

// 使用Barrett规约的蝶形变换核函数
__global__ void ntt_butterfly_kernel(int *a, int n, int len, int wlen, const Barrett bt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = len >> 1;
    int stride = n / len;
    
    if (idx >= stride * half) return;
    
    int group = idx / half;
    int pos = idx % half;
    int base = group * len;
    
    // 计算单位根的幂次
    int w_now = 1;
    if (pos > 0) {
        int temp_wlen = wlen;
        int temp_pos = pos;
        while (temp_pos) {
            if (temp_pos & 1) 
                w_now = barrett_reduce((int64_t)w_now * temp_wlen, bt);
            temp_wlen = barrett_reduce((int64_t)temp_wlen * temp_wlen, bt);
            temp_pos >>= 1;
        }
    }
    
    int u_idx = base + pos;
    int v_idx = base + pos + half;
    
    int u = a[u_idx];
    int v = barrett_reduce((int64_t)a[v_idx] * w_now, bt);
    int sum = u + v;
    int diff = u - v;
    
    a[u_idx] = (sum >= bt.p) ? sum - bt.p : sum;
    a[v_idx] = (diff < 0) ? diff + bt.p : diff;
}

// 使用Barrett规约的点乘核函数
__global__ void pointwise_multiply_kernel(int *a, int *b, int n, const Barrett bt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) 
        a[idx] = barrett_reduce((int64_t)a[idx] * b[idx], bt);
}

// 使用Barrett规约的标准化核函数
__global__ void normalize_kernel(int *a, int n, int inv_n, const Barrett bt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) 
        a[idx] = barrett_reduce((int64_t)a[idx] * inv_n, bt);
}

// 优化的NTT设备函数
void ntt_cuda_device(int *d_a, int n, int w, const Barrett bt) {
    int blockSize = 512;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // 位反转置换
    bit_reverse_kernel<<<gridSize, blockSize>>>(d_a, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // 蝶形变换
    for (int len = 2; len <= n; len <<= 1) {
        int wlen = mod_pow_host(w, (bt.p - 1) / len, bt.p);
        int half = len >> 1;
        int total_threads = (n / len) * half;
        gridSize = (total_threads + blockSize - 1) / blockSize;
        
        ntt_butterfly_kernel<<<gridSize, blockSize>>>(d_a, n, len, wlen, bt);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
}

// 优化的多项式乘法函数
void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    if (n <= 0 || p <= 0) {
        std::cerr << "错误：n和p必须为正整数" << std::endl;
        return;
    }
    
    // 确定NTT长度
    int m = 1;
    while (m < 2 * n - 1) m <<= 1;
    
    // 初始化Barrett结构
    Barrett bt(p);
    
    // 分配设备内存
    int *d_ta, *d_tb;
    CHECK_CUDA_ERROR(cudaMalloc(&d_ta, m * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_tb, m * sizeof(int)));
    
    // 准备输入数据
    std::vector<int> ta(m, 0);
    std::vector<int> tb(m, 0);
    for (int i = 0; i < n; i++) {
        ta[i] = ((a[i] % p) + p) % p;
        tb[i] = ((b[i] % p) + p) % p;
    }
    
    // 拷贝数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_ta, ta.data(), m * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tb, tb.data(), m * sizeof(int), cudaMemcpyHostToDevice));
    
    // 计算NTT参数
    int root = 3; // 常用原根
    int inv_root = mod_inv(root, p);
    int inv_m = mod_inv(m, p);
    
    // 执行正向NTT
    ntt_cuda_device(d_ta, m, root, bt);
    ntt_cuda_device(d_tb, m, root, bt);
    
    // 点乘
    int blockSize = 512;
    int gridSize = (m + blockSize - 1) / blockSize;
    pointwise_multiply_kernel<<<gridSize, blockSize>>>(d_ta, d_tb, m, bt);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // 执行逆向NTT
    ntt_cuda_device(d_ta, m, inv_root, bt);
    
    // 标准化
    normalize_kernel<<<gridSize, blockSize>>>(d_ta, m, inv_m, bt);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // 拷贝结果回主机
    CHECK_CUDA_ERROR(cudaMemcpy(ab, d_ta, (2 * n - 1) * sizeof(int), cudaMemcpyDeviceToHost));
    
    // 后处理确保结果在[0, p-1]范围内
    for (int i = 0; i < 2 * n - 1; i++) {
        ab[i] = (ab[i] % p + p) % p;
    }
    
    // 释放设备内存
    CHECK_CUDA_ERROR(cudaFree(d_ta));
    CHECK_CUDA_ERROR(cudaFree(d_tb));
}
