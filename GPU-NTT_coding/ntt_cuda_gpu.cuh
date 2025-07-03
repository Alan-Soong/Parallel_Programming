#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

// 快速幂求模逆 (主机函数)
__host__ int mod_pow(int base, int exp, int mod) {
    int result = 1;
    while (exp) {
        if (exp & 1) result = (long long)result * base % mod;
        base = (long long)base * base % mod;
        exp >>= 1;
    }
    return result;
}

// 简单优化：只改变线程块大小的位反转
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

// 优化蝶形变换：改进内存访问模式
__global__ void ntt_butterfly_kernel(int *a, int n, int len, int wlen, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = len >> 1;
    int stride = n / len;
    
    if (idx >= stride * half) return;
    
    int group = idx / half;
    int pos = idx % half;
    int base = group * len;
    
    // 优化：直接计算单位根幂次，避免循环
    int w_now = 1;
    if (pos > 0) {
        // 使用更高效的幂计算
        int temp_wlen = wlen;
        int temp_pos = pos;
        while (temp_pos) {
            if (temp_pos & 1) w_now = (long long)w_now * temp_wlen % p;
            temp_wlen = (long long)temp_wlen * temp_wlen % p;
            temp_pos >>= 1;
        }
    }
    
    int u_idx = base + pos;
    int v_idx = base + pos + half;
    
    int u = a[u_idx];
    int v = (long long)a[v_idx] * w_now % p;
    
    a[u_idx] = (u + v) % p;
    a[v_idx] = (u - v + p) % p;
}

// 简化点乘内核
__global__ void pointwise_multiply_kernel(int *a, int *b, int n, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    a[idx] = (long long)a[idx] * b[idx] % p;
}

// 简化标准化内核
__global__ void normalize_kernel(int *a, int n, int inv_n, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    a[idx] = (long long)a[idx] * inv_n % p;
}

// 优化的GPU NTT函数
void ntt_cuda_device(int *d_a, int n, int w, int p) {
    // 使用更大的线程块以提高占用率
    int blockSize = 1024;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // 位反转置换
    bit_reverse_kernel<<<gridSize, blockSize>>>(d_a, n);
    cudaDeviceSynchronize();
    
    // 蝶形变换
    for (int len = 2; len <= n; len <<= 1) {
        int wlen = mod_pow(w, (p - 1) / len, p);
        int half = len >> 1;
        int stride = n / len;
        int total_threads = stride * half;
        int butterGridSize = (total_threads + blockSize - 1) / blockSize;
        
        ntt_butterfly_kernel<<<butterGridSize, blockSize>>>(d_a, n, len, wlen, p);
        cudaDeviceSynchronize();
    }
}

// 优化版CUDA多项式乘法
void poly_multiply_cuda(int *a, int *b, int *ab, int n, int p) {
    int m = 1;
    while (m < 2 * n - 1) m <<= 1;

    // 分配主机内存
    int *ta = (int*)calloc(m, sizeof(int));
    int *tb = (int*)calloc(m, sizeof(int));
    memcpy(ta, a, n * sizeof(int));
    memcpy(tb, b, n * sizeof(int));

    int root = 3;
    int inv_root = mod_pow(root, p - 2, p);

    // 分配GPU内存
    int *d_ta, *d_tb;
    size_t size = m * sizeof(int);
    cudaMalloc(&d_ta, size);
    cudaMalloc(&d_tb, size);
    
    // 数据传输到GPU
    cudaMemcpy(d_ta, ta, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tb, tb, size, cudaMemcpyHostToDevice);
    
    // 执行正向NTT
    ntt_cuda_device(d_ta, m, root, p);
    ntt_cuda_device(d_tb, m, root, p);
    
    // 点乘
    int blockSize = 1024;
    int gridSize = (m + blockSize - 1) / blockSize;
    pointwise_multiply_kernel<<<gridSize, blockSize>>>(d_ta, d_tb, m, p);
    cudaDeviceSynchronize();
    
    // 逆向NTT
    ntt_cuda_device(d_ta, m, inv_root, p);
    
    // 标准化
    int inv_m = mod_pow(m, p - 2, p);
    normalize_kernel<<<gridSize, blockSize>>>(d_ta, m, inv_m, p);
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
    poly_multiply_cuda(a, b, ab, n, p);
}

void ntt_cuda(int *a, int n, int w, int p) {
    int *d_a;
    size_t size = n * sizeof(int);
    
    cudaMalloc(&d_a, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    ntt_cuda_device(d_a, n, w, p);
    
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}

void ntt(int *a, int n, int w, int p) {
    ntt_cuda(a, n, w, p);
}
