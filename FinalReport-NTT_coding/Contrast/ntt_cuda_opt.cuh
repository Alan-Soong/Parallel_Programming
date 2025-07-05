#include <cuda.h>
#include <curand.h>
#include <cufft.h>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <iostream>
#include <cmath>

#define CHECK_CUDA_ERROR(err) (checkCudaError(err, __FILE__, __LINE__))

// 错误检查函数
void checkCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 主机和设备均可使用的模幂运算
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

// 计算模逆元
int mod_inv(int x, int p) {
    return mod_pow(x, p - 2, p);
}

// 在设备上执行的模幂运算
__device__ int device_mod_pow(int base, int exp, int mod) {
    int res = 1;
    base = base % mod;
    while (exp) {
        if (exp & 1)
            res = (int)(((int64_t)res * base) % mod);
        base = (int)(((int64_t)base * base) % mod);
        exp >>= 1;
    }
    return res;
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
__global__ void butterflyTransform(int *data, int n, int len, int root, int prime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int group = idx / (len/2);
    int j = idx % (len/2);
    
    if (group >= n/len) return;
    
    int base = group * len;
    int idx1 = base + j;
    int idx2 = base + j + len/2;
    
    int w = device_mod_pow(root, j * (n/len), prime);
    int u = data[idx1];
    int v = (int)((int64_t)data[idx2] * w % prime);
    
    data[idx1] = (u + v) % prime;
    data[idx2] = (u - v + prime) % prime;
}

// 主机端NTT实现
void ntt(int *data, int size, int root, int prime) {
    int *d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice));
    
    // 计算log2(size)
    int log2n = (int)log2f((float)size);
    
    // 执行位反转
    dim3 blockSize(1024);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    bitReverse<<<gridSize, blockSize>>>(d_data, size, log2n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // 执行蝶形变换
    for (int len = 2; len <= size; len <<= 1) {
        int num_ops = size / 2;
        int num_blocks = (num_ops + blockSize.x - 1) / blockSize.x;
        butterflyTransform<<<num_blocks, blockSize>>>(d_data, size, len, root, prime);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    
    CHECK_CUDA_ERROR(cudaMemcpy(data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
}

// CUDA核函数：多项式乘法
__global__ void polynomialMultiplyKernel(int *a, int *b, int *ab, int m, int p) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < m) {
        ab[i] = (int)((int64_t)a[i] * b[i] % p);
    }
}

// CUDA核函数：缩放结果
__global__ void scaleResultKernel(int *data, int size, int inv_size, int p) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        data[i] = (int)((int64_t)data[i] * inv_size % p);
    }
}

// 多项式乘法主函数
void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    if (n <= 0) {
        std::cerr << "错误：n必须为正整数" << std::endl;
        return;
    }

    int m = 1;
    while (m < 2 * n - 1) {
        m <<= 1;
    }

    // 分配设备内存
    int *d_a, *d_b, *d_ab;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, m * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, m * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ab, m * sizeof(int)));
    
    // 初始化设备数据
    CHECK_CUDA_ERROR(cudaMemset(d_a, 0, m * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_b, 0, m * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice));

    // 计算生成元
    int generator = 3;
    int root = mod_pow(generator, (p-1)/m, p);
    int inv_root = mod_inv(root, p);
    int inv_m = mod_inv(m, p);

    // 执行NTT
    ntt(d_a, m, root, p);
    ntt(d_b, m, root, p);

    // 执行点乘操作
    dim3 blockSize(1024);
    dim3 gridSize((m + blockSize.x - 1) / blockSize.x);
    polynomialMultiplyKernel<<<gridSize, blockSize>>>(d_a, d_b, d_ab, m, p);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 执行逆NTT
    ntt(d_ab, m, inv_root, p);
    
    // 缩放结果
    scaleResultKernel<<<gridSize, blockSize>>>(d_ab, m, inv_m, p);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 复制结果回主机
    CHECK_CUDA_ERROR(cudaMemcpy(ab, d_ab, m * sizeof(int), cudaMemcpyDeviceToHost));

    // 清理设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ab);
    
    // 边界检查和修正
    for (int i = 0; i < m; ++i) {
        if (ab[i] < 0 || ab[i] >= p) {
            ab[i] = (ab[i] % p + p) % p;
        }
    }
}