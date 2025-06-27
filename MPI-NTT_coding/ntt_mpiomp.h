#ifndef NTT_MPI_H
#define NTT_MPI_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <omp.h> // 添加 OpenMP 头文件

// 模幂运算（保持不变）
inline int mod_pow(int base, int exp, int mod) {
    long long res = 1, b = base % mod;
    while (exp) {
        if (exp & 1) res = res * b % mod;
        b = b * b % mod;
        exp >>= 1;
    }
    return res < 0 ? res + mod : res;
}

// 简化的位反转置换
inline void bit_reverse(std::vector<int>& a) {
    const int n = a.size();
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

// 并行化的NTT实现
void ntt(std::vector<int>& a, bool invert, int root, int mod) {
    const int n = a.size();
    bit_reverse(a);

    #pragma omp parallel default(none) shared(a, n, invert, root, mod)
    {
        for (int len = 2; len <= n; len <<= 1) {
            const int half = len >> 1;
            const int wlen = invert ? 
                mod_pow(mod_pow(root, mod - 2, mod), (mod - 1) / len, mod) :
                mod_pow(root, (mod - 1) / len, mod);
            
            #pragma omp for schedule(static)
            for (int i = 0; i < n; i += len) {
                int w = 1;
                for (int j = 0; j < half; ++j) {
                    const int u = a[i + j];
                    const int v = (int)((1LL * a[i + j + half] * w) % mod);
                    a[i + j] = (u + v) % mod;
                    a[i + j + half] = (u - v + mod) % mod;
                    w = (int)((1LL * w * wlen) % mod);
                }
            }
            #pragma omp barrier
        }

        if (invert) {
            const int inv_n = mod_pow(n, mod - 2, mod);
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                a[i] = (int)((1LL * a[i] * inv_n) % mod);
            }
        }
    }
}

// MPI 并行 NTT（保持简化为完整 NTT）
void ntt_mpi(std::vector<int>& a, bool invert, int root, int mod, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 确保所有进程都有完整的数据
    MPI_Bcast(a.data(), a.size(), MPI_INT, 0, comm);

    // 每个进程执行完整 NTT（已包含 OpenMP 并行）
    ntt(a, invert, root, mod);
}

// 优化多项式乘法
void poly_multiply(int *a, int *b, int *ab, int n, int p, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int lim = 1;
    while (lim < 2 * n - 1) lim <<= 1;

    // 预分配空间
    std::vector<int> A(lim), B(lim);

    // 根进程初始化数据
    if (rank == 0) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            A[i] = a[i];
            B[i] = b[i];
        }
    }

    // 广播输入数据
    MPI_Bcast(A.data(), lim, MPI_INT, 0, comm);
    MPI_Bcast(B.data(), lim, MPI_INT, 0, comm);

    // 选择合适的原根
    const int root = (p == 7340033) ? 5 : 3;

    // 并行NTT变换
    ntt(A, false, root, p);
    ntt(B, false, root, p);

    // MPI分块点值乘法
    const int block_size = (lim + size - 1) / size;
    const int start = rank * block_size;
    const int end = std::min(start + block_size, lim);
    
    std::vector<int> local_C(lim, 0);
    #pragma omp parallel for schedule(static)
    for (int i = start; i < end; i++) {
        local_C[i] = (int)((1LL * A[i] * B[i]) % p);
    }

    // 归约结果
    std::vector<int> C(lim);
    MPI_Allreduce(local_C.data(), C.data(), lim, MPI_INT, MPI_SUM, comm);

    // 逆变换
    ntt(C, true, root, p);

    // 收集结果
    if (rank == 0) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < 2 * n - 1; i++) {
            ab[i] = C[i];
        }
    }
}

#endif // NTT_MPI_H