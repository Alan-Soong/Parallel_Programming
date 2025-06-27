#ifndef NTT_MPI_H
#define NTT_MPI_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <mpi.h>

// 巴雷特规约函数 (内联)
inline long long barrett_reduce(long long x, long long p, long long mu, int k) {
    if (p == 1) return 0; // 模1时结果为0
    long long q1 = x >> (k - 1);
    long long q = (q1 * mu) >> (k + 1);
    long long r = x - q * p;
    if (r < 0) r += p;
    if (r >= p) r -= p;
    return r;
}

// 模幂运算 (优化版)
inline long long mod_pow(long long base, long long exp, long long mod) {
    long long res = 1;
    base %= mod;
    while (exp) {
        if (exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return res < 0 ? res + mod : res;
}

// 位反转置换
inline void bit_reverse(std::vector<long long>& a) {
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

// 串行 NTT (使用巴雷特规约优化)
void ntt(std::vector<long long>& a, bool invert, long long root, long long mod, long long mu, int k) {
    int n = a.size();
    if (n == 0) return;  // 防止空向量
    
    bit_reverse(a);
    
    for (int len = 2; len <= n; len <<= 1) {
        long long wlen = mod_pow(root, (mod - 1) / len, mod);
        if (invert) wlen = mod_pow(wlen, mod - 2, mod);
        
        for (int i = 0; i < n; i += len) {
            long long w = 1;
            for (int j = 0; j < len / 2; ++j) {
                long long u = a[i + j];
                long long v = barrett_reduce(a[i + j + len/2] * w, mod, mu, k);
                a[i + j] = u + v;
                if (a[i + j] >= mod) a[i + j] -= mod;
                a[i + j + len/2] = u - v + mod;
                if (a[i + j + len/2] >= mod) a[i + j + len/2] -= mod;
                w = barrett_reduce(w * wlen, mod, mu, k);
            }
        }
    }
    
    if (invert) {
        long long inv_n = mod_pow(n, mod - 2, mod);
        for (int i = 0; i < n; ++i) {
            a[i] = barrett_reduce(a[i] * inv_n, mod, mu, k);
        }
    }
}

// 并行多项式乘法 (使用巴雷特规约优化)
void poly_multiply(int *a, int *b, int *ab, int n, int p, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 计算扩展长度
    int lim = 1;
    while (lim < 2 * n - 1) lim <<= 1;
    
    // 确保lim是size的倍数 (减少负载不均衡)
    int padded_lim = lim;
    if (lim % size != 0) {
        padded_lim = ((lim + size - 1) / size) * size;
    }

    // 预计算巴雷特规约参数 (k 和 mu)
    int k = 0;
    long long mu = 0;
    if (rank == 0) {
        if (p > 1) {
            long long temp = p;
            while (temp) {
                k++;
                temp >>= 1;
            }
            long long divisor = 1LL << (2 * k);
            mu = divisor / p;
        } else if (p == 1) {
            k = 1; // 避免除以零
        } else {
            std::cerr << "Error: modulus p must be positive" << std::endl;
            MPI_Abort(comm, 1);
        }
    }
    MPI_Bcast(&k, 1, MPI_INT, 0, comm);
    MPI_Bcast(&mu, 1, MPI_LONG_LONG, 0, comm);

    // 主进程初始化数据
    std::vector<long long> A(padded_lim, 0);
    std::vector<long long> B(padded_lim, 0);
    
    if (rank == 0) {
        if (n <= 0) {
            std::cerr << "Error: n must be positive" << std::endl;
            MPI_Abort(comm, 1);
        }
        for (int i = 0; i < n; ++i) {
            A[i] = a[i] % p;
            B[i] = b[i] % p;
        }
    }

    // 广播数据
    MPI_Bcast(&padded_lim, 1, MPI_INT, 0, comm);
    if (rank != 0) {
        A.resize(padded_lim);
        B.resize(padded_lim);
    }
    MPI_Bcast(A.data(), padded_lim, MPI_LONG_LONG, 0, comm);
    MPI_Bcast(B.data(), padded_lim, MPI_LONG_LONG, 0, comm);

    // 所有进程执行完整NTT
    long long root_val = 3; // 对于给定的模数p，3是有效的原根
    
    // 检查向量大小
    if (A.size() != padded_lim || B.size() != padded_lim) {
        std::cerr << "Rank " << rank << ": Vector size mismatch! A=" 
                  << A.size() << ", B=" << B.size() 
                  << ", padded_lim=" << padded_lim << std::endl;
        MPI_Abort(comm, 1);
    }
    
    ntt(A, false, root_val, p, mu, k);
    ntt(B, false, root_val, p, mu, k);

    // 点乘计算 (并行分块，使用巴雷特规约)
    int chunk_size = padded_lim / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? padded_lim : start + chunk_size;
    end = std::min(end, padded_lim);
    
    std::vector<long long> local_C(padded_lim, 0);
    for (int i = start; i < end; ++i) {
        local_C[i] = barrett_reduce(A[i] * B[i], p, mu, k);
    }

    // 归约求和
    std::vector<long long> C(padded_lim, 0);
    MPI_Reduce(local_C.data(), C.data(), padded_lim, MPI_LONG_LONG, MPI_SUM, 0, comm);

    // 主进程执行逆变换
    if (rank == 0) {
        ntt(C, true, root_val, p, mu, k);
        
        // 输出结果 (只取有效部分)
        int result_size = 2 * n - 1;
        for (int i = 0; i < result_size; ++i) {
            ab[i] = static_cast<int>(C[i] % p);
            if (ab[i] < 0) ab[i] += p;
        }
    }
}

#endif // NTT_MPI_H