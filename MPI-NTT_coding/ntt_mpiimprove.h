// #ifndef NTT_MPI_H
// #define NTT_MPI_H

// #include <vector>
// #include <algorithm>
// #include <iostream>
// #include <mpi.h>

// // 模幂运算
// inline int mod_pow(int base, int exp, int mod) {
//     long long res = 1, b = base % mod;
//     while (exp) {
//         if (exp & 1) res = res * b % mod;
//         b = b * b % mod;
//         exp >>= 1;
//     }
//     return res < 0 ? res + mod : res;
// }

// // 位反转置换
// inline void bit_reverse(std::vector<int>& a) {
//     int n = a.size();
//     int j = 0;
//     for (int i = 1; i < n; ++i) {
//         int bit = n >> 1;
//         while (j & bit) {
//             j ^= bit;
//             bit >>= 1;
//         }
//         j ^= bit;
//         if (i < j) std::swap(a[i], a[j]);
//     }
// }

// // 获取原根
// inline int get_primitive_root(int p) {
//     return 3; // 对 p=7340033 有效
// }

// // 串行 NTT
// void ntt(std::vector<int>& a, bool invert, int root, int mod) {
//     int n = a.size();
//     bit_reverse(a);
//     for (int len = 2; len <= n; len <<= 1) {
//         int wlen = mod_pow(root, (mod - 1) / len, mod);
//         if (invert) wlen = mod_pow(wlen, mod - 2, mod);
//         for (int i = 0; i < n; i += len) {
//             int w = 1;
//             for (int j = 0; j < len / 2; ++j) {
//                 int u = a[i + j];
//                 int v = (int)((1LL * a[i + j + len / 2] * w) % mod);
//                 a[i + j] = (u + v) % mod;
//                 a[i + j + len / 2] = (u - v + mod) % mod;
//                 w = (int)((1LL * w * wlen) % mod);
//             }
//         }
//     }
//     if (invert) {
//         int inv_n = mod_pow(n, mod - 2, mod);
//         for (int& x : a) x = (int)((1LL * x * inv_n) % mod);
//     }
// }

// // 并行多项式乘法 (修复版)
// void poly_multiply(int *a, int *b, int *ab, int n, int p, MPI_Comm comm) {
//     int rank, size;
//     MPI_Comm_rank(comm, &rank);
//     MPI_Comm_size(comm, &size);

//     // 计算扩展长度
//     int lim = 1;
//     while (lim < 2 * n - 1) lim <<= 1;

//     // 主进程初始化数据
//     std::vector<int> A(lim), B(lim);
//     if (rank == 0) {
//         for (int i = 0; i < n; ++i) {
//             A[i] = a[i] % p;
//             B[i] = b[i] % p;
//         }
//     }

//     // 广播数据
//     MPI_Bcast(A.data(), lim, MPI_INT, 0, comm);
//     MPI_Bcast(B.data(), lim, MPI_INT, 0, comm);

//     // 所有进程执行完整NTT（关键修复：保持数据完整性）
//     int root = get_primitive_root(p);
//     ntt(A, false, root, p);
//     ntt(B, false, root, p);

//     // 点值乘法（并行分块）
//     int chunk = (lim + size - 1) / size;
//     int start = rank * chunk;
//     int end = std::min(start + chunk, lim);
//     int local_size = end - start;
    
//     std::vector<int> local_C(lim, 0); // 全零初始化
    
//     for (int i = start; i < end; ++i) {
//         local_C[i] = static_cast<int>(1LL * A[i] * B[i] % p);
//     }

//     // 归约求和
//     std::vector<int> C(lim, 0);
//     MPI_Reduce(local_C.data(), C.data(), lim, MPI_INT, MPI_SUM, 0, comm);

//     // 主进程执行逆变换
//     if (rank == 0) {
//         ntt(C, true, root, p);
//         for (int i = 0; i < 2 * n - 1; ++i) {
//             ab[i] = C[i] % p;
//         }
//     }
// }

// #endif // NTT_MPI_H

#ifndef NTT_MPI_H
#define NTT_MPI_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <mpi.h>

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

// 串行 NTT (优化版)
void ntt(std::vector<long long>& a, bool invert, long long root, long long mod) {
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
                long long v = (a[i + j + len/2] * w) % mod;
                a[i + j] = (u + v) % mod;
                a[i + j + len/2] = (u - v + mod) % mod;
                w = (w * wlen) % mod;
            }
        }
    }
    
    if (invert) {
        long long inv_n = mod_pow(n, mod - 2, mod);
        for (int i = 0; i < n; ++i) {
            a[i] = (a[i] * inv_n) % mod;
        }
    }
}

// 并行多项式乘法 (修复版)
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

    // 主进程初始化数据
    std::vector<long long> A(padded_lim, 0);
    std::vector<long long> B(padded_lim, 0);
    
    if (rank == 0) {
        // 确保n大于0
        if (n <= 0) {
            std::cerr << "Error: n must be positive" << std::endl;
            MPI_Abort(comm, 1);
        }
        
        for (int i = 0; i < n; ++i) {
            A[i] = a[i] % p;
            B[i] = b[i] % p;
        }
    }

    // 广播数据大小信息
    MPI_Bcast(&padded_lim, 1, MPI_INT, 0, comm);
    
    // 调整向量大小
    if (rank != 0) {
        A.resize(padded_lim);
        B.resize(padded_lim);
    }
    
    // 广播数据
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
    
    ntt(A, false, root_val, p);
    ntt(B, false, root_val, p);

    // 点乘计算 (并行分块)
    int chunk_size = padded_lim / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? padded_lim : start + chunk_size;
    
    // 确保结束位置不超过向量大小
    end = std::min(end, padded_lim);
    
    // 每个进程初始化局部结果向量
    std::vector<long long> local_C(padded_lim, 0);
    
    // 计算点乘
    for (int i = start; i < end; ++i) {
        local_C[i] = (A[i] * B[i]) % p;
    }

    // 归约求和
    std::vector<long long> C(padded_lim, 0);
    MPI_Reduce(local_C.data(), C.data(), padded_lim, MPI_LONG_LONG, MPI_SUM, 0, comm);

    // 主进程执行逆变换
    if (rank == 0) {
        ntt(C, true, root_val, p);
        
        // 输出结果 (只取有效部分)
        int result_size = 2 * n - 1;
        for (int i = 0; i < result_size; ++i) {
            ab[i] = static_cast<int>(C[i] % p);
            if (ab[i] < 0) ab[i] += p;
        }
    }
}

#endif // NTT_MPI_H