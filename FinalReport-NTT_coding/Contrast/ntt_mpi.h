#ifndef NTT_MPI_H
#define NTT_MPI_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <mpi.h>

// 模幂运算
inline int mod_pow(int base, int exp, int mod) {
    long long res = 1, b = base % mod;
    while (exp) {
        if (exp & 1) res = res * b % mod;
        b = b * b % mod;
        exp >>= 1;
    }
    return res < 0 ? res + mod : res;
}

// 位反转置换
inline void bit_reverse(std::vector<int>& a) {
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

// 获取原根
inline int get_primitive_root(int p) {
    return 3; // 对 p=7340033 有效
}

// 串行 NTT
void ntt(std::vector<int>& a, bool invert, int root, int mod) {
    int n = a.size();
    bit_reverse(a);
    for (int len = 2; len <= n; len <<= 1) {
        int wlen = mod_pow(root, (mod - 1) / len, mod);
        if (invert) wlen = mod_pow(wlen, mod - 2, mod);
        for (int i = 0; i < n; i += len) {
            int w = 1;
            for (int j = 0; j < len / 2; ++j) {
                int u = a[i + j];
                int v = (int)((1LL * a[i + j + len / 2] * w) % mod);
                a[i + j] = (u + v) % mod;
                a[i + j + len / 2] = (u - v + mod) % mod;
                w = (int)((1LL * w * wlen) % mod);
            }
        }
    }
    if (invert) {
        int inv_n = mod_pow(n, mod - 2, mod);
        for (int& x : a) x = (int)((1LL * x * inv_n) % mod);
    }
}

// MPI 并行 NTT（简化版：每个进程执行完整 NTT）
void ntt_mpi(std::vector<int>& a, bool invert, int root, int mod, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 确保所有进程都有完整的数据
    MPI_Bcast(a.data(), a.size(), MPI_INT, 0, comm);

    // 每个进程执行完整 NTT
    ntt(a, invert, root, mod);
}

// 并行多项式乘法
void poly_multiply(int *a, int *b, int *ab, int n, int p, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 步骤 1：扩展到最小的2的幂
    int lim = 1;
    while (lim < 2 * n - 1) lim <<= 1;

    // 步骤 2：初始化数组
    std::vector<int> A(lim), B(lim), C(lim);
    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            A[i] = a[i] % p;
            B[i] = b[i] % p;
        }
        for (int i = n; i < lim; ++i) A[i] = B[i] = 0;
    }

    // 步骤 3：广播 A 和 B
    MPI_Bcast(A.data(), lim, MPI_INT, 0, comm);
    MPI_Bcast(B.data(), lim, MPI_INT, 0, comm);

    // 步骤 4：执行 NTT
    int root = get_primitive_root(p);
    ntt_mpi(A, false, root, p, comm);
    ntt_mpi(B, false, root, p, comm);

    // 步骤 5：点值乘法
    int chunk = (lim + size - 1) / size; // 上取整
    int start = rank * chunk;
    int end = std::min(start + chunk, lim);
    std::vector<int> local_C(chunk, 0);
    for (int i = start; i < end; ++i) {
        local_C[i - start] = (int)((1LL * A[i] * B[i]) % p);
    }

    // 步骤 6：收集点值乘法结果
    std::vector<int> recv_counts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
        int i_start = i * chunk;
        int i_end = std::min(i_start + chunk, lim);
        recv_counts[i] = i_end - i_start;
        displs[i] = i_start;
    }
    MPI_Allgatherv(local_C.data(), end - start, MPI_INT, C.data(), recv_counts.data(), displs.data(), MPI_INT, comm);

    // 步骤 7：逆 NTT
    ntt_mpi(C, true, root, p, comm);

    // 步骤 8：主进程输出结果
    if (rank == 0) {
        for (int i = 0; i < 2 * n - 1; ++i) {
            ab[i] = C[i] % p;
        }
    }
}

#endif // NTT_MPI_H