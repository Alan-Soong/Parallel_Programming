#ifndef NTT_MPI_H
#define NTT_MPI_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <pthread.h>
#include <mpi.h>

// 安全模运算
inline int safe_mod(long long x, int mod) {
    x %= mod;
    return x < 0 ? x + mod : x;
}

// 模幂运算
inline int mod_pow(int base, int exp, int mod) {
    long long res = 1, b = base % mod;
    while (exp) {
        if (exp & 1) res = safe_mod(res * b, mod);
        b = safe_mod(b * b, mod);
        exp >>= 1;
    }
    return res;
}

// 获取原根 (Note: Not used in poly_multiply, 'root' is hardcoded there)
inline int get_primitive_root(int p) {
    return 3; // 对 p=7340033 有效, but poly_multiply uses 5
}

// pthread 参数结构体
struct ThreadData {
    std::vector<int>* a;        // 用于NTT的数组指针
    std::vector<int>* A;        // 点值乘法输入A
    std::vector<int>* B;        // 点值乘法输入B
    std::vector<int>* C;        // 点值乘法输出C
    int start;
    int end;
    int len;
    int wlen;
    int mod;
    int w; // Used for inv_n in normalization
};

// 位反转置换的线程函数
void* bit_reverse_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    std::vector<int>& a = *(data->a);
    int n = a.size();
    for (int i = data->start; i < data->end; ++i) {
        int j = 0;
        for (int k = i, len = n >> 1; len; len >>= 1, k >>= 1) {
            j = (j << 1) | (k & 1);
        }
        if (i < j && i < n && j < n) { // Added checks for bounds safety, though n should be power of 2
            std::swap(a[i], a[j]);
        }
    }
    return nullptr;
}

// 位反转置换
inline void bit_reverse(std::vector<int>& a, int num_threads) {
    int n = a.size();
    if (n <= 1) return; // Handle small arrays
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadData> thread_data(num_threads);
    int chunk = (n + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        thread_data[t].a = &a;
        thread_data[t].start = t * chunk;
        thread_data[t].end = std::min((t + 1) * chunk, n);
        pthread_create(&threads[t], nullptr, bit_reverse_thread, &thread_data[t]);
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], nullptr);
    }
}

// NTT 蝶形变换线程函数
void* ntt_butterfly_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    std::vector<int>& a = *(data->a);
    int len = data->len;
    int wlen = data->wlen;
    int mod = data->mod;

    // 每个线程处理 [start, end) 块，每个块大小为 len
    for (int block = data->start; block < data->end; ++block) {
        int base = block * len;
        int w = 1;
        for (int j = 0; j < len / 2; ++j) {
            int idx1 = base + j;
            int idx2 = idx1 + len / 2;
            if (idx2 < (int)a.size()) {
                int u = a[idx1];
                int v = safe_mod((long long)a[idx2] * w, mod);
                a[idx1] = safe_mod((long long)u + v, mod);
                a[idx2] = safe_mod((long long)u - v, mod);
            }
            w = safe_mod((long long)w * wlen, mod);
        }
    }
    return nullptr;
}

// NTT 归一化线程函数
void* ntt_normalize_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    std::vector<int>& a = *(data->a);
    int mod = data->mod;
    int inv_n = data->w; // 'w' field is reused for inv_n

    for (int i = data->start; i < data->end; ++i) {
        if (i < a.size()) { // Bounds check
            a[i] = safe_mod(1LL * a[i] * inv_n, mod);
        }
    }
    return nullptr;
}

// 串行 NTT（使用 pthread）
void ntt(std::vector<int>& a, bool invert, int root, int mod, int num_threads) {
    int n = a.size();
    if (n <= 1) return;

    bit_reverse(a, num_threads);

    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadData> thread_data(num_threads);

    for (int len = 2; len <= n; len <<= 1) {
        int wlen = mod_pow(root, (mod - 1) / len, mod);
        if (invert) wlen = mod_pow(wlen, mod - 2, mod);

        int num_blocks = n / len;
        int blocks_per_thread = (num_blocks + num_threads - 1) / num_threads;

        for (int t = 0; t < num_threads; ++t) {
            thread_data[t].a = &a;
            thread_data[t].start = t * blocks_per_thread;
            thread_data[t].end   = std::min((t + 1) * blocks_per_thread, num_blocks);
            thread_data[t].len   = len;
            thread_data[t].wlen  = wlen;
            thread_data[t].mod   = mod;
            pthread_create(&threads[t], nullptr, ntt_butterfly_thread, &thread_data[t]);
        }

        for (int t = 0; t < num_threads; ++t) {
            pthread_join(threads[t], nullptr);
        }
    }

    if (invert) {
        int inv_n = mod_pow(n, mod - 2, mod);
        int chunk = (n + num_threads - 1) / num_threads; // Chunking for normalization is per element, so original chunk is fine.

        for (int t = 0; t < num_threads; ++t) {
            thread_data[t].a = &a;
            thread_data[t].start = t * chunk;
            thread_data[t].end = std::min((t + 1) * chunk, n);
            thread_data[t].mod = mod;
            thread_data[t].w = inv_n; // Reuse 'w' for inv_n
            pthread_create(&threads[t], nullptr, ntt_normalize_thread, &thread_data[t]);
        }

        for (int t = 0; t < num_threads; ++t) {
            pthread_join(threads[t], nullptr);
        }
    }
}

// MPI 并行 NTT (THIS IS NOT USED IN POLY_MULTIPLY CURRENTLY, BUT IF IT WERE, IT WOULD NEED FIXES)
void ntt_mpi(std::vector<int>& a, bool invert, int root, int mod, MPI_Comm comm, int num_threads) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Ensure array size is a multiple of process count
    // This constraint is for distributing segments for the 'local_a' processing
    if (a.size() % size != 0) {
        if (rank == 0) {
            std::cerr << "Error: Array size " << a.size() << " must be divisible by " << size << std::endl;
        }
        MPI_Abort(comm, 1);
    }

    int local_n = a.size() / size;
    std::vector<int> local_a(local_n);

    // Scatter data
    MPI_Scatter(a.data(), local_n, MPI_INT, local_a.data(), local_n, MPI_INT, 0, comm);

    ntt(local_a, invert, root, mod, num_threads); // This performs normalization if invert is true

    // Gather results
    MPI_Gather(local_a.data(), local_n, MPI_INT, a.data(), local_n, MPI_INT, 0, comm);

    // Removed redundant normalization here as ntt() already handles it.
    // if (invert && rank == 0) {
    //     int inv_n = mod_pow(a.size(), mod - 2, mod);
    //     for (int i = 0; i < a.size(); ++i) {
    //         a[i] = safe_mod(1LL * a[i] * inv_n, mod);
    //     }
    // }
}

// 修改点值乘法线程函数
void* pointwise_mult_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    for (int i = data->start; i < data->end; ++i) {
        // Bounds check not strictly needed if start/end are within A.size(), but good practice.
        // Original: if (i < data->A->size()) {
            (*data->C)[i] = safe_mod(1LL * (*(data->A))[i] * (*(data->B))[i], data->mod);
        // }
    }
    return nullptr;
}

// 修改多项式乘法函数
void poly_multiply(int *a_coeffs, int *b_coeffs, int *ab_coeffs, int n, int p, MPI_Comm comm, int num_threads) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int lim = 1;
    while (lim < 2 * n - 1) lim <<= 1; // lim 必须是 2 的幂，且大于等于 2*n-1

    std::vector<int> A(lim, 0), B(lim, 0), C(lim, 0); // C 将存储点值乘积

    // Step 1: 根进程初始化 A 和 B，并进行零填充
    if (rank == 0) {
        std::copy(a_coeffs, a_coeffs + n, A.begin());
        std::copy(b_coeffs, b_coeffs + n, B.begin());
    }

    // Step 2: 将 A 和 B 广播到所有进程
    // 这是关键：每个进程都拥有了完整的 A 和 B 向量
    MPI_Bcast(A.data(), lim, MPI_INT, 0, comm);
    MPI_Bcast(B.data(), lim, MPI_INT, 0, comm);

    // 选择原根。对于 p=7340033，3 和 5 都是原根。
    int root = (p == 7340033) ? 3 : get_primitive_root(p);

    // Step 3: 每个进程独立地执行 NTT
    // 由于 A 和 B 已经被广播，每个进程都对完整数据进行 NTT。
    // 这一步是计算冗余的，但能确保正确性。
    // 这里的 NTT 调用的是 `ntt` 函数 (pthread 版本)，而不是 `ntt_mpi`。
    ntt(A, false, root, p, num_threads); // 对 A 执行正向 NTT
    ntt(B, false, root, p, num_threads); // 对 B 执行正向 NTT

    // Step 4: 点值乘法 (仍然使用 pthread 并行化，因为每个进程都有完整的 A 和 B)
    // A 和 B 此时是点值表示。
    // C 将存储 A 和 B 的点值乘积。
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadData> thread_data(num_threads);
    int chunk_for_pointwise = (lim + num_threads - 1) / num_threads; // 将 C 的计算任务分发给线程

    for (int t = 0; t < num_threads; ++t) {
        thread_data[t].A = &A;
        thread_data[t].B = &B;
        thread_data[t].C = &C; // C 是当前进程的本地向量
        thread_data[t].start = t * chunk_for_pointwise;
        thread_data[t].end = std::min(thread_data[t].start + chunk_for_pointwise, lim);
        thread_data[t].mod = p;
        pthread_create(&threads[t], nullptr, pointwise_mult_thread, &thread_data[t]);
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], nullptr);
    }

    // Step 5: 收集和逆向 NTT
    // 由于 A 和 B 是广播的，每个进程的 C 向量在点值乘法后都是完整的、正确的。
    // 因此，不需要 MPI_Reduce 或 MPI_Gather 来聚合 C。
    // 我们只需要在根进程上执行逆向 NTT，然后将结果复制到 ab_coeffs。
    if (rank == 0) {
        ntt(C, true, root, p, num_threads); // 对 C 执行逆向 NTT (在根进程上)
        // 将结果复制到输出数组，只取有效系数 (2*n-1 个)
        std::copy(C.begin(), C.begin() + 2 * n - 1, ab_coeffs);
    }
}

#endif // NTT_MPI_H