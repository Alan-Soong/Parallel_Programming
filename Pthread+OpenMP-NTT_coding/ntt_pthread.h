// #include <pthread.h>
// #include <vector>

// const int MAXN = 1 << 18;
// __int128 A[MAXN], B[MAXN];
// int THREADS = 8;

// struct NTTParams {
//     __int128 *a;
//     int len; // 当前层的蝶形运算长度
//     int p;   // 模数
//     __int128 wlen; // 单位根
//     int start;     // 起始索引
//     int num_blocks; // 该线程处理的块数
//     int block_size; // 每块的大小
// };

// __int128 mod_pow(__int128 base, __int128 exp, __int128 mod) {
//     __int128 res = 1;
//     base %= mod;
//     while (exp > 0) {
//         if (exp & 1) res = res * base % mod;
//         base = base * base % mod;
//         exp >>= 1;
//     }
//     return res;
// }

// __int128 mod_inv(__int128 x, __int128 p) {
//     return mod_pow(x, p - 2, p);
// }

// void* ntt_layer_thread(void* arg) {
//     NTTParams* params = (NTTParams*)arg;
//     __int128* a = params->a;
//     int len = params->len;
//     int p = params->p;
//     __int128 wlen = params->wlen;
//     int start = params->start;
//     int num_blocks = params->num_blocks;
//     int block_size = params->block_size;

//     // 每个线程处理 num_blocks 个蝶形运算单元
//     for (int b = 0; b < num_blocks; ++b) {
//         int i = start + b * block_size; // 每个块的起始索引
//         __int128 w = 1;
//         for (int j = 0; j < len / 2; ++j) {
//             int l = i + j;
//             int r = i + j + len / 2;
//             __int128 u = a[l], v = a[r] * w % p;
//             a[l] = (u + v) % p;
//             a[r] = (u - v + p) % p;
//             w = w * wlen % p;
//         }
//     }
//     return nullptr;
// }

// void ntt(__int128 *a, int n, int p, int g, bool invert) {
//     // Bit-reversal
//     for (int i = 1, j = 0; i < n; ++i) {
//         int bit = n >> 1;
//         for (; j & bit; bit >>= 1) j ^= bit;
//         j ^= bit;
//         if (i < j) std::swap(a[i], a[j]);
//     }

//     // 逐层进行蝶形运算
//     for (int len = 2; len <= n; len <<= 1) {
//         __int128 wlen = mod_pow(g, (p - 1) / len, p);
//         if (invert) wlen = mod_inv(wlen, p);

//         pthread_t threads[THREADS];
//         NTTParams params[THREADS];

//         // 计算每个线程的任务
//         int num_blocks = n / len; // 总共的蝶形运算单元数
//         int blocks_per_thread = (num_blocks + THREADS - 1) / THREADS; // 每个线程的块数（向上取整）
//         int block_size = len; // 每个蝶形运算单元的大小

//         for (int t = 0; t < THREADS; ++t) {
//             int start_block = t * blocks_per_thread;
//             int thread_blocks = std::min(blocks_per_thread, num_blocks - start_block);
//             if (thread_blocks <= 0) {
//                 params[t] = {a, len, p, wlen, 0, 0, block_size};
//             } else {
//                 params[t] = {a, len, p, wlen, start_block * len, thread_blocks, block_size};
//             }
//             pthread_create(&threads[t], nullptr, ntt_layer_thread, &params[t]);
//         }

//         for (int t = 0; t < THREADS; ++t)
//             pthread_join(threads[t], nullptr);
//     }

//     if (invert) {
//         __int128 inv_n = mod_inv(n, p);
//         for (int i = 0; i < n; ++i)
//             a[i] = a[i] * inv_n % p;
//     }
// }

// void poly_multiply(int *a, int *b, int *ab, int n, int p) {
//     int len = 1;
//     while (len < 2 * n) len <<= 1;
//     int g = 3;

//     // 初始化 A 和 B
//     for (int i = 0; i < len; ++i) {
//         A[i] = i < n ? (__int128)a[i] : 0;
//         B[i] = i < n ? (__int128)b[i] : 0;
//     }

//     // 前向 NTT
//     ntt(A, len, p, g, false);
//     ntt(B, len, p, g, false);

//     // 点值乘法
//     for (int i = 0; i < len; ++i)
//         A[i] = A[i] * B[i] % p;

//     // 逆 NTT
//     ntt(A, len, p, g, true);

//     // 输出结果
//     for (int i = 0; i < 2 * n - 1; ++i)
//         ab[i] = (int)(A[i] % p);
// }

#include <pthread.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <thread>

const int MAXN = 1 << 18;
int THREADS = std::min(8, (int)std::thread::hardware_concurrency());

struct NTTParams {
    __int128 *a;
    int len; // Length of butterfly operation
    int p;   // Modulus
    __int128 wlen; // Root of unity
    int start;     // Starting index
    int num_blocks; // Number of blocks for this thread
    int block_size; // Size of each block
};

// Barrett Reduction for faster modulo
struct Barrett {
    __int128 m; // Modulus
    __int128 k; // Precomputed for reduction
    Barrett(__int128 mod) : m(mod), k(((__int128)1 << 64) / mod) {}
    __int128 reduce(__int128 x) const {
        __int128 q = (x * k) >> 64;
        __int128 r = x - q * m;
        return r >= m ? r - m : r < 0 ? r + m : r;
    }
};

__int128 mod_pow(__int128 base, __int128 exp, __int128 mod, const Barrett& br) {
    __int128 res = 1;
    base = br.reduce(base);
    while (exp > 0) {
        if (exp & 1) res = br.reduce(res * base);
        base = br.reduce(base * base);
        exp >>= 1;
    }
    return res;
}

__int128 mod_inv(__int128 x, __int128 p, const Barrett& br) {
    return mod_pow(x, p - 2, p, br);
}

void* ntt_layer_thread(void* arg) {
    NTTParams* params = (NTTParams*)arg;
    __int128* a = params->a;
    int len = params->len;
    int p = params->p;
    __int128 wlen = params->wlen;
    int start = params->start;
    int num_blocks = params->num_blocks;
    int block_size = params->block_size;
    Barrett br(p);

    for (int b = 0; b < num_blocks; ++b) {
        int i = start + b * block_size;
        __int128 w = 1;
        for (int j = 0; j < len / 2; ++j) {
            int l = i + j;
            int r = i + j + len / 2;
            __int128 u = a[l], v = br.reduce(a[r] * w);
            a[l] = br.reduce(u + v);
            a[r] = br.reduce(u - v + p);
            w = br.reduce(w * wlen);
        }
    }
    return nullptr;
}

void ntt(__int128 *a, int n, int p, int g, bool invert, const std::vector<int>& rev) {
    if (n <= 0 || (n & (n - 1)) != 0) return; // Ensure n is power of 2
    Barrett br(p);

    // Bit-reversal using precomputed table
    for (int i = 1; i < n; ++i) {
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    }

    // Precompute roots of unity
    std::vector<__int128> roots;
    for (int len = 2; len <= n; len <<= 1) {
        __int128 wlen = mod_pow(g, (p - 1) / len, p, br);
        if (invert) wlen = mod_inv(wlen, p, br);
        roots.push_back(wlen);
    }

    // Process each layer
    for (int l = 0, len = 2; len <= n; len <<= 1, ++l) {
        int num_blocks = n / len;
        int blocks_per_thread = (num_blocks + THREADS - 1) / THREADS;
        int block_size = len;

        pthread_t threads[THREADS];
        NTTParams params[THREADS];
        int active_threads = 0;

        // Create threads for this layer
        for (int t = 0; t < THREADS; ++t) {
            int start_block = t * blocks_per_thread;
            int thread_blocks = std::min(blocks_per_thread, num_blocks - start_block);
            params[t] = {a, len, p, roots[l], start_block * len, thread_blocks, block_size};
            if (thread_blocks > 0) {
                pthread_create(&threads[t], nullptr, ntt_layer_thread, &params[t]);
                active_threads++;
            }
        }

        // Join threads for this layer
        for (int t = 0; t < active_threads; ++t) {
            pthread_join(threads[t], nullptr);
        }
    }

    if (invert) {
        __int128 inv_n = mod_inv(n, p, br);
        for (int i = 0; i < n; ++i) {
            a[i] = br.reduce(a[i] * inv_n);
        }
    }
}

void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    if (n <= 0 || p <= 0) return; // Basic input validation
    int len = 1;
    while (len < 2 * n) len <<= 1;
    int g = 3;

    // Dynamic allocation
    std::vector<__int128> A(len), B(len);

    // Precompute bit-reversal
    std::vector<int> rev(len);
    for (int i = 1, j = 0; i < len; ++i) {
        int bit = len >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        rev[i] = j;
    }

    // Initialize A and B
    for (int i = 0; i < n; ++i) {
        A[i] = a[i];
        B[i] = b[i];
    }

    // Forward NTT
    ntt(A.data(), len, p, g, false, rev);
    ntt(B.data(), len, p, g, false, rev);

    // Pointwise multiplication
    Barrett br(p);
    for (int i = 0; i < len; ++i) {
        A[i] = br.reduce(A[i] * B[i]);
    }

    // Inverse NTT
    ntt(A.data(), len, p, g, true, rev);

    // Output result
    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = (int)(A[i] % p);
    }
}