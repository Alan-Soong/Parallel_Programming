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
    int len; // 蝴蝶变换的长度
    int p;   // 模数
    __int128 wlen; // 单位根
    int start;     // 起始索引
    int num_blocks; // 该进程的block数
    int block_size; // 每个block的大小
};

// 快速幂取模
__int128 mod_pow(__int128 base, __int128 exp, __int128 mod) {
    __int128 res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return res;
}

// 模逆元
__int128 mod_inv(__int128 x, __int128 p) {
    return mod_pow(x, p - 2, p);
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

    for (int b = 0; b < num_blocks; ++b) {
        int i = start + b * block_size;
        __int128 w = 1;
        for (int j = 0; j < len / 2; ++j) {
            int l = i + j;
            int r = i + j + len / 2;
            __int128 u = a[l], v = (a[r] * w) % p;
            a[l] = (u + v) % p;
            a[r] = (u - v + p) % p;
            w = (w * wlen) % p;
        }
    }
    return nullptr;
}

void ntt(__int128 *a, int n, int p, int g, bool invert, const std::vector<int>& rev) {
    if (n <= 0 || (n & (n - 1)) != 0) return; // 确保2的幂次

    // 位反转
    for (int i = 1; i < n; ++i) {
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    }

    // 预计算单位根
    std::vector<__int128> roots;
    for (int len = 2; len <= n; len <<= 1) {
        __int128 wlen = mod_pow(g, (p - 1) / len, p);
        if (invert) wlen = mod_inv(wlen, p);
        roots.push_back(wlen);
    }

    // 处理每一层
    for (int l = 0, len = 2; len <= n; len <<= 1, ++l) {
        int num_blocks = n / len;
        int blocks_per_thread = (num_blocks + THREADS - 1) / THREADS;
        int block_size = len;

        pthread_t threads[THREADS];
        NTTParams params[THREADS];
        int active_threads = 0;

        // 创建线程
        for (int t = 0; t < THREADS; ++t) {
            int start_block = t * blocks_per_thread;
            int thread_blocks = std::min(blocks_per_thread, num_blocks - start_block);
            params[t] = {a, len, p, roots[l], start_block * len, thread_blocks, block_size};
            if (thread_blocks > 0) {
                pthread_create(&threads[t], nullptr, ntt_layer_thread, &params[t]);
                active_threads++;
            }
        }

        // 等待线程完成
        for (int t = 0; t < active_threads; ++t) {
            pthread_join(threads[t], nullptr);
        }
    }

    if (invert) {
        __int128 inv_n = mod_inv(n, p);
        for (int i = 0; i < n; ++i) {
            a[i] = (a[i] * inv_n) % p;
        }
    }
}

void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    if (n <= 0 || p <= 0) return; // 验证有效
    int len = 1;
    while (len < 2 * n) len <<= 1;
    int g = 3;

    std::vector<__int128> A(len), B(len);

    // 预计算位反转
    std::vector<int> rev(len);
    for (int i = 1, j = 0; i < len; ++i) {
        int bit = len >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        rev[i] = j;
    }

    // 初始化A和B
    for (int i = 0; i < n; ++i) {
        A[i] = a[i];
        B[i] = b[i];
    }

    // 正NTT
    ntt(A.data(), len, p, g, false, rev);
    ntt(B.data(), len, p, g, false, rev);

    // 点乘
    for (int i = 0; i < len; ++i) {
        A[i] = (A[i] * B[i]) % p;
    }

    // 逆NTT
    ntt(A.data(), len, p, g, true, rev);

    // 输出结果
    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = (int)(A[i] % p);
    }
}