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

// 快速取模的巴雷特归约
struct Barrett {
    __int128 m; // 模数
    __int128 k; // 规约的预计算
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
    if (n <= 0 || (n & (n - 1)) != 0) return; // 确保2的幂次
    Barrett br(p);

    // 预计算表的位反转
    for (int i = 1; i < n; ++i) {
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    }

    // 单位根的预计算
    std::vector<__int128> roots;
    for (int len = 2; len <= n; len <<= 1) {
        __int128 wlen = mod_pow(g, (p - 1) / len, p, br);
        if (invert) wlen = mod_inv(wlen, p, br);
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

        // 对这一层创造线程
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
    if (n <= 0 || p <= 0) return;   // 验证有效
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
    Barrett br(p);
    for (int i = 0; i < len; ++i) {
        A[i] = br.reduce(A[i] * B[i]);
    }

    // 逆NTT
    ntt(A.data(), len, p, g, true, rev);

    // 输出结果
    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = (int)(A[i] % p);
    }
}