#ifndef NTT_CRT_PTHREAD_H
#define NTT_CRT_PTHREAD_H

#include <vector>
#include <pthread.h>
#include <algorithm>
#include <iostream>
#include <cstring>

// 定义多模数NTT所需的小模数集合 (use int64_t where possible)
const int64_t MOD1 = 998244353;
const int64_t MOD2 = 754974721;
const int64_t MOD3 = 167772161;
const int64_t MOD4 = 469762049;
const int64_t MOD5 = 1004535809;

// 对应的原根
const int PRIMITIVE_ROOT1 = 3;
const int PRIMITIVE_ROOT2 = 11;
const int PRIMITIVE_ROOT3 = 3;
const int PRIMITIVE_ROOT4 = 3;
const int PRIMITIVE_ROOT5 = 3;

// 线程数量
const int NUM_THREADS = 8;

// 最大多项式长度
const int MAXN = 1 << 18;

// 预计算的单位根表
static std::vector<std::vector<int64_t>> roots_cache(5, std::vector<int64_t>(MAXN));
static std::vector<std::vector<int64_t>> inv_roots_cache(5, std::vector<int64_t>(MAXN));
static bool roots_initialized = false;

struct ThreadParams {
    const int* a;
    const int* b;
    int* result;
    int n;
    int64_t mod;
    int primitive_root;
    int* ta; // Thread-local buffer
    int* tb; // Thread-local buffer
    int mod_idx; // Index for accessing roots_cache
};

struct CRTThreadParams {
    const std::vector<int*>* results;
    const std::vector<int64_t>* moduli;
    int* final_result;
    __int128 p;
    int start_idx;
    int end_idx;
    int num_mods;
    __int128 M;
    const std::vector<__int128>* Mi;
    const std::vector<__int128>* Mi_inv;
};

// 快速幂计算 (a^b) % mod
int64_t mod_pow(int64_t a, int64_t b, int64_t mod) {
    int64_t result = 1;
    a %= mod;
    while (b > 0) {
        if (b & 1) 
            result = (result * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return result;
}

// 扩展欧几里得算法求模逆元
void extended_gcd(int64_t a, int64_t b, int64_t& x, int64_t& y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return;
    }
    int64_t x1, y1;
    extended_gcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
}

int64_t mod_inverse(int64_t a, int64_t mod) {
    int64_t x, y;
    extended_gcd(a, mod, x, y);
    return (x % mod + mod) % mod;
}

// 预计算វ计算单位根
void init_roots_cache() {
    if (roots_initialized) return;
    roots_initialized = true;
    static const int64_t moduli[] = {MOD1, MOD2, MOD3, MOD4, MOD5};
    static const int roots[] = {PRIMITIVE_ROOT1, PRIMITIVE_ROOT2, PRIMITIVE_ROOT3, 
                               PRIMITIVE_ROOT4, PRIMITIVE_ROOT5};
    
    for (int i = 0; i < 5; ++i) {
        int64_t mod = moduli[i];
        int64_t root = roots[i];
        for (int len = 2; len <= MAXN; len <<= 1) {
            int64_t wlen = mod_pow(root, (mod - 1) / len, mod);
            roots_cache[i][len] = wlen;
            inv_roots_cache[i][len] = mod_inverse(wlen, mod);
        }
    }
}

// 单模数下的NTT变换
void ntt(int* a, int n, int64_t mod, int mod_idx, bool inverse) {
    // 位逆序置换
    for (int i = 0, j = 0; i < n; ++i) {
        if (i < j) std::swap(a[i], a[j]);
        for (int k = n >> 1; (j ^= k) < k; k >>= 1);
    }
    
    // 蝶形运算
    for (int len = 2; len <= n; len <<= 1) {
        int64_t wlen = inverse ? inv_roots_cache[mod_idx][len] : roots_cache[mod_idx][len];
        
        for (int i = 0; i < n; i += len) {
            int64_t w = 1;
            for (int j = 0; j < len / 2; ++j) {
                int64_t u = a[i + j];
                int64_t v = (int64_t)a[i + j + len / 2] * w % mod;
                a[i + j] = (u + v) % mod;
                a[i + j + len / 2] = (u - v + mod) % mod;
                w = (w * wlen) % mod;
            }
        }
    }
    
    // 逆变换处理
    if (inverse) {
        int64_t n_inv = mod_inverse(n, mod);
        for (int i = 0; i < n; ++i) {
            a[i] = (int64_t)a[i] * n_inv % mod;
        }
    }
}

// 单模数下的多项式乘法
void poly_multiply_single_mod(const int* a, const int* b, int* result, int n, 
                             int64_t mod, int primitive_root, int* ta, int* tb, int mod_idx) {
    int len = 1;
    while (len < 2 * n) len <<= 1;
    
    std::fill(ta, ta + len, 0);
    std::fill(tb, tb + len, 0);
    
    for (int i = 0; i < n; ++i) {
        ta[i] = a[i] % mod;
        tb[i] = b[i] % mod;
    }
    
    ntt(ta, len, mod, mod_idx, false);
    ntt(tb, len, mod, mod_idx, false);
    
    for (int i = 0; i < len; ++i) {
        ta[i] = (int64_t)ta[i] * tb[i] % mod;
    }
    
    ntt(ta, len, mod, mod_idx, true);
    
    for (int i = 0; i < 2 * n - 1; ++i) {
        result[i] = ta[i];
    }
}

// 线程函数：在特定模数下执行多项式乘法
void* thread_poly_multiply(void* arg) {
    ThreadParams* params = reinterpret_cast<ThreadParams*>(arg);
    poly_multiply_single_mod(params->a, params->b, params->result, 
                            params->n, params->mod, 
                            params->primitive_root, params->ta, params->tb, params->mod_idx);
    return nullptr;
}

// CRT合并线程函数
void* thread_crt_merge(void* arg) {
    CRTThreadParams* params = reinterpret_cast<CRTThreadParams*>(arg);
    const int num_mods = params->num_mods;
    const std::vector<int*>& results = *(params->results);
    const std::vector<int64_t>& moduli = *(params->moduli);
    const std::vector<__int128>& Mi = *(params->Mi);
    const std::vector<__int128>& Mi_inv = *(params->Mi_inv);
    __int128 M = params->M;
    
    for (int idx = params->start_idx; idx < params->end_idx; ++idx) {
        __int128 x = 0;
        for (int j = 0; j < num_mods; ++j) {
            __int128 term = (__int128)results[j][idx] * Mi[j] % M;
            term = term * Mi_inv[j] % M;
            x = (x + term) % M;
        }
        
        // 调整到[-M/2, M/2)区间
        if (x >= M/2) x -= M;
        
        // 对目标模数p取模
        __int128 p = params->p;
        x %= p;
        if (x < 0) x += p;
        
        params->final_result[idx] = (int)x;
    }
    
    return nullptr;
}

// 使用CRT合并的多模数NTT多项式乘法
void poly_multiply(int* a, int* b, int* result, int n, __int128 p) {
    init_roots_cache(); // 初始化单位根表
    
    int num_mods = (p > ((int64_t)1 << 32)) ? 5 : 3;

    static const int64_t all_moduli[] = {MOD1, MOD2, MOD3, MOD4, MOD5};
    static const int all_roots[] = {PRIMITIVE_ROOT1, PRIMITIVE_ROOT2, 
                                   PRIMITIVE_ROOT3, PRIMITIVE_ROOT4, PRIMITIVE_ROOT5};

    std::vector<int64_t> moduli(all_moduli, all_moduli + num_mods);
    std::vector<int> primitive_roots(all_roots, all_roots + num_mods);

    pthread_t threads[5];
    ThreadParams params[5];
    std::vector<int*> results(num_mods);
    
    // 分配线程本地缓冲区
    std::vector<std::vector<int>> ta_buffers(num_mods, std::vector<int>(2 * MAXN));
    std::vector<std::vector<int>> tb_buffers(num_mods, std::vector<int>(2 * MAXN));
    std::vector<std::vector<int>> result_buffers(num_mods, std::vector<int>(2 * MAXN));

    // 第一阶段：并行计算各模数下的NTT
    for (int i = 0; i < num_mods; ++i) {
        results[i] = result_buffers[i].data();
        params[i] = {a, b, results[i], n, moduli[i], primitive_roots[i], 
                     ta_buffers[i].data(), tb_buffers[i].data(), i};
        pthread_create(&threads[i], nullptr, thread_poly_multiply, &params[i]);
    }

    // 等待所有NTT线程完成
    for (int i = 0; i < num_mods; ++i) {
        pthread_join(threads[i], nullptr);
    }

    // 预计算CRT常量
    __int128 M = 1;
    for (int i = 0; i < num_mods; ++i) {
        M *= moduli[i];
    }
    
    std::vector<__int128> Mi(num_mods);
    std::vector<__int128> Mi_inv(num_mods);
    for (int i = 0; i < num_mods; ++i) {
        Mi[i] = M / moduli[i];
        Mi_inv[i] = mod_inverse(Mi[i], moduli[i]);
    }

    const int total_coeffs = 2 * n - 1;
    const int coeffs_per_thread = (total_coeffs + NUM_THREADS - 1) / NUM_THREADS;
    
    pthread_t crt_threads[NUM_THREADS];
    CRTThreadParams crt_params[NUM_THREADS];

    // 第二阶段：并行CRT合并
    for (int i = 0; i < NUM_THREADS; ++i) {
        int start = i * coeffs_per_thread;
        int end = std::min(start + coeffs_per_thread, total_coeffs);
        
        crt_params[i] = {
            &results,
            &moduli,
            result,
            p,
            start,
            end,
            num_mods,
            M,
            &Mi,
            &Mi_inv
        };
        
        pthread_create(&crt_threads[i], nullptr, thread_crt_merge, &crt_params[i]);
    }

    // 等待所有CRT线程完成
    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(crt_threads[i], nullptr);
    }
}

#endif // NTT_CRT_PTHREAD_H
