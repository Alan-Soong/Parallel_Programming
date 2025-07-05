#ifndef NTT_CRT_OPENMP_H
#define NTT_CRT_OPENMP_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <omp.h>

// 定义多模数NTT所需的小模数集合
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
static std::vector<std::vector<int64_t>> roots_cache(5, std::vector<int64_t>(MAXN + 1));
static std::vector<std::vector<int64_t>> inv_roots_cache(5, std::vector<int64_t>(MAXN + 1));
static bool roots_initialized = false;

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
int64_t mod_inverse(int64_t a, int64_t mod) {
    int64_t t = 0, newt = 1;
    int64_t r = mod, newr = a;
    
    while (newr != 0) {
        int64_t quotient = r / newr;
        int64_t tmp_t = t;
        t = newt;
        newt = tmp_t - quotient * newt;
        
        int64_t tmp_r = r;
        r = newr;
        newr = tmp_r - quotient * newr;
    }
    
    if (r > 1) return -1; // 无逆元
    if (t < 0) t += mod;
    return t;
}

// 预计算单位根
void init_roots_cache() {
    if (roots_initialized) return;
    roots_initialized = true;
    static const int64_t moduli[] = {MOD1, MOD2, MOD3, MOD4, MOD5};
    static const int roots[] = {PRIMITIVE_ROOT1, PRIMITIVE_ROOT2, PRIMITIVE_ROOT3, 
                               PRIMITIVE_ROOT4, PRIMITIVE_ROOT5};
    
    #pragma omp parallel for schedule(static)
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

// 位反转置换
void bit_reverse(int* a, int n) {
    int logn = 0;
    while ((1 << logn) < n) logn++;
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int j = 0;
        for (int k = 0; k < logn; k++) {
            j |= ((i >> k) & 1) << (logn - 1 - k);
        }
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
}

// 单模数下的NTT变换
void ntt(int* a, int n, int64_t mod, int mod_idx, bool inverse) {
    bit_reverse(a, n);
    
    // 蝶形运算
    for (int len = 2; len <= n; len <<= 1) {
        int64_t wlen = inverse ? inv_roots_cache[mod_idx][len] : roots_cache[mod_idx][len];
        int half_len = len / 2;
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i += len) {
            int64_t w = 1;
            for (int j = 0; j < half_len; j++) {
                int idx1 = i + j;
                int idx2 = i + j + half_len;
                int64_t u = a[idx1];
                int64_t v = (int64_t)a[idx2] * w % mod;
                
                a[idx1] = (u + v) % mod;
                a[idx2] = (u - v + mod) % mod;
                w = w * wlen % mod;
            }
        }
    }
    
    // 逆变换处理
    if (inverse) {
        int64_t n_inv = mod_inverse(n, mod);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            a[i] = (int64_t)a[i] * n_inv % mod;
        }
    }
}

// 单模数下的多项式乘法
void poly_multiply_single_mod(const int* a, const int* b, int* result, int n, 
                             int64_t mod, int mod_idx, int* ta, int* tb) {
    int len = 1;
    while (len < 2 * n) len <<= 1;
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < len; i++) {
        ta[i] = (i < n) ? a[i] % mod : 0;
        tb[i] = (i < n) ? b[i] % mod : 0;
    }
    
    ntt(ta, len, mod, mod_idx, false);
    ntt(tb, len, mod, mod_idx, false);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < len; i++) {
        ta[i] = (int64_t)ta[i] * tb[i] % mod;
    }
    
    ntt(ta, len, mod, mod_idx, true);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < 2 * n - 1; i++) {
        result[i] = ta[i];
    }
}

// 使用CRT合并的多模数NTT多项式乘法
void poly_multiply(int* a, int* b, int* result, int n, int64_t p) {
    init_roots_cache(); // 初始化单位根表
    
    int num_mods = (p > (1LL << 32)) ? 5 : 3;
    static const int64_t all_moduli[] = {MOD1, MOD2, MOD3, MOD4, MOD5};
    
    // 设置OpenMP线程数
    omp_set_num_threads(NUM_THREADS);
    
    // 为每个模数分配缓冲区
    std::vector<std::vector<int>> ta_buffers(num_mods, std::vector<int>(MAXN * 2));
    std::vector<std::vector<int>> tb_buffers(num_mods, std::vector<int>(MAXN * 2));
    std::vector<std::vector<int>> result_buffers(num_mods, std::vector<int>(MAXN * 2));
    
    // 第一阶段：并行计算各模数下的NTT（使用OpenMP并行）
    #pragma omp parallel for schedule(static)
    for (int mod_idx = 0; mod_idx < num_mods; mod_idx++) {
        int64_t mod = all_moduli[mod_idx];
        poly_multiply_single_mod(a, b, result_buffers[mod_idx].data(), n, 
                               mod, mod_idx,
                               ta_buffers[mod_idx].data(),
                               tb_buffers[mod_idx].data());
    }
    
    // 预计算CRT常量
    __int128 M = 1;
    for (int i = 0; i < num_mods; i++) {
        M *= all_moduli[i];
    }
    
    std::vector<__int128> Mi(num_mods);
    std::vector<__int128> Mi_inv(num_mods);
    for (int i = 0; i < num_mods; i++) {
        Mi[i] = M / all_moduli[i];
        Mi_inv[i] = mod_inverse(Mi[i], all_moduli[i]);
    }

    const int total_coeffs = 2 * n - 1;
    
    // 第二阶段：并行CRT合并（使用OpenMP并行）
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total_coeffs; idx++) {
        __int128 x = 0;
        for (int j = 0; j < num_mods; j++) {
            __int128 term = (__int128)result_buffers[j][idx] * Mi[j] % M;
            term = term * Mi_inv[j] % M;
            x = (x + term) % M;
        }
        
        // 调整到[-M/2, M/2)区间
        if (x >= M/2) x -= M;
        
        // 对目标模数p取模
        x %= p;
        if (x < 0) x += p;
        
        result[idx] = (int)x;
    }
}

#endif // NTT_CRT_OPENMP_H