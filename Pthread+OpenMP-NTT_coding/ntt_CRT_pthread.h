#ifndef NTT_CRT_PTHREAD_H
#define NTT_CRT_PTHREAD_H

#include <vector>
#include <pthread.h>
#include <algorithm>
#include <iostream>
#include <cstring>

// 定义多模数NTT所需的小模数集合
// 这些模数都满足形如 a * 2^k + 1 的形式，且原根为3
const __int128 MOD1 = 998244353;      // 2^23 * 119 + 1
const __int128 MOD2 = 754974721;      // 2^24 * 45 + 1
const __int128 MOD3 = 167772161;      // 2^25 * 5 + 1
const __int128 MOD4 = 469762049;      // 2^26 * 7 + 1
const __int128 MOD5 = 1004535809;     // 2^21 * 479 + 1

// 原根均为3
const int PRIMITIVE_ROOT = 3;

// 线程数量
const int NUM_THREADS = 8;

// 最大多项式长度
const int MAXN = 1 << 18;

// NTT层参数结构体
struct NTTLayerParams {
    int* a;                // 数组指针
    int len;               // 当前层长度
    __int128 mod;          // 模数
    __int128 wlen;         // 单位根
    int start;             // 起始位置
    int num_blocks;        // 块数
    int block_size;        // 块大小
};

// 多项式乘法线程参数结构体
struct PolyMultiplyParams {
    int* a;                // 多项式A系数数组
    int* b;                // 多项式B系数数组
    int* result;           // 结果数组
    int n;                 // 多项式长度
    __int128 mod;          // 模数
    int thread_id;         // 线程ID
};

// CRT合并线程参数结构体
struct CRTParams {
    int** results;         // 各模数下的结果数组
    int* final_result;     // 最终结果数组
    int start;             // 起始位置
    int end;               // 结束位置
    std::vector<__int128> moduli; // 模数数组
    __int128 p;            // 最终取模的模数
};

// 快速幂计算 (a^b) % mod
__int128 mod_pow(__int128 a, __int128 b, __int128 mod) {
    __int128 result = 1;
    a %= mod;
    while (b > 0) {
        if (b & 1) result = (result * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return result;
}

// 扩展欧几里得算法求模逆元
void extended_gcd(__int128 a, __int128 b, __int128& x, __int128& y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return;
    }
    __int128 x1, y1;
    extended_gcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
}

__int128 mod_inverse(__int128 a, __int128 mod) {
    __int128 x, y;
    extended_gcd(a, mod, x, y);
    return (x % mod + mod) % mod;
}

// 中国剩余定理合并结果
__int128 crt(const std::vector<__int128>& remainders, const std::vector<__int128>& moduli) {
    __int128 result = 0;
    __int128 M = 1;
    
    // 计算所有模数的乘积
    for (const auto& mod : moduli) {
        M *= mod;
    }
    
    // 应用中国剩余定理
    for (size_t i = 0; i < remainders.size(); ++i) {
        __int128 Mi = M / moduli[i];
        __int128 Mi_inv = mod_inverse(Mi, moduli[i]);
        result = (result + (remainders[i] * Mi % M) * Mi_inv % M) % M;
    }
    
    return result;
}

// NTT层线程函数
void* ntt_layer_thread(void* arg) {
    NTTLayerParams* params = (NTTLayerParams*)arg;
    int* a = params->a;
    int len = params->len;
    __int128 mod = params->mod;
    __int128 wlen = params->wlen;
    int start = params->start;
    int num_blocks = params->num_blocks;
    int block_size = params->block_size;
    
    // 每个线程处理指定数量的蝶形运算块
    for (int b = 0; b < num_blocks; ++b) {
        int i = start + b * block_size;
        __int128 w = 1;
        for (int j = 0; j < len / 2; ++j) {
            int l = i + j;
            int r = i + j + len / 2;
            __int128 u = a[l];
            __int128 v = (a[r] * w) % mod;
            a[l] = (u + v) % mod;
            a[r] = (u - v + mod) % mod;
            w = (w * wlen) % mod;
        }
    }
    
    return nullptr;
}

// 并行NTT变换
void parallel_ntt(int* a, int n, __int128 mod, bool inverse) {
    // 位逆序置换
    for (int i = 0, j = 0; i < n; ++i) {
        if (i < j) std::swap(a[i], a[j]);
        for (int k = n >> 1; (j ^= k) < k; k >>= 1);
    }
    
    // 并行蝶形运算
    for (int len = 2; len <= n; len <<= 1) {
        __int128 wlen = mod_pow(PRIMITIVE_ROOT, (mod - 1) / len, mod);
        if (inverse) wlen = mod_inverse(wlen, mod);
        
        pthread_t threads[NUM_THREADS];
        NTTLayerParams params[NUM_THREADS];
        
        // 计算每个线程的任务
        int num_blocks = n / len; // 总蝶形运算块数
        int blocks_per_thread = (num_blocks + NUM_THREADS - 1) / NUM_THREADS; // 每线程块数
        int block_size = len; // 每块大小
        
        for (int t = 0; t < NUM_THREADS; ++t) {
            int start_block = t * blocks_per_thread;
            int thread_blocks = std::min(blocks_per_thread, num_blocks - start_block);
            
            if (thread_blocks <= 0) {
                params[t] = {a, len, mod, wlen, 0, 0, block_size};
            } else {
                params[t] = {a, len, mod, wlen, start_block * len, thread_blocks, block_size};
            }
            
            pthread_create(&threads[t], nullptr, ntt_layer_thread, &params[t]);
        }
        
        // 等待所有线程完成当前层
        for (int t = 0; t < NUM_THREADS; ++t) {
            pthread_join(threads[t], nullptr);
        }
    }
    
    // 如果是逆变换，需要乘以n的逆元
    if (inverse) {
        __int128 n_inv = mod_inverse(n, mod);
        for (int i = 0; i < n; ++i) {
            a[i] = (a[i] * n_inv) % mod;
        }
    }
}

// 单模数下的多项式乘法线程函数
void* thread_poly_multiply(void* arg) {
    PolyMultiplyParams* params = (PolyMultiplyParams*)arg;
    int* a = params->a;
    int* b = params->b;
    int* result = params->result;
    int n = params->n;
    __int128 mod = params->mod;
    
    int len = 1;
    while (len < 2 * n) len <<= 1;
    
    // 创建临时数组
    int* ta = new int[len]();
    int* tb = new int[len]();
    
    // 复制输入数据
    for (int i = 0; i < n; ++i) {
        ta[i] = a[i] % mod;
        tb[i] = b[i] % mod;
    }
    
    // 前向NTT
    parallel_ntt(ta, len, mod, false);
    parallel_ntt(tb, len, mod, false);
    
    // 点值乘法
    for (int i = 0; i < len; ++i) {
        ta[i] = (__int128)ta[i] * tb[i] % mod;
    }
    
    // 逆NTT
    parallel_ntt(ta, len, mod, true);
    
    // 复制结果
    for (int i = 0; i < 2 * n - 1; ++i) {
        result[i] = ta[i];
    }
    
    // 释放内存
    delete[] ta;
    delete[] tb;
    
    return nullptr;
}

// CRT合并线程函数
void* thread_crt_merge(void* arg) {
    CRTParams* params = (CRTParams*)arg;
    int** results = params->results;
    int* final_result = params->final_result;
    int start = params->start;
    int end = params->end;
    std::vector<__int128> moduli = params->moduli;
    __int128 p = params->p;
    
    for (int i = start; i < end; ++i) {
        std::vector<__int128> remainders;
        for (size_t j = 0; j < moduli.size(); ++j) {
            remainders.push_back(results[j][i]);
        }
        
        // 应用CRT合并
        __int128 crt_result = crt(remainders, moduli);
        
        // 如果需要对输入模数取模
        if (p != 0) {
            crt_result %= p;
        }
        
        // 存储最终结果
        final_result[i] = crt_result;
    }
    
    return nullptr;
}

// 使用CRT合并的多模数NTT多项式乘法（多线程优化版本）
void poly_multiply(int* a, int* b, int* result, int n, __int128 p) {
    // 确定使用的模数数量（根据输入模数大小决定）
    int num_mods = 3; // 默认使用3个模数
    
    // 如果输入模数超过32位，增加使用的模数数量
    if (p > ((__int128)1 << 32)) {
        num_mods = 5;
    }
    
    // 创建模数数组
    std::vector<__int128> moduli = {MOD1, MOD2, MOD3, MOD4, MOD5};
    moduli.resize(num_mods);
    
    // 创建线程和参数
    pthread_t mod_threads[num_mods];
    PolyMultiplyParams mod_params[num_mods];
    
    // 为每个模数分配结果数组
    int* results[5];
    for (int i = 0; i < num_mods; ++i) {
        results[i] = new int[2 * n]();
    }
    
    // 创建线程执行不同模数下的NTT
    for (int i = 0; i < num_mods; ++i) {
        mod_params[i] = {a, b, results[i], n, moduli[i], i};
        pthread_create(&mod_threads[i], nullptr, thread_poly_multiply, &mod_params[i]);
    }
    
    // 等待所有模数线程完成
    for (int i = 0; i < num_mods; ++i) {
        pthread_join(mod_threads[i], nullptr);
    }
    
    // 并行CRT合并
    int result_size = 2 * n - 1;
    int chunk_size = (result_size + NUM_THREADS - 1) / NUM_THREADS;
    
    pthread_t crt_threads[NUM_THREADS];
    CRTParams crt_params[NUM_THREADS];
    
    for (int t = 0; t < NUM_THREADS; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, result_size);
        
        if (start >= result_size) {
            crt_params[t] = {results, result, 0, 0, moduli, p};
        } else {
            crt_params[t] = {results, result, start, end, moduli, p};
        }
        
        pthread_create(&crt_threads[t], nullptr, thread_crt_merge, &crt_params[t]);
    }
    
    // 等待所有CRT合并线程完成
    for (int t = 0; t < NUM_THREADS; ++t) {
        pthread_join(crt_threads[t], nullptr);
    }
    
    // 释放内存
    for (int i = 0; i < num_mods; ++i) {
        delete[] results[i];
    }
}

#endif // NTT_CRT_PTHREAD_H
