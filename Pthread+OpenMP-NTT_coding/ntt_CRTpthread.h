#ifndef NTT_CRT_PTHREAD_H
#define NTT_CRT_PTHREAD_H

#include <vector>
#include <pthread.h>
#include <algorithm>
#include <iostream>
#include <cstring>

// 定义多模数NTT所需的小模数集合
// 这些模数都满足形如 a * 2^k + 1 的形式
const __int128 MOD1 = 998244353;      // 2^23 * 119 + 1
const __int128 MOD2 = 754974721;      // 2^24 * 45 + 1
const __int128 MOD3 = 167772161;      // 2^25 * 5 + 1
const __int128 MOD4 = 469762049;      // 2^26 * 7 + 1
const __int128 MOD5 = 1004535809;     // 2^21 * 479 + 1

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

struct ThreadParams {
    const int* a;
    const int* b;
    int* result;
    int n;
    __int128 mod;
    int primitive_root;
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
__int128 crt(const std::vector<__int128>& remainders, const std::vector<__int128>& moduli, __int128 target_mod) {
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
        
        // 注意这里的计算顺序，避免中间结果溢出
        __int128 term = remainders[i];
        term = (term * Mi) % M;
        term = (term * Mi_inv) % M;
        result = (result + term) % M;
    }
    
    // 如果结果需要对目标模数取模
    if (target_mod > 0) {
        // 处理结果可能超过目标模数一半的情况
        if (result >= M / 2) {
            result = result - M;
        }
        
        // 对目标模数取模，处理负数情况
        result = ((result % target_mod) + target_mod) % target_mod;
    }
    
    return result;
}

// 单模数下的NTT变换
void ntt(int* a, int n, __int128 mod, int pri_root, bool inverse) {
    // 位逆序置换
    for (int i = 0, j = 0; i < n; ++i) {
        if (i < j) std::swap(a[i], a[j]);
        for (int k = n >> 1; (j ^= k) < k; k >>= 1);
    }
    
    // 蝶形运算
    for (int len = 2; len <= n; len <<= 1) {
        __int128 wlen = mod_pow(pri_root, (mod - 1) / len, mod);
        if (inverse) wlen = mod_inverse(wlen, mod);
        
        for (int i = 0; i < n; i += len) {
            __int128 w = 1;
            for (int j = 0; j < len / 2; ++j) {
                __int128 u = a[i + j];
                __int128 v = (a[i + j + len / 2] * w) % mod;
                a[i + j] = (u + v) % mod;
                a[i + j + len / 2] = (u - v + mod) % mod;
                w = (w * wlen) % mod;
            }
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

// 单模数下的多项式乘法
void poly_multiply_single_mod(int* a, int* b, int* result, int n, __int128 mod, int primitive_root) {
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
    ntt(ta, len, mod, primitive_root, false);
    ntt(tb, len, mod, primitive_root, false);
    
    // 点值乘法
    for (int i = 0; i < len; ++i) {
        ta[i] = (__int128)ta[i] * tb[i] % mod;
    }
    
    // 逆NTT
    ntt(ta, len, mod, primitive_root, true);
    
    // 复制结果
    for (int i = 0; i < 2 * n - 1; ++i) {
        result[i] = ta[i];
    }
    
    // 释放内存
    delete[] ta;
    delete[] tb;
}

// 线程函数：在特定模数下执行多项式乘法
void* thread_poly_multiply(void* arg) {
    ThreadParams* params = reinterpret_cast<ThreadParams*>(arg);
    poly_multiply_single_mod(const_cast<int*>(params->a), const_cast<int*>(params->b),
                             params->result, params->n, params->mod, params->primitive_root);
    return nullptr;
}


// 使用CRT合并的多模数NTT多项式乘法
void poly_multiply(int* a, int* b, int* result, int n, __int128 p) {
    int num_mods = (p > ((__int128)1 << 32)) ? 5 : 3;

    static const __int128 all_moduli[] = {MOD1, MOD2, MOD3, MOD4, MOD5};
    static const int all_roots[] = {PRIMITIVE_ROOT1, PRIMITIVE_ROOT2, PRIMITIVE_ROOT3, PRIMITIVE_ROOT4, PRIMITIVE_ROOT5};

    std::vector<__int128> moduli(all_moduli, all_moduli + num_mods);
    std::vector<int> primitive_roots(all_roots, all_roots + num_mods);

    pthread_t threads[5];
    ThreadParams params[5];

    // 为每个模数准备一个结果数组
    std::vector<int*> results(num_mods);
    for (int i = 0; i < num_mods; ++i) {
        results[i] = new int[2 * n]();
        params[i] = {a, b, results[i], n, moduli[i], primitive_roots[i]};
        pthread_create(&threads[i], nullptr, thread_poly_multiply, &params[i]);
    }

    // 等待线程完成
    for (int i = 0; i < num_mods; ++i) {
        pthread_join(threads[i], nullptr);
    }

    // 合并结果
    for (int i = 0; i < 2 * n - 1; ++i) {
        std::vector<__int128> remainders(num_mods);
        for (int j = 0; j < num_mods; ++j) {
            remainders[j] = results[j][i];
        }
        __int128 crt_result = crt(remainders, moduli, p);
        result[i] = crt_result;
    }

    // 清理内存
    for (int i = 0; i < num_mods; ++i) {
        delete[] results[i];
    }
}


#endif // NTT_CRT_PTHREAD_H
