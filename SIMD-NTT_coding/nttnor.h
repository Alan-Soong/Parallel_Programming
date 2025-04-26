#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
// #include <sys/time.h>
// #include <omp.h>

// 快速幂求模逆
int mod_pow(int base, int exp, int mod) {
    int result = 1;
    while (exp) {
        if (exp & 1) result = (long long)result * base % mod;
        base = (long long)base * base % mod;
        exp >>= 1;
    }
    return result;
}

// NTT 变换（DIT Cooley-Tukey）
void ntt(int *a, int n, int w, int p) {
    // Bit-reversal permutation
    for (int i = 0, j = 0; i < n; ++i) {
        if (i < j) {
            int t = a[i]; a[i] = a[j]; a[j] = t;
        }
        for (int k = n >> 1; (j ^= k) < k; k >>= 1);
    }

    // Butterfly
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1;
        int wlen = mod_pow(w, (p - 1) / len, p);  // 正确计算该层单位根
        for (int i = 0; i < n; i += len) {
            int w_now = 1;
            for (int j = 0; j < half; ++j) {
                int u = a[i + j];
                int v = (long long)a[i + j + half] * w_now % p;
                a[i + j] = (u + v) % p;
                a[i + j + half] = (u - v + p) % p;
                w_now = (long long)w_now * wlen % p;
            }
        }
    }
}

// 多项式乘法
void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    int m = 1;
    while (m < 2 * n - 1) m <<= 1;

    int *ta = (int*)calloc(m, sizeof(int));
    int *tb = (int*)calloc(m, sizeof(int));
    memcpy(ta, a, n * sizeof(int));
    memcpy(tb, b, n * sizeof(int));

    int root = 3; // 原根，依赖具体模数
    int inv_root = mod_pow(root, p - 2, p);

    ntt(ta, m, root, p);
    ntt(tb, m, root, p);

    for (int i = 0; i < m; ++i)
        ta[i] = (long long)ta[i] * tb[i] % p;

    ntt(ta, m, inv_root, p);

    int inv_m = mod_pow(m, p - 2, p);
    for (int i = 0; i < 2 * n - 1; ++i)
        ab[i] = (long long)ta[i] * inv_m % p;

    free(ta);
    free(tb);
}