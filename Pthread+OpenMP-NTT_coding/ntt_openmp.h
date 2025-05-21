#include <cstring>
#include <omp.h>

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
    // Bit-reversal permutation（顺序执行，避免复杂同步）
    for (int i = 0, j = 0; i < n; ++i) {
        if (i < j) {
            int t = a[i]; a[i] = a[j]; a[j] = t;
        }
        for (int k = n >> 1; (j ^= k) < k; k >>= 1);
    }

    // Butterfly
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1;
        int wlen = mod_pow(w, (p - 1) / len, p);
        #pragma omp parallel for
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

    // 初始化输入
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        ta[i] = a[i];
        tb[i] = b[i];
    }

    int root = 3; // 原根，依赖具体模数
    int inv_root = mod_pow(root, p - 2, p);

    // 执行NTT
    ntt(ta, m, root, p);
    ntt(tb, m, root, p);

    // 点值乘法
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        ta[i] = (long long)ta[i] * tb[i] % p;
    }

    // 逆NTT
    ntt(ta, m, inv_root, p);

    // 归一化
    int inv_m = mod_pow(m, p - 2, p);
    #pragma omp parallel for
    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = (long long)ta[i] * inv_m % p;
    }

    free(ta);
    free(tb);
}