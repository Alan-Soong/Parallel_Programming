#ifndef NTT_SERIAL_H
#define NTT_SERIAL_H

#include <vector>
#include <algorithm>
#include <cstdint>
#include <iostream>

// Barrett Reduction
class BarrettReducer {
public:
    int mod;
    uint64_t m;

    explicit BarrettReducer(int p) : mod(p) {
        m = uint64_t((__uint128_t(1) << 64) / p);
    }

    inline int reduce(uint64_t a) const {
        uint64_t q = ((__uint128_t)a * m) >> 64;
        int r = int(a - q * mod);
        return r < mod ? (r < 0 ? r + mod : r) : r - mod;
    }

    inline int mul(int a, int b) const {
        return reduce((uint64_t)a * b);
    }
};

// 快速模幂
inline int mod_pow(int base, int exp, int mod) {
    long long res = 1, b = base % mod;
    while (exp) {
        if (exp & 1) res = res * b % mod;
        b = b * b % mod;
        exp >>= 1;
    }
    return res < 0 ? res + mod : res;
}

// 位反转置换
inline void bit_reverse(std::vector<int>& a) {
    int n = a.size();
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

// 原根（假设是 3）
inline int get_primitive_root(int p) {
    return 3;
}

// NTT
void ntt(std::vector<int>& a, bool invert, int root, int mod) {
    int n = a.size();
    BarrettReducer br(mod);
    bit_reverse(a);
    for (int len = 2; len <= n; len <<= 1) {
        int wlen = mod_pow(root, (mod - 1) / len, mod);
        if (invert) wlen = mod_pow(wlen, mod - 2, mod);
        for (int i = 0; i < n; i += len) {
            int w = 1;
            for (int j = 0; j < len / 2; ++j) {
                int u = a[i + j];
                int v = br.mul(a[i + j + len / 2], w);
                a[i + j] = (u + v < mod) ? (u + v) : (u + v - mod);
                a[i + j + len / 2] = (u - v >= 0) ? (u - v) : (u - v + mod);
                w = br.mul(w, wlen);
            }
        }
    }
    if (invert) {
        int inv_n = mod_pow(n, mod - 2, mod);
        for (int& x : a) x = br.mul(x, inv_n);
    }
}

// 多项式乘法（序列化版本）
void poly_multiply(int* a, int* b, int* ab, int n, int p) {
    int lim = 1;
    while (lim < 2 * n - 1) lim <<= 1;

    std::vector<int> A(lim, 0), B(lim, 0);
    for (int i = 0; i < n; ++i) {
        A[i] = a[i] % p;
        B[i] = b[i] % p;
    }

    int root = get_primitive_root(p);
    ntt(A, false, root, p);
    ntt(B, false, root, p);

    BarrettReducer br(p);
    for (int i = 0; i < lim; ++i)
        A[i] = br.mul(A[i], B[i]);

    ntt(A, true, root, p);
    for (int i = 0; i < 2 * n - 1; ++i)
        ab[i] = A[i];
}

#endif // NTT_SERIAL_H
