#include <pthread.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>
using namespace std;

using i64 = long long;
using i128 = __int128;

const int MAXN = 1 << 20;
int rev[MAXN];

// 快速幂
i128 qpow(i128 a, i128 b, i128 mod) {
    i128 res = 1;
    while (b) {
        if (b & 1) res = res * a % mod;
        a = a * a % mod;
        b >>= 1;
    }
    return res;
}

// 反转位序
void bit_reverse(i128 *a, int n) {
    for (int i = 0; i < n; ++i)
        if (i < rev[i])
            swap(a[i], a[rev[i]]);
}

// NTT 实现
void ntt(i128 *a, int n, bool invert, i128 mod, i128 root) {
    bit_reverse(a, n);
    for (int len = 2; len <= n; len <<= 1) {
        i128 wlen = qpow(root, (mod - 1) / len, mod);
        if (invert) wlen = qpow(wlen, mod - 2, mod);
        for (int i = 0; i < n; i += len) {
            i128 w = 1;
            for (int j = 0; j < len / 2; ++j) {
                i128 u = a[i + j];
                i128 v = a[i + j + len / 2] * w % mod;
                a[i + j] = (u + v) % mod;
                a[i + j + len / 2] = (u - v + mod) % mod;
                w = w * wlen % mod;
            }
        }
    }
    if (invert) {
        i128 inv_n = qpow(n, mod - 2, mod);
        for (int i = 0; i < n; ++i)
            a[i] = a[i] * inv_n % mod;
    }
}

// 主函数，接口不变
void poly_multiply(int *a, int *b, int *ab, int n, __int128 p) {
    i128 root;
    if (p == 7340033) root = 3;
    else if (p == 104857601) root = 3;
    else if (p == 469762049) root = 3;
    // else {
    //     cerr << "Unsupported modulus.\n";
    //     exit(1);
    // }
    else if (p = 1337006139375617) root = 3;

    int size = 1, logn = 0;
    while (size < 2 * n) size <<= 1, ++logn;
    for (int i = 0; i < size; ++i)
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (logn - 1));

    static i128 A[MAXN], B[MAXN];
    for (int i = 0; i < size; ++i) {
        A[i] = i < n ? a[i] : 0;
        B[i] = i < n ? b[i] : 0;
    }

    ntt(A, size, false, p, root);
    ntt(B, size, false, p, root);

    for (int i = 0; i < size; ++i)
        A[i] = A[i] * B[i] % p;

    ntt(A, size, true, p, root);

    for (int i = 0; i < 2 * n - 1; ++i)
        ab[i] = (int)(A[i] % p);
}