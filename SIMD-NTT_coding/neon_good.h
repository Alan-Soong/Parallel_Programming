#pragma once
#include <cstdint>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <arm_neon.h>
#include <iostream>

// 模幂运算
static inline int mod_pow(int base, int exp, int mod) {
    int result = 1;
    while (exp) {
        if (exp & 1) result = (int)((int64_t)result * base % mod);
        base = (int)((int64_t)base * base % mod);
        exp >>= 1;
    }
    return result;
}

int mod_inv(int x, int p) {
    return mod_pow(x, p - 2, p);
}

// Montgomery 类实现
struct Montgomery {
    using u64 = uint64_t;
    using u128 = __uint128_t;

    int mod;
    u64 inv_factor;      // -mod^{-1} mod 2^64
    u64 r2;       // 2^64 % mod

    Montgomery(int p) : mod(p) {
        assert((p & 1) && p < (1LL << 31)); // 必须是奇数且不超过 2^31
        inv_factor = mont_inv(p);
        r2 = -u64(p) % p;
        r2 = (u64(r2) << 32) % p;
        r2 = (u64(r2) << 32) % p;
    }

    u64 to(u64 x) const {
        return reduce(u128(x) * r2);
    }

    int from(u64 x) const {
        return reduce(x);
    }

    u64 reduce(u128 x) const {
        u64 q = (u64)x * inv_factor;
        u128 t = (x + u128(q) * mod) >> 64;
        return t >= mod ? t - mod : t;
    }

    int mul(int a, int b) const {
        return reduce(u128(a) * b);
    }

    int pow(int x, int e) const {
        u64 res = to(1);
        u64 base = to(x);
        while (e) {
            if (e & 1) res = reduce(u128(res) * base);
            base = reduce(u128(base) * base);
            e >>= 1;
        }
        return res;
    }

    int inv(int x) const {
        return pow(x, mod - 2);
    }

private:
    static u64 mont_inv(u64 p) {
        u64 res = p;
        for (int i = 0; i < 6; ++i) {
            res *= 2 - p * res;
        }
        return -res;
    }
};

// NEON优化的模约减
static inline int32x4_t mod_reduce(int64x2_t v0, int64x2_t v1, int p) {
    int32x4_t mod = vdupq_n_s32(p);
    int32x4_t t = { (int32_t)(v0[0] % p), (int32_t)(v0[1] % p),
                     (int32_t)(v1[0] % p), (int32_t)(v1[1] % p) };
    t = vbslq_s32(vcltq_s32(t, vdupq_n_s32(0)), vaddq_s32(t, mod), t);
    return t;
}

// 单层NEON优化蝶形变换（处理4个点）
void butterfly_layer_simd(int *a, int n, int len, int wlen, int p) {
    int *twiddles = (int*)aligned_alloc(16, (len / 2) * sizeof(int));
    if (!twiddles) {
        fprintf(stderr, "内存分配失败\n");
        return;
    }
    twiddles[0] = 1;
    for (int j = 1; j < len / 2; ++j) {
        twiddles[j] = (int)((int64_t)twiddles[j - 1] * wlen % p);
    }

    for (int i = 0; i < n; i += len) {
        for (int j = 0; j < len / 2; j += 4) {
            int base = i + j;
            int offset = base + len / 2;

            if (j + 3 < len / 2) {
                int32x4_t u = vld1q_s32(&a[base]);
                int32x4_t v = vld1q_s32(&a[offset]);
                int32x4_t w = vld1q_s32(&twiddles[j]);

                int64x2_t v0 = vmull_s32(vget_low_s32(v), vget_low_s32(w));
                int64x2_t v1 = vmull_s32(vget_high_s32(v), vget_high_s32(w));
                int32x4_t vmod = mod_reduce(v0, v1, p);

                int32x4_t sum = vaddq_s32(u, vmod);
                int32x4_t diff = vsubq_s32(u, vmod);

                int32x4_t mod = vdupq_n_s32(p);
                sum = vbslq_s32(vcgeq_s32(sum, mod), vsubq_s32(sum, mod), sum);
                diff = vbslq_s32(vcltq_s32(diff, vdupq_n_s32(0)), vaddq_s32(diff, mod), diff);

                vst1q_s32(&a[base], sum);
                vst1q_s32(&a[offset], diff);
            } else {
                for (int k = j; k < len / 2; ++k) {
                    int u = a[i + k];
                    int v = (int)((int64_t)a[i + k + len / 2] * twiddles[k] % p);
                    a[i + k] = (u + v) % p;
                    a[i + k + len / 2] = (u - v + p) % p;
                }
                break;
            }
        }
    }
    free(twiddles);
}

// SIMD优化的NTT
void ntt_simd(int *a, int n, int w, int p) {
    if (n <= 0 || (n & (n - 1)) != 0) {
        fprintf(stderr, "错误：n必须为2的幂\n");
        return;
    }

    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) j ^= bit, bit >>= 1;
        j ^= bit;
        if (i < j) {
            int tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        int wlen = mod_pow(w, (p - 1) / len, p);
        if (len / 2 >= 4) {
            butterfly_layer_simd(a, n, len, wlen, p);
        } else {
            for (int i = 0; i < n; i += len) {
                int w_now = 1;
                for (int j = 0; j < len / 2; ++j) {
                    int u = a[i + j];
                    int v = (int)((int64_t)a[i + j + len / 2] * w_now % p);
                    a[i + j] = (u + v) % p;
                    a[i + j + len / 2] = (u - v + p) % p;
                    w_now = (int)((int64_t)w_now * wlen % p);
                }
            }
        }
    }
}

// 多项式乘法（使用NTT）
void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    if (n <= 0) {
        fprintf(stderr, "错误：n必须为正整数\n");
        return;
    }

    int m = 1;
    while (m < 2 * n - 1) m <<= 1;

    int *ta = (int*)aligned_alloc(16, m * sizeof(int));
    int *tb = (int*)aligned_alloc(16, m * sizeof(int));
    if (!ta || !tb) {
        fprintf(stderr, "内存分配失败\n");
        free(ta);
        free(tb);
        return;
    }
    memset(ta, 0, m * sizeof(int));
    memset(tb, 0, m * sizeof(int));
    memcpy(ta, a, n * sizeof(int));
    memcpy(tb, b, n * sizeof(int));

    int root = 3;
    int inv_root = mod_inv(root, p);

    ntt_simd(ta, m, root, p);
    ntt_simd(tb, m, root, p);

    for (int i = 0; i < m; ++i) {
        ab[i] = (int)((int64_t)ta[i] * tb[i] % p);
    }

    int inv_m = mod_inv(m, p);
    for (int i = 0; i < m; ++i) {
        ab[i] = (int)((int64_t)ab[i] * inv_m % p);
    }
    ntt_simd(ab, m, inv_root, p);

    free(ta);
    free(tb);
}
