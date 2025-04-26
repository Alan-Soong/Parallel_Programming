#include <arm_neon.h>
// #include <omp.h>
#include <stdint.h>
#include <string.h>
#include <cmath>
#include <iostream>


// 模幂运算
int64_t mod_pow(int64_t base, int64_t exp, int64_t mod) {
    int64_t result = 1;
    while (exp) {
        if (exp & 1) result = (int)((int64_t)result * base % mod);
        base = (int64_t)((int64_t)base * base % mod);
        exp >>= 1;
    }
    return result;
}

// 模逆（基于费马小定理）
int64_t mod_inv(int64_t x, int64_t p) {
    return mod_pow(x, p - 2, p);
}

// NEON优化的模约减
static inline int32x4_t mod_reduce(int64x2_t v0, int64x2_t v1, int64_t p) {
    int32x4_t mod = vdupq_n_s32(p);
    int32x4_t t = { (int32_t)(v0[0] % p), (int32_t)(v0[1] % p),
                     (int32_t)(v1[0] % p), (int32_t)(v1[1] % p) };
    t = vbslq_s32(vcltq_s32(t, vdupq_n_s32(0)), vaddq_s32(t, mod), t);
    return t;
}

// 单层NEON优化蝶形变换（处理4个点）
void butterfly_layer_simd(int32_t *a, int n, int len, int64_t wlen, int64_t p) {
    // 使用对齐内存分配twiddle因子
    int32_t *twiddles = (int32_t*)aligned_alloc(16, (len / 2) * sizeof(int));
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
                // SIMD处理4个元素
                int32x4_t u = vld1q_s32(&a[base]);
                int32x4_t v = vld1q_s32(&a[offset]);
                int32x4_t w = vld1q_s32(&twiddles[j]);

                int64x2_t v0 = vmull_s32(vget_low_s32(v), vget_low_s32(w));
                int64x2_t v1 = vmull_s32(vget_high_s32(v), vget_high_s32(w));
                int32x4_t vmod = mod_reduce(v0, v1, p);

                int32x4_t sum = vaddq_s32(u, vmod);
                int32x4_t diff = vsubq_s32(u, vmod);

                // 模约减
                int32x4_t mod = vdupq_n_s32(p);
                sum = vbslq_s32(vcgeq_s32(sum, mod), vsubq_s32(sum, mod), sum);
                diff = vbslq_s32(vcltq_s32(diff, vdupq_n_s32(0)), vaddq_s32(diff, mod), diff);

                vst1q_s32(&a[base], sum);
                vst1q_s32(&a[offset], diff);
            } else {
                // 标量处理剩余元素
                for (int k = j; k < len / 2; ++k) {
                    int64_t u = a[i + k];
                    int64_t v = (int)((int64_t)a[i + k + len / 2] * twiddles[k] % p);
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
void ntt_simd(int32_t *a, int n, int64_t w, int64_t p) {
    // 验证n为2的幂
    if (n <= 0 || (n & (n - 1)) != 0) {
        fprintf(stderr, "错误：n必须为2的幂\n");
        return;
    }

    // 位逆序排列
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

    // 蝶形变换
    for (int len = 2; len <= n; len <<= 1) {
        int64_t wlen = mod_pow(w, (p - 1) / len, p);
        if (len / 2 >= 4) {
            butterfly_layer_simd(a, n, len, wlen, p);
        } else {
            // 标量回退
            for (int i = 0; i < n; i += len) {
                int64_t w_now = 1;
                for (int j = 0; j < len / 2; ++j) {
                    int64_t u = a[i + j];
                    int64_t v = (int)((int64_t)a[i + j + len / 2] * w_now % p);
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
    // 验证输入
    if (n <= 0) {
        fprintf(stderr, "错误：n必须为正整数\n");
        return;
    }

    // 找到最小的2的幂 >= 2n-1
    int m = 1;
    while (m < 2 * n - 1) m <<= 1;

    // 分配对齐内存
    int32_t *ta = (int32_t*)aligned_alloc(16, m * sizeof(int32_t));
    int32_t *tb = (int32_t*)aligned_alloc(16, m * sizeof(int32_t));
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

    // 原根及其逆（针对p = 998244353）
    int root = 3;
    int inv_root = mod_inv(root, p);

    // 前向NTT
    ntt_simd(ta, m, root, p);
    ntt_simd(tb, m, root, p);

    // 点值乘法
    for (int i = 0; i < m; ++i) {
        ta[i] = (int32_t)((int64_t)ta[i] * tb[i] % p);
    }

    // 逆NTT
    ntt_simd(ta, m, inv_root, p);

    // 归一化结果
    int64_t inv_m = mod_inv(m, p);
    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = (int)((int64_t)ta[i] * inv_m % p);
    }

    free(ta);
    free(tb);
}

