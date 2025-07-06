#include <cstring>
#include <iostream>
#include <arm_neon.h>
#include <cstdlib>
#include <omp.h>

// ======================= Montgomery Reduction =======================
struct Montgomery {
    int p;
    uint32_t inv_p;
    int r2;

    Montgomery(int mod) : p(mod) {
        inv_p = -modinv(p);
        r2 = ((int64_t(1) << 31) % p) * 2 % p; // 2^32 % p
    }

    uint32_t modinv(uint32_t a) {
        uint32_t x = a;
        for (int i = 0; i < 5; ++i) x *= 2 - a * x;
        return x;
    }

    inline int montgomery_reduce(uint64_t T) const {
        uint32_t m = (uint32_t)T * inv_p;
        uint64_t t = (T + (uint64_t)m * p) >> 32;
        return (t >= p) ? t - p : t;
    }

    int transform(int a) const {
        return montgomery_reduce((uint64_t)a * r2);
    }

    int reduce(int a) const {
        return montgomery_reduce((uint64_t)a);
    }

    int mul(int a, int b) const {
        return montgomery_reduce((uint64_t)a * b);
    }
};

void butterfly_layer_simd(int *a, int n, int len, int wlen, int p, Montgomery &mont) {
    int *twiddles = (int*)aligned_alloc(16, (len / 2) * sizeof(int));
    twiddles[0] = 1;
    for (int j = 1; j < len / 2; ++j)
        twiddles[j] = (int)((int64_t)twiddles[j - 1] * wlen % p);

    int blocks = n / len;

    #pragma omp parallel for schedule(static)
    for (int blk = 0; blk < blocks; ++blk) {
        int i = blk * len;
        for (int j = 0; j < len / 2; j += 4) {
            int base = i + j;
            int offset = base + len / 2;

            if (j + 3 < len / 2) {
                int32x4_t u = vld1q_s32(&a[base]);
                int32x4_t v = vld1q_s32(&a[offset]);
                int32x4_t w = vld1q_s32(&twiddles[j]);

                int64x2_t v0 = vmull_s32(vget_low_s32(v), vget_low_s32(w));
                int64x2_t v1 = vmull_s32(vget_high_s32(v), vget_high_s32(w));

                int32_t vres[4] = {
                    (int32_t)((int64_t)v0[0] % p),
                    (int32_t)((int64_t)v0[1] % p),
                    (int32_t)((int64_t)v1[0] % p),
                    (int32_t)((int64_t)v1[1] % p)
                };
                int32x4_t vmod = vld1q_s32(vres);

                int32x4_t sum = vaddq_s32(u, vmod);
                int32x4_t diff = vsubq_s32(u, vmod);
                int32x4_t modv = vdupq_n_s32(p);
                sum = vbslq_s32(vcgeq_s32(sum, modv), vsubq_s32(sum, modv), sum);
                diff = vbslq_s32(vcltq_s32(diff, vdupq_n_s32(0)), vaddq_s32(diff, modv), diff);

                vst1q_s32(&a[base], sum);
                vst1q_s32(&a[offset], diff);
            } else {
                for (int k = j; k < len / 2; ++k) {
                    int u = a[i + k];
                    int v = (int)((int64_t)a[i + k + len / 2] * twiddles[k] % p);
                    a[i + k] = (u + v >= p) ? u + v - p : u + v;
                    a[i + k + len / 2] = (u - v < 0) ? u - v + p : u - v;
                }
                break;
            }
        }
    }

    free(twiddles);
}

int mod_pow(int base, int exp, int mod) {
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

void ntt_simd(int *a, int n, int w, int p, Montgomery &mont) {
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) j ^= bit, bit >>= 1;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        int wlen = mod_pow(w, (p - 1) / len, p);
        if (len / 2 >= 4)
            butterfly_layer_simd(a, n, len, wlen, p, mont);
        else {
            for (int i = 0; i < n; i += len) {
                int w_now = 1;
                for (int j = 0; j < len / 2; ++j) {
                    int u = a[i + j];
                    int v = (int)((int64_t)a[i + j + len / 2] * w_now % p);
                    a[i + j] = (u + v >= p) ? u + v - p : u + v;
                    a[i + j + len / 2] = (u - v < 0) ? u - v + p : u - v;
                    w_now = (int)((int64_t)w_now * wlen % p);
                }
            }
        }
    }
}

void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    int m = 1;
    while (m < 2 * n - 1) m <<= 1;

    int *ta = (int*)aligned_alloc(16, m * sizeof(int));
    int *tb = (int*)aligned_alloc(16, m * sizeof(int));
    memset(ta, 0, m * sizeof(int));
    memset(tb, 0, m * sizeof(int));
    memcpy(ta, a, n * sizeof(int));
    memcpy(tb, b, n * sizeof(int));

    Montgomery mont(p);

    int root = 3;
    int inv_root = mod_inv(root, p);

    ntt_simd(ta, m, root, p, mont);
    ntt_simd(tb, m, root, p, mont);

    for (int i = 0; i < m; ++i)
        ta[i] = (int)((int64_t)ta[i] * tb[i] % p);

    ntt_simd(ta, m, inv_root, p, mont);

    int inv_m = mod_inv(m, p);
    for (int i = 0; i < 2 * n - 1; ++i)
        ab[i] = (int)((int64_t)ta[i] * inv_m % p);

    free(ta);
    free(tb);
}
