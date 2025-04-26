#include <arm_neon.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
using namespace std;

// Modular exponentiation
int64_t mod_pow(int64_t base, int64_t exp, int64_t mod) {
    int64_t result = 1;
    base %= mod;
    while (exp) {
        if (exp & 1) result = (result * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return result;
}

// Modular inverse using Fermat's Little Theorem
int64_t mod_inv(int64_t x, int64_t p) {
    return mod_pow(x, p - 2, p);
}

// NEON-optimized modulo reduction
static inline int32x4_t mod_reduce(int64x2_t v0, int64x2_t v1, int64_t p) {
    int32x4_t mod = vdupq_n_s32(p);
    int32x4_t t = { (int32_t)v0[0], (int32_t)v0[1], (int32_t)v1[0], (int32_t)v1[1] };
    t = vsubq_s32(t, vmulq_n_s32(vshrq_n_s32(t, 31), p)); // Handle negative
    t = vsubq_s32(t, vmulq_n_s32(vcgeq_s32(t, mod), p));   // Reduce mod p
    return t;
}

// SIMD butterfly layer (extended to handle len/2 >= 2)
void butterfly_layer_simd(int32_t *a, int n, int len, int64_t wlen, int64_t p) {
    int32_t *twiddles = (int32_t*)aligned_alloc(16, ((len / 2) * sizeof(int32_t) + 15) & ~15);
    if (!twiddles) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    twiddles[0] = 1;
    for (int j = 1; j < len / 2; ++j) {
        twiddles[j] = (int32_t)((int64_t)twiddles[j - 1] * wlen % p);
    }

    for (int i = 0; i < n; i += len) {
        for (int j = 0; j < len / 2; j += 4) {
            int base = i + j;
            int offset = base + len / 2;

            if (j + 3 < len / 2) {
                // Process 4 elements with SIMD
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
                // Scalar fallback for remaining elements
                for (int k = j; k < len / 2; ++k) {
                    int64_t u = a[i + k];
                    int64_t v = (int64_t)a[i + k + len / 2] * twiddles[k] % p;
                    a[i + k] = (u + v) % p;
                    a[i + k + len / 2] = (u - v + p) % p;
                }
                break;
            }
        }
    }
    free(twiddles);
}

// SIMD-optimized NTT
void ntt_simd(int32_t *a, int n, int64_t w, int64_t p) {
    if (n <= 0 || (n & (n - 1)) != 0) {
        fprintf(stderr, "Error: n must be a power of 2\n");
        return;
    }

    // Precomputed bit-reversal table
    static vector<int> rev;
    if (rev.size() != n) {
        rev.resize(n);
        rev[0] = 0;
        for (int i = 1, j = 0; i < n; ++i) {
            int bit = n >> 1;
            while (j & bit) j ^= bit, bit >>= 1;
            j ^= bit;
            rev[i] = j;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (i < rev[i]) swap(a[i], a[rev[i]]);
    }

    // Butterfly layers
    for (int len = 2; len <= n; len <<= 1) {
        int64_t wlen = mod_pow(w, (p - 1) / len, p);
        if (len / 2 >= 2) { // Extended to len/2 >= 2
            butterfly_layer_simd(a, n, len, wlen, p);
        } else {
            for (int i = 0; i < n; i += len) {
                int64_t w_now = 1;
                for (int j = 0; j < len / 2; ++j) {
                    int64_t u = a[i + j];
                    int64_t v = (int64_t)a[i + j + len / 2] * w_now % p;
                    a[i + j] = (u + v) % p;
                    a[i + j + len / 2] = (u - v + p) % p;
                    w_now = (int64_t)w_now * wlen % p;
                }
            }
        }
    }
}

// Polynomial multiplication (unchanged signature)
void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    if (n <= 0 || !a || !b || !ab || p != 998244353) {
        fprintf(stderr, "Invalid input or unsupported modulus\n");
        return;
    }

    int m = 1;
    while (m < 2 * n - 1) m <<= 1;

    int32_t *ta = nullptr, *tb = nullptr;
    ta = (int32_t*)aligned_alloc(16, ((m * sizeof(int32_t) + 15) & ~15));
    tb = (int32_t*)aligned_alloc(16, ((m * sizeof(int32_t) + 15) & ~15));
    if (!ta || !tb) {
        fprintf(stderr, "Memory allocation failed\n");
        free(ta);
        free(tb);
        return;
    }
    memset(ta, 0, m * sizeof(int32_t));
    memset(tb, 0, m * sizeof(int32_t));
    memcpy(ta, a, n * sizeof(int));
    memcpy(tb, b, n * sizeof(int));

    int64_t root = 3;
    int64_t inv_root = mod_inv(root, p);

    ntt_simd(ta, m, root, p);
    ntt_simd(tb, m, root, p);

    for (int i = 0; i < m; ++i) {
        ta[i] = (int32_t)((int64_t)ta[i] * tb[i] % p);
    }

    ntt_simd(ta, m, inv_root, p);

    int64_t inv_m = mod_inv(m, p);
    for (int i = 0; i < 2 * n - 1; ++i) {
        int64_t val = (int64_t)ta[i] * inv_m % p;
        ab[i] = val < 0 ? val + p : val;
    }

    free(ta);
    free(tb);
}
