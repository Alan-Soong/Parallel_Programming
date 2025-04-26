#pragma once
#include <cstdint>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <arm_neon.h>
#include <iostream>
#include <algorithm> // For std::min

// 模幂运算优化实现
static inline int mod_pow(int base, int exp, int mod) {
    int res = 1;
    while (exp) {
        res = (exp & 1) ? ((int)((int64_t)res * base % mod)) : res;
        base = (int)((int64_t)base * base % mod);
        exp >>= 1;
    }
    return res;
}

int mod_inv(int x, int p) {
    return mod_pow(x, p - 2, p);
}

// 蒙哥马利约减重实现
class MontgomeryApprox {
    using U64 = uint64_t;
    using U128 = __uint128_t;

    int modulus;
    U64 invMFactor;
    U64 rSquare;

    static U64 computeMontInv(U64 p) {
        U64 res = p;
        for (int i = 0; i < 6; ++i) res *= 2 - p * res;
        return -res;
    }

public:
    MontgomeryApprox(int p) : modulus(p) {
        assert((modulus & 1) && modulus < (1LL << 31));
        invMFactor = computeMontInv(p);
        rSquare = computeRSquare(p);
    }

    U64 computeRSquare(int m) const {
        U64 r = -U64(m) % m;
        r = (U64(r) << 32) % m;
        return (U64(r) << 32) % m;
    }

    U64 convertTo(U64 x) const {
        return approxReduce(U128(x) * rSquare);
    }

    int convertFrom(U64 x) const {
        return approxReduce(x);
    }

    U64 approxReduce(U128 x) const {
        U64 q = (U64)x * invMFactor;
        U128 t = (x + U128(q) * modulus) >> 64;
        return t >= modulus ? t - modulus : t;
    }

    int multiply(int a, int b) const {
        return approxReduce(U128(a) * b);
    }

    int exponentiate(int x, int e) const {
        U64 res = convertTo(1);
        U64 base = convertTo(x);
        while (e) {
            if (e & 1) res = approxReduce(U128(res) * base);
            base = approxReduce(U128(base) * base);
            e >>= 1;
        }
        return convertFrom(res);
    }

    int inverse(int x) const {
        return exponentiate(x, modulus - 2);
    }
};

// NEON优化模约减改进实现
static inline int32x4_t approxModReduce(int64x2_t val0, int64x2_t val1, int mod) {
    int32x4_t currentMod = vdupq_n_s32(mod);
    int32x4_t result = {
        (int32_t)(val0[0] % mod), (int32_t)(val0[1] % mod),
        (int32_t)(val1[0] % mod), (int32_t)(val1[1] % mod)
    };

    // Fix: Use uint32x4_t for comparison results
    uint32x4_t isNegative = vcltq_s32(result, vdupq_n_s32(0));
    result = vbslq_s32(isNegative, vaddq_s32(result, currentMod), result);

    return result;
}

// SIMD优化蝶形变换重实现
void optimizedButterflyTransform(int *data, int size, int length, int root, int prime) {
    int *twiddles = (int*)aligned_alloc(16, (length / 2) * sizeof(int));
    if (!twiddles) {
        std::cerr << "Memory allocation failed\n";
        return;
    }

    twiddles[0] = 1;
    for (int j = 1; j < length / 2; ++j) {
        twiddles[j] = (int)((int64_t)twiddles[j - 1] * root % prime);
    }

    for (int i = 0; i < size; i += length) {
        for (int j = 0; j < length / 2; j += 4) {
            int base = i + j;
            int offset = base + length / 2;

            if (j + 3 < length / 2) {
                int32x4_t u = vld1q_s32(&data[base]);
                int32x4_t v = vld1q_s32(&data[offset]);
                int32x4_t w = vld1q_s32(&twiddles[j]);

                int64x2_t val0 = vmull_s32(vget_low_s32(v), vget_low_s32(w));
                int64x2_t val1 = vmull_s32(vget_high_s32(v), vget_high_s32(w));
                int32x4_t modRes = approxModReduce(val0, val1, prime);

                int32x4_t sum = vaddq_s32(u, modRes);
                int32x4_t diff = vsubq_s32(u, modRes);

                // Fix: Use uint32x4_t for comparison results
                uint32x4_t isGreaterEqual = vcgeq_s32(sum, vdupq_n_s32(prime));
                sum = vbslq_s32(isGreaterEqual, vsubq_s32(sum, vdupq_n_s32(prime)), sum);

                uint32x4_t isNegative = vcltq_s32(diff, vdupq_n_s32(0));
                diff = vbslq_s32(isNegative, vaddq_s32(diff, vdupq_n_s32(prime)), diff);

                vst1q_s32(&data[base], sum);
                vst1q_s32(&data[offset], diff);
            } else {
                for (int k = j; k < length / 2; ++k) {
                    int u = data[i + k];
                    int v = (int)((int64_t)data[i + k + length / 2] * twiddles[k] % prime);
                    data[i + k] = (u + v) % prime;
                    data[i + k + length / 2] = (u - v + prime) % prime;
                }
                break;
            }
        }
    }
    free(twiddles);
}

// 改进的SIMD优化NTT
void optimizedNtt(int *data, int size, int root, int prime) {
    if (size <= 0 || (size & (size - 1)) != 0) {
        std::cerr << "Error: size must be a power of two\n";
        return;
    }

    for (int i = 1, j = 0; i < size; ++i) {
        int bit = size >> 1;
        while (j & bit) j ^= bit, bit >>= 1;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }

    for (int len = 2; len <= size; len <<= 1) {
        int rootLen = mod_pow(root, (prime - 1) / len, prime);
        if (len / 2 >= 4) {
            optimizedButterflyTransform(data, size, len, rootLen, prime);
        } else {
            for (int i = 0; i < size; i += len) {
                int currentRoot = 1;
                for (int j = 0; j < len / 2; ++j) {
                    int u = data[i + j];
                    int v = (int)((int64_t)data[i + j + len / 2] * currentRoot % prime);
                    data[i + j] = (u + v) % prime;
                    data[i + j + len / 2] = (u - v + prime) % prime;
                    currentRoot = (int)((int64_t)currentRoot * rootLen % prime);
                }
            }
        }
    }
}

// 改进的多项式乘法实现
void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    if (n <= 0) {
        fprintf(stderr, "错误：n必须为正整数\n");
        return;
    }

    int m = 1;
    while (m < 2 * n - 1) {
        m <<= 1; // 计算大于等于2n-1的最小2的幂
    }

    // 检查内存是否分配成功
    int *ta = (int*)aligned_alloc(16, m * sizeof(int));
    int *tb = (int*)aligned_alloc(16, m * sizeof(int));
    if (!ta || !tb) {
        fprintf(stderr, "内存分配失败\n");
        free(ta);
        free(tb);
        return;
    }

    // 初始化临时数组
    memset(ta, 0, m * sizeof(int));
    memset(tb, 0, m * sizeof(int));
    memcpy(ta, a, n * sizeof(int));
    memcpy(tb, b, n * sizeof(int));

    int generator = 3;
    int invGen = mod_inv(generator, p);

    // 执行NTT变换
    optimizedNtt(ta, m, generator, p);
    optimizedNtt(tb, m, generator, p);

    // 点乘操作
    for (int i = 0; i < m; ++i) {
        ab[i] = (int)((int64_t)ta[i] * tb[i] % p);
    }

    // 执行逆变换前的缩放
    int invSize = mod_inv(m, p);
    for (int i = 0; i < m; ++i) {
        ab[i] = (int)((int64_t)ab[i] * invSize % p);
    }

    // 执行逆变换
    optimizedNtt(ab, m, invGen, p);

    // 清理临时数组
    free(ta);
    free(tb);

    // 添加额外的边界检查
    for (int i = 0; i < m; ++i) {
        if (ab[i] < 0 || ab[i] >= p) {
            fprintf(stderr, "警告: 结果超出模范围, ab[%d] = %d\n", i, ab[i]);
            ab[i] = (ab[i] % p + p) % p;
        }
    }
}