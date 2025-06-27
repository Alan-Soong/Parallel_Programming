#ifndef NTT_MPI_SIMD_H
#define NTT_MPI_SIMD_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <arm_neon.h>
#include <cstring>
#include <cstdlib>
#include <cstdint>

// 模幂运算
inline int64_t mod_pow(int64_t base, int64_t exp, int64_t mod) {
    if (mod <= 0) return 0; // 验证模数
    int64_t res = 1;
    base = (base % mod + mod) % mod; // 确保基数非负
    while (exp) {
        if (exp & 1) res = (__int128_t)res * base % mod;
        base = (__int128_t)base * base % mod;
        exp >>= 1;
    }
    return res < 0 ? res + mod : res;
}

// 模逆运算（使用费马小定理）
inline int64_t mod_inv(int64_t x, int64_t p) {
    if (x == 0 || p <= 1) return 0; // 验证输入
    return mod_pow(x, p - 2, p);
}

// Optimized NEON modulo reduction
static inline int32x4_t mod_reduce_neon(int32x4_t val, int32_t p) {
    // 将 int32x4_t 转为 float32x4_t 做近似除法
    float32x4_t val_f = vcvtq_f32_s32(val);
    float32x4_t p_f   = vdupq_n_f32((float)p);
    float32x4_t div_f = vdivq_f32(val_f, p_f);
    // 转回 int，得到近似整除结果
    int32x4_t approx_div = vcvtq_s32_f32(div_f);
    
    // val -= approx_div * p
    int32x4_t p_vec = vdupq_n_s32(p);
    int32x4_t product = vmulq_s32(approx_div, p_vec);
    val = vsubq_s32(val, product);
    
    // 如果 val < 0，则加上 p
    int32x4_t zero_vec = vdupq_n_s32(0);
    uint32x4_t negative_mask = vcltq_s32(val, zero_vec);
    int32x4_t negative_mask_s32 = vreinterpretq_s32_u32(negative_mask);
    int32x4_t add_val = vmulq_s32(negative_mask_s32, p_vec);
    val = vaddq_s32(val, add_val);
    
    return val;
}

// NEON优化的蝶形变换层
void butterfly_layer_simd(int32_t *a, int n, int len, int64_t wlen, int64_t p) {
    if (!a || n <= 0 || len <= 0 || p <= 0 || p > (1LL << 30)) {
        fprintf(stderr, "butterfly_layer_simd 参数无效或模数过大\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    // 分配对齐的扭转因子数组
    int32_t *twiddles = (int32_t*)aligned_alloc(16, ((len / 2) * sizeof(int32_t)));
    if (!twiddles) {
        fprintf(stderr, "扭转因子内存分配失败\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    twiddles[0] = 1;
    for (int j = 1; j < len / 2; ++j) {
        twiddles[j] = (__int128_t)twiddles[j - 1] * wlen % p;
    }

    int32x4_t p_vec = vdupq_n_s32(p);
    for (int i = 0; i < n; i += len) {
        for (int j = 0; j < len / 2; j += 4) {
            int base = i + j;
            int offset = base + len / 2;

            if (j + 3 < len / 2) {
                int32x4_t u = vld1q_s32(&a[base]);
                int32x4_t v = vld1q_s32(&a[offset]);
                int32x4_t w = vld1q_s32(&twiddles[j]);

                // 优化乘法和模运算
                int64x2_t v_low = vmull_s32(vget_low_s32(v), vget_low_s32(w));
                int64x2_t v_high = vmull_s32(vget_high_s32(v), vget_high_s32(w));
                
                int32x4_t v_times_w = vcombine_s32(
                    vdup_n_s32((int32_t)(v_low[0] % p)),
                    vdup_n_s32((int32_t)(v_high[0] % p))
                );
                v_times_w = vsetq_lane_s32((int32_t)(v_low[1] % p), v_times_w, 1);
                v_times_w = vsetq_lane_s32((int32_t)(v_high[1] % p), v_times_w, 3);

                int32x4_t sum = mod_reduce_neon(vaddq_s32(u, v_times_w), p);
                int32x4_t diff = mod_reduce_neon(vsubq_s32(u, v_times_w), p);

                vst1q_s32(&a[base], sum);
                vst1q_s32(&a[offset], diff);
            } else {
                for (int k = j; k < len / 2; ++k) {
                    int64_t u_scalar = a[i + k];
                    int64_t v_scalar = (__int128_t)a[i + k + len / 2] * twiddles[k] % p;
                    a[i + k] = (u_scalar + v_scalar) % p;
                    a[i + k + len / 2] = (u_scalar - v_scalar + p) % p;
                }
                break;
            }
        }
    }
    free(twiddles);
}

// SIMD优化的NTT变换
void ntt_simd(int32_t *a, int n, int64_t root, int64_t mod, bool invert) {
    if (!a || n <= 0 || (n & (n - 1)) != 0 || mod <= 0 || mod > (1LL << 30)) {
        fprintf(stderr, "NTT 参数无效或模数过大\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    // 位反转置换
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    // 蝶形变换层
    for (int len = 2; len <= n; len <<= 1) {
        int64_t wlen = mod_pow(root, (mod - 1) / len, mod);
        if (invert) wlen = mod_inv(wlen, mod);

        // 修改判断条件为 (len >= 8)
        if (len >= 8) {
            butterfly_layer_simd(a, n, len, wlen, mod);
        } else {
            for (int i = 0; i < n; i += len) {
                int64_t w_now = 1;
                for (int k = 0; k < len / 2; ++k) {
                    int64_t u = a[i + k];
                    int64_t v = (__int128_t)a[i + k + len / 2] * w_now % mod;
                    a[i + k] = (u + v) % mod;
                    a[i + k + len / 2] = (u - v + mod) % mod;
                    w_now = (__int128_t)w_now * wlen % mod;
                }
            }
        }
    }

    if (invert) {
        int64_t inv_n = mod_inv(n, mod);
        int32x4_t inv_n_vec = vdupq_n_s32(inv_n);
        for (int i = 0; i < n; i += 4) {
            if (i + 3 < n) {
                int32x4_t vec_a = vld1q_s32(&a[i]);
                int64x2_t prod_low = vmull_s32(vget_low_s32(vec_a), vget_low_s32(inv_n_vec));
                int64x2_t prod_high = vmull_s32(vget_high_s32(vec_a), vget_high_s32(inv_n_vec));
                
                int32x4_t result_vec = vcombine_s32(
                    vdup_n_s32((int32_t)(prod_low[0] % mod)),
                    vdup_n_s32((int32_t)(prod_high[0] % mod))
                );
                result_vec = vsetq_lane_s32((int32_t)(prod_low[1] % mod), result_vec, 1);
                result_vec = vsetq_lane_s32((int32_t)(prod_high[1] % mod), result_vec, 3);
                vst1q_s32(&a[i], result_vec);
            } else {
                for (int k = i; k < n; ++k) {
                    a[k] = (__int128_t)a[k] * inv_n % mod;
                }
            }
        }
    }
}

// 获取原根（常见NTT模数）
inline int get_primitive_root(int p) {
    switch (p) {
        case 998244353: return 3;
        case 7340033: return 3;
        case 104857601: return 3;
        case 469762049: return 3;
        default:
            fprintf(stderr, "错误：模数 %d 无效或未定义原根\n", p);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return -1; // 不会执行到此处
    }
}

// MPI并行NTT封装
void ntt_mpi_simd_wrapper(int32_t* a_data, int n, bool invert, int root, int mod, MPI_Comm comm) {
    if (!a_data || n <= 0 || mod <= 0 || mod > (1LL << 30)) {
        fprintf(stderr, "ntt_mpi_simd_wrapper 参数无效或模数过大\n");
        MPI_Abort(comm, 1);
        return;
    }

    // 验证模数是否支持
    root = get_primitive_root(mod);
    if (root == -1) {
        fprintf(stderr, "错误：模数 %d 不支持\n", mod);
        MPI_Abort(comm, 1);
        return;
    }

    ntt_simd(a_data, n, root, mod, invert);
}

// 并行多项式乘法（MPI + NEON SIMD）
void poly_multiply(int *a, int *b, int *ab, int n, int p, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 放宽对 p 和 n 的检查
    if (!a || !b || !ab || n <= 0 || p <= 0) {
        fprintf(stderr, "poly_multiply 参数无效\n");
        MPI_Abort(comm, 1);
        return;
    }

    // 步骤 1：扩展到最小的2的幂
    int lim = 1;
    while (lim < 2 * n - 1) lim <<= 1;

    // 验证内存需求
    if (lim > (1 << 20)) {
        fprintf(stderr, "扩展规模 lim=%d 过大，超过限制\n", lim);
        MPI_Abort(comm, 1);
        return;
    }

    // 步骤 2：初始化对齐数组
    int32_t *A_aligned = (int32_t*)aligned_alloc(16, lim * sizeof(int32_t));
    int32_t *B_aligned = (int32_t*)aligned_alloc(16, lim * sizeof(int32_t));
    int32_t *C_aligned = (int32_t*)aligned_alloc(16, lim * sizeof(int32_t));

    if (!A_aligned || !B_aligned || !C_aligned) {
        fprintf(stderr, "多项式乘法内存分配失败\n");
        free(A_aligned); free(B_aligned); free(C_aligned);
        MPI_Abort(comm, 1);
        return;
    }

    // 初始化为零
    memset(A_aligned, 0, lim * sizeof(int32_t));
    memset(B_aligned, 0, lim * sizeof(int32_t));
    memset(C_aligned, 0, lim * sizeof(int32_t));

    // 步骤 3：主进程复制输入数据
    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            A_aligned[i] = a[i] % p;
            B_aligned[i] = b[i] % p;
        }
    }

    // 步骤 4：广播 A 和 B
    MPI_Bcast(A_aligned, lim, MPI_INT, 0, comm);
    MPI_Bcast(B_aligned, lim, MPI_INT, 0, comm);

    // 步骤 5：执行 NTT
    int root = get_primitive_root(p);
    ntt_mpi_simd_wrapper(A_aligned, lim, false, root, p, comm);
    ntt_mpi_simd_wrapper(B_aligned, lim, false, root, p, comm);

    // 步骤 6：点值乘法
    int chunk = (lim + size - 1) / size;
    int start = rank * chunk;
    int end = std::min(start + chunk, lim);
    std::vector<int32_t> local_C(end - start);

    for (int i = start; i < end; i += 4) {
        if (i + 3 < end) {
            int32x4_t val_A = vld1q_s32(&A_aligned[i]);
            int32x4_t val_B = vld1q_s32(&B_aligned[i]);
            int64x2_t prod_low = vmull_s32(vget_low_s32(val_A), vget_low_s32(val_B));
            int64x2_t prod_high = vmull_s32(vget_high_s32(val_A), vget_high_s32(val_B));

            int32x4_t result_vec = vcombine_s32(
                vdup_n_s32((int32_t)(prod_low[0] % p)),
                vdup_n_s32((int32_t)(prod_high[0] % p))
            );
            result_vec = vsetq_lane_s32((int32_t)(prod_low[1] % p), result_vec, 1);
            result_vec = vsetq_lane_s32((int32_t)(prod_high[1] % p), result_vec, 3);
            vst1q_s32(&local_C[i - start], result_vec);
        } else {
            for (int k = i; k < end; ++k) {
                local_C[k - start] = (__int128_t)A_aligned[k] * B_aligned[k] % p;
            }
        }
    }

    // 步骤 7：收集点值乘法结果（优化避免内存重叠）
    std::vector<int> recv_counts(size);
    std::vector<int> displs(size);
    int total_count = 0;
    for (int i = 0; i < size; ++i) {
        int i_start = i * chunk;
        int i_end = std::min(i_start + chunk, lim);
        recv_counts[i] = i_end - i_start;
        displs[i] = i_start;
        total_count += recv_counts[i];
    }

    if (total_count != lim) {
        fprintf(stderr, "错误：recv_counts 总和 %d 不等于 lim %d\n", total_count, lim);
        free(A_aligned); free(B_aligned); free(C_aligned);
        MPI_Abort(comm, 1);
        return;
    }

    // 使用临时缓冲区避免内存重叠
    int32_t *temp_C = (int32_t*)aligned_alloc(16, lim * sizeof(int32_t));
    if (!temp_C) {
        fprintf(stderr, "临时缓冲区内存分配失败\n");
        free(A_aligned); free(B_aligned); free(C_aligned);
        MPI_Abort(comm, 1);
        return;
    }
    memset(temp_C, 0, lim * sizeof(int32_t));

    MPI_Allgatherv(local_C.data(), end - start, MPI_INT, temp_C, 
                   recv_counts.data(), displs.data(), MPI_INT, comm);

    // 复制到 C_aligned
    memcpy(C_aligned, temp_C, lim * sizeof(int32_t));
    free(temp_C);

    // 步骤 8：逆 NTT
    ntt_mpi_simd_wrapper(C_aligned, lim, true, root, p, comm);

    // 步骤 9：主进程输出结果
    if (rank == 0) {
        for (int i = 0; i < 2 * n - 1; ++i) {
            ab[i] = (C_aligned[i] % p + p) % p; // 确保结果非负
        }
    }

    // 释放内存
    free(A_aligned);
    free(B_aligned);
    free(C_aligned);
}

#endif // NTT_MPI_SIMD_H