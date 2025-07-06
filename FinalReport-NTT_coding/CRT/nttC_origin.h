#ifndef NTT_CRT_SERIAL_H
#define NTT_CRT_SERIAL_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <cstring>

// Define moduli and primitive roots
const int64_t MOD1 = 998244353;
const int64_t MOD2 = 754974721;
const int64_t MOD3 = 167772161;
const int64_t MOD4 = 469762049;
const int64_t MOD5 = 1004535809;

const int PRIMITIVE_ROOT1 = 3;
const int PRIMITIVE_ROOT2 = 11;
const int PRIMITIVE_ROOT3 = 3;
const int PRIMITIVE_ROOT4 = 3;
const int PRIMITIVE_ROOT5 = 3;

// Maximum polynomial length
const int MAXN = 1 << 18;

// Precomputed twiddle factor caches and n_inv cache
static std::vector<std::vector<int64_t>> roots_cache(5, std::vector<int64_t>(MAXN + 1));
static std::vector<std::vector<int64_t>> inv_roots_cache(5, std::vector<int64_t>(MAXN + 1));
static std::vector<std::vector<int64_t>> n_inv_cache(5, std::vector<int64_t>(MAXN + 1));
static bool caches_initialized = false;

// Static buffers to avoid repeated allocation
static std::vector<std::vector<int>> ta_buffers(5, std::vector<int>(MAXN * 2));
static std::vector<std::vector<int>> tb_buffers(5, std::vector<int>(MAXN * 2));
static std::vector<std::vector<int>> result_buffers(5, std::vector<int>(MAXN * 2));

// Modular exponentiation
inline int64_t mod_pow(int64_t a, int64_t b, int64_t mod) {
    int64_t result = 1;
    a %= mod;
    if (a < 0) a += mod;
    while (b > 0) {
        if (b & 1) result = (result * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return result;
}

// Modular inverse using extended GCD
inline int64_t mod_inverse(int64_t a, int64_t mod) {
    int64_t t = 0, newt = 1;
    int64_t r = mod, newr = a;
    while (newr != 0) {
        int64_t quotient = r / newr;
        std::swap(t, newt);
        newt -= quotient * t;
        std::swap(r, newr);
        newr -= quotient * r;
    }
    if (r > 1) return -1; // No inverse
    if (t < 0) t += mod;
    return t;
}

// Initialize caches for twiddle factors and n_inv
void init_caches() {
    if (caches_initialized) return;
    caches_initialized = true;
    static const int64_t moduli[] = {MOD1, MOD2, MOD3, MOD4, MOD5};
    static const int roots[] = {PRIMITIVE_ROOT1, PRIMITIVE_ROOT2, PRIMITIVE_ROOT3, 
                               PRIMITIVE_ROOT4, PRIMITIVE_ROOT5};
    
    for (int i = 0; i < 5; ++i) {
        int64_t mod = moduli[i];
        int64_t root = roots[i];
        for (int len = 2; len <= MAXN; len <<= 1) {
            int64_t wlen = mod_pow(root, (mod - 1) / len, mod);
            roots_cache[i][len] = wlen;
            inv_roots_cache[i][len] = mod_inverse(wlen, mod);
            n_inv_cache[i][len] = mod_inverse(len, mod);
        }
    }
}

// Bit-reversal permutation with cached logn
inline void bit_reverse(int* a, int n, int logn) {
    for (int i = 0; i < n; ++i) {
        int j = 0;
        for (int k = 0; k < logn; ++k) {
            j |= ((i >> k) & 1) << (logn - 1 - k);
        }
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
}

// Single modulus NTT
void ntt(int* a, int n, int64_t mod, int mod_idx, bool inverse, int logn) {
    bit_reverse(a, n, logn);
    
    // Butterfly operations
    for (int len = 2; len <= n; len <<= 1) {
        int64_t wlen = inverse ? inv_roots_cache[mod_idx][len] : roots_cache[mod_idx][len];
        int half_len = len / 2;
        for (int i = 0; i < n; i += len) {
            int64_t w = 1;
            for (int j = 0; j < half_len; ++j) {
                int idx1 = i + j;
                int idx2 = i + j + half_len;
                int64_t u = a[idx1];
                int64_t v = (int64_t)a[idx2] * w % mod;
                a[idx1] = (u + v < mod) ? u + v : u + v - mod;
                a[idx2] = (u >= v) ? u - v : u - v + mod;
                w = (w * wlen) % mod;
            }
        }
    }
    
    if (inverse) {
        int64_t n_inv = n_inv_cache[mod_idx][n];
        for (int i = 0; i < n; ++i) {
            a[i] = (int64_t)a[i] * n_inv % mod;
        }
    }
}

// Single modulus polynomial multiplication
void poly_multiply_single_mod(const int* a, const int* b, int* result, int n, 
                             int64_t mod, int mod_idx, int* ta, int* tb) {
    int len = 1;
    int logn = 0;
    while (len < 2 * n) {
        len <<= 1;
        ++logn;
    }
    
    // Initialize buffers
    for (int i = 0; i < n; ++i) {
        ta[i] = a[i] % mod;
        if (ta[i] < 0) ta[i] += mod;
        tb[i] = b[i] % mod;
        if (tb[i] < 0) tb[i] += mod;
    }
    std::fill(ta + n, ta + len, 0);
    std::fill(tb + n, tb + len, 0);
    
    ntt(ta, len, mod, mod_idx, false, logn);
    ntt(tb, len, mod, mod_idx, false, logn);
    
    for (int i = 0; i < len; ++i) {
        ta[i] = (int64_t)ta[i] * tb[i] % mod;
    }
    
    ntt(ta, len, mod, mod_idx, true, logn);
    
    for (int i = 0; i < 2 * n - 1; ++i) {
        result[i] = ta[i];
    }
}

// CRT-based polynomial multiplication
void poly_multiply(int* a, int* b, int* result, int n, int64_t p) {
    init_caches(); // Initialize caches
    
    int num_mods = (p > (1LL << 32)) ? 5 : 3;
    static const int64_t all_moduli[] = {MOD1, MOD2, MOD3, MOD4, MOD5};
    
    // Compute NTT for each modulus
    for (int mod_idx = 0; mod_idx < num_mods; ++mod_idx) {
        poly_multiply_single_mod(a, b, result_buffers[mod_idx].data(), n, 
                                all_moduli[mod_idx], mod_idx,
                                ta_buffers[mod_idx].data(),
                                tb_buffers[mod_idx].data());
    }
    
    // Precompute CRT constants
    __int128 M = 1;
    for (int i = 0; i < num_mods; ++i) {
        M *= all_moduli[i];
    }
    
    std::vector<__int128> Mi(num_mods);
    std::vector<__int128> Mi_inv(num_mods);
    for (int i = 0; i < num_mods; ++i) {
        Mi[i] = M / all_moduli[i];
        Mi_inv[i] = mod_inverse(Mi[i], all_moduli[i]);
    }

    // CRT merging with block processing
    const int total_coeffs = 2 * n - 1;
    const int block_size = 1024; // Adjust based on cache size
    for (int start = 0; start < total_coeffs; start += block_size) {
        int end = std::min(start + block_size, total_coeffs);
        for (int idx = start; idx < end; ++idx) {
            __int128 x = 0;
            for (int j = 0; j < num_mods; ++j) {
                __int128 term = (__int128)result_buffers[j][idx] * Mi[j] % M;
                term = term * Mi_inv[j] % M;
                x = (x + term) % M;
            }
            if (x >= M / 2) x -= M;
            x %= p;
            if (x < 0) x += p;
            result[idx] = (int)x;
        }
    }
}

#endif // NTT_CRT_SERIAL_H