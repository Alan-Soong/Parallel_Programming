#ifndef NTT_MPI_H
#define NTT_MPI_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <cstdint>  // for uint64_t
#include <omp.h>

// Barrett Reduction wrapper
class BarrettReducer {
public:
    int mod;
    uint64_t m; // precomputed floor(2^64 / mod)

    explicit BarrettReducer(int p) : mod(p) {
        m = (uint64_t((__uint128_t(1) << 64) / mod));
    }

    inline int reduce(uint64_t a) const {
        uint64_t q = ((__uint128_t)a * m) >> 64;
        int r = (int)(a - q * mod);
        return r < mod ? (r < 0 ? r + mod : r) : r - mod;
    }

    inline int mul(int a, int b) const {
        return reduce((uint64_t)a * b);
    }
};

// Modulo exponentiation
inline int mod_pow(int base, int exp, int mod) {
    long long res = 1, b = base % mod;
    while (exp) {
        if (exp & 1) res = res * b % mod;
        b = b * b % mod;
        exp >>= 1;
    }
    return res < 0 ? res + mod : res;
}

// Bit-reversal permutation
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

// Get primitive root (for p=7340033, 3 is valid)
inline int get_primitive_root(int p) {
    return 3;
}

// Serial NTT using Barrett reduction
void ntt(std::vector<int>& a, bool invert, int root, int mod) {
    int n = a.size();
    BarrettReducer br(mod);

    bit_reverse(a);
    for (int len = 2; len <= n; len <<= 1) {
        int wlen = mod_pow(root, (mod - 1) / len, mod);
        if (invert) wlen = mod_pow(wlen, mod - 2, mod);
        for (int i = 0; i < n; i += len) {
            int w = 1;
            // #pragma omp parallel for
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

// MPI Parallel NTT (simplified: each process executes full NTT)
void ntt_mpi(std::vector<int>& a, bool invert, int root, int mod, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Ensure all processes have complete data
    MPI_Bcast(a.data(), a.size(), MPI_INT, 0, comm);

    // Each process executes full NTT
    ntt(a, invert, root, mod);
}

// Parallel polynomial multiplication
void poly_multiply(int *a, int *b, int *ab, int n, int p, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int lim = 1;
    while (lim < 2 * n - 1) lim <<= 1;

    std::vector<int> A(lim), B(lim), C(lim);
    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            A[i] = a[i] % p;
            B[i] = b[i] % p;
        }
        for (int i = n; i < lim; ++i) A[i] = B[i] = 0;
    }

    MPI_Bcast(A.data(), lim, MPI_INT, 0, comm);
    MPI_Bcast(B.data(), lim, MPI_INT, 0, comm);

    int root = get_primitive_root(p);
    ntt_mpi(A, false, root, p, comm);
    ntt_mpi(B, false, root, p, comm);

    int chunk = (lim + size - 1) / size;
    int start = rank * chunk;
    int end = std::min(start + chunk, lim);
    std::vector<int> local_C(chunk, 0);

    BarrettReducer br(p);
    #pragma omp parallel for
    for (int i = start; i < end; ++i) {
        local_C[i - start] = br.mul(A[i], B[i]);
    }

    std::vector<int> recv_counts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
        int i_start = i * chunk;
        int i_end = std::min(i_start + chunk, lim);
        recv_counts[i] = i_end - i_start;
        displs[i] = i_start;
    }
    MPI_Allgatherv(local_C.data(), end - start, MPI_INT, C.data(), recv_counts.data(), displs.data(), MPI_INT, comm);

    ntt_mpi(C, true, root, p, comm);

    if (rank == 0) {
        #pragma omp parallel for
        for (int i = 0; i < 2 * n - 1; ++i) {
            ab[i] = C[i] % p;
        }
    }
}

#endif // NTT_MPI_H
