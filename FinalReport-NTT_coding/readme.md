# 期末大报告关于代码的说明

我们这次的实验使用了诸多头文件进行实验的汇总，具体讲解如下：

## 朴素算法的加速

- #include "Contrast/ntt_origin.h" //这是 ntt 的普通代码（没有并行）
- #include "Contrast/ntt_simd.h" //这是 ntt 的 SIMD 代码
- #include "Contrast/ntt_pthread.h" //这是 ntt 的 Pthread 并行代码
- #include "Contrast/ntt_openmp.h" //这是 ntt 的 OpenMP 并行代码
- #include "Contrast/ntt_mpi.h" //这是 ntt 的 MPI 并行代码
- #include "Contrast/ntt_cuda_opt.cuh" //这是 ntt 的 CUDA 并行代码

## Montgomery规约

- #include "Montgomery/nttM_simd.h" //这是 ntt（基于 SIMD 优化）的 Montgomery 规约代码
- #include "Montgomery/ntt_simd+pthread.h" //这是 ntt（基于 SIMD 和 Pthread 优化）的 Montgomery 规约代码
- #include "Montgomery/ntt_simd+openmp.h" //这是 ntt（基于 SIMD 和 OpenMP 优化）的 Montgomery 规约代码

## Barrett规约

- #include "Barrett/nttB_origin.h" //这是 ntt 的 Barrett 规约代码
- #include "Barrett/nttB_mpi.h" //这是 ntt（基于 MPI 优化）的 Barrett 规约代码
- #include "Barrett/nttB_mpi+pthread.h" //这是 ntt（基于 MPI 和 Pthread 优化）的 Barrett 规约代码
- #include "Barrett/nttB_mpi+openmp.h" //这是 ntt（基于 MPI 和 OpenMP 优化）的 Barrett 规约代码
- #include "Barrett/nttB_triple.h" //这是 ntt（基于 MPI 和 Pthread、OpenMP 优化）的 Barrett 规约代码

> GPU 的代码由于时间关系，我没有在报告中体现，可以后续拓展

## CRT的使用

- #include "CRT/nttC_origin.h" //这是 ntt 的 CRT 代码
- #include "CRT/nttC_pthread.h" //这是 ntt（基于 Pthread 优化）的 CRT 代码
- #include "CRT/nttC_openmp.h" //这是 ntt（基于 OpenMP 优化）的 CRT 代码
- #include "CRT/nttC_pthread+openmp.h" //这是 ntt（基于 Pthread 和 OpenMP 优化）的 CRT 代码

## 其他

- main.cc // 主体函数，负责普通的算法及一系列头文件
- mpi_main.cc // 主体函数，负责 MPI 的算法及一系列头文件
- main.cu // 主体函数，负责 CUDA 的算法及一系列头文件

在最后使用的时候，可以根据需要选择不同的头文件进行编译和运行。
