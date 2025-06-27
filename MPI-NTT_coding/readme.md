# MPI关于代码的说明

如`main`可执行文件所述，分类原因同上：

* #include "nttnor.h" //这是ntt的普通代码（没有并行）

* #include "ntt_mpi.h" //这是ntt的mpi代码（并行）

* #include "ntt_mpiB.h" // 这是实现了巴雷特规约的代码（并行）

* #include "ntt_mpiimprove.h"

* #include "ntt_mpipth.h" // 这是ntt的mpi+pthread代码（并行）

* ##include "ntt_mpiomp.h"  // 这是ntt的mpi+openmp代码（并行）

由于内部名称冲突所以注释了多个保留一个，运行时根据不同需要选择即可。

> 可执行文件是默认最优的编译结果。
