# GPU关于代码的说明

如`main`可执行文件所述，分类原因同上：

* #include "nttnor.h" //这是ntt的普通代码（没有并行）

* #include "ntt_cuda_gpu.h" //这是ntt的cuda代码（并行）
* #include "ntt_cuda_single_opt.h" //这是ntt的cuda代码（并行）
* #include "ntt_cuda_gpu.h" //这是ntt的cuda代码（并行）

* #include "ntt_cuda_B.h" // 这是实现了Barrett规约的代码（并行，有一个fixed的版本是待修正的版本）

* #include "ntt_cuda_M.h" // 这是实现了Montgomery规约的代码（并行，有一个fixed的版本是待修正的版本）

由于内部名称冲突所以注释了多个保留一个，运行时根据不同需要选择即可。

> 前面的三个GPU本质没有太多区别，存在一定的加速比优化，但是总体差不多
> 可执行文件是默认最优的编译结果。
