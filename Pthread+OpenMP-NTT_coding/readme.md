# 关于代码的说明

如`main`可执行文件所述，由于代码很多影响观看，

所以采用分头文件的方式将几次迭代的代码进行分类：

* #include "nttnor.h" //这是ntt的普通代码（没有并行）

* #include "ntt_pthread.h" // 这是ntt的pthread代码（并行）

* #include "ntt_fourDivide.h" // 这是四除法的ntt代码（并行）

* #include "ntt_CRTpthread.h" // 这是ntt的CRT代码（并行）

* #include "ntt_openmp.h"  // 这是ntt的openmp代码（并行）

由于内部名称冲突所以注释了三个保留一个，运行时根据不同需要选择即可。

> 可执行文件是默认最优的编译结果（提交测试的时候，GitHub里面是第一版的）。
