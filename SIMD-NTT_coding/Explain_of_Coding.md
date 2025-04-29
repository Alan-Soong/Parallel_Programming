# 关于代码的说明

如`main`可执行文件所述，由于代码很多影响观看，

所以采用分头文件的方式将几次迭代的代码进行分类：

- #include "nttnor.h" //这是ntt的普通代码（没有并行）

- #include "neon_good.h" // 这是ntt的neon代码（并行）

- #include "neon_improve.h"    // 这是ntt的neon代码（并行，只有蝴蝶）

- #include "ntt_neon_cmgood.h" // 这是ntt的neon代码（并行，实现Montgomery+蝴蝶）

由于内部名称冲突所以注释了三个保留一个，运行时根据不同需要选择即可。

> 可执行文件是默认最优的编译结果。