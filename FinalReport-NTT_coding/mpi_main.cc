#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
// 可以自行添加需要的头文件
// #include "nttnor.h"
// # include "Barrett/nttB_mpi.h"
// # include "nttB_mpi+openmp.h"
// # include "nttB_mpi+pthread.h"
# include "Barrett/nttB_triple.h"
// #include "Barrett/nttB_mpi+openmp.h" // 这是ntt的mpi+openmp代码（并行）
// mpic++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11
// qsub qsub_mpi.sh
// perf record -e cpu-clock,cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./main
// perf report > 文件名.txt

void fRead(int *a, int *b, int *n, int *p, int input_id, int rank){
    // 数据输入函数，只在根进程上读取
    
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin>>*n>>*p;
    for (int i = 0; i < *n; i++){
        fin>>a[i];
    }
    for (int i = 0; i < *n; i++){   
        fin>>b[i];
    }
    

    // 广播n和p的值给所有进程
    MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(p, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void fCheck(int *ab, int n, int input_id, int rank){
    // 判断多项式乘法结果是否正确，只在根进程上执行
    
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin>>x;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
    
}

void fWrite(int *ab, int n, int input_id, int rank){
    // 数据输出函数, 只在根进程上执行
    
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++){
        fout<<ab[i]<<'\n';
    }
    
}

int a[300000], b[300000], ab[300000];

int main(int argc, char *argv[])
{
    // 初始化MPI环境
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 创建输出目录
    if (rank == 0) {
        if (system("mkdir -p files") != 0) {
            std::cerr << "创建输出目录失败" << std::endl;
        }
    }

    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 263882790666241
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
    int test_begin = 0;
    int test_end = 4;

    for(int i = test_begin; i <= test_end; ++i){
        double total_time = 0.0;
        int n_, p_;
        long double ans = 0;

        // 清空结果数组
        if (rank == 0) {
            memset(ab, 0, sizeof(ab));
        }

        // 读取输入数据
        fRead(a, b, &n_, &p_, i, rank);

        // 同步所有进程
        MPI_Barrier(MPI_COMM_WORLD);

        // 记录开始时间
        double start_time = MPI_Wtime();

        // 执行MPI版本的多项式乘法
        poly_multiply(a, b, ab, n_, p_, MPI_COMM_WORLD);

        // 记录结束时间
        double end_time = MPI_Wtime();
        double elapsed_time = (end_time - start_time) * 1000; // 转换为毫秒
        ans += elapsed_time; // 转换为微秒

        // 收集所有进程的时间，计算平均值
        MPI_Reduce(&elapsed_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // 验证结果并输出
        if (rank == 0) {
            fCheck(ab, n_, i, rank);
            // std::cout << "MPI进程数: " << size << ", n = " << n_ << ", p = " << p_
            //           << ", 执行时间: " << total_time << " ms" << std::endl;
            std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
            fWrite(ab, n_, i, rank);
        }
    }

    // 结束MPI环境
    MPI_Finalize();
    return 0;
}