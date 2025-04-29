#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cstring>
#include <windows.h>  // 添加Windows高精度计时头文件
#include <stdlib.h>

using namespace std;
using namespace chrono;

// 矩阵向量内积的平凡算法（行优先访问）
void mat_vec(double** matrix, double* vec, double* result, int n) {
    for (int i = 0; i < n; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            result[i] += matrix[j][i] * vec[j];
        }
    }
}

// 矩阵向量内积的优化算法（行优先访问）
void optimized_mat_vec(double** matrix, double* vec, double* result, int n) {
    memset(result, 0, n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
}

// 求和的平凡累加
double normal_sum(double* arr, int n, double sum) {
    for (int i = 0; i < n; i += 1) {
        sum += arr[i];
    }
    return sum;
}

// 求和的两路并行累加
double optimized_sum(double* arr, int n, double sum) {
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; i += 2) {
        sum1 += arr[i];
        if (i + 1 < n) sum2 += arr[i + 1];
    }
    sum = sum1 + sum2;
    return sum;
}

// 求和的递归累加
// double recursion_sum(double* arr, int n, double sum) {
//     if(n == 1 || n == 0){
//         return sum;
//     }
//     else{
//         for(int i = 0; i < n/2; i++){
//             sum = arr[i] + arr[n-i-1];
//         }
//         n = n/2;
//         return recursion_sum(arr, n, sum);
//     }
// }
//double recursion_sum(double* arr, int n, double sum) {
//    if(n == 0) return 0.0;  // 处理边界情况
//    if(n == 1) return sum + arr[0];
//    double temp = 0.0;
//    for(int i = 0; i < n/2; i++){
//        temp += arr[i] + arr[n - i - 1];
//    }
//    n = n/2;
//    return recursion_sum(arr, n, sum + temp);  // ✅ 累加中间结果
//}

double recursion_sum(double* arr, int n, double sum) {
    if (n == 0) return sum;           // 若数组为空，直接返回累加器
    if (n == 1) return sum + arr[0];    // 基础情况：仅有1个元素

    // 计算新数组的大小：如果 n 为偶数，则为 n/2；如果为奇数，则多一个中间元素
    int new_size = n / 2 + (n % 2);
    double* new_arr = new double[new_size];

    // 对称配对求和：前半部分与后半部分对称元素相加
    for (int i = 0; i < n / 2; i++) {
        new_arr[i] = arr[i] + arr[n - i - 1];
    }
    // 如果是奇数，保留中间的那个元素
    if (n % 2 != 0) {
        new_arr[n / 2] = arr[n / 2];
    }

    // 递归求新数组的总和，注意新调用时将累加器置为0，
    // 最后将本层累加器 sum 与递归结果相加返回
    double result = recursion_sum(new_arr, new_size, 0.0);
    delete[] new_arr;
    return sum + result;
}



// 求和的二重循环累加
double sec_roll_sum(double* arr, int n, double sum) {
    vector<double> summ(n);
    copy(arr, arr + n, summ.begin());
    for (int m = n; m > 1; m /= 2) {
        for (int i = 0; i < m / 2; i++) {
            summ[i] = summ[i] + summ[m - i - 1];
        }
    }
    sum = summ[0];
    return sum;
}

// 生成全索引数组
double* generate_vec(int size) {
    double* arr = new double[size];
    for (int i = 0; i < size; i++) {
        arr[i] = i;
    }
    return arr;
}

// 生成矩阵（元素值等于行索引 + 列索引）
double** generate_matrix(int size) {
    double** matrix = new double* [size];
    for (int i = 0; i < size; ++i) {
        matrix[i] = new double[size];
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = i + j;
        }
    }
    return matrix;
}

// 验证两个向量是否相等
bool verify_mat_vec_result(double* result1, double* result2, int n) {
    const double eps = 1e-9;
    for (int i = 0; i < n; ++i) {
        if (std::abs(result1[i] - result2[i]) > eps) {
            return false;
        }
    }
    return true;
}

// 矩阵-向量乘法测试
template <typename Func>
void test_mat_vec(Func func, ofstream& file) {
    vector<int> test_sizes = { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
    //vector<int> test_sizes = {8, 16, 32, 64};
    //const int reps = 100;
    const int Reps[10] = { 51200, 25600, 12800, 6400, 3200, 1600, 800, 400, 200, 100 };

    file << "Matrix-Vector Multiplication Test\n";
    file << "Size,Time (μs)\n";

    //int n : test_sizes
    for (int i = 0; i < 10; i++) {
        int n = test_sizes[i];
        int reps = Reps[i];
        double** matrix = generate_matrix(n);
        double* vec = generate_vec(n);
        double* res = new double[n];
        double* ref_res = new double[n];

        LARGE_INTEGER freq,head,tail;

        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);

        // 计算平凡算法的结果
        mat_vec(matrix, vec, ref_res, n);

        // 时间测量部分修改
        //using big_microseconds = chrono::duration<long, std::micro>;  // 定义更大的时间类型
        //auto start = high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) {
            func(matrix, vec, res, n);
        }
        //auto end = high_resolution_clock::now();
        //auto time_taken = duration_cast<big_microseconds>(end - start).count() / reps;

        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        auto time_taken = (tail.QuadPart - head.QuadPart) * 1000000.0 / freq.QuadPart;
        cout << "Col: " << time_taken << "us" << endl;
        bool is_correct = verify_mat_vec_result(res, ref_res, n);
        file << n << "," << reps << "," << time_taken << "," << (is_correct ? "True" : "False") << "\n";



        for (int i = 0; i < n; i++) {
            delete[] matrix[i];
        }
        delete[] matrix;

        delete[] vec;
        delete[] res;
    }
}

// 验证两个标量是否相等
bool verify_sum_result(double result1, double result2) {
    const double eps = 1e-9;
    return std::abs(result1 - result2) <= eps;
}

// 数组求和测试
template <typename Func>
void test_sum(Func func, ofstream& file) {
    vector<int> test_sizes = { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
    //vector<int> test_sizes = {8, 16, 32, 64};
    //const int reps = 100;
    const int Reps[10] = { 51200, 25600, 12800, 6400, 3200, 1600, 800, 400, 200, 100 };

    file << "Array Sum Test\n";
    file << "Size,Time (μs)\n";

    //int n : test_sizes
    for (int i = 0; i < 10; i++) {
        int n = test_sizes[i] * 64;
        int reps = Reps[i] * 10;
        double* arr = generate_vec(n);
        double ref_sum = 0.0;
        volatile double dummy0 = normal_sum(arr, n, ref_sum);
        volatile double dummy;
        LARGE_INTEGER freq,head,tail;

        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        // 时间测量部分修改
        //using big_microseconds = chrono::duration<long, std::micro>;  // 定义更大的时间类型
        //auto start = high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) {
            double sum = 0.0;
            dummy = func(arr, n, sum);
        }
        //auto end = high_resolution_clock::now();
        //auto time_taken = duration_cast<big_microseconds>(end - start).count() / reps;

        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        auto time_taken = (tail.QuadPart - head.QuadPart) * 1000000.0 / freq.QuadPart;
        cout << "Col: " << time_taken << "us" << endl;
        bool is_correct = verify_sum_result(dummy, dummy0);
        file << n << "," << reps << "," << time_taken << "," << (is_correct ? "True" : "False") << "\n";


        delete[] arr;
    }
}

int main() {

    cout << "Program 1 results start saving......" << endl;

    ofstream mat_vec_file("matrix_vector_results().csv");
    test_mat_vec(mat_vec, mat_vec_file);
    //cout<<endl;
    //test_mat_vec(optimized_mat_vec, mat_vec_file);
    mat_vec_file.close();

    return 0;
}

/*
#include <iostream>
#include <windows.h>

using namespace std;

void high_precision_timing() {
    LARGE_INTEGER freq, start, end;
    if (!QueryPerformanceFrequency(&freq)) {
        cerr << "Error: High-resolution counter not supported!" << endl;
        return;
    }

    // 绑定线程到 CPU 0（可选）
    DWORD_PTR oldMask = SetThreadAffinityMask(GetCurrentThread(), 0x1);

    QueryPerformanceCounter(&start);

    // 模拟被测代码
    volatile int sum = 0; // 阻止优化
    for (int i = 0; i < 1000000; i++) {
        sum += i;
    }

    QueryPerformanceCounter(&end);

    // 恢复线程关联性（可选）
    SetThreadAffinityMask(GetCurrentThread(), oldMask);

    LONGLONG diff = end.QuadPart - start.QuadPart;
    if (diff < 0) {
        cerr << "Error: Counter wrap-around detected!" << endl;
        return;
    }

    double time_us = diff * 1e6 / freq.QuadPart;
    cout << "Time taken: " << time_us << "us" << endl;
}

int main() {
    high_precision_timing();
    return 0;
}*/
