// Matlab-mex
#include "mex.h"
// C
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <stdbool.h>

typedef double data_type;

// 小于(等于号不可以丢)
bool less_than(data_type l, data_type r) {
    return l <= r;
}

// 大于(等于号不可以丢)
bool greater_than(data_type l, data_type r) {
    return l >= r;
}

// 其实可以只写一个一维的函数, 水平和竖直方向, 只要把第一次滤波的结果取 T, 就可以了, 用 Matlab 的话, 代码可以简洁一点
// Matlab 矩阵是按照列存储的
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs < 2) {
        mexErrMsgIdAndTxt("error : ", "args: (image, kernel_size, use_min(optional))");
        return;
    } 
    bool use_min = true;       // 默认用最小值滤波
    int H = mxGetM(prhs[0]);   // 获取图像高
    int W = mxGetN(prhs[0]);   // 获取图像宽
    int kernel_size = mxGetScalar(prhs[1]);  // 获取滤波核的边长
    if(nrhs > 2) use_min = mxGetScalar(prhs[2]); // 如果有第三个参数, 则用第三个参数
    
    printf("execute fast minimum filtering for image with  (%d X %d) using a kernel of (%d X %d) \n", H, W, kernel_size, kernel_size);

    // 获取被滤波图像的指针
    double* src = mxGetPr(prhs[0]);

    // 获取中间参数(如果是最大值滤波, 把下面的 less_than 改成 greater_than; 然后 EXTREMUM 取反
    int radius = (kernel_size - 1) >> 1;
    bool (*comp)(data_type, data_type) = use_min ? less_than: greater_than;
    double EXTREMUM = use_min ? 1.79769313486231570e+308: -1.79769313486231570e+308;
    
    // 对数据做 padding(注意数据是按照列存储的)
    int H2 = H + 2 * radius;
    int W2 = W + 2 * radius;
    int pad_size = H2 > W2 ? H2 : W2;
    data_type buffer[pad_size];
    for(int i = 0; i < pad_size; ++i)  
        buffer[i] = EXTREMUM;

    // 声明单调队列的参数
    int win_len = kernel_size;
    int front = 0;
    int back = 0;
    int capacity = kernel_size + 1;
    int Q[kernel_size];

    // 准备一个结果
    plhs[0] = mxCreateDoubleMatrix((mwSize)H, (mwSize)W, mxREAL);
    double* result = mxGetPr(plhs[0]);

    // 先做竖直方向的最小值滤波
    for(int i = 0; i < W; ++i) {
        // 把这一列数据拷贝到缓冲区 buffer 中
        memcpy(buffer + radius, src + i * H, sizeof(data_type) * H);
        data_type* res_ptr = result + i * H;
        // 初始化单调队列
        front = back = 0;
        // 先放第一个元素
        back = (back + 1) % capacity;
        Q[back] = 0;
        // 接下来移动窗口, 记录窗口内的单调递减序列
        for(int j = 1; j < H2; ++j) {
            if(j >= kernel_size)
                res_ptr[j - kernel_size] = buffer[Q[(front + 1) % capacity]];
            // 如果当前元素比前一个元素小, 则把单调队列中, 大于当前元素 data[i] 的都 pop 掉
            int pos = j;
            while(front != back) {
                int tail = Q[back];
                if(comp(buffer[pos], buffer[tail]))  // 如果队列中都比当前元素小, 停止
                    back = (back - 1 + capacity) % capacity;
                else break; // 否则, 把队列中大于当前元素的 popback 掉
            }
            // 当前元素的坐标放到这里
            back = (back + 1) % capacity;
            Q[back] = pos;
            // 如果当前维护的区间长度超出了窗口
            int front_next = (front + 1) % capacity;
            if(pos - Q[front_next] == win_len)
                front = front_next;
        }
        res_ptr[H2 - kernel_size] = buffer[Q[(front + 1) % capacity]];    
    }

    // 缓冲区重新 padding
    for(int i = 0; i < pad_size; ++i)  buffer[i] = EXTREMUM;

    // 做水平方向的最小值滤波
    win_len = kernel_size; // 这里不用乘以 W
    for(int i = 0; i < H; ++i) {
        // 把这一行数据拷贝到 buffer 中
        for(int k = 0; k < W; ++k) 
            buffer[radius + k] = result[i + k * H];
        // 找到这一行被滤波数据的起点
        data_type* res_ptr = result + i;
        // 重置单调队列的数据指针到这一列 
        front = back = 0;
        // 放第一个元素
        back = (back + 1) % capacity;
        Q[back] = 0;
        for(int j = 1; j < W2; ++j) {
            if(j >= kernel_size) 
                res_ptr[(j - kernel_size) * H] = buffer[Q[(front + 1) % capacity]];
            int pos = j;
            while(front != back) {
                int tail = Q[back];
                if(comp(buffer[pos], buffer[tail]))  // 如果队列中都比当前元素小, 停止
                    back = (back - 1 + capacity) % capacity;
                else break; // 否则, 把队列中大于当前元素的 popback 掉
            }
            // 当前元素的坐标放到这里
            back = (back + 1) % capacity;
            Q[back] = pos;
            // 如果当前维护的区间长度超出了窗口
            int front_next = (front + 1) % capacity;
            if(pos - Q[front_next] == win_len)
                front = front_next;
        }
        res_ptr[(W2 - kernel_size) * H] = buffer[Q[(front + 1) % capacity]];   
    }

}