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

// 这个 1 维的其实跟 2 维的一次滤波思路类似, 但也有一点点差别, 这个一维的只在被滤波的维度上做 padding
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs < 2) {
        mexErrMsgIdAndTxt("error : ", "args: (image, kernel_size)");
        return;
    }
    int H = mxGetM(prhs[0]);
    int W = mxGetN(prhs[0]);
    int kernel_size = mxGetScalar(prhs[1]);
    bool use_min = true;
    if(nrhs > 2) use_min = mxGetScalar(prhs[2]);
    
    printf("execute fast minimum filtering for image with  (%d X %d) using a kernel of (1 X %d) \n", H, W, kernel_size);

    // 获取被滤波图像的指针
    double* src = mxGetPr(prhs[0]);
    const int length = H * W;

    // 获取中间参数(如果是最大值滤波, 把下面的 less_than 改成 greater_than; 然后 EXTREMUM 取反
    int radius = (kernel_size - 1) >> 1;
    bool (*comp)(data_type, data_type) = use_min ? less_than: greater_than;
    double EXTREMUM = use_min ? 1.79769313486231570e+308: -1.79769313486231570e+308;
    
    // 对数据做 padding(注意数据是按照列存储的)
    int H2 = H + 2 * radius;
    int pad_size = H2 * W;
    data_type* padded_data = (data_type*)malloc(sizeof(data_type) * pad_size);
    for(int i = 0; i < pad_size; ++i)  padded_data[i] = EXTREMUM;
    for(int i = 0; i < W; ++i)
        memcpy(padded_data + i * H2 + radius, src + i * H, sizeof(data_type) * H);

    // 声明单调队列的参数
    int win_len = kernel_size;
    int front = 0;
    int back = 0;
    int capacity = kernel_size + 1;
    int* Q = (int*)malloc(sizeof(int) * kernel_size);

    // 先做竖直方向的最小值滤波
    plhs[0] = mxCreateDoubleMatrix((mwSize)H, (mwSize)W, mxREAL);
    double* temp = mxGetPr(plhs[0]);

    for(int i = 0; i < W; ++i) {
        data_type* col_ptr = padded_data + i * H2;
        data_type* res_ptr = temp + i * H;
        // 初始化单调队列
        front = back = 0;
        // 先放第一个元素
        back = (back + 1) % capacity;
        Q[back] = 0;
        // 接下来移动窗口, 记录窗口内的单调递减序列
        for(int j = 1; j < H2; ++j) {
            if(j >= kernel_size)
                res_ptr[j - kernel_size] = col_ptr[Q[(front + 1) % capacity]];
            // 如果当前元素比前一个元素小, 则把单调队列中, 大于当前元素 data[i] 的都 pop 掉
            int pos = j;
            while(front != back) {
                int back_index = Q[back];
                if(comp(col_ptr[pos], col_ptr[back_index]))  // 如果队列中都比当前元素小, 停止
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
        res_ptr[H2 - kernel_size] = col_ptr[Q[(front + 1) % capacity]];    
    }

    // 释放中间变量内存
    free(Q);
    free(padded_data);
}