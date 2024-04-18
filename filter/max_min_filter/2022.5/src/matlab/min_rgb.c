// Matlab-mex
#include "mex.h"
// C
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <string.h>




// 其实可以只写一个一维的函数, 水平和竖直方向, 只要把第一次滤波的结果取 T, 就可以了, 用 Matlab 的话, 代码可以简洁一点
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs < 1) {
        mexErrMsgIdAndTxt("erro : ", "args: (image, channels)");
        return;
    }
    size_t C = nrhs > 1 ? mxGetScalar(prhs[1]): 3;
    size_t H = mxGetM(prhs[0]);
    size_t W = mxGetN(prhs[0]) / C;

    printf("compute minmum of rgb for image with (%d X %d X %d)\n", H, W, C);
    
    // 获取被滤波图像的指针
    double* src = mxGetPr(prhs[0]);

    // 准备一个结果
    plhs[0] = mxCreateDoubleMatrix((mwSize)H, (mwSize)W, mxREAL);
    double* result = mxGetPr(plhs[0]);

    size_t length = H * W;
    size_t length_2 = 2 * length;
    for(size_t i = 0; i < length; ++i) 
    	result[i] = fmin(src[i], fmin(src[i + length], src[i + length_2]));
}
