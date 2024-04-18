#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>


void compute_dark(float* pad_ptr, float* res_ptr, const int H2, const int W2, const int radius) {
	// 算原来的高和宽
    const int H = H2 - 2 * radius;
    const int W = W2 - 2 * radius;
    // 准备一个临时结果
    float* temp_ptr = (float*)malloc(sizeof(float) * H2 * W);
    // 开始最小值滤波
    int cnt = 0;
    for(int i = 0;i < H2; ++i) {
        float* row_ptr = pad_ptr + i * W2 + radius;
        for(int j = 0;j < W; ++j) {
            float min_value = 1e7;
            for(int k = -radius; k <= radius; ++k)
                min_value = fmin(min_value, row_ptr[j + k]);
            temp_ptr[cnt++] = min_value;
        }
    }
    for(int j = 0;j < W; ++j) {
        for(int i = 0;i < H; ++i) {
            float min_value = 1e7;
            const int offset = (radius + i) * W + j;
            for(int k = -radius; k <= radius; ++k)
                min_value = fmin(min_value, temp_ptr[offset + k * W]);
            res_ptr[i * W + j] = min_value; 
        }
    }
    free(temp_ptr);
    temp_ptr = NULL;
}