#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>


// 找最小值
float min_in_array(float* data_ptr, int length) {
    if(length < 1) return 0;
    float min_value = data_ptr[0];
    for(int i = 1;i < length; ++i)
        if(data_ptr[i] < min_value)
            min_value = data_ptr[i];
    return min_value;
}

// 找最大值
float max_in_array(float* data_ptr, int length) {
    if(length < 1) return 0;
    float max_value = data_ptr[0];
    for(int i = 1;i < length; ++i)
        if(data_ptr[i] > max_value)
            max_value = data_ptr[i];
    return max_value;
}

float* assign_memory(int length) {
    int byte_num = sizeof(float) * length;
    float* found_memory = (float*)malloc(byte_num);
    memset(found_memory, 0, byte_num);
    return found_memory;
}

// 交换两个指针
void swap_pointer(float **a, float **b) {
    float *tmp = *a;
    *a = *b;
    *b = tmp;
}

// 截断函数
float clip_float(float x, float low, float high) {
    if(x < low) x = low;
    else if(x > high) x = high;
    return x;
}

// 截断函数
int clip_int(int x, int low, int high) {
    if(x < low) x = low;
    else if(x > high) x = high;
    return x;
}

// 求网格坐标 (_x, _y, _z) 在网格中的地址(偏移量)
int get_index(int _x, int _y, int _z, int grid_width, int grid_value) {
    return (_x * grid_width + _y) * grid_value + _z;
}

// 根据非整数的映射坐标 (x, y, z), 在 wi_grid 和 w_grid 中分别做插值, 插值结果归一化
float trilinear_interpolate(
        float* wi_grid,
        float* w_grid,
        float x, float y, float z,
        int grid_height, int grid_width, int grid_value) {
    // 计算这个小数坐标 (x, y, z) 在网格中, 在三个方向上的上界和下界
    int x_down = clip_int(floor(x), 0, grid_height - 1);
    int x_up   = clip_int(x_down + 1, 0, grid_height - 1);
    int y_down = clip_int(floor(y), 0, grid_width - 1);
    int y_up   = clip_int(y_down + 1, 0, grid_width - 1);
    int z_down = clip_int(floor(z), 0, grid_value - 1);
    int z_up   = clip_int(z_down + 1, 0, grid_value - 1);
    // 获取这个小数坐标在 x, y, z 方向上的权重量
    float x_weight = abs(x - x_down);
    float y_weight = abs(y - y_down);
    float z_weight = abs(z - z_down);
    // 准备立方体 8 个点坐标对应的偏移量
    int offsets[8] = {
        get_index(x_down, y_down, z_down, grid_width, grid_value),
        get_index(x_up,   y_down, z_down, grid_width, grid_value),
        get_index(x_down, y_up,   z_down, grid_width, grid_value),
        get_index(x_down, y_down, z_up, grid_width, grid_value),
        get_index(x_up,   y_up,   z_down, grid_width, grid_value),
        get_index(x_up,   y_down, z_up, grid_width, grid_value),
        get_index(x_down, y_up,   z_up, grid_width, grid_value),
        get_index(x_up,   y_up,   z_up, grid_width, grid_value)
    };
    // 准备立方体 8 个点坐标对应的加权值
    float weights[8] = {
        (1.f - x_weight) * (1.f - y_weight) * (1.f - z_weight),
        x_weight         * (1.f - y_weight) * (1.f - z_weight),
        (1.f - x_weight) * y_weight         * (1.f - z_weight),
        (1.f - x_weight) * (1.f - y_weight) * z_weight,
        x_weight         * y_weight         * (1.f - z_weight),
        x_weight         * (1.f - y_weight) * z_weight,
        (1.f - x_weight) * y_weight         * z_weight,
        x_weight         * y_weight         * z_weight
    };
    // 两个网格的插值共用一套加权参数
    float wi_interpolated = 0.f;
    for(int i = 0;i < 8; ++i) wi_interpolated += weights[i] * wi_grid[offsets[i]];
    float w_interpolated = 0.f;
    for(int i = 0;i < 8; ++i) w_interpolated += weights[i] * w_grid[offsets[i]];
    // 插值结果相除, 归一化
    return wi_interpolated / w_interpolated;
}


void fast_bilateral_approximation(
        float* res_ptr, float* input_ptr, float* refer_ptr, 
        int H, int W, 
        float spatial_sample, float range_sample, int grid_padding) {
    // 【1】********************** 收集图像信息 **********************
    int length = H * W;

    // 【2】********************** 根据图像的宽高, 亮度构建双边网格 **********************
    // 计算图像中的取值范围, 用于定义值域网格
    float range_min = min_in_array(refer_ptr, length);
    float range_max = max_in_array(refer_ptr, length);
    float range_interval = range_max - range_min;
    {
        printf("intensity  :  [%f, %f]\n", range_min, range_max);
    }
    // 决定下采样网格的大小
    int grid_height = floor((H - 1) / spatial_sample) + 1 + 2 * grid_padding;
    int grid_width = floor((W - 1) / spatial_sample) + 1 + 2 * grid_padding;
    int grid_value = floor(range_interval / range_sample) + 1 + 2 * grid_padding;
    // 创建 grid, 一个是分母的加权部分, 另一个是分子(齐次的 1)
    int grid_size = grid_height * grid_width * grid_value;
    float* wi_grid = assign_memory(grid_size);
    float* w_grid = assign_memory(grid_size);
    {
        printf("grid  :  \n");
        printf("\theight     :  %d\n", grid_height);
        printf("\twidth      :  %d\n", grid_width);
        printf("\tvalue      :  %d\n", grid_value);
        printf("\tgrid_size  :  %d\n", grid_size);
    }
    // 根据参考图像的信息, 将输入图像下采样填充到 grid 网格中
    for(int i = 0;i < H; ++i) {
        int x = floor(i / spatial_sample) + grid_padding + 1;  // 图像第 i 行映射到网格中的坐标
        float* I_ptr = input_ptr + i * W;  // 输入图象在第 i 行的指针
        float* R_ptr = refer_ptr + i * W;           // 参考图像在第 i 行的指针
        for(int j = 0;j < W; ++j) {
            int y = floor(j / spatial_sample) + grid_padding + 1;  // 图像第 j 列映射到网格中的坐标
            int z = floor((R_ptr[j] - range_min) * 1.f / range_sample) + grid_padding + 1;  // 图像中点 (i,j) 的亮度值映射到网格的 z 维的坐标
            int grid_pos = (x * grid_width + y) * grid_value + z;
            wi_grid[grid_pos] += I_ptr[j];
            w_grid[grid_pos] += 1;
        }
    }
    {
        printf("filling values to gird is completed\n");
    }


    // ********************** 在网格上做卷积, 低通滤波 **********************
    int offset[3] = {grid_width * grid_value, grid_value, 1};
    float* wi_grid_buffer = assign_memory(grid_size);
    float* w_grid_buffer = assign_memory(grid_size);

    for(int dimension = 0;dimension < 3; ++dimension) {
        int _offset = offset[dimension];  // 当前维度 +1, -1 在网格中的偏移量
        for(int iter = 0;iter < 4; ++iter) {     // 实际半径为 2 倍的 1, 下面的滤波半径都是 1
            swap_pointer(&wi_grid, &wi_grid_buffer);
            swap_pointer(&w_grid, &w_grid_buffer);       // 这个交换很巧妙, 第一次卷积的结果存放在 buffer, 第二次从 buffer 中再卷积一次放在网格中
            // 开始三维卷积
            for(int i = 1, i_MAX = grid_height - 1; i < i_MAX; ++i) {
                for(int j = 1, j_MAX = grid_width - 1; j < j_MAX; ++j) {
                    int start = (i * grid_width + j) * grid_value; // 当前网格在第(i, j)个格子的偏移量
                    float* wi = wi_grid + start;
                    float* wi_buf = wi_grid_buffer + start;       // 加权的网格 和 它的上一次卷积结果, 在第(i, j)个格子的偏移地址
                    float* w = w_grid + start;
                    float* w_buf = w_grid_buffer + start;         // 齐次的网格 和 它的上一次卷积结果, 在第(i, j)个格子的偏移地址
                    for(int k = 1, k_MAX = grid_value - 1; k < k_MAX; ++k) {
                        // 每次卷积, dimension 这个维度上前一个像素 + 后一个像素 和 当前像素做加权平均, 平滑
                        wi[k] = 0.25 * (2.0 * wi_buf[k] + wi_buf[k - _offset] + wi_buf[k + _offset]);
                        w[k] = 0.25 * (2.0 * w_buf[k] + w_buf[k - _offset] + w_buf[k + _offset]);
                    }
                }
            }
        }
    }
    {
        printf("low-pass convolution on grid is completed!\n");
    }


    // ********************** 网格做了低通滤波之后, 根据参考图从网格中插值得到每一个目标点的值 **********************
    int cnt = 0;
    for(int i = 0;i < H; ++i) {
        for(int j = 0;j < W; ++j) {
            // 计算这个点在网格中的坐标
            float x = i * 1.f / spatial_sample + grid_padding;
            float y = j * 1.f / spatial_sample + grid_padding;
            float z = (refer_ptr[i * W + j] - range_min) / range_sample + grid_padding;
            // 三次线性插值, 两个分支
            float interp_res = trilinear_interpolate(wi_grid, w_grid, x, y, z, grid_height, grid_width, grid_value);
            // wi / w 是最终的加权结果
            res_ptr[cnt++] = clip_float(interp_res, 0, 1);
        }
    }
    {
        printf("interpolation over !\n");
    }

    // 如果是视频的话, 这里的内存分配和销毁可以放到一个类里, 当作缓存, 没必要运行一次就分配一次内存, 程序改成 C++
    free(wi_grid_buffer);
    free(w_grid_buffer);
    free(wi_grid);
    free(w_grid);
    wi_grid = w_grid = wi_grid_buffer = w_grid_buffer = NULL;
}

