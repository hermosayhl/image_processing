// C++
#include <assert.h>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
// OpenCV
#include <opencv2/opencv.hpp>




namespace {
    void run(const std::function<void()>& work=[]{}, const std::string message="") {
        auto start = std::chrono::steady_clock::now();
        work();
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << message << " " << duration.count() << " ms" <<  std::endl;
    }
    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    cv::Mat cv_concat(const std::vector<cv::Mat>& images){
        cv::Mat display;
        cv::hconcat(images, display);
        return display;
    }
    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    template<typename T>
    T min_in_array(const T* const data_ptr, const int length) {
        if(length < 1) return 0;
        T min_value = data_ptr[0];
        for(int i = 1;i < length; ++i)
            if(data_ptr[i] < min_value)
                min_value = data_ptr[i];
        return min_value;
    }
    template<typename T>
    T max_in_array(const T* const data_ptr, const int length) {
        if(length < 1) return 0;
        T max_value = data_ptr[0];
        for(int i = 1;i < length; ++i)
            if(data_ptr[i] > max_value)
                max_value = data_ptr[i];
        return max_value;
    }

    template<typename T>
    inline T clip(T x, T down, T up) {
        if(x < down) x = down;
        else if(x > up) x = up;
        return x;
    }
}



cv::Mat fast_bilateral_approximation(
        const cv::Mat& input,
        const cv::Mat& refer,
        // 采样率
        const float spatial_sample=8,
        const float range_sample=0.1,
        // 网格做滤波时候的半径
        const int grid_padding=2,
        // 打印信息
        const bool verbose=true) {

    using pointer_type = const float* const;

    // 【1】********************** 收集图像信息 **********************
    const int H = input.rows;
    const int W = input.cols;
    const int length = H * W;
    assert(H == refer.rows and W == refer.cols and "shapes of input and refer must be the same");
    assert(input.channels() == 1 and refer.channels() == 1 and "only gray images are supported");

    // 【2】********************** 根据图像的宽高, 亮度构建双边网格 **********************
    // 计算图像中的取值范围, 用于定义值域网格
    pointer_type refer_ptr = refer.ptr<float>();
    const float range_min = min_in_array(refer_ptr, length);
    const float range_max = max_in_array(refer_ptr, length);
    const float range_interval = range_max - range_min;
    if(verbose) {
        std::cout << "intensity    :  [" << range_min << ", " << range_max << "]" << std::endl;
    }
    // 决定下采样网格的大小
    const int grid_height = std::floor((H - 1) / spatial_sample) + 1 + 2 * grid_padding;
    const int grid_width = std::floor((W - 1) / spatial_sample) + 1 + 2 * grid_padding;
    const int grid_value = std::floor(range_interval / range_sample) + 1 + 2 * grid_padding;
    // 创建 grid, 一个是分母的加权部分, 另一个是分子(齐次的 1)
    const int grid_size = grid_height * grid_width * grid_value;
    std::vector<float> wi_grid(grid_size, 0);
    std::vector<float> w_grid(grid_size, 0);
    if(verbose) {
        std::cout << "grid  :  \n";
        std::cout << "\theight     :  " << grid_height << std::endl;
        std::cout << "\twidth      :  " << grid_width << std::endl;
        std::cout << "\tvalue      :  " << grid_value << std::endl;
        std::cout << "\tgrid_size  :  "<< grid_size << std::endl;
    }
    // 根据参考图像的信息, 将输入图像下采样填充到 grid 网格中
    for(int i = 0;i < H; ++i) {
        const int x = std::floor(i / spatial_sample) + grid_padding + 1;  // 图像第 i 行映射到网格中的坐标
        pointer_type I_ptr = input.ptr<float>() + i * W;  // 输入图象在第 i 行的指针
        pointer_type R_ptr = refer_ptr + i * W;           // 参考图像在第 i 行的指针
        for(int j = 0;j < W; ++j) {
            const int y = std::floor(j / spatial_sample) + grid_padding + 1;  // 图像第 j 列映射到网格中的坐标
            const int z = std::floor((R_ptr[j] - range_min) * 1.f / range_sample) + grid_padding + 1;  // 图像中点 (i,j) 的亮度值映射到网格的 z 维的坐标
            const int grid_pos = (x * grid_width + y) * grid_value + z;
            wi_grid[grid_pos] += I_ptr[j];
            w_grid[grid_pos] += 1;
        }
    }
    if(verbose) {
        int effective_count = 0;
        for(int i = 0;i < grid_size; ++i)
            if(w_grid[i] > 0) ++effective_count;
        std::cout << "filling proportion of the grid is  " << effective_count * 1.f / grid_size << std::endl;
    }

    // ********************** 在网格上做卷积, 低通滤波 **********************
    std::vector<int> offset({grid_width * grid_value, grid_value, 1});
    std::vector<float> wi_grid_buffer(grid_size, 0); // 用于存储多维分离卷积的上一次结果
    std::vector<float> w_grid_buffer(grid_size, 0);
    for(int dimension = 0;dimension < 3; ++dimension) {
        const int _offset = offset[dimension];  // 当前维度 +1, -1 在网格中的偏移量
        for(int iter = 0;iter < 4; ++iter) {     // 实际半径为 2 倍的 1, 下面的滤波半径都是 1
            wi_grid.swap(wi_grid_buffer);
            w_grid.swap(w_grid_buffer);       // 这个交换很巧妙, 第一次卷积的结果存放在 buffer, 第二次从 buffer 中再卷积一次放在网格中
            // 开始三维卷积
            for(int i = 1, i_MAX = grid_height - 1; i < i_MAX; ++i) {
                for(int j = 1, j_MAX = grid_width - 1; j < j_MAX; ++j) {
                    const int start = (i * grid_width + j) * grid_value; // 当前网格在第(i, j)个格子的偏移量
                    float* wi = wi_grid.data() + start;
                    float* wi_buf = wi_grid_buffer.data() + start;       // 加权的网格 和 它的上一次卷积结果, 在第(i, j)个格子的偏移地址
                    float* w = w_grid.data() + start;
                    float* w_buf = w_grid_buffer.data() + start;         // 齐次的网格 和 它的上一次卷积结果, 在第(i, j)个格子的偏移地址
                    for(int k = 1, k_MAX = grid_value - 1; k < k_MAX; ++k) {
                        // 每次卷积, dimension 这个维度上前一个像素 + 后一个像素 和 当前像素做加权平均, 平滑
                        wi[k] = 0.25 * (2.0 * wi_buf[k] + wi_buf[k - _offset] + wi_buf[k + _offset]);
                        w[k] = 0.25 * (2.0 * w_buf[k] + w_buf[k - _offset] + w_buf[k + _offset]);
                    }
                }
            }
        }
    }
    if(verbose) {
        std::cout << "low-pass convolution on grid is completed" << std::endl;
    }

    // ********************** 网格做了低通滤波之后, 根据参考图从网格中插值得到每一个目标点的值 **********************
    auto trilinear_interpolate = [](
            const std::vector<float>& wi_grid,
            const std::vector<float>& w_grid,
            const float x, const float y, const float z,
            std::vector<int> border) ->float {
        // 计算这个小数坐标 (x, y, z) 在网格中, 在三个方向上的上界和下界
        const int x_down = clip<int>(std::floor(x), 0, border[0] - 1);
        const int x_up   = clip<int>(x_down + 1, 0, border[0] - 1);
        const int y_down = clip<int>(std::floor(y), 0, border[1] - 1);
        const int y_up   = clip<int>(y_down + 1, 0, border[1] - 1);
        const int z_down = clip<int>(std::floor(z), 0, border[2] - 1);
        const int z_up   = clip<int>(z_down + 1, 0, border[2] - 1);
        // 获取这个小数坐标在 x, y, z 方向上的权重量
        const float x_weight = std::abs(x - x_down);
        const float y_weight = std::abs(y - y_down);
        const float z_weight = std::abs(z - z_down);
        // 计算 (__x, __y, __z) 在网格中的偏移地址
        auto index = [&](const int _x, const int _y, const int _z) ->int {
            return (_x * border[1] + _y) * border[2] + _z;
        };
        // 准备立方体 8 个点坐标对应的偏移量
        std::vector<int> offsets = {
            index(x_down, y_down, z_down),
            index(x_up,   y_down, z_down),
            index(x_down, y_up,   z_down),
            index(x_down, y_down, z_up),
            index(x_up,   y_up,   z_down),
            index(x_up,   y_down, z_up),
            index(x_down, y_up,   z_up),
            index(x_up,   y_up,   z_up)
        };
        // 准备立方体 8 个点坐标对应的加权值
        std::vector<float> weights = {
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
    };

    int cnt = 0;
    cv::Mat result(H, W, CV_8UC1);  // 准备一个结果
    uchar* const res_ptr = result.ptr<uchar>();
    for(int i = 0;i < H; ++i) {
        for(int j = 0;j < W; ++j) {
            // 计算这个点在网格中的坐标
            const float x = i * 1.f / spatial_sample + grid_padding;
            const float y = j * 1.f / spatial_sample + grid_padding;
            const float z = (refer_ptr[i * W + j] - range_min) / range_sample + grid_padding;
            // 三次线性插值, 两个分支
            float interp_res = trilinear_interpolate(wi_grid, w_grid, x, y, z, {grid_height, grid_width, grid_value});
            // wi / w 是最终的加权结果
            res_ptr[cnt++] = cv::saturate_cast<uchar>(255 * interp_res);
        }
    }
    return result;
}


int main() {
    std::setbuf(stdout, 0);

    // 读取图像
    cv::Mat color_image = cv::imread("./images/input/greekdome.ppm");
    assert(not color_image.empty() and "failed to load image");

    // 单通道
    if(true) {
        // 转化成灰度图
        cv::Mat gray_image;
        cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);
        // 转成 float 数据
        gray_image.convertTo(gray_image, CV_32FC1);
        gray_image /= 255;
        // 开始快速滤波
        cv::Mat result;
        run([&](){
            result = fast_bilateral_approximation(
                gray_image, gray_image,
                4, 0.05,
                2, true
            );
        }, "fast bilateral approximation using sampling grid");
        cv_show(result);
    }
    else {
        // 拆成三通道
        std::vector<cv::Mat> bgr_channels;
        cv::split(color_image, bgr_channels);
        // 存放结果
        std::vector<cv::Mat> result_temp;
        result_temp.reserve(3);
        for(auto& channel : bgr_channels) {
            channel.convertTo(channel, CV_32FC1);
            channel /= 255;
            result_temp.emplace_back(fast_bilateral_approximation(
                channel, channel, 4, 0.05, 1, false
            ));
        }
        // 合并成多通道
        cv::Mat result;
        cv::merge(result_temp, result);
        cv_show(cv_concat({color_image, result}));
        cv_write(result, "./images/output/result.png");
    }

    return 0;
}



