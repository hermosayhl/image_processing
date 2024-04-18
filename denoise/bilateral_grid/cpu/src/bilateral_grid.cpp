#include "utils.h"

namespace {
    template<typename T>
    inline T square(const T x) {
        return x * x;
    }

    inline double fast_exp(const double y) {
        double d;
        *(reinterpret_cast<int*>(&d) + 0) = 0;
        *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * y + 1072632447);
        return d;
    }

    template<typename T>
    T _fast_exp(T x) {
        x = 1.0 + x / 256;
        for(int i = 0;i < 8; ++i) x *= x;
        return x;
    }
}



cv::Mat bilateral_mean_filtering(
        cv::Mat noisy_image,
        const float spatial_sigma=1.0,
        const int intensity_padding=30) {
    assert(noisy_image.channels() == 1 and noisy_image.type() == CV_8UC1 and "only gray images are supported !");
    // 获取图像信息
    const int H = noisy_image.rows;
    const int W = noisy_image.cols;
    // 获取滤波核参数
    const int spatial_padding = std::ceil(3 * spatial_sigma);
    // 这里需要对图像做 paddding
    cv::copyMakeBorder(noisy_image, noisy_image, spatial_padding, spatial_padding, spatial_padding, spatial_padding, cv::BORDER_REFLECT);
    const int W2 = noisy_image.cols;
    // 准备一个偏移量模板
    int max_k = 0;
    std::vector<int> offset(square<int>(2 * spatial_padding + 1));
    for(int i = -spatial_padding; i <= spatial_padding; ++i)
        for(int j = -spatial_padding; j <= spatial_padding; ++j)
            offset[max_k++] = i * W2 + j;
    // 准备一个结果
    int cnt = 0;
    cv::Mat result(H, W, noisy_image.type());
    uchar* const res_ptr = result.ptr<uchar>();
    // 开始卷积
    for(int i = 0; i < H; ++i) {
        uchar* const row_ptr = noisy_image.ptr<uchar>() + (spatial_padding + i) * W2 + spatial_padding;
        for(int j = 0; j < W; ++j) {
            // 获取这个点的亮度
            int intensity = row_ptr[j];
            // 做低通滤波
            float weight_sum = 0.f;
            float intensity_sum = 0.f;
            for(int k = 0; k < max_k; ++k) {
                const int pos = offset[k];
                // 如果邻域点和当前点的亮度差小于一定程度, 才参与加权
                if(std::abs(row_ptr[j + pos] - intensity) <= intensity_padding) {
                    weight_sum += 1;
                    intensity_sum += row_ptr[j + pos];
                }
                // 注释上面三行, 放下面两行代码, 做朴素的均值滤波
                // weight_sum += 1;
                // intensity_sum += row_ptr[j + pos];
            }
            res_ptr[cnt++] = cv::saturate_cast<uchar>(intensity_sum / weight_sum);
        }
    }
    return result;
}


cv::Mat bilateral_grid_mean_filtering(
        cv::Mat& noisy_image,
        const float spatial_sigma=1.0,
        const int intensity_padding=30,
        const int intensity_level=255) {
    assert(noisy_image.channels() == 1 and noisy_image.type() == CV_8UC1 and "only gray images are supported !");
    // 获取图像信息
    const int H = noisy_image.rows;
    const int W = noisy_image.cols;
    // 获取滤波核参数
    const int spatial_padding = std::ceil(3 * spatial_sigma);
    // 构造一个网格
    const int grid_height = H + 2 * spatial_padding;
    const int grid_width = W + 2 * spatial_padding;
    const int grid_intensity = intensity_level + 2 * intensity_padding;
    const int grid_size = grid_height * grid_width * grid_intensity;
    std::vector<float> grid(grid_size, 0);
    // 把图像信息填充到网格中
    for(int i = 0;i < H; ++i) {
        uchar* const row_ptr = noisy_image.ptr<uchar>() + i * W;
        for(int j = 0;j < W; ++j) {
            int intensity = row_ptr[j];
            grid[((i + spatial_padding) * grid_width + j + spatial_padding) * grid_intensity + intensity + intensity_padding] += intensity;
        }
    }
    // 构造一个偏移量模板
    int max_k = 0;
    std::vector<int> offset((2 * spatial_padding + 1) * (2 * spatial_padding + 1) * (2 * intensity_padding + 1));
    for(int i = -spatial_padding;i <= spatial_padding; ++i)
        for(int j = -spatial_padding;j <= spatial_padding; ++j)
            for(int k = -intensity_padding;k <= intensity_padding; ++k)
                offset[max_k++] = (i * grid_width + j) * grid_intensity + k;
    // 准备一个结果
    int cnt = 0;
    cv::Mat result(H, W, noisy_image.type());
    uchar* const res_ptr = result.ptr<uchar>();
    // 开始卷积
    for(int i = spatial_padding, max_i = grid_height - spatial_padding; i < max_i; ++i) {
        for(int j = spatial_padding, max_j = grid_width - spatial_padding; j < max_j; ++j) {
            // 获取这个位置的亮度值
            int intensity = noisy_image.data[(i - spatial_padding) * W + j - spatial_padding];
            // 定位网格
            float* const grid_start = grid.data() + (i * grid_width + j) * grid_intensity + intensity_padding;
            // 以 grid_start[intensity] 为中心, 做以此低通滤波
            float weight_sum = 0.f;
            float intensity_sum = 0.f;
            for(int k = 0;k < max_k; ++k) {
                const int pos = offset[k];
                if(grid_start[intensity + pos] > 0) {
                    weight_sum += 1;
                    intensity_sum += grid_start[intensity + pos];
                }
            }
            // 卷积输出填充到对应的位置上
            res_ptr[cnt++] = cv::saturate_cast<uchar>(intensity_sum / weight_sum);
        }
    }
    return result;
}




cv::Mat bilateral_grid_mean_filtering_faster(
        cv::Mat noisy_image,
        const float spatial_sigma=1.0,
        const int intensity_padding=7,
        const int intensity_level=64) {
    assert(noisy_image.channels() == 1 and noisy_image.type() == CV_8UC1 and "only gray images are supported !");
    // 获取图像信息
    const int H = noisy_image.rows;
    const int W = noisy_image.cols;
    // 如果是其他数据类型, 还得转成 float
    // 获取滤波核参数
    const int spatial_padding = std::ceil(3 * spatial_sigma);
    // 构造一个网格
    const int grid_height = H + 2 * spatial_padding;
    const int grid_width = W + 2 * spatial_padding;
    const int grid_intensity = intensity_level + 2 * intensity_padding;
    const int grid_size = grid_height * grid_width * grid_intensity;
    std::vector<float> grid(grid_size, 0);
    std::vector<float> grid_weight(grid_size, 0);
    // 每个网格的长度
    const int grid_interval = std::ceil(255 / intensity_level);
    // 把图像信息填充到网格中
    for(int i = 0;i < H; ++i) {
        uchar* const row_ptr = noisy_image.ptr<uchar>() + i * W;
        for(int j = 0;j < W; ++j) {
            int intensity = row_ptr[j];
            int pos = static_cast<int>(intensity * 1.f / float(grid_interval));
            pos = ((i + spatial_padding) * grid_width + j + spatial_padding) * grid_intensity + pos + intensity_padding;
            grid[pos] += intensity;
            grid_weight[pos] += 1;
        }
    }
    // 准备一个偏移量模板
    int max_k = 0;
    std::vector<int> offset(square<int>(2 * spatial_padding + 1) * (2 * intensity_padding + 1));
    for(int i = -spatial_padding; i <= spatial_padding; ++i)
        for(int j = -spatial_padding; j <= spatial_padding; ++j)
            for(int k = -intensity_padding; k <= intensity_padding; ++k)
                offset[max_k++] = (i * grid_width + j) * grid_intensity + k;
    // 开始在网格中卷积
    std::vector<float> grid_result(grid_size, 0);
    std::vector<float> grid_weight_result(grid_size, 0);

    for(int i = spatial_padding, max_i = grid_height - spatial_padding; i < max_i; ++i) {
        for(int j = spatial_padding, max_j = grid_width - spatial_padding; j < max_j; ++j) {
            // 获取这个点在网格中的位置
            for(int pos = intensity_padding, max_p = grid_intensity - intensity_padding; pos < max_p; ++pos) {
                // 获取在网格中的偏移量
                float* const grid_ptr = grid.data() + (i * grid_width + j) * grid_intensity + pos + intensity_padding;
                float* const weight_ptr = grid_weight.data() + (i * grid_width + j) * grid_intensity + pos + intensity_padding;
                // 开始卷积一个点
                float weight_sum = 0.f;
                float intensity_sum = 0.f;
                for(int k = 0; k < max_k; ++k) {
                    const int p = offset[k];
                    {
                        weight_sum += weight_ptr[p];
                        intensity_sum += grid_ptr[p];
                    }
                }
                // 卷积结束, 得到这个格子的值
//                std::cout << value << std::endl;
                // 根据结果来网格中求值
                grid_result[(i * grid_width + j) * grid_intensity + pos + intensity_padding] = intensity_sum / max_k;
                grid_weight_result[(i * grid_width + j) * grid_intensity + pos + intensity_padding] = weight_sum / max_k;
            }
        }
    }


    // 准备一个结果
    int cnt = 0;
    cv::Mat result(H, W, CV_8UC1);
    uchar* const res_ptr = result.ptr<uchar>();
    // 对于结果中每一个点, 去 grid_result 中去插值得到结果
    for(int i = 0;i < H; ++i) {
        for(int j = 0;j < W; ++j) {
            // 计算这个点在网格中的坐标
            const float x = i + spatial_padding;
            const float y = j + spatial_padding;
            const float z = noisy_image.data[i * W + j] * 1.f / float(grid_interval) + intensity_padding;
            // 三次线性插值, 两个分支
            // wi / w 是最终的加权结果
//            res_ptr[cnt++] = cv::saturate_cast<uchar>(interp_res);
        }
    }
    return result;
}














int main() {
    std::setbuf(stdout, 0);

    // 读取图像
    cv::Mat noisy_image = cv::imread("./images/input/demo.png", 0);
    cv::resize(noisy_image, noisy_image, {50, 50});

    // 用最暴力的网格做均值滤波
//    auto smoothed = bilateral_grid_mean_filtering(
//            noisy_image, 1.5, 30, 255);

    // 优化上面, 对亮度域做分级, 加速
    auto smoothed = bilateral_grid_mean_filtering_faster(
            noisy_image, 1.0, 7.0);



    // 单纯用均值滤波
    // auto smoothed = bilateral_mean_filtering(noisy_image, 1.5, 30);



    cv_show(cv_concat({noisy_image, smoothed}));

    return 0;
}