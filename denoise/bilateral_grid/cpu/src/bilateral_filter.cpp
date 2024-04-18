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


// 朴素版的双边滤波
cv::Mat bilateral_filtering(
        // 这里不做引用, 直接用于 padding
        cv::Mat noisy_image,
        const float spatial_sigma=2.f,
        const float range_sigma=0.1f) {
    // 类型检查
    assert(noisy_image.channels() == 1 and "Only images of one channel are supported !");
    assert(noisy_image.type() == CV_8UC1 and "images with dynamic range of 0-255 are supported !");

    // 亮度域转到 0~1, 数据 float
    noisy_image.convertTo(noisy_image, CV_32FC1);
    noisy_image /= 255;

    // 获取图像信息
    const int H = noisy_image.rows;
    const int W = noisy_image.cols;

    // 根据标准差参数准备一些中间参数
    const int radius = std::ceil(3 * spatial_sigma); // 空间域滤波半径

    // 对原图做 padding
    cv::copyMakeBorder(noisy_image, noisy_image, radius, radius, radius, radius, cv::BORDER_REFLECT);
    const int W2 = noisy_image.cols; // 这里改了, 注意

    // 准备一个空间率滤波模板
    const int kernel_size = square<int>(2 * radius + 1);
    int max_k = 0;
    std::vector<int> offset(kernel_size, 0);
    std::vector<float> spatial_weights(kernel_size, 0.f);
    const float spatial_sigma_inv = -0.5f / (spatial_sigma * spatial_sigma);
    for(int i = -radius; i <= radius; ++i) {
        for(int j = -radius; j <= radius; ++j) {
            if(i == 0 and j == 0) continue;
            spatial_weights[max_k] = std::exp(spatial_sigma_inv * (i * i + j * j));
            offset[max_k] = i * W2 + j;  // 注意这里每一行是 W2 个像素
            ++max_k;
        } // 不用对 spatial_weight 归一化, 最后卷积的时候归一化, 结果一样的
    }

    // 准备值域权重需要的中间参数, 注意这里是 double, 不是 float
    const double range_sigma_inv = -0.5 / (range_sigma * range_sigma);

    // 准备一个结果存放滤波结果
    int cnt = 0;
    cv::Mat result(H, W, CV_8UC1);
    uchar* res_ptr = result.ptr<uchar>();

    // 开始双边滤波
    for(int i = 0; i < H; ++i) {
        float* const row_ptr = noisy_image.ptr<float>() + (radius + i) * W2 + radius;  // padding 图像中第 i 行第一个滤波中心的地址偏移量
        for(int j = 0; j < W; ++j) {
            float center = row_ptr[j];   // 当前滤波的中心点的亮度值
            float weight_sum = 1.0;      // 累加所有点的权重, 中心点权重 1.0
            float intensity_sum = center;// 加权亮度, 初始化值是中心点的亮度
            for(int k = 0;k < max_k; ++k) {
                float neighbor = row_ptr[j + offset[k]];  // 局部窗口的邻域点
                float range_weight = fast_exp(range_sigma_inv * square<float>(neighbor - center)); // 该邻域点跟中心点在亮度上计算权重
                float weight = spatial_weights[k] * range_weight;  // 空间域权重 * 亮度域权重
                weight_sum += weight;
                intensity_sum += neighbor * weight;
            }
            res_ptr[cnt++] = cv::saturate_cast<uchar>(255 * intensity_sum / weight_sum);
        }
    }
    return result;
}



cv::Mat bilateral_filtering_color(
        cv::Mat& noisy_image, const float spatial_sigma=4.0, const float range_sigma=0.1) {
    // 类型检查
    assert(noisy_image.channels() == 3 and "Only images of one channel are supported !");
    assert(noisy_image.type() == CV_8UC3 and "images with dynamic range of 0-255 are supported !");

    // 亮度域转到 0~1, 数据 float
    noisy_image.convertTo(noisy_image, CV_32FC3);
    noisy_image /= 255;

    // 获取图像信息
    const int H = noisy_image.rows;
    const int W = noisy_image.cols;
    const int C = noisy_image.channels();

    // 根据标准差参数准备一些中间参数
    const int radius = std::ceil(3 * spatial_sigma); // 空间域滤波半径

    // 对原图做 padding
    cv::copyMakeBorder(noisy_image, noisy_image, radius, radius, radius, radius, cv::BORDER_REFLECT);
    const int W2 = noisy_image.cols; // 这里改了, 注意

    // 准备一个空间率滤波模板
    const int kernel_size = square<int>(2 * radius + 1);
    int max_k = 0;
    std::vector<int> offset(kernel_size, 0);
    std::vector<float> spatial_weights(kernel_size, 0.f);
    const float spatial_sigma_inv = -0.5f / (spatial_sigma * spatial_sigma);
    for(int i = -radius; i <= radius; ++i) {
        for(int j = -radius; j <= radius; ++j) {
            spatial_weights[max_k] = std::exp(spatial_sigma_inv * (i * i + j * j));
            offset[max_k] = (i * W2 + j) * C;  // 注意这里每一行是 W2 个像素
            ++max_k;
        } // 不用对 spatial_weight 归一化, 最后卷积的时候归一化, 结果一样的
    }

    // 准备值域权重需要的中间参数, 注意这里是 double, 不是 float
    const double range_sigma_inv = -0.5 / (range_sigma * range_sigma);

    // 准备一个结果存放滤波结果
    int cnt = 0;
    cv::Mat result(H, W, CV_8UC3);
    uchar* res_ptr = result.ptr<uchar>();

    // 开始双边滤波
    for(int i = 0; i < H; ++i) {
        float* const row_ptr = noisy_image.ptr<float>() + (radius + i) * W2 * C + radius * C;  // padding 图像中第 i 行第一个滤波中心的地址偏移量
        for(int j = 0; j < W; ++j) {
            const int p = C * j; // 当前偏移量
            std::vector<float> intensity_sum(C, 0.f); // 存储这一个点, 在 r, g, b 三通道上的累计灰度值
            float weight_sum = 0.f;   // 存储这个位置, 在空间域上的权重 * rgb 三通道在亮度域上的权重之积
            for(int k = 0; k < max_k; ++k) {
                float weight = 1.f;   // 初始化为 1
                for(int ch = 0; ch < C; ++ch) {
                    float diff = row_ptr[p + ch + offset[k]] - row_ptr[p + ch]; // r, g, b 对应像素做亮度域权重的计算
                    weight *= fast_exp(range_sigma_inv * square<float>(diff)); // 做累乘
                }
                weight *= spatial_weights[k];  // 乘以空间域上的权重
                weight_sum += weight;          // 权重累加, 最后归一化
                for(int ch = 0; ch < C; ++ch)  // 遍历这个窗口, r, g, b 每个点各自计算邻域的加权和
                    intensity_sum[ch] += weight * row_ptr[p + ch + offset[k]];  // 当前点 p, 的第 ch 个通道, 平面偏移 k
            }
            for(int ch = 0; ch < C; ++ch)
                res_ptr[cnt++] = cv::saturate_cast<uchar>(255 * intensity_sum[ch] / weight_sum);
        }
    }
    return result;
}


int main() {
    std::setbuf(stdout, 0);

    // 测试单通道
    if(false) {
        // 读取图像
        cv::Mat noisy_image = cv::imread("./images/input/example.png", 0);
        auto smoothed = bilateral_filtering(
                noisy_image, 4.0, 0.2);
        cv_show(smoothed);
        cv_write(smoothed, "./images/output/pure_bilateral_result.png");
    }
    // 测试三通道一起做双边滤波的情形
    else {
        cv::Mat noisy_image = cv::imread("./images/input/demo.png");
        assert(not noisy_image.empty() and "failed to load image");
        auto smoothed = bilateral_filtering_color(noisy_image, 4.0, 0.2);
        cv_show(smoothed);
    }

    return 0;
}