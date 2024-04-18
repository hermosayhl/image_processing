// C++
#include <cmath>
#include <vector>
#include <iostream>
#include <assert.h>
#include <filesystem>
// 图像处理
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

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REFLECT);
        return padded_image;
    }

}





inline float _min(const float x, const float y) {
    return x > y ? y : x;
}

inline float _max(const float x, const float y) {
    return x < y ? y : x;
}

cv::Mat bilinear_interpolate(const cv::Mat& origin, const std::pair<int, int>& _size) {
    // 获取信息
    const int H0 = origin.rows;
    const int W0 = origin.cols;
    const int C = origin.channels();
    const int h = _size.first;
    const int w = _size.second;
    // 计算 x 方向和 y 方向上的比率
    const float x_rate = H0 > h ? H0 * 1.f / h : (H0 - 1) * 1.f / h;
    const float y_rate = W0 > w ? W0 * 1.f / w : (W0 - 1) * 1.f / w;
    // 准备一个结果
    cv::Mat result(h, w, CV_8UC3);
    uchar* const res_ptr = result.data;
    // 用来计算在结果 .data 存放的位置
    int cnt = 0;
    // 一共要插值 h X w 次
    for(int x = 0; x < h; ++x) {
        // 找到结果中 x, 对应原图中的 x 坐标
        float x_pos = x_rate * (x + 0.5f) - 0.5f;
        // 找到这个 x_pos 的上下界
        const int x_down = _max(std::floor(x_pos), 0);
        const int x_up = _min(x_down + 1, H0 - 1);
        // 计算 x 方向上的插值参数
        float x_left = x_up - x_pos;
        float x_right = x_pos - x_down;  // 1 - x_left
        // 原图中(x_down 和 x_up) 两行的指针
        const uchar* const ori_ptr = origin.data + x_down * W0 * C;
        const uchar* const ori_ptr_2 = ori_ptr + W0 * C;
        // 填充结果的第 x 行的 y 个像素
        for(int y = 0;y < w; ++y) {
            // 计算 y 对应原图中 y 的坐标, 放大或者缩小
            float y_pos = y_rate * (y + 0.5f) - 0.5f;
            // 计算 y 的上下界, 此时映射到原图中 (x_pos, y_pos) 的周围四个点都找到了
            const int y_down = std::floor(y_pos);
            const int y_up = _min(y_down + 1, W0 - 1);
            // 计算 y 方向上的插值参数
            float y_left = y_up - y_pos;
            float y_right = y_pos - y_down;  // 1 - y_left
            // 多个通道分别计算
            for(int c = 0;c < C; ++c) {
                // y 方向上第一次线性插值, 得到两个值
                float f_E = y_left * ori_ptr[y_down * C + c] + y_right * ori_ptr[y_up * C + c];
                float f_F = y_left * ori_ptr_2[y_down * C + c] + y_right * ori_ptr_2[y_up * C + c];
                // x 方向上第二次线性插值
                float target = x_left * f_E + x_right * f_F;
                res_ptr[cnt++] = cv::saturate_cast<uchar>(target);
            }
        }
    }
    return result;
}



inline float cubic(const float x) {
    return x * x * x;
}

inline float square(const float x) {
    return x * x;
}


template<typename T>
float make_cubic_interpolation(const std::vector<T>& F, const float input) {
    // 根据这 4 个点计算三次函数的参数
    float a = -0.5 * F[0] + 1.5 * F[1] - 1.5 * F[2] + 0.5 * F[3];
    float b = F[0] - 2.5 * F[1] + 2 * F[2] - 0.5 * F[3];
    float c = -0.5 * F[0] + 0.5 * F[2];
    float d = F[1];
    // 根据三次函数, 插值算这个点 input 的值
    return a * cubic(input) + b * square(input) + c * input + d;
}



cv::Mat bicubic_interpolate(const cv::Mat& origin, const std::pair<int, int>& _size) {
    // 获取信息
    int H = origin.rows;
    int W = origin.cols;
    const int H2 = _size.first;
    const int W2 = _size.second;
    const int C = origin.channels();
    // 计算纵向跟横向的缩放比
    const float h_ratio = H2 > H ? (H - 1) * 1.f / H2 : H * 1.f / H2;
    const float w_ratio = W2 > W ? (W - 1) * 1.f / W2 : W * 1.f / W2;
    // 计算纵向跟横向需要偏移的距离
    const float h_add = 0.5 * ((H - 1) - (H2 - 1) * h_ratio);
    const float w_add = 0.5 * ((W - 1) - (W2 - 1) * w_ratio);
    // 做 padding, 因为是周围 16 个点做插值
    const int pad = 1;
    const auto padded_image = make_pad(origin, pad, pad);
    const uchar* const pad_ptr = padded_image.ptr<uchar>();
    // 准备一个结果
    cv::Mat result(H2, W2, origin.type());
    uchar* const res_ptr = result.ptr<uchar>();
    int cnt = 0;
    // 准备几个临时变量
    std::vector<uchar> temp_Y(4);  // 存储横向一次插值的结果
    std::vector<float> temp_X(4);  // 存储纵向一次插值的四个点
    // 插值每一个行
    for(int x = 0;x < H2; ++x) {
        // 算这一行在 3 x 3 中的位置, 下界和偏移
        float x_pos = x * h_ratio + h_add;
        int x_down = std::floor(x_pos);
        const float x_offset = x_pos - x_down;
        // 插值每一个点
        for(int y = 0;y < W2; ++y) {
            float y_pos = y * w_ratio + w_add;
            int y_down = std::floor(y_pos);
            const float y_offset = y_pos - y_down;
            // 多通道
            for(int ch = 0;ch < 3; ++ch) {
                // 首先, 计算从第 x_down 行开始, [-1, 0, 1, 2] 的插值
                for(int i = -1; i <= 2; ++i) {
                    // x_down + i 是在 3 x 3 图像中的坐标, pad 是做了 padding 的偏移量
                    const int X = x_down + i + pad;
                    // 找到 x_down + i 行的数据起始的指针, + pad * C 是因为有横向 pad
                    const uchar* const X_ptr = pad_ptr + X * padded_image.cols * C + pad * C;
                    // 插值 (x_down + i, y_pos), 需要找到 (x_down + i, y_pos) 的四个点, 存储在 temp_Y 中
                    for(int j = -1; j <= 2; ++j)
                        temp_Y[j + 1] = X_ptr[(y_down + j) * C + ch];
                    // 这 4 个点做 cubic 插值, 作为 (x_down + i, y_pos) 的结果
                    temp_X[i + 1] = make_cubic_interpolation<uchar>(temp_Y, y_offset);
                }
                // 得到了 (x_down + i, y_pos) 四个点的插值结果, 做一次 cubic 插值, 作为 (x_pos, y_pos) 的插值结果
                const float one = make_cubic_interpolation<float>(temp_X, x_offset);
                res_ptr[cnt++] = cv::saturate_cast<uchar>(one);
            }
        }
    }
    return result;
}



int main() {
	// 读取图像
	const std::string image_path("../images/input/a1016-050716_115658__I2E4159.png");
	cv::Mat origin_image = cv::imread(image_path);
    assert(not origin_image.empty());
    cv_show(origin_image);

    // 先把图变小
    const auto small = bicubic_interpolate(origin_image, {200, 300});

    // 双立方插值, 把图变大
    constexpr int target_h = 1200;
    constexpr int target_w = 1800;
    const auto big = bicubic_interpolate(small, {target_h, target_w});

    // 和双线性插值对比
    const auto bilinear_big = bilinear_interpolate(small, {target_h, target_w});

    // 和 OpenCV 内置实现对比
    cv::Mat cv_big;
    cv::resize(small, cv_big, {target_w, target_h}, cv::INTER_CUBIC);

    // 展示
    cv_show(small);
    cv_show(bilinear_big);
    cv_show(big);
    cv_show(cv_big);
    cv::Mat concat;
    cv::vconcat(std::vector<cv::Mat>({big, cv_big}), concat);
    cv_show(concat);

    // 保存结果
    const std::string output_path("../images/output/");
    cv_write(small, output_path + "small.png");
    cv_write(big, output_path + "big.png");
    return 0;
}











