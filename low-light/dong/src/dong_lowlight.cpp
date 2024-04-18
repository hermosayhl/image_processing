// C++
#include <list>
#include <vector>
#include <iostream>
#include <assert.h>
// OpenCV
#include <opencv2/opencv.hpp>
// self
#include "guided_filter.hpp"


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


template<typename T>
cv::Mat compute_dark_channel(const cv::Mat& origin, const int H, const int W, const int radius=7) {
    const T* const ori_ptr = origin.ptr<T>();
    // 决定数据类型, 是 uchar 还是 float
    const int TYPE = origin.type() == CV_8UC3 ? CV_8UC1 : CV_32FC1;
    // 【1】先 R, G, B 三通道求一个最小图
    cv::Mat min_bgr(H, W, TYPE);
    T* const min_bgr_ptr = min_bgr.ptr<T>();
    const int length = H * W;
    for(int i = 0;i < length; ++i) {
        const int p = 3 * i;
        min_bgr_ptr[i] = std::min(ori_ptr[p], std::min(ori_ptr[p + 1], ori_ptr[p + 2]));
    }
    // 【2】min_bgr 中每个点, 在窗口中找一个最小值
    // 先对图像做 padding
    auto pad_min_bgr = make_pad(min_bgr, radius, radius);
    const int H2 = H + 2 * radius;
    const int W2 = W + 2 * radius;
    // 存放第一次横向找最小值地结果
    cv::Mat temp(H2, W, TYPE);
    T* const temp_ptr = temp.ptr<T>();
    int cnt = 0;
    // 第一次, 横向找 H2 次
    for(int i = 0;i < H2; ++i) {
        T* const row_ptr = pad_min_bgr.ptr<T>() + i * W2 + radius;
        for(int j = 0;j < W; ++j) {
            T min_value = 255;
            for(int k = -radius; k <= radius; ++k)
                min_value = std::min(min_value, row_ptr[j + k]);
            temp_ptr[cnt++] = min_value;
        }
    }
    // 释放空间
    pad_min_bgr.release();
    // 第二次, 竖向比较
    for(int j = 0;j < W; ++j) {
        for(int i = 0;i < H; ++i) {
            T min_value = 255;
            const int offset = (radius + i) * W + j;
            for(int k = -radius; k <= radius; ++k)
                min_value = std::min(min_value, temp_ptr[offset + k * W]);
            min_bgr_ptr[i * W + j] = min_value;  // 结果直接存放到 min_bgr 里面, 节约空间
        }
    }
    return min_bgr;
}



using details_type = std::list< std::pair<std::string, cv::Mat> >;

details_type dong_enhance(
        const cv::Mat& low_light,
        const int radius=3,
        const int A_pixels=100,
        const float weight=0.8,
        const float border=0.5,
        const bool denoise=false) {
    details_type collections;
    // 获取信息
    const int H = low_light.rows;
    const int W = low_light.cols;
    const int C = low_light.channels();
    const int length = H * W;
    const int total_length = length * C;
    assert(low_light.type() == 16 and "Only CV_8U3(BGR) are supported !");

    //【1】首先 255 - low_light, 得到反转图像
    cv::Mat inverse(H, W, low_light.type());
    uchar* const inv_ptr = inverse.ptr<uchar>();
    for(int i = 0;i < total_length; ++i)
        inv_ptr[i] = 255 - low_light.data[i];
    collections.emplace_back("inverse", inverse);

    // 【2】对 inverse 去雾, 首先求 inverse 的暗通道
    const auto dark_channel = compute_dark_channel<uchar>(inverse, H, W, radius);
    collections.emplace_back("dark_channel", dark_channel);

    // 【3】根据暗通道, 排序筛选最大值的前 100 个点
    // 在 有雾图像中去找 rgb 之和最大值作为对三个通道大气光 A 的估计
    std::vector< std::vector<int> > book(256);
    const uchar* const dark_ptr = dark_channel.ptr<uchar>();
    for(int i = 0;i < length; ++i)
        book[dark_ptr[i]].emplace_back(i);
    int cnt = 0;
    std::vector<int> index(A_pixels);  // 找到暗通道最大的 100 个点坐标
    for(int i = 255; i >= 0; --i) {
        const int _size = book[i].size(); // 这里是 O(1)
        for(int t = 0;t < _size and cnt < A_pixels; ++t)
            index[cnt++] = book[i][t];
    }
    int max_bgr_sum = 0;  // 从这 100 个暗通道最大值对应 有雾图像 中, 找 r, g, b 之和最大的点
    int max_index = -1;
    for(int i = 0;i < A_pixels; ++i) {
        const int p = 3 * index[i];
        const int cur_sum = inv_ptr[p] + inv_ptr[p + 1] + inv_ptr[p + 2];
        if(cur_sum > max_bgr_sum) {
            max_bgr_sum = cur_sum;
            max_index = index[i];
        }
    }
    // 现在 max_index 是最大的 r, g, b 的通道
    int A[3] = {inv_ptr[max_index * 3], inv_ptr[max_index * 3 + 1], inv_ptr[max_index * 3 + 2]};
    std::cout << "三个通道的全局大气光估计值 " << A[0] << ", " << A[1] << ", " << A[2] << "\n";

    // 【4】估计 t
    // 准备一个 float 图, 每个通道的像素除以对应的全局大气光
    cv::Mat R_A(H, W, CV_32FC3);
    float* const R_A_ptr = R_A.ptr<float>();
    for(int i = 0;i < length; ++i) {
        const int p = 3 * i;
        for(int c = 0;c < 3; ++c)
            R_A_ptr[p + c] = inv_ptr[p + c] * 1.f / A[c];
    }
    // 根据比值的暗通道, 得到透射率(这里的 R_A 是 CV_32FC1 了, 节约空间)
    R_A = compute_dark_channel<float>(R_A, H, W, radius);
    float* t_ptr = R_A.ptr<float>();

    // 对透射率做点小改变, 远景增强地更厉害点
    auto discount = [border](const float x) ->float {
        return x;
        if(x >= 0 and x < border) return 2 * x * x;
        else return x;
    };
    for(int i = 0;i < length; ++i)
        t_ptr[i] = discount(1.f - weight * t_ptr[i]);

    // 开始求解 J(x), 然后取反
    cv::Mat result(H, W, CV_8UC3);
    uchar* const res_ptr = result.ptr<uchar>();
    for(int i = 0;i < length; ++i) {
        const int p = 3 * i;
        for(int c = 0;c < 3; ++c) {
            const float J = (inv_ptr[p + c] - A[c]) * 1.f / t_ptr[i] + A[c];  // 偏大
            res_ptr[p + c] = cv::saturate_cast<uchar>(J);
        }
    }
    // 做去噪
    if(denoise) {
        cv::fastNlMeansDenoisingColored(result, result, 5, 5, 5, 15);
    }
    collections.emplace_back("dehazed", result.clone());

    // 反转图像
    for(int i = 0;i < total_length; ++i)
        res_ptr[i] = 255 - res_ptr[i]; // 更大
    collections.emplace_back("enhanced", result.clone());
    return collections;
}




void demo_lowlight_enhance(const std::string& image_path="./images/input/a4542-Duggan_080411_6019.png") {
    // 读取图像
    cv::Mat low_light = cv::imread(image_path);

    cv_show(low_light);

    // 开始增强
    auto collections = dong_enhance(low_light, 3, 100, 0.8, 0.5, false);

    cv::Mat concat;
    cv::hconcat(std::vector<cv::Mat>({low_light, collections.back().second}), concat);
    cv_show(concat);

    // 展示
    for(const auto& item : collections) {
        cv_show(item.second, item.first.c_str());
        cv_write(item.second, "./images/output/" + item.first + ".png");
    }
}



void demo_lowlight_dehaze(const std::string& image_path) {
    // 读取图像
    cv::Mat hazy_image = cv::imread(image_path);

    // cv_show(hazy_image);

    // 获取一张逆图像
    cv::Mat low_light = cv::Mat(hazy_image.rows, hazy_image.cols, CV_8UC3, cv::Scalar(255, 255, 255)) - hazy_image;
    // cv_show(low_light);

    // 开始增强
    auto collections = dong_enhance(low_light, 3, 100, 0.8, 0.7, false);

    cv::Mat concat;
    cv::hconcat(std::vector<cv::Mat>({low_light, collections.back().second}), concat);
    cv_show(concat);

    // 展示
    for(const auto& item : collections) {
        cv_show(item.second, item.first.c_str());
        cv_write(item.second, "./images/output/dehaze_" + item.first + ".png");
    }


}




int main() {
	// demo_lowlight_enhance("./images/input/a4542-Duggan_080411_6019.png");

    // demo_lowlight_dehaze("./images/input/8729.png");
    demo_lowlight_dehaze("./images/input/tree2.png");
    // demo_lowlight_dehaze("tiananmen1.bmp");

    return 0;
}











