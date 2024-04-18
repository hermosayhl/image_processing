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
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
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



int main() {
	// 读取图像
	const std::string image_path("./images/input/a1016-050716_115658__I2E4159.png");
	cv::Mat origin_image = cv::imread(image_path);
    assert(not origin_image.empty());

    // 做插值
    const auto result_big = bilinear_interpolate(origin_image, {1600, 2400});
    const auto result_small = bilinear_interpolate(origin_image, {140, 200});

    // 展示
    cv_show(result_big);
    cv_show(result_small);

    // 保存结果
    const std::string output_path("./images/output/");
    cv_write(result_big, output_path + "big.png");
    cv_write(result_small, output_path + "small.png");
    return 0;
}











