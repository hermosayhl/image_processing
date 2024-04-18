// C++
#include <cmath>
#include <vector>
#include <iostream>
#include <assert.h>
#include <filesystem>
// 矩阵运算
#include <Eigen/Sparse>
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

    void cv_info(const cv::Mat& one_image) {
        std::cout << "高  :  " << one_image.rows << "\n宽  :  " << one_image.cols << "\n通道 :  " << one_image.channels() << std::endl;
        std::cout << "步长 :  " << one_image.step << std::endl;
        std::cout << "是否连续" << std::boolalpha << one_image.isContinuous() << std::endl;
    }

    cv::Mat cv_concat(const std::vector<cv::Mat> images, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(images, result);
        else cv::vconcat(images, result);
        return result;
    }

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }

    inline int square(const int x) {
        return x * x;
    }
}



float compute_psnr(const cv::Mat& lhs, const cv::Mat& rhs) {
    const int H = lhs.rows;
    const int W = rhs.cols;
    const int C = lhs.channels();
    assert(rhs.rows == H and rhs.cols == W and rhs.channels() == C);
    const int length = H * W;
    const uchar* const l_ptr = lhs.data;
    const uchar* const r_ptr = rhs.data;
    int mse = 0;
    for(int i = 0;i < length; ++i) {
        const int p = C * i;
        for(int j = 0;j < C; ++j)
            mse += square(l_ptr[p + j] - r_ptr[p + j]);
    }
    double __mse = std::sqrt(mse * 1.0 / (length * C));
    // 代入 psnr 公式  20 * log10(255 / std::sqrt(MSE)),
    // 注意 PSNR 越高, mse 越小, 所以 MSE 放在分母
    // 还可以发现, PSNR 每高 1 个 db, 则有 1 / 10 ** 0.5 = 1 / 1.22 倍的误差
    double psnr_value = 20 * std::log10(255 / __mse);
    // std::cout << cv::PSNR(lhs, rhs) << std::endl;
    return psnr_value;
}





float __ssim(const cv::Mat& lhs, const cv::Mat& rhs, const int window_size=11) {
    // 获取信息
    const int H = lhs.rows;
    const int W = lhs.cols;
    const int radius = (window_size - 1) >> 1;
    std::cout << lhs.channels() << std::endl;
    // 对图像做 padding
    cv::Mat left = make_pad(lhs, radius, radius);
    cv::Mat right = make_pad(rhs, radius, radius);
    const int H2 = left.rows;
    const int W2 = left.cols;
    // 转成 float 数据
    left.convertTo(left, CV_32F);
    right.convertTo(right, CV_32F);
    // 做一个 offset
    int cnt = 0;
    const int win_len = square(window_size);
    int offset[win_len];
    std::cout << "win_len = " << win_len << std::endl;
    for(int i = -radius;i <= radius; ++i)
        for(int j = -radius;j <= radius; ++j)
            offset[cnt++] = i * W + j;
    // 准备一个均值图, 存放 left 图的均值
    auto compute_mean = [&](const cv::Mat& target) -> cv::Mat {
        cv::Mat result(H, W, CV_32F);
        for(int i = 0;i < H; ++i) {
            const float* in_ptr = target.ptr<float>() + (i + radius) * W2 + radius;
            float* res_ptr = result.ptr<float>() + i * W;
            for(int j = 0;j < W; ++j) {
                // 遍历局部窗口
                float sum_value = 0;
                for(int k = 0;k < win_len; ++k)
                    sum_value += in_ptr[j + offset[k]];
                res_ptr[j] = sum_value / win_len;
            }
        }
        return result;
    };
    const auto left_mean = compute_mean(left);
    const auto right_mean = compute_mean(right);
    // 计算二者的方差,
    const int length = H * W;
    auto compute_sigma = [&](const cv::Mat& target, const cv::Mat& mean) ->cv::Mat {
        cv::Mat result(H, W, CV_32F);
        for(int i = 0;i < H; ++i) {
            const float* mean_ptr = mean.ptr<float>() + i * W;
            const float* in_ptr = target.ptr<float>() + (i + radius) * W2 + radius;
            float* res_ptr = result.ptr<float>() + i * W;
            for(int j = 0;j < W; ++j) {
                // 遍历局部窗口
                float sum_value = 0;
                for(int k = 0;k < win_len; ++k) {
                    float diff = in_ptr[j + offset[k]] - mean_ptr[j];
                    sum_value += diff * diff;
                }
                res_ptr[j] = sum_value / win_len;
            }
        }
        return result;
    };
    const auto left_sigma = compute_sigma(left, left_mean);
    const auto right_sigma = compute_sigma(right, right_mean);
    // 计算协方差
    cv::Mat cov(H, W, CV_32F);
    for(int i = 0;i < H; ++i) {
        const float* left_ptr = left.ptr<float>() + (radius + i) * W;
        const float* right_ptr = right.ptr<float>() + (radius + i) * W;
        const float* left_mean_ptr = left_mean.ptr<float>() + i * W;
        const float* right_mean_ptr = right_mean.ptr<float>() + i * W;
        float* res_ptr = cov.ptr<float>() + i * W;
        for(int j = 0;j < W; ++j) {
            // 遍历局部窗口
            float sum_value = 0;
            for(int k = 0;k < win_len; ++k) {
                float diff = (left_ptr[j + offset[k]] - left_mean_ptr[j]) * (right_ptr[j + offset[k]] - right_mean_ptr[j]);
                sum_value += diff;
            }
            res_ptr[j] = sum_value / win_len;
        }
    }

    // 计算均值的平方
    cv::Mat left_mean_2(H, W, CV_32F);
    for(int i = 0;i < length; ++i) left_mean_2.data[i] = left_mean.data[i] * left_mean.data[i];
    cv::Mat right_mean_2(H, W, CV_32F);
    for(int i = 0;i < length; ++i) right_mean_2.data[i] = right_mean.data[i] * right_mean.data[i];
    // 计算方差的平方
    cv::Mat left_sigma_2(H, W, CV_32F);
    for(int i = 0;i < length; ++i) left_sigma_2.data[i] = left_sigma.data[i] * left_sigma.data[i];
    cv::Mat right_sigma_2(H, W, CV_32F);
    for(int i = 0;i < length; ++i) right_sigma_2.data[i] = right_sigma.data[i] * right_sigma.data[i];
    // 计算 ssim, 平均值
    cv::Mat ssim_matrix = (2 * left_mean * right_mean + 0.01 * 0.01) / (left_mean_2 + right_mean_2 + 0.01 * 0.01);
    ssim_matrix = ssim_matrix * (2 * cov + 0.03 * 0.03) / (left_sigma_2 + right_sigma_2 + 0.03 * 0.03);
    return 0;
}

float compute_ssim(const cv::Mat& lhs, const cv::Mat& rhs, const int window_size=11) {
    assert(lhs.rows == rhs.rows and lhs.cols == rhs.cols and lhs.channels() == rhs.channels());
    // 转成三通道
    std::vector<cv::Mat> lhs_temp, rhs_temp;
    cv::split(lhs, lhs_temp);
    cv::split(rhs, rhs_temp);
    // 对应通道之间计算 ssim
    const int C = lhs.channels();
    float mean_ssim = 0;
    for(int i = 0;i < C; ++i) {
        mean_ssim += __ssim(lhs_temp[0], rhs_temp[0], window_size);
    }
    return mean_ssim / C;
}



int main() {

    // 读取图像
    cv::Mat label = cv::imread("./images/ground_truth.png");
    cv::Mat result = cv::imread("./images/pre-trained.png");
    assert(not label.empty() and not result.empty());

    // 计算 PSNR
    const float psnr_value = compute_psnr(label, result);
    printf("PSNR    :  %.5f db\n", psnr_value);
    printf("OpenCV  :  %.5f db\n\n", cv::PSNR(label, result));

    // 计算 SSIM
    const float ssim_value = compute_ssim(label, result);
    printf("SSIM    :  %.5f\n", ssim_value);

    return 0;
}