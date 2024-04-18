// C++
#include <cmath>
#include <vector>
#include <iostream>
// self
#include "non_local_means.h"


namespace {
    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }
    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    inline double fast_exp(const double y) {
        double d;
        *(reinterpret_cast<int*>(&d) + 0) = 0;
        *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * y + 1072632447);
        return d;
    }
    // 均值滤波
    void box_filter(const double* const new_source, double* const sum_ptr, const int radius_h, const int radius_w, const int H, const int W) {
        // 先对图像做 padding
        const int new_H = H + 2 * radius_h;
        const int new_W = W + 2 * radius_w;
        std::vector<double> padding_image(new_H * new_W, 0);
        double* const padding_ptr = padding_image.data();
        // 先把已有内容填上
        for(int i = 0;i < H; ++i) {
            double* const row_ptr = padding_ptr + (radius_h + i) * new_W + radius_w;
            const double* const src_row_ptr = new_source + i * W;
            std::memcpy(row_ptr, src_row_ptr, sizeof(double) * W);
        }
        // 填充上面的边界
        for(int i = 0;i < radius_h; ++i) {
            std::memcpy(padding_ptr + (radius_h - 1 - i) * new_W + radius_w, new_source + i * W, sizeof(double) * W);
            std::memcpy(padding_ptr + (new_H - radius_h + i) * new_W + radius_w, new_source + (H - i - 1) * W, sizeof(double) * W);
        }
        // 填充左右两边的边界, 这次没法 memcpy 了, 内存不是连续的
        for(int j = 0;j < radius_w; ++j) {
            double* const _beg = padding_ptr + radius_h * new_W + radius_w - 1 - j;
            for(int i = 0;i < H; ++i)
                _beg[i * new_W] = new_source[i * W + j];
        }
        for(int j = 0;j < radius_w; ++j) {
            double* const _beg = padding_ptr + radius_h * new_W + radius_w + W + j;
            for(int i = 0;i < H; ++i)
                _beg[i * new_W] = new_source[i * W + W - 1 - j];
        }
        // 现在图像的高和宽分别是 new_H, new_W, 草稿画一下图就知道
        const int kernel_h = (radius_h << 1) + 1;
        const int kernel_w = (radius_w << 1) + 1;
        // 准备 buffer 和每一个点代表的 box 之和
        std::vector<double> buffer(new_W, 0.0);
        // 首先求目标(结果的)第一行的 buffer
        for(int i = 0;i < kernel_h; ++i) {
            const double* const row_ptr = padding_ptr + i * new_W;
            for(int j = 0;j < new_W; ++j) buffer[j] += row_ptr[j];
        }
        // 求每一行的每个点的 box 的和
        for(int i = 0;i < H; ++i) {
            // 当前 kernel_w 个 buffer 点的累加值
            double cur_sum = 0;
            // 这一行第一个 box 的 cur_sum, 前 kernel_w 个 buffer 点的累加值
            for(int j = 0;j < kernel_w; ++j) cur_sum += buffer[j];
            // 记录这第一个 box 的值
            const int _beg = i * W;
            sum_ptr[_beg] = cur_sum;
            // 向右边挪动, 减去最左边的值, 加上最右边要加进来的值
            for(int j = 1;j < W; ++j) {
                cur_sum = cur_sum - buffer[j - 1] + buffer[j - 1 + kernel_w];
                sum_ptr[_beg + j] = cur_sum;
            }
            // 这一行的点的 sum 都记下来了, 准备换行, 更新 buffer ==> 减去最上面的值, 加上新一行对应的值
            // 最后一次不需要更新......
            if(i != H - 1) {
                const double* const up_ptr = padding_ptr + i * new_W;
                const double* const down_ptr = padding_ptr + (i + kernel_h) * new_W;
                for(int j = 0;j < new_W; ++j) buffer[j] = buffer[j] - up_ptr[j] + down_ptr[j];
            }
        }
        // sum 其实就是最后的矩阵, 现在要除以 area, 每个 box 的面积
        const int area = kernel_h * kernel_w;
        const int length = H * W;
        for(int i = 0;i < length; ++i)
            sum_ptr[i] /= area;
	}
}



/*
 * 历程卡了我两天, 原来 cv::Rect 出来的图像不是连续存储的, 坑爹啊, 所以每次得到的图像就很奇怪, 我怎么看偏移都没错, 但结果就是错的
 * 还有一个 padded_image 的宽不是 W, 不是 W ! 不是 W !!!!
 */
cv::Mat fast_non_local_means_gray_1(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma, const bool use_fast_exp) {
    // 获取图像信息
    const int H = noise_image.rows;
    const int W = noise_image.cols;
    const int length = H * W;
    // 对图像做补齐操作
    const int relative_pos = search_radius + radius;
    const auto padded_image = make_pad(noise_image, relative_pos, relative_pos);
    const int W2 = padded_image.cols;
    // 存储每个点的当前求和, 以及当前权重之和
    std::vector<double> cur_sum(length, 0);
    std::vector<double> weight_sum(length, 0);
    // 当前相对位置对应的图像
    cv::Mat relative_image = noise_image.clone();
    // 当前相对位置对应 MSE, 和平均 MSE
    std::vector<double> relative_errors(length, 0);
    std::vector<double> relative_mean(length, 0);
    const double sigma_inv = 1. / (sigma * sigma);
    // 遍历每一个相对位置
    for(int x = -search_radius; x <= search_radius; ++x) {
        for(int y = -search_radius; y <= search_radius; ++y) {
            // 首先把当前相对位置对应的图像抠出来(这里有点消耗时间)
            for(int t = 0;t < H; ++t)
                std::memcpy(relative_image.data + t * W, padded_image.data + (relative_pos + x + t) * W2 + relative_pos + y, W * sizeof(uchar));
            // 二者计算 mse
            for(int i = 0;i < length; ++i) {
                const double error = relative_image.data[i] - noise_image.data[i];
                relative_errors[i] = error * error;
            }
            // 求 mse 的平均图
            box_filter(relative_errors.data(), relative_mean.data(), radius, radius, H, W);
            // 遍历图像中每个点, 记录当前相对位置对这个点的权重
            for(int i = 0;i < length; ++i) {
                const double w = fast_exp(-relative_mean[i] * sigma_inv);
                cur_sum[i] += w * relative_image.data[i];
                weight_sum[i] += w;
            }
        }
    }
    auto denoised = noise_image.clone();
    for(int i = 0;i < length; ++i)
        denoised.data[i] = cv::saturate_cast<uchar>(cur_sum[i] / weight_sum[i]);
    return denoised;
}
//    for (int x = -search_radius; x <= search_radius; ++x) {
//        for (int y = -search_radius; y <= search_radius; ++y)  {
//            // 不拷贝, 存指针 速度并没有快多少
//            uchar* _beg = padded_image.data + (relative_pos + x) * W2 + relative_pos + y;
//            uchar* row_ptr = _beg;
//            for(int i = 0, cnt = 0;i < length; ++i) {
//                const double temp = noise_image.data[i] - row_ptr[cnt];
//                residual_image[i] = temp * temp;
//                if(++cnt == W) cnt = 0, row_ptr += W2;
//            }
//            box_filter(residual_image.data(), residual_mean.data(), radius, radius, H, W);
//            row_ptr = _beg;
//            for(int i = 0, cnt = 0;i < length; ++i) {
//                double distance = - residual_mean[i] * sigma_inv;
//                double w = std::exp(distance);
//                weight_sum[i] += w;
//                cur_sum[i] += w * row_ptr[cnt];
//                if(++cnt == W) cnt = 0, row_ptr += W2;
//            }
//        }
//    }


// Eigen3 矩阵优化, 后面再说吧, 有点复杂了, box_filter 也得跟着优化


// CUDA 版本的优化空间还很大

// KDTree