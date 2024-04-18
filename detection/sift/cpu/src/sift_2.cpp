//C++
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// Eigen3
#include <Eigen/Core>
#include <Eigen/Dense>
// opencv
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>


namespace {
    void run(const std::function<void()>& work=[]{}, const std::string message="") {
        auto start = std::chrono::steady_clock::now();
        work();
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << message << " " << duration.count() << " ms" <<  std::endl;
    }

    void cv_info(const cv::Mat& one_image) {
        std::cout << "高  :  " << one_image.rows << "\n宽  :  " << one_image.cols << "\n通道 :  " << one_image.channels() << std::endl;
        std::cout << "步长 :  " << one_image.step << std::endl;
    }

    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    cv::Mat cv_resize(cv::Mat& one_image, const int height, const int width, const int _interpolation=cv::INTER_LINEAR) {
		cv::Mat result;
		cv::resize(one_image, result, cv::Size(width, height), 0, 0, _interpolation);
		return result;
	}

	cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
		cv::Mat padded_image;
		cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
		return padded_image;
	}

	cv::Mat cv_concat(std::vector<cv::Mat> sequence) {
        cv::Mat result;
        cv::hconcat(sequence, result);
        return result;
    }
}



/*
 * 参考资料
 * 1. https://www.cnblogs.com/ronny/p/4028776.html
 * 2. https://blog.csdn.net/shiyongraow/article/details/78238764
 * 3. https://blog.csdn.net/shiyongraow/article/details/78296710
 * 4. 下一步我可以考虑, 不用那个差分的 sigma, 也不用第一张图, 直接用原图试试, 还有假如我不用 DOG 直接用 LOG 会怎么样
*/
std::vector< std::vector<cv::Mat> > build_DOG_pyramid(
        const cv::Mat& source, const int S=3, const float sigma=1.6, const int min_size=4, const bool use_diff=true) {
    // 首先上采样得到第一张图
    cv::Mat first_image;
    cv::resize(source, first_image, {0, 0}, 2, 2, cv::INTER_LINEAR);
    const float first_sigma = std::sqrt(sigma * sigma - (2 * 0.5) * (2 * 0.5)); // 1.2489995996796799
    cv::GaussianBlur(first_image, first_image, {0, 0}, first_sigma, first_sigma);
    // 计算有几组图像, 最小分辨率是 4x4
    const int min_length = std::min(first_image.rows, first_image.cols);
    const int octaves_num = std::floor(std::log(min_length) / std::log(2) - min_size);
    // 计算任意一组图像的方差序列
    const int images_num = S + 3;
    const float k = std::pow(2, 1.f / S); // 尺度的累乘系数, 每次尺度 *= k, k = 1.25, k^2 = 1.58, 跟 first_sigma = 1.6 十分接近, 大致保持了 k 倍关系
    std::vector<float> sigmas_list(images_num, 0);
    sigmas_list[0] = sigma;
    for(int i = 1; i < images_num; ++i) {
        const float temp = std::pow(k, i - 1) * sigma; // 当前这张图象可以通过第一张图象以 temp 的高斯模糊生成
        sigmas_list[i] = use_diff ? std::sqrt((k * k - 1) * temp * temp) : temp; // 是否用 diff, 使用高斯模糊的半群性质
    }
    // 共 octaves_num 组, 每组有 S + 3 张图像, 对应尺度逐一做高斯模糊
    std::vector< std::vector<cv::Mat> > gaussi_scaled_pyramid;
    cv::Mat cur_scale, temp;
    for(int i = 0;i < octaves_num; ++i) {
        std::vector<cv::Mat> this_octave;
        this_octave.reserve(images_num);
        if(i == 0) cur_scale = first_image.clone(); // 如果是第一组图像的第一张图象, 直接用上采样两倍的那个图像
        else { // 否则, 直接取上一组的倒数第三张图像作为起始图像, 然后下采样为原来的一半, 相当于尺度乘以 2
            const cv::Mat& refer = gaussi_scaled_pyramid[i - 1][(images_num - 1) - 3];
            cv::resize(refer, cur_scale, {refer.cols / 2, refer.rows / 2}, 0, 0, cv::INTER_LINEAR);
        }
        this_octave.emplace_back(cur_scale.clone()); // 起始图像都已经做了高斯模糊了, first_image 和上一组的倒数第三张都做过高斯模糊
        for(int j = 1; j < images_num; ++j) {
            if(use_diff) { // 如果方差用的 diff, 就用上一张图像继续高斯模糊
                cv::GaussianBlur(cur_scale, cur_scale, {0, 0}, sigmas_list[j], sigmas_list[j]);
                this_octave.emplace_back(cur_scale.clone()); // 这里不用 clone() 很坑爹啊
            } else { // 否则直接从这一组的第一张图象, 直接高斯模糊
                 cv::GaussianBlur(cur_scale, temp, {0, 0}, sigmas_list[j], sigmas_list[j]);
                 this_octave.emplace_back(temp.clone());
            }
        }
        gaussi_scaled_pyramid.emplace_back(std::move(this_octave)); // 记录这一组的高斯模糊图像
    }
    // 得到高斯差分金字塔
    std::vector< std::vector<cv::Mat> > DOG_pyramid;
    for(int i = 0;i < octaves_num; ++i) {
        std::vector<cv::Mat> this_octave;
        this_octave.reserve(images_num - 1);
        for(int j = 1;j < images_num; ++j)
            this_octave.emplace_back(gaussi_scaled_pyramid[i][j] - gaussi_scaled_pyramid[i][j - 1]);
        DOG_pyramid.emplace_back(std::move(this_octave));
    }
    return DOG_pyramid;
}


// 判断坐标 j 的点, 其值为 center 是否是上下三层的 3x3x3 的局部极值
inline bool is_local_extremum(const float center, const float* const down, const float* const mid, const float* const up, const int j, const int W) {
    return (center > 0 and center > mid[j - 1] and center > mid[j + 1] and
           center > mid[j - 1 - W] and center > mid[j - W] and center > mid[j + 1 - W] and
           center > mid[j - 1 + W] and center > mid[j + W] and center > mid[j + 1 + W] and
           center > down[j - 1] and center > down[j] and center > down[j + 1] and
           center > down[j - 1 - W] and center > down[j - W] and center > down[j + 1 - W] and
           center > down[j - 1 + W] and center > down[j + W] and center > down[j + 1 + W] and
           center > up[j - 1] and center > up[j] and center > up[j + 1] and
           center > up[j - 1 - W] and center > up[j - W] and center > up[j + 1 - W] and
           center > up[j - 1 + W] and center > up[j + W] and center > up[j + 1 + W])
           or
           (center < 0 and center < mid[j - 1] and center < mid[j + 1] and
           center < mid[j - 1 - W] and center < mid[j - W] and center < mid[j + 1 - W] and
           center < mid[j - 1 + W] and center < mid[j + W] and center < mid[j + 1 + W] and
           center < down[j - 1] and center < down[j] and center < down[j + 1] and
           center < down[j - 1 - W] and center < down[j - W] and center < down[j + 1 - W] and
           center < down[j - 1 + W] and center < down[j + W] and center < down[j + 1 + W] and
           center < up[j - 1] and center < up[j] and center < up[j + 1] and
           center < up[j - 1 - W] and center < up[j - W] and center < up[j + 1 - W] and
           center < up[j - 1 + W] and center < up[j + W] and center < up[j + 1 + W]);
}


struct keypoint_type {
    cv::Point pos; // 位置坐标
    int size;      // 关键点的尺寸
    float response;// 插值之后的 DOG 响应值
    keypoint_type(const int x, const int y, const int _size, const float _response)
        : pos(x, y), size(_size), response(_response) {}
};

std::vector<keypoint_type> sift_detect_keypoints(
        const cv::Mat& _source,
        const int S=3,
        const float sigma=1.6,
        const int min_size=4,
        const float dog_threshhold=0.09,
        const float gamma=10.0) {
    // 转化成灰度图, 类型 float
    cv::Mat source;
    if(_source.channels() > 1)
        cv::cvtColor(_source, source, cv::COLOR_BGR2GRAY);
    assert(source.channels() == 1);
    source.convertTo(source, CV_32FC1);
    // 首先构建差分金字塔
    const auto DOG_pyramid = build_DOG_pyramid(source, S, sigma, min_size);
    // 寻找尺度空间极值 (x, y, sigma)
    const float threshold = std::floor(0.5 * dog_threshhold / S * 255);
    std::vector<keypoint_type> keypoints;
    const int octaves_num = DOG_pyramid.size(); // 几组
    const int images_num = DOG_pyramid[0].size(); // 每组有几张图像
    for(int o = 0;o < octaves_num; ++o) { // 遍历每一组
        for(int s = 1;s < images_num - 1; ++s) { // 从中间的几层图像开始, 所以是 1 ~ images_num - 1
            // 获取上中下三组图像的引用
            const auto& down_image = DOG_pyramid[o][s - 1];
            const auto& mid_image = DOG_pyramid[o][s];
            const auto& up_image = DOG_pyramid[o][s + 1];
            const int H = mid_image.rows, W = mid_image.cols;
            const int H_1 = H - 1, W_1 = W - 1;
            // 遍历每一个点, 判断每个点是不是局部极值
            for(int i = 1;i < H_1; ++i) {
                const float* const down = down_image.ptr<float>() + i * W;
                const float* const mid = mid_image.ptr<float>() + i * W;
                const float* const up = up_image.ptr<float>() + i * W;
                for(int j = 1;j < W_1; ++j) {
                    const float center = mid[j];
                    if(std::abs(center) < threshold)  // 去掉一些响应值过小的关键点, 这里的响应值是 DOG 响应值
                        continue;
                    if(is_local_extremum(center, down, mid, up, j, W)) { // 如果是局部极值
                        /*
                        const float temp = std::pow(2, o - 1);
                        const int size = sigma * std::pow(2, s / S)  * temp * 2; // 1.414
                        keypoints.emplace_back(j * temp, i * temp, size, center);
                        */
                        // 做更精准的插值, 找到更好的坐标以及尺度
                        constexpr int max_iters = 5;  // 最多迭代 5 次
                        bool convergence = false; // 是否收敛
                        float response; // 插值之后的响应值极值
                        int i2 = i, j2 = j, s2 = s;  // 注意这里的 i2, j2, s2 是在不断更新的, 相邻三层 down, mid, up 的行指针也要重新赋值
                        float D_xx, D_yy, D_sigma_2, D_xy, D_x_sigma, D_y_sigma;
                        for(int t = 0;t < max_iters; ++t) {
                            // 首先获取新的指针
                            const float* mid2 = mid_image.ptr<float>() + i2 * W;
                            const float* down2 = down_image.ptr<float>() + i2 * W;
                            const float* up2 = up_image.ptr<float>() + i2 * W;
                            // 首先对 i2, j2, s2  这个点局部求一阶导数和二阶海斯矩阵
                            Eigen::Matrix<float, 3, 1> D_x;
                            D_x << 0.5f * (mid2[j2 + 1] - mid2[j2 - 1]), 0.5f * (mid2[j2 + W] - mid2[j2 - W]), 0.5f * (up2[j2] - down2[j2]);
                            Eigen::Matrix<float, 3, 3> DD_x;
                            D_xx = mid2[j2 + 1] + mid2[j2 - 1] - 2 * mid2[j2];
                            D_yy = mid2[j2 + W] + mid2[j2 - W] - 2 * mid2[j2];
                            D_sigma_2 = up2[j2] + down2[j2] - 2 * mid2[j2];
                            D_xy = (mid2[j2 + W + 1] + mid2[j2 - W - 1] - mid2[j2 + W - 1] - mid2[j2 - W + 1]) / 4.f;
                            D_x_sigma = (up2[j2 + 1] + down2[j2 - 1] - up2[j2 - 1] - down2[j2 + 1]) / 4.f;
                            D_y_sigma = (up2[j2 + W] + down2[j2 - W] - up2[j2 - W] - down2[j2 + W]) / 4.f;
                            DD_x << D_xx, D_xy, D_x_sigma,
                                    D_xy, D_yy, D_y_sigma,
                                    D_x_sigma, D_y_sigma, D_sigma_2;
                            // 求极值点的偏移量
                            const Eigen::Matrix<float, 3, 1> offset = - DD_x.inverse() * D_x;
                            // 判断是否收敛
                            if(std::abs(offset(0)) < 0.5f and std::abs(offset(1)) < 0.5f and std::abs(offset(2)) < 0.5f) {
                                convergence = true;
                                response = mid2[j2] + 0.5 * D_x.transpose() * offset; // 更新这个地方的极值
                                break;
                            }
                            // 根据偏移量关键点在尺度空间的位置
                            j2 += int(offset(0));
                            i2 += int(offset(1));
                            s2 += int(offset(2));
                            // 如果越界了, 退出迭代
                            if(s2 < 1 or s2 > images_num - 2 or i2 < 1 or i2 >= H_1 or j2 < 1 or j2 >= W_1)
                                break;
                        }
                        // 如果结果是收敛了
                        if(convergence) {
                            // 继续下一步, 去除边缘响应太强的点
                            const float trace = D_xx + D_yy;
                            const float det = D_xx * D_yy - D_xy * D_xy;
                            if(det > 0 and (trace * trace) / det < (gamma + 1) * (gamma + 1) / gamma) {
                                // 根据极值点 (j2, i2, s2) 恢复到原始分辨率的大小
                                const float temp = std::pow(2, o - 1);
                                const int size = sigma * std::pow(2, s2 / S)  * temp * 2;
                                keypoints.emplace_back(j2 * temp, i2 * temp, size, response);
                            }
                        }
                    }
                }
            }
        }
    }
    return keypoints;
}


int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;
    // 根据图片路径读取图像
    const char* source_path = "./images/input/a1476-IMG_2647.png";
    const auto source_image = cv::imread(source_path);
    assert(not source_image.empty() and "读取图像失败 !");
    // sift 检测关键点
    const auto keypoints = sift_detect_keypoints(source_image);
    // 展示与保存
    auto display = source_image.clone();
    for(const auto& point : keypoints)
        cv::circle(display, point.pos, point.size, CV_RGB(0, 255, 0), 1);
    cv_show(display);
    cv_write(display, "./images/output/keypoints_12.png");
    return 0;
}
