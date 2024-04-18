//C++
#include <cmath>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <iostream>
// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Eigen3
#include <Eigen/Core>
#include <Eigen/Dense>


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

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
		cv::Mat padded_image;
		cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
		return padded_image;
	}

    cv::Mat cv_concat(const std::vector<cv::Mat> images, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(images, result);
        else cv::vconcat(images, result);
        return result;
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    template<typename T>
    cv::Mat toint8(const std::vector<T>& source, const int H, const int W) {
        cv::Mat result(H, W, CV_8UC1);
        const int length = H * W;
        for(int i = 0;i < length; ++i) result.data[i] = std::abs(source[i]);
        return result;
    }

    cv::Mat get_rotated(const cv::Mat& source, const int angle, const cv::Size& _size, const cv::Point2f& center) {
        cv::Mat rotated_image;
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::warpAffine(source, rotated_image, rot_mat, _size, cv::INTER_LINEAR);
        return rotated_image;
    }

    cv::Mat my_rotate(const cv::Mat& source) {
        const int H = source.rows;
        const int W = source.cols;
        cv::Mat res(W, H, CV_8UC3);
        for(int i = 0;i < H; ++i) {
            for(int j = 0;j < W; ++j) {
                res.at<cv::Vec3b>(W - 1 - j, i)[0] = source.at<cv::Vec3b>(i, j)[0];
                res.at<cv::Vec3b>(W - 1 - j, i)[1] = source.at<cv::Vec3b>(i, j)[1];
                res.at<cv::Vec3b>(W - 1 - j, i)[2] = source.at<cv::Vec3b>(i, j)[2];
            }
        }
        return res;
    }
    // 代码取自 https://blog.csdn.net/qq_34784753/article/details/69379135
    double generateGaussianNoise(double mu, double sigma) {
        const double epsilon = std::numeric_limits<double>::min();
        static double z0, z1;
        static bool flag = false;
        flag = !flag;
        if (!flag) return z1 * sigma + mu;
        double u1, u2;
        do {
            u1 = std::rand() * (1.0 / RAND_MAX);
            u2 = std::rand() * (1.0 / RAND_MAX);
        } while (u1 <= epsilon);
        z0 = std::sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2);
        z1 = std::sqrt(-2.0 * log(u1)) * sin(2 * CV_PI * u2);
        return z0 * sigma + mu;
    }
    // 为图像添加高斯噪声
    // 代码取自 https://blog.csdn.net/qq_34784753/article/details/69379135
    cv::Mat add_gaussian_noise(const cv::Mat &source) {
        cv::Mat res = source.clone();
        int channels = res.channels();
        int rows_number = res.rows;
        int cols_number = res.cols * channels;
        if (res.isContinuous()) {
            cols_number *= rows_number;
            rows_number = 1;
        }
        for (int i = 0; i < rows_number; i++) {
            for (int j = 0; j < cols_number; j++) {
                int val = res.ptr<uchar>(i)[j] +
                    generateGaussianNoise(2, 1.5) * 16;
                res.ptr<uchar>(i)[j] = cv::saturate_cast<uchar>(val);
            }
        }
        return res;
    }

    // 画一下 harris 响应值的热力图, 可以参考之前何恺明去雾的那次代码
}



// 高斯模糊
void gausssi_filter(int* const source, double* const target, const int radius, const int H, const int W) {
    // 首先创建一维的高斯模板
    const double sigma = radius * 1. / 3;
    const int kernel_size = (radius << 1) + 1;
    std::vector<double> weights(kernel_size, 0);
    for(int i = -radius;i <= radius; ++i)
        weights[i + radius] = 1. / sigma * std::exp(-i * i / (2 * sigma * sigma));
    double weights_sum = 0.0;
    for(int i = 0;i < kernel_size; ++i) weights_sum += weights[i];
    if(std::abs(weights_sum - 0) > 1e-10) for(int i = 0;i < kernel_size; ++i) weights[i] /= weights_sum;
    // 先开始 x 方向的高斯卷积
    std::vector<double> temp(H * W);
    const int W2 = W - radius;
    for(int i = 0;i < H; ++i) {
        int* const row_ptr = source + i * W;
        double* const res_ptr = temp.data() + i * W;
        for(int j = radius; j < W2; ++j) {
            double cur_sum = 0.0;
            for(int k = -radius; k <= radius; ++k)
                cur_sum += weights[radius + k] * row_ptr[j + k];
            res_ptr[j] = cur_sum;
        }
    }
    // 继续做 y 方向上的高斯卷积
    const int H2 = H - radius;
    for(int j = radius; j < W2; ++j) {
        for(int i = radius; i < H2; ++i) {
            double cur_sum = 0.0;
            for(int k = -radius; k <= radius; ++k)
                cur_sum += weights[radius + k] * temp[i * W + j + k];
            target[i * W + j] = cur_sum;
        }
    }
}



void box_filter(int* const source, double* const target, const int radius, const int H, const int W) {
    // 储存这一行的结果
    const int kernel_size = (radius << 1) + 1;
    std::vector<double> buffer(W);
    // 遍历前 kernel_size 行, 计算每一列的和
    for(int i = 0; i < kernel_size; ++i) {
        int* const row_ptr = source + i * W;
        for(int j = 0; j < W; ++j) buffer[j] += row_ptr[j];
    }
    // 计算剩下每一行的
    const int H2 = H - radius;
    const int W2 = W - 2 * radius;
    for(int i = radius;i < H2; ++i) {
        // 计算第一个位置的和
        double cur_sum = 0;
        for(int j = 0;j < kernel_size; ++j) cur_sum += buffer[j];
        // 记录这第一个位置的结果
        const int _beg = i * W + radius;
        target[_beg] = cur_sum;
        // 开始向右挪动
        for(int j = 1; j < W2; ++j) {
            cur_sum = cur_sum - buffer[j - 1] + buffer[j - 1 + kernel_size];
            target[_beg + j] = cur_sum;
        }
        // 这一行移动完毕, 换到下一行, 更新 buffer
        if(i != H2 - 1) {
            int* up_ptr = source + (i - radius) * W;
            int* down_ptr = source + (i + radius + 1) * W;
            for(int j = 0;j < W; ++j) buffer[j] = buffer[j] - up_ptr[j] + down_ptr[j];
        }
    }
    const int length = H * W;
    const int area = kernel_size * kernel_size;
    for(int i = 0;i < length; ++i) target[i] /= area;
}


// 给定图像 source, 计算 Ix^2, Iy^2, IxIy 的加权结果
std::tuple< std::vector<double>, std::vector<double>, std::vector<double> > compute_weighted_IxIy(
        const cv::Mat& source, const int radius=2, const bool use_gaussi=false) {
    // 获取图像信息
    const int H = source.rows;
    const int W = source.cols;
    const int length = H * W;
    // 先获取 x, y 两个方向上的梯度, 这里用的是 Sobel
    std::vector<int> gradients_x(length, 0);
    std::vector<int> gradients_y(length, 0);
    const int H_1 = H - 1, W_1 = W - 1;
    for(int i = 1; i < H_1; ++i) {
        const uchar* const row_ptr = source.data + i * W;
        int* const x_ptr = gradients_x.data() + i * W;
        int* const y_ptr = gradients_y.data() + i * W;
        for(int j = 1; j < W_1; ++j) {
            // 计算 Sobel 梯度
            x_ptr[j] = 2 * row_ptr[j + 1] + row_ptr[j + 1 + W] + row_ptr[j + 1 - W] - (2 * row_ptr[j - 1] + row_ptr[j - 1 + W] + row_ptr[j - 1 - W]);
            y_ptr[j] = 2 * row_ptr[j + W] + row_ptr[j + W + 1] + row_ptr[j + W - 1] - (2 * row_ptr[j - W] + row_ptr[j - W + 1] + row_ptr[j - W - 1]);
        }
    }
    // 计算 xx, yy, xy
    std::vector<int> gradients_xx(length, 0), gradients_yy(length, 0), gradients_xy(length, 0);
    for(int i = 0;i < length; ++i) gradients_xx[i] = gradients_x[i] * gradients_x[i];
    for(int i = 0;i < length; ++i) gradients_yy[i] = gradients_y[i] * gradients_y[i];
    for(int i = 0;i < length; ++i) gradients_xy[i] = gradients_x[i] * gradients_y[i];
    // 计算每一个点的加权之和, 先储存起来, 窗口函数可以是均值, 也可以是高斯加权
    std::vector<double> xx_sum(length, 0), yy_sum(length, 0), xy_sum(length, 0);
    const auto window_function = use_gaussi ? gausssi_filter : box_filter;
    window_function(gradients_xx.data(), xx_sum.data(), radius, H, W);
    window_function(gradients_yy.data(), yy_sum.data(), radius, H, W);
    window_function(gradients_xy.data(), xy_sum.data(), radius, H, W);

    return std::make_tuple(xx_sum, yy_sum, xy_sum);
}



using key_points_type = std::vector< std::tuple<double, int, int> >;
key_points_type harris_corner_detect(
        const cv::Mat& source,
        const int radius=2,
        const double alpha=0.04,
        const double threshold=1e5,
        const int point_num=-1,
        const bool use_gaussi=false) {
    if(source.channels() > 1) {
        std::cout << "只接收单通道灰度图的角点检测 !" << std::endl;
        return key_points_type();
    }
    // 获取图像信息
    const int H = source.rows;
    const int W = source.cols;
    const int length = H * W;
    // 计算 Ix, Iy, IxIy 的加权结果
    const auto gradients_info = compute_weighted_IxIy(source, radius, use_gaussi);
    // 开始计算每一个点的 harris 响应值
    std::vector<double> R(length, 0);
    const int H_radius = H - radius;
    const int W_radius = W - radius;
    for(int i = radius; i < H_radius; ++i) {
        const double* const xx = std::get<0>(gradients_info).data() + i * W;
        const double* const yy = std::get<1>(gradients_info).data() + i * W;
        const double* const xy = std::get<2>(gradients_info).data() + i * W;
        double* const res_ptr = R.data() + i * W;
        for(int j = radius; j < W_radius; ++j) {
            // 计算这个点所在窗口的加权和
            const double A = xx[j] / 255;
            const double B = yy[j] / 255;
            const double C = xy[j] / 255;
            // 计算 λ1 和 λ2
            const double det = A * B - C * C;
            const double trace = A + B;
            res_ptr[j] = det - alpha * (trace * trace);
        }
    }
    // 在这里看看边缘
    float new_threshold = 3.f;
    do {
        cv::Mat display = source.clone();
        for(int i = 0;i < length; ++i) {
            if(R[i] < -new_threshold) {
                cv::circle(display, cv::Point(i % W, i / W), 2, cv::Scalar(0, 0, 255), 1);
            }
        }
        cv_show(display);
    } while (0);
    // 准备一个结果
    key_points_type detection;
    // 需要进行局部非极大化抑制
    const int H_1 = H - 1, W_1 = W - 1;
    for(int i = 1; i < H_1; ++i) {
        double* row_ptr = R.data() + i * W;
        for(int j = 1; j < W_1; ++j) {
            const double center = row_ptr[j];
            if(center > threshold) {
                if(center > row_ptr[j - 1] and center > row_ptr[j + 1] and
                   center > row_ptr[j - 1 - W] and center > row_ptr[j - W] and center > row_ptr[j + 1 - W] and
                   center > row_ptr[j - 1 + W] and center > row_ptr[j + W] and center > row_ptr[j + 1 + W])
                    detection.emplace_back(center, i, j);
            }
        }
    }
    // 取前 point_sum 个
    if(point_num > 0 and detection.size() > point_num) {
        // 按照响应值大小排序
        std::sort(detection.begin(), detection.end());
        std::reverse(detection.begin(), detection.end());
        detection.erase(detection.begin() + point_num, detection.end());
        detection.shrink_to_fit();
    }
    std::cout << "收集到  " << detection.size() << " 个角点 " << std::endl;
    return detection;
}


// Shi - Tomasi 的角点检测
key_points_type shi_tomasi_corner_detect(
        const cv::Mat& source,
        const int radius=2,
        const double threshold=1e5,
        const int point_num=-1,
        const bool use_gaussi=false) {
    // 获取图像信息
    const int H = source.rows;
    const int W = source.cols;
    const int length = H * W;
    // 计算 Ix, Iy, IxIy 的加权结果
    const auto gradients_info = compute_weighted_IxIy(source, radius, use_gaussi);
    // 开始计算每一个点的 harris 响应值
    std::vector<double> R(length, 0);
    const int H_radius = H - radius;
    const int W_radius = W - radius;
    for(int i = radius; i < H_radius; ++i) {
        const double* const xx = std::get<0>(gradients_info).data() + i * W;
        const double* const yy = std::get<1>(gradients_info).data() + i * W;
        const double* const xy = std::get<2>(gradients_info).data() + i * W;
        double* const res_ptr = R.data() + i * W;
        for(int j = radius; j < W_radius; ++j) {
            // 构建 M 矩阵
            Eigen::MatrixXd M(2, 2);
            M << xx[j] / 255, xy[j] / 255, xy[j] / 255, yy[j] / 255;
            // 求 M 的特征值
            Eigen::EigenSolver<Eigen::MatrixXd> solver(M);
            const auto& result = solver.eigenvalues().real();
            res_ptr[j] = std::min(result[0], result[1]);
        }
    }
    // 准备一个结果
    key_points_type detection;
    // 需要进行局部非极大化抑制
    const int H_1 = H - 1, W_1 = W - 1;
    for(int i = 1; i < H_1; ++i) {
        double* row_ptr = R.data() + i * W;
        for(int j = 1; j < W_1; ++j) {
            const double center = row_ptr[j];
            if(center > threshold) {
                if(center > row_ptr[j - 1] and center > row_ptr[j + 1] and
                   center > row_ptr[j - 1 - W] and center > row_ptr[j - W] and center > row_ptr[j + 1 - W] and
                   center > row_ptr[j - 1 + W] and center > row_ptr[j + W] and center > row_ptr[j + 1 + W])
                    detection.emplace_back(center, i, j);
            }
        }
    }
    // 取前 point_sum 个
    if(point_num > 0 and detection.size() > point_num) {
        // 按照响应值大小排序
        std::sort(detection.begin(), detection.end());
        std::reverse(detection.begin(), detection.end());
        detection.erase(detection.begin() + point_num, detection.end());
        detection.shrink_to_fit();
    }
    std::cout << "收集到  " << detection.size() << " 个角点 " << std::endl;
    return detection;
}




void demo_1() {
    const std::string save_dir("./images/output/1/");
    std::string origin_path("./images/input/harris_demo_1.png"); // harris_demo_1.png
    const auto origin_image = cv::imread(origin_path);
    if(origin_image.empty()) {
        std::cout << "读取图像 " << origin_path << " 失败 !" << std::endl;
        return;
    }
    // 转成灰度图
    cv::Mat origin_gray;
    cv::cvtColor(origin_image, origin_gray, cv::COLOR_BGR2GRAY);

    // 写一个展示用的函数
    auto show_harris = [](const cv::Mat& source, const key_points_type& harris_result, const std::string save_name, const int radius=2, const int thickness=4)
            -> void {
        // 画出来
        cv::Mat display = source.clone();
        for(const auto& item : harris_result)
            cv::circle(display, cv::Point(std::get<2>(item), std::get<1>(item)), radius, cv::Scalar(0, 0, 255), thickness);
        // 展示
        cv_show(display);
        cv_write(display, save_name);
    };

    // 检测 Harris 角点
    auto harris_result = harris_corner_detect(origin_gray, 2, 0.04, 4e3, 0);
    show_harris(origin_image, harris_result, save_dir + "original.png");
    const int best_size = harris_result.size();

    // 增大局部窗口的半径
    harris_result = harris_corner_detect(origin_gray, 5, 0.04, 4e3, 0);
    show_harris(origin_image, harris_result, save_dir + "original_radius.png");

    // 增加参数 α
    harris_result = harris_corner_detect(origin_gray, 2, 0.06, 4e3, 0);
    show_harris(origin_image, harris_result, save_dir + "original_alpha.png");

    // 证明对亮度或对比度改变的影响
    cv::Mat darken_image = 0.5 * origin_image, darken_gray;
    darken_image.convertTo(darken_image, CV_8UC3);
    cv::cvtColor(darken_image, darken_gray, cv::COLOR_BGR2GRAY);
    harris_result = harris_corner_detect(darken_gray, 2, 0.04, 2e2, best_size);
    show_harris(darken_image, harris_result, save_dir + "darken.png");
    cv::Mat darken_image_2 = origin_image - 50, darken_gray_2;
    darken_image_2.convertTo(darken_image_2, CV_8UC3);
    cv::cvtColor(darken_image, darken_gray_2, cv::COLOR_BGR2GRAY);
    harris_result = harris_corner_detect(darken_gray_2, 2, 0.04, 2e2, best_size);
    show_harris(darken_image_2, harris_result, save_dir + "darken_2.png");


    // 证明旋转不变性
    const auto my_rotated_image = my_rotate(origin_image);
    cv::Mat my_rotated_gray;
    cv::cvtColor(my_rotated_image, my_rotated_gray, cv::COLOR_BGR2GRAY);
    harris_result = harris_corner_detect(my_rotated_gray, 2, 0.04, 4e3, 0);
    show_harris(my_rotated_image, harris_result, save_dir + "rotated_1.png");
    // 内容有损的旋转
    const int H = origin_image.rows;
    const int W = origin_image.cols;
    const auto rotated_image = get_rotated(origin_image, 45, cv::Size(W, H), cv::Point2f(W / 2, H / 2));
    cv::Mat rotated_gray;
    cv::cvtColor(rotated_image, rotated_gray, cv::COLOR_BGR2GRAY);
    harris_result = harris_corner_detect(rotated_gray, 2, 0.04, 4e3, 0);
    show_harris(rotated_image, harris_result, save_dir + "rotated_2.png");

    // 镜像
    cv::Mat flipped_image, flipped_gray;
    cv::flip(origin_image, flipped_image, 1);
    cv::cvtColor(flipped_image, flipped_gray, cv::COLOR_BGR2GRAY);
    harris_result = harris_corner_detect(flipped_gray, 2, 0.04, 4e3, 0);
    show_harris(flipped_image, harris_result, save_dir + "flip_1.png");

    // 证明不满足 尺度不变性
    cv::Mat resized_image, resized_gray;
    cv::resize(origin_image, resized_image, cv::Size(W / 8, H / 8));
    cv::cvtColor(resized_image, resized_gray, cv::COLOR_BGR2GRAY);
    harris_result = harris_corner_detect(resized_gray, 2, 0.04, 4e3, 0);
    show_harris(resized_image, harris_result, save_dir + "scaled.png", 1, 1);

    // 考虑抗噪性
    const auto noise_image = add_gaussian_noise(origin_image);
    cv::Mat noise_gray;
    cv::cvtColor(noise_image, noise_gray, cv::COLOR_BGR2GRAY);
    harris_result = harris_corner_detect(noise_gray, 2, 0.04, 4e3, 0);
    show_harris(noise_image, harris_result, save_dir + "noisy.png");

    // Shi-Tomasi 角点检测方法
    harris_result = shi_tomasi_corner_detect(origin_gray, 2, 30, best_size);
    show_harris(origin_image, harris_result, save_dir + "shi_tomasi.png");


    // Harris 检测边缘 ????

}





int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    // Laplace 检测边缘
    demo_1();

    return 0;
}
