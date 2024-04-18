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
    cv::Mat touint8(const std::vector<T>& source, const int H, const int W) {
        cv::Mat res(H, W, CV_8UC1);
        const int length = H * W;
        for(int i = 0;i < length; ++i) res.data[i] = std::abs(source[i]);
        return res;
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
}




namespace SUSAN_CORNER {

    using key_points_type = std::vector< std::pair<int, int> >;
    key_points_type susan_corner_detect(const cv::Mat& source, const int radius, const int nms_radius=5, const int threshold=20) {
        // 获取图像信息
        const int H = source.rows;
        const int W = source.cols;
        const int length = H * W;
        cv::Mat gray_image;
        if(source.channels() == 3) cv::cvtColor(source, gray_image, cv::COLOR_BGR2GRAY);
        else gray_image = source;
        // 准备一个模板
        const int max_size = (2 * radius + 1) * (2 * radius + 1);
        std::vector<int> weights(max_size, 0);
        std::vector<int> offset(max_size, 0);
        const int radius_2 = (radius + 0.5) * (radius + 0.5);
        int cnt = 0;
        for(int i = -radius;i <= radius; ++i)
            for(int j = -radius; j <= radius; ++j)
                if(i * i + j * j <= radius_2) {
                    weights[cnt] = 1;
                    offset[cnt++] = i * W + j;
                }
        // 判断是否是同质点的阈值
        const int half_area = cnt / 2;
        // 存放每个点的响应值
        std::vector<int> response(length, 0);
        // 开始计算角点响应值
        const int H_radius = H - radius, W_radius = W - radius;
        for(int i = radius; i < H_radius; ++i) {
            uchar* const row_ptr = gray_image.data + i * W;
            int* const res_ptr = response.data() + i * W;
            for(int j = radius; j < W_radius; ++j) {
                // 首先, 判断这个圆形窗口和中心点的差小于阈值的
                const uchar center = row_ptr[j];
                double number = 0;
                for(int k = 0;k < cnt; ++k)
                    if(std::abs(row_ptr[j + offset[k]] - center) < threshold)
                        number += weights[k];
                // 把中心点减去
                --number;
                // 如果同质点的加权之和小于一半的面积
                if(number < half_area)
                    res_ptr[j] = half_area - number;
            }
        }
        // cv_show(20 * touint8(response, H, W));
        // cv_write(30 * touint8(response, H, W), "./images/output/corner_detection/SUSAN_response.png");
        // 准备结果
        key_points_type detection;
        // 非极大值抑制
        for(int i = nms_radius; i < H - nms_radius; ++i) {
            const int* const row_ptr = response.data() + i * W;
            for(int j = nms_radius; j < W - nms_radius; ++j) {
                // 找局部区域的所有点判断是否是极大值
                const int center = row_ptr[j];
                // center = 0 的点很可能就是平坦区域, 不参与比较
                if(center > 1) {
                    bool flag = true;
                    for(int x = -nms_radius; x <= nms_radius; ++x) {
                        const int* const cur_ptr = row_ptr + j + x * W;
                        for(int y = -nms_radius; y <= nms_radius; ++y) {
                            if(cur_ptr[y] >= center) {
                                if(x == 0 and y == 0) continue;
                                flag = false; break;
                            }
                        }
                        if(!flag) break;
                    }
                    // 如果真的是极大值
                    if(flag) detection.emplace_back(i, j);
                }
            }
        }
        return detection;
    }


    void corner_detect_demo() {
        const std::string save_dir("./images/output/corner_detection/");
        std::string origin_path("./images/input/corner/harris_demo_1.png");
        const auto origin_image = cv::imread(origin_path);
        if(origin_image.empty()) {
            std::cout << "读取图像 " << origin_path << " 失败 !" << std::endl;
            return;
        }
        // 写一个展示的函数
        auto corner_display = [](const cv::Mat& source, const key_points_type& detection, const std::string save_path, const int radius=3, const int thickness=4)
                -> void {
            cv::Mat display = source.clone();
            for(const auto& item : detection)
            cv::circle(display, cv::Point(item.second, item.first), radius, cv::Scalar(0, 255, 0), thickness);
            cv_show(display);
            cv_write(display, save_path);
        };
        cv::Mat another_image;

        // 朴素的 SUSAN
        auto detection = susan_corner_detect(origin_image, 3, 10, 45);
        corner_display(origin_image, detection, save_dir + "horse.png");

        another_image = cv::imread("./images/input/corner/corner_2.png");
        detection = susan_corner_detect(another_image, 3, 10, 45);
        corner_display(another_image, detection, save_dir + "table.png");

        another_image = cv::imread("./images/input/corner/corner_1.png");
        detection = susan_corner_detect(another_image, 3, 5, 10);
        corner_display(another_image, detection, save_dir + "toy.png");

        another_image = cv::imread("./images/input/corner/corner_3.png");
        detection = susan_corner_detect(another_image, 3, 4, 30);
        corner_display(another_image, detection, save_dir + "house.png", 2, 2);

        another_image = cv::imread("./images/input/corner/a0515-NKIM_MG_6602.png");
        detection = susan_corner_detect(another_image, 3, 8, 50);
        corner_display(another_image, detection, save_dir + "French.png", 2, 2);

        another_image = cv::imread("./images/input/corner/a0423-07-06-02-at-07h35m36-s_MG_1355.png");
        detection = susan_corner_detect(another_image, 3, 7, 45);
        corner_display(another_image, detection, save_dir + "gugong.png", 2, 2);

        another_image = cv::imread("./images/input/corner/a0367-IMG_0338.png");
        detection = susan_corner_detect(another_image, 3, 7, 60);
        corner_display(another_image, detection, save_dir + "city.png", 2, 2);

        another_image = cv::imread("./images/input/corner/a0516-IMG_4420.png");
        detection = susan_corner_detect(another_image, 3, 10, 30);
        corner_display(another_image, detection, save_dir + "car.png", 2, 2);

        // 如果是噪声的话
        const auto noisy_image = add_gaussian_noise(cv::imread("./images/input/corner/a0515-NKIM_MG_6602.png"));
        detection = susan_corner_detect(noisy_image, 3, 10, 30);
        corner_display(noisy_image, detection, save_dir + "noisy.png", 2, 2);

        // 如果是旋转的话
        const auto rotated_image = my_rotate(cv::imread("./images/input/corner/a0515-NKIM_MG_6602.png"));
        detection = susan_corner_detect(rotated_image, 3, 10, 45);
        corner_display(rotated_image, detection, save_dir + "rotated.png", 2, 2);

        // 如果是灰度变化
        cv::Mat darken_image = 0.5 * cv::imread("./images/input/corner/a0515-NKIM_MG_6602.png");
        darken_image.convertTo(darken_image, CV_8UC3);
        detection = susan_corner_detect(darken_image, 3, 10, 23);
        corner_display(darken_image, detection, save_dir + "darken.png", 2, 2);

        // 如果是尺度变化
        cv::Mat scaled_image = cv::imread("./images/input/corner/a0515-NKIM_MG_6602.png");
        cv::resize(scaled_image, scaled_image, cv::Size(scaled_image.cols / 4, scaled_image.rows / 4));
        scaled_image.convertTo(scaled_image, CV_8UC3);
        detection = susan_corner_detect(scaled_image, 3, 2, 45);
        corner_display(scaled_image, detection, save_dir + "scaled.png", 1, 1);
    }
}





namespace SUSAN_EDGE {

    cv::Mat susan_edge_detect(const cv::Mat& source, const int radius=3, const int threshold=30, const double edge_ratio=0.75) {
        // 获取图像信息
        cv::Mat gray;
        if(source.channels() == 3) cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);
        else gray = source;
        const int H = gray.rows;
        const int W = gray.cols;
        const int length = H * W;
        // 准备一个圆形模板
        const int max_size = (2 * radius + 1) * (2 * radius + 1);
        std::vector<int> weights(max_size, 0);
        std::vector<int> offset(max_size, 0);
        std::vector<int> dis_i(max_size, 0), dis_j(max_size, 0);
        const int radius_2 = (radius + 0.5) * (radius + 0.5);
        int cnt = 0;
        for(int i = -radius;i <= radius; ++i) // 竖直方向
            for(int j = -radius; j <= radius; ++j) // 水平方向
                if(i * i + j * j <= radius_2) {
                    weights[cnt] = 1;
                    offset[cnt] = i * W + j;
                    dis_i[cnt] = i, dis_j[cnt] = j;
                    ++cnt;
                }
        // 计算边缘响应值
        // 判断是否是同质点的阈值
        const int half_area = cnt / 2;
        const int similar_threshold = edge_ratio * cnt; // 0.75 个总和
        // 存放每个点的响应值
        std::vector<int> response(length, 0);
        // 存放每个点的边缘方向
        std::vector<double> direction(length, 0);
        // 开始计算角点响应值
        const int H_radius = H - radius, W_radius = W - radius;
        for(int i = radius; i < H_radius; ++i) {
            uchar* const row_ptr = gray.data + i * W;
            int* const res_ptr = response.data() + i * W;
            double* const direct_ptr = direction.data() + i * W;
            for(int j = radius; j < W_radius; ++j) {
                // 首先, 寻找这个圆形窗口和中心点的差小于阈值的点
                double number = 0;
                const uchar center = row_ptr[j];
                std::vector<int> book(cnt, 0); // 记录哪些点是同质点
                for(int k = 0;k < cnt; ++k) {
                    // 如果是同质点
                    if(std::abs(row_ptr[j + offset[k]] - center) < threshold) {
                        number += weights[k];
                        book[k] = 1;
                    }
                }
                --number; // 把中心点减去
                // 这里区分平坦区域和边缘, 大于 0.75 的是平坦区域
                if(number < similar_threshold) {
                    // 记录边缘响应值
                    res_ptr[j] = similar_threshold - number;
                    // 【阶跃边缘】
                    if(number >= half_area) {
                        // 找同质区域的引力中心坐标
                        int core_i = 0, core_j = 0;
                        int core_num = 0;
                        for(int k = 0;k < cnt; ++k) {
                            if(book[k] == 1) {
                                ++core_num;
                                core_i += i + dis_i[k];
                                core_j += j + dis_i[k];
                            }
                        }
                        core_i /= core_num, core_j /= core_num;
                        // 引力中心(core_x, core_y) 与 当前中心(i, j) 的连线垂直于边缘, 所以角度就是 x / y, 而不是 y / x
                        // 记录角度大小和正负
                        direct_ptr[j] = (core_j - j) * 1.0 / ((core_i - i) + 1e-12);
                    }
                    // 【过渡边缘】
                    else {
                        // 这时候, 需要计算两个方向的 x 和 y 之差, 只需要统计范围内的同质点就好
                        int trend_i = 0;
                        int trend_j = 0;
                        double is_positive = 0;
                        for(int k = 0;k < cnt; ++k) {
                            if(book[k] == 1) {
                                trend_i += dis_i[k] * dis_i[k];
                                trend_j += dis_j[k] * dis_j[k];
                                is_positive += dis_i[k] * dis_j[k];
                            }
                        }
                        // 求出角度大小
                        direct_ptr[j] = trend_j * 1. / (trend_i + 1e-12);
                        // 求出正负
                        if(is_positive < 0) direct_ptr[j] = -direct_ptr[j];
                    }
                }
            }
        }
        // cv_show(10 * touint8(response, H, W));
        cv_write(10 * touint8(response, H, W), "./images/output/edge_detection/1_edge_response.png");
        // 现在已经知道每一个点的方向, 沿着方向做非极大化抑制
        std::vector<int> nms_result;
        nms_result.resize(length);
        std::copy(response.begin(), response.end(), nms_result.begin());
        // 开始非极大化抑制
        for(int i = 1; i < H - 1; ++i) {
            const int* const row_ptr = response.data() + i * W;
            const double* const direct_ptr = direction.data() + i * W;
            int* const res_ptr = nms_result.data() + i * W;
            for(int j = 1; j < W - 1; ++j) {
                // 开始检查方向
                double lhs, rhs;
                const float ratio = direct_ptr[j];
                // 靠近 x 轴, 1 - 3 象限
                if(0 <= ratio and ratio < 1) {
                    lhs = ratio * row_ptr[j - 1 + W] + (1 - ratio) * row_ptr[j - 1];
                    rhs = ratio * row_ptr[j + 1 - W] + (1 - ratio) * row_ptr[j + 1];
                }
                // 靠近 y 轴, 1 - 3 象限
                else if(ratio >= 1) {
                    const float ratio_inv = 1. / ratio;
                    lhs = ratio_inv * row_ptr[j - 1 + W] * (1 - ratio_inv) * row_ptr[j + W];
                    rhs = ratio_inv * row_ptr[j + 1 - W] * (1 - ratio_inv) * row_ptr[j - W];
                }
                // 靠近 x 轴, 2 - 4 象限
                else if(ratio > -1 and ratio < 0) {
                    lhs = -ratio * row_ptr[j - 1 - W] + (1 + ratio) * row_ptr[j - 1];
                    rhs = -ratio * row_ptr[j + 1 + W] + (1 + ratio) * row_ptr[j + 1];
                }
                // 靠近 y 轴, 2 - 4 象限
                else if(ratio <= -1) {
                    const float ratio_inv = 1. / ratio;
                    rhs = -ratio_inv * row_ptr[j - 1 - W] + (1 + ratio_inv) * row_ptr[j - W];
                    lhs = -ratio_inv * row_ptr[j + 1 + W] + (1 + ratio_inv) * row_ptr[j + W];
                }
                // 判断是否是这个梯度方向上的局部极值
                if(row_ptr[j] < lhs or row_ptr[j] < rhs) {
                    res_ptr[j] = 0;
                }
            }
        }
        // cv_show(10 * touint8(nms_result, H, W));
        cv_write(10 * touint8(nms_result, H, W), "./images/output/edge_detection/2_nms_result.png");
        // 高低阈值, 扩大对比度
        const double low_threshold = 5;
        const double high_threshold = 14;
        cv::Mat refined(H, W, CV_8UC1);
        for(int i = 0;i < length; ++i) {
            if(nms_result[i] > high_threshold) refined.data[i] = 255;
            else if(nms_result[i] > low_threshold) refined.data[i] = 128;
            else refined.data[i] = 0;
        }
        // cv_show(refined);
        cv_write(refined, "./images/output/edge_detection/3_low_high_threshold.png");
        // 连接高低阈值, 删除孤立点
        for(int i = 1; i < H - 1; ++i) {
            uchar* const res_ptr = refined.data + i * W;
            for(int j = 1;j < W - 1; ++j) {
                if(res_ptr[j] == 128) {
                    if(res_ptr[j - 1] == 255 or res_ptr[j + 1] == 255 or
                       res_ptr[j - 1 - W] == 255 or res_ptr[j - W] == 255 or res_ptr[j + 1 - W] == 255 or
                       res_ptr[j - 1 + W] == 255 or res_ptr[j + W] == 255 or res_ptr[j + 1 + W] == 255)
                        res_ptr[j] = 255;
                }
            }
        }
        for(int i = 1; i < H - 1; ++i) {
            uchar* const res_ptr = refined.data + i * W;
            for(int j = W - 2;j > 0; --j) {
                if(res_ptr[j] == 128) {
                    if(res_ptr[j - 1] == 255 or res_ptr[j + 1] == 255 or
                       res_ptr[j - 1 - W] == 255 or res_ptr[j - W] == 255 or res_ptr[j + 1 - W] == 255 or
                       res_ptr[j - 1 + W] == 255 or res_ptr[j + W] == 255 or res_ptr[j + 1 + W] == 255)
                        res_ptr[j] = 255;
                }
            }
        }
        for(int i = H - 2; i >= 1; --i) {
            uchar* const res_ptr = refined.data + i * W;
            for(int j = 1;j < W - 1; ++j) {
                if(res_ptr[j] == 128) {
                    if(res_ptr[j - 1] == 255 or res_ptr[j + 1] == 255 or
                       res_ptr[j - 1 - W] == 255 or res_ptr[j - W] == 255 or res_ptr[j + 1 - W] == 255 or
                       res_ptr[j - 1 + W] == 255 or res_ptr[j + W] == 255 or res_ptr[j + 1 + W] == 255)
                        res_ptr[j] = 255;
                }
            }
        }
        for(int i = H - 2; i >= 1; --i) {
            uchar* const res_ptr = refined.data + i * W;
            for(int j = W - 2;j >= 1; --j) {
                if(res_ptr[j] == 128) {
                    if(res_ptr[j - 1] == 255 or res_ptr[j + 1] == 255 or
                       res_ptr[j - 1 - W] == 255 or res_ptr[j - W] == 255 or res_ptr[j + 1 - W] == 255 or
                       res_ptr[j - 1 + W] == 255 or res_ptr[j + W] == 255 or res_ptr[j + 1 + W] == 255)
                        res_ptr[j] = 255;
                    else res_ptr[j] = 0;
                }
            }
        }
        cv_write(refined, "./images/output/edge_detection/4_connect.png");
        // 准备一个结果
        return refined;
    }

    void edge_detect_demo() {
        const std::string save_dir("./images/output/edge_detection/");
        std::string origin_path("./images/input/corner/a2992-NKIM_MG_7779.png");
        const auto origin_image = cv::imread(origin_path);
        if(origin_image.empty()) {
            std::cout << "读取图像 " << origin_path << " 失败 !" << std::endl;
            return;
        }
        auto edge_extraction = susan_edge_detect(origin_image, 3, 30);
        cv_show(edge_extraction);
        cv_write(edge_extraction, save_dir + "french.png");

        // 添加噪声
        const auto noisy_image = add_gaussian_noise(origin_image);
        cv_show(noisy_image);
        edge_extraction = susan_edge_detect(noisy_image, 3, 30);
        cv_show(edge_extraction);
        cv_write(edge_extraction, save_dir + "noisy.png");

        // 旋转
        const auto rotated_image = my_rotate(origin_image);
        cv_show(rotated_image);
        edge_extraction = susan_edge_detect(rotated_image, 3, 30);
        cv_show(edge_extraction);
        cv_write(edge_extraction, save_dir + "rotate.png");

        // 灰度变换
        const cv::Mat darken_image = 0.5 * origin_image;
        darken_image.convertTo(darken_image, CV_8UC3);
        cv_show(darken_image);
        edge_extraction = susan_edge_detect(darken_image, 3, 15);
        cv_show(edge_extraction);
        cv_write(edge_extraction, save_dir + "darken.png");

        // 尺度
        cv::Mat scaled_image;
        cv::resize(origin_image, scaled_image, cv::Size(origin_image.cols / 4, origin_image.rows / 4));
        cv_show(scaled_image);
        edge_extraction = susan_edge_detect(scaled_image, 3, 30);
        cv_show(edge_extraction);
        cv_write(edge_extraction, save_dir + "rotate.png");
    }
}


namespace SUSAN_DENOISE {

    cv::Mat make_pad(const cv::Mat &one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }

    inline double fast_exp(const double y) {
        double d;
        *(reinterpret_cast<int*>(&d) + 0) = 0;
        *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * y + 1072632447);
        return d;
    }

    cv::Mat susan_edge_preserving_denoise(
            const cv::Mat& source,
            const int radius=3,
            const double sigma=1.0,
            const double similar_threshold=5) {
        // 单通道的去噪
        auto susan_edge_preserving_denoise_gray = [radius, sigma, similar_threshold](const cv::Mat& gray) -> cv::Mat {
            // 做 padding
            const auto padded = make_pad(gray, radius, radius);
            const int W2 = padded.cols;
            const int W2_radius = W2 - radius;
            const int H2_radius = padded.rows - radius;
            // 值域滤波没有滤波模板, 但可以准备局部区域的定位
            int cnt = 0;
            const int max_size = (2 * radius + 1) * (2 * radius + 1);
            std::vector<int> offset(max_size, 0);
            const int radius_2 = (radius + 0.5) * (radius + 0.5);
            // 圆形模板
            for(int i = -radius;i <= radius; ++i)
                for(int j = -radius; j <= radius; ++j)
                    if(i * i + j * j <= radius_2) {
                        if(i == 0 and j == 0) continue;
                        offset[cnt++] = i * W2 + j;
                    }
            // 开始滤波
            cv::Mat result = gray.clone();
            const double constant = fast_exp(-radius * radius / (2 * sigma * sigma));
            const double threshold_inv = 1. / (similar_threshold * similar_threshold);
            for(int i = radius;i < H2_radius; ++i) {
                const uchar* const row_ptr = padded.data + i * W2;
                uchar* const res_ptr = result.data + (i - radius) * gray.cols;
                for(int j = radius; j < W2_radius; ++j) {
                    double weight_sum = 0;
                    double intensity_sum = 0;
                    const int center = row_ptr[j];
                    for(int k = 0;k < cnt; ++k) {
                        const int residual = row_ptr[j + offset[k]] - center;
                        const double weight = constant * fast_exp( - residual * residual * threshold_inv);
                        weight_sum += weight;
                        intensity_sum += weight * row_ptr[j + offset[k]];
                    }
                    res_ptr[j - radius] = cv::saturate_cast<uchar>(intensity_sum / weight_sum);
                }
            }
            return result;
        };
        const int C = source.channels();
        // 单通道
        if(C == 1)
            return susan_edge_preserving_denoise_gray(source);
        // 多通道
        std::vector<cv::Mat> channels;
        cv::split(source, channels);
        std::vector<cv::Mat> denoised_channels;
        for(const auto& ch : channels) {
            denoised_channels.emplace_back(susan_edge_preserving_denoise_gray(ch));
        }
        // 多通道合并
        cv::Mat denoised;
        cv::merge(denoised_channels, denoised);
        return denoised;
    }

    void edge_preserving_denoise_demo() {
        const std::string save_dir("./images/output/denoise/");
        std::string origin_path("./images/input/denoise/woman_1.png");
        const auto origin_image = cv::imread(origin_path);
        if(origin_image.empty()) {
            std::cout << "读取图像 " << origin_path << " 失败 !" << std::endl;
            return;
        }
        cv::Mat denoised, comparison, another_image;
        denoised = susan_edge_preserving_denoise(origin_image, 9, 3.0, 30);
        comparison = cv_concat({origin_image, denoised});
        cv_show(comparison);
        cv_write(comparison, save_dir + "woman_1.png");

        another_image = cv::imread("./images/input/denoise/woman_3.jpg");
        denoised = susan_edge_preserving_denoise(another_image, 6, 2.0, 30);
        comparison = cv_concat({another_image, denoised});
        cv_show(comparison);
        cv_write(comparison, save_dir + "woman_3.png");

        another_image = cv::imread("./images/input/denoise/Kodak24/22.png");
        denoised = susan_edge_preserving_denoise(another_image, 6, 2.0, 30);
        comparison = cv_concat({another_image, denoised});
        cv_show(comparison);
        cv_write(comparison, save_dir + "Kodak24_22.png");

        another_image = cv::imread("./images/input/denoise/Kodak24/18.png");
        denoised = susan_edge_preserving_denoise(another_image, 6, 2.0, 30);
        comparison = cv_concat({another_image, denoised});
        cv_show(comparison);
        cv_write(comparison, save_dir + "Kodak24_18.png");
    }
}



int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    // 角点检测
    SUSAN_CORNER::corner_detect_demo();

    // 边缘检测
    SUSAN_EDGE::edge_detect_demo();

    // 去噪
    SUSAN_DENOISE::edge_preserving_denoise_demo();

    return 0;
}
