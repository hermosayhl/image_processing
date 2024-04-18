//C++
#include <list>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <vector>
#include <iostream>
// Eigen3
#include <Eigen/Core>
#include <Eigen/Dense>
//OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// self


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
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_CONSTANT, 0);
        return padded_image;
    }

    template<typename T>
    cv::Mat toint8(const std::vector<T>& source, const int H, const int W, const int C, const int _type) {
        cv::Mat result(H, W, _type);
        const int length = H * W * C;
        for(int i = 0;i < length; ++i) result.data[i] = cv::saturate_cast<uchar>(std::abs(source[i]));
        return result;
    }
}



namespace {

    template<typename T>
    inline T square(const T x) {
        return x * x;
    }
    
    template<typename T>
    inline float norm(const T x) {
        return x * 1.f / 255;
    }

    // 将值限定在 0-1 之间
    inline float clip(const float x) {
        if(x < 0) return 0;
        else if(x > 1) return 1;
        return x;
    }

    // 每次都会修改 cur_unknown; 这个函数是找到当前未知区域的最外面一圈(理想状况下)
    std::vector<int> find_unknown_edge(cv::Mat& cur_unknown, const int H, const int W, const uchar* const um_ptr) {
        // 周围 8 个像素
        cv::Mat new_cur_unknown = make_pad(cur_unknown, 1, 1);
        const int W2 = new_cur_unknown.cols;
        // 每个点都检查自身周围8 个像素, 只要有一个是已求解的, 就把当前点标记一下, 是这一次要求解的交界处的点
        for(int i = 0; i < H; ++i) {
            uchar* const row_ptr = new_cur_unknown.data + (1 + i) * W2 + 1;
            uchar* const res_ptr = cur_unknown.data + i * W;
            for(int j = 0; j < W; ++j) {
                if(row_ptr[j] == 255) {
                    if(row_ptr[j - 1] == 0 or row_ptr[j + 1] == 0 or
                       row_ptr[j - 1 - W2] == 0 or row_ptr[j - W2] == 0 or row_ptr[j + 1 - W2] == 0 or
                       row_ptr[j - 1 + W2] == 0 or row_ptr[j + W2] == 0 or row_ptr[j + 1 + W2] == 0)
                        res_ptr[j] = 0; // 这个点在交界处, 记为 0
                }
            }
        }
        const int length = H * W;
        std::vector<int> unknown_edge;
        for(int i = 0;i < length; ++i)
            if(cur_unknown.data[i] == 0 and um_ptr[i] == 255) // 被标记的的一圈和未求解的交集
                unknown_edge.emplace_back(i);
        return unknown_edge;
    }

    // 对前景和后景进行聚类, 定义每个簇的属性
    struct cluster {
    public:
        // 要保存的变量
        Eigen::Matrix3f cov;      // 这个簇的协方差矩阵
        Eigen::Vector3f mean;     // 这个簇的像素均值, 是个 3 维向量
        Eigen::Vector3f eigen_vec;// 协方差矩阵的特征向量
        float threshold;          // 用来分裂的, 和中心向量的结果做比较
        float max_eigen_value;    // 协方差矩阵最大的特征值
        // 不可变量
        const int pixel_num;      // 这个簇的像素数目
        const uchar* const img_ptr; // 这些像素所在图像的指针
        const std::vector<int> pixels; // 这些像素在图像中的位置
        const std::vector<float> W; // 这些像素占的比重, 由 alpha 和高斯距离合成
    public:
        cluster(const uchar* const _img_ptr, std::vector<int>& _pixels, std::vector<float>& _W)
                : pixel_num(_pixels.size()), img_ptr(_img_ptr), pixels(std::move(_pixels)), W(std::move(_W)) {
            // 首先求这些像素的颜色均值
            float weight_sum = 0.0;
            std::vector<float> mu(3, 0);
            for(int i = 0;i < pixel_num; ++i) {
                const int pos = 3 * pixels[i]; // 获取像素位置
                mu[0] += norm(img_ptr[pos] * W[i]);
                mu[1] += norm(img_ptr[pos + 1] * W[i]);
                mu[2] += norm(img_ptr[pos + 2] * W[i]);
                weight_sum += W[i];
            }
            for(int ch = 0; ch < 3; ++ch) mu[ch] /= weight_sum;
            // 得到这个簇的像素均值向量
            for(int ch = 0;ch < 3; ++ch) this->mean(ch) = mu[ch];
            // 求协方差矩阵
            Eigen::MatrixXf residual(pixel_num, 3);
            for(int i = 0;i < pixel_num; ++i) {
                const int pos = 3 * pixels[i];
                const float sw = std::sqrt(W[i]);
                residual(i, 0) = sw * (norm(img_ptr[pos]) - mu[0]);
                residual(i, 1) = sw * (norm(img_ptr[pos + 1]) - mu[1]);
                residual(i, 2) = sw * (norm(img_ptr[pos + 2]) - mu[2]);
            }
            this->cov = residual.transpose() * residual / weight_sum;
            this->cov += 1e-5 * Eigen::MatrixXf::Identity(3, 3);
            // 解特征值
            Eigen::EigenSolver< Eigen::Matrix<float, 3, 3> > eigen_solver(this->cov);
            const auto& eigen_values = eigen_solver.eigenvalues().real();
            const auto& eigen_vectors = eigen_solver.eigenvectors().real();
            const int eigen_size = eigen_values.size();
            std::vector<float> eigen_values_std({abs(eigen_values(0)), abs(eigen_values(1)), abs(eigen_values(2))});
            // 找最大的特征值的位置
            int max_index = 0;
            this->max_eigen_value = eigen_values_std[0];
            for(int i = 1;i < eigen_size; ++i) {
                if(this->max_eigen_value < eigen_values_std[i]) {
                    this->max_eigen_value = eigen_values_std[i];
                    max_index = i;
                }
            }
            // 赋值得到最大特征值对应的特征向量
            for(int ch = 0;ch < 3; ++ch) this->eigen_vec(ch) = eigen_vectors(max_index, ch);
            // 二者的结果作为标准
            this->threshold = this->mean.transpose() * this->eigen_vec;
        }
        // 比较符重载
        bool operator<(const cluster& rhs) const {
            return this->max_eigen_value < rhs.max_eigen_value;
        }
    };

    std::vector< std::pair<Eigen::Vector3f, Eigen::Matrix3f> > make_clusters(
        const uchar* const img_ptr, std::vector<int>& pixels, std::vector<float>& W, const bool single=false) {
        // 先收集一个 cluster
        std::list<cluster> nodes({cluster(img_ptr, pixels, W)});
        // 准备分成俩个簇
        std::vector<int> lhs, rhs;
        std::vector<float> lhs_w, rhs_w;
        // 如果考虑多个前景背景分布
        if(not single) {
            while(true) {
                // 找到当前特征值最大的簇
                const auto max_index = std::max_element(nodes.begin(), nodes.end());
                // 判断是否要分裂
                if(max_index->max_eigen_value > 0.05) {
                    // 做分裂, 所有像素都遍历一遍
                    const int pixel_num = max_index->pixel_num;
                    for(int i = 0;i < pixel_num; ++i) {
                        const int pos = 3 * max_index->pixels[i]; // 这个像素的起始位置, 0, 1, 2 分别是 B, G, R
                        float temp = 0;
                        temp += norm(max_index->eigen_vec(0) * max_index->img_ptr[pos]);
                        temp += norm(max_index->eigen_vec(1) * max_index->img_ptr[pos + 1]);
                        temp += norm(max_index->eigen_vec(2) * max_index->img_ptr[pos + 2]);
                        if(temp <= max_index->threshold) {
                            lhs.emplace_back(max_index->pixels[i]);
                            lhs_w.emplace_back(max_index->W[i]);
                        } else {
                            rhs.emplace_back(max_index->pixels[i]);
                            rhs_w.emplace_back(max_index->W[i]);
                        }
                    }
                    // 只要不是空的
                    if(lhs.empty() or rhs.empty()) break;
                    if(not lhs.empty()) nodes.emplace_back(cluster(max_index->img_ptr, lhs, lhs_w));
                    if(not rhs.empty()) nodes.emplace_back(cluster(max_index->img_ptr, rhs, rhs_w));
                    nodes.erase(max_index); // 删除这个被分裂的簇
                }
                else break; // 最分散的簇达不到分裂的标准, 退出分裂
            }
        }
        std::vector< std::pair<Eigen::Vector3f, Eigen::Matrix3f> > results;
        for(const auto& one : nodes)
            results.emplace_back(one.mean, one.cov); // 返回每个簇的均值和方差
        return results;
    };
}


cv::Mat bayers_matting(
        const cv::Mat& _observation,
        const cv::Mat& _trimap,
        const bool single=false,
        const int radius=12,
        const float sigma=8.0,
        const int min_cluster_num=8,
        const int max_iterations=37,
        const float sigma_c = 0.01) {
    // 处理异常情形
    assert(_observation.channels() == 3 and _trimap.channels() == 1 and "观测图像必须是 3 通道, Trimap 图必须是单通道!");
    assert(_observation.rows == _trimap.rows and _observation.cols == _trimap.cols and "观测图像和 trimap 尺寸不一致 !");
    // 做 padding, 这样一来任意一个要求解的点都有一个完整的局部窗口
    const auto observation = make_pad(_observation, radius, radius);
    const auto trimap = make_pad(_trimap, radius, radius);
    const uchar* const trimap_ptr = trimap.data;
    // 获取图像信息
    const int H = observation.rows;
    const int W = observation.cols;
    const int length = H * W;
    // 根据 trimap 提取前景和背景, 未知区域
    cv::Mat fore_mask(H, W, CV_8UC1);
    cv::Mat back_mask(H, W, CV_8UC1);
    cv::Mat unknown_mask(H, W, CV_8UC1);
    uchar* const fm_ptr = fore_mask.data, * const bm_ptr = back_mask.data, * const um_ptr = unknown_mask.data;
    for(int i = 0;i < length; ++i) {
        if(trimap_ptr[i] == 255) fm_ptr[i] = 255; // 前景
        else if(trimap_ptr[i] == 0) bm_ptr[i] = 255; // 背景
        else um_ptr[i] = 255; // 未知, 要求解, 除了 255 其它都为 0
    }
    cv::Mat foreground(H, W, observation.type());
    cv::Mat background(H, W, observation.type());
    observation.copyTo(foreground, fore_mask); // 根据 mask 把前景抠出来
    observation.copyTo(background, back_mask); // 同上, 抠背景
    // 准备一个高斯核模板
    const int filter_size = square(radius * 2 + 1);
    std::vector<float> gaussi_filter(filter_size, 0); // 高斯核模板, 大小是 (2 * radius + 1)^2
    std::vector<int> gaussi_offset(filter_size, 0); // 准备该局部窗口内部的偏移量, 一次循环即可遍历整个局部窗口
    int offset = 0;
    const float sigma_inv = 1. / (2 * sigma * sigma);
    for(int i = -radius; i <= radius; ++i) {
        for(int j = -radius; j <= radius; ++j) {
            gaussi_filter[offset] = std::exp(-1.f * (i * i + j * j) * sigma_inv);
            gaussi_offset[offset] = i * W + j; // 距离相差 i 行和 j 个像素
            ++offset;
        }
    }
    float weight_sum = 0;
    for(int i = 0;i < filter_size; ++i) weight_sum += gaussi_filter[i];
    for(int i = 0;i < filter_size; ++i) gaussi_filter[i] /= weight_sum; // 归一化
    // 根据 trimap 得到初始的 alpha
    cv::Mat alpha = fore_mask.clone() / 255;
    alpha.convertTo(alpha, CV_32FC1); // 转成 float 数据
    float* const alpha_ptr = alpha.ptr<float>();  // 获取 alpha 图像的指针
    for(int i = 0;i < length; ++i) if(um_ptr[i] == 255) alpha_ptr[i] = -1; // 标识这个是未知的, 要求解的
    // 获取要求解的点的数目
    int targets_num = 0;
    for(int i = 0;i < length; ++i) if(um_ptr[i] == 255) ++targets_num;
    std::cout << "待求解的点共有  " << targets_num << " 个!\n";
    // 求解需要的中间变量
    int obtained = 0;
    int last_unknown_edge_size = 0;
    cv::Mat cur_unknown = unknown_mask.clone();
    // 开始求解
    while(obtained < targets_num) {
        // 首先, 获取未知区域最外面一圈的边缘
        const auto unknown_edge = find_unknown_edge(cur_unknown, H, W, um_ptr);
        // 求解这一圈的未知像素的 alpha
        const int unknown_edge_size = unknown_edge.size();
        std::cout << obtained << "/" << targets_num << "   " << "to solve  " << unknown_edge.size() << std::endl;
        // 如果这一次要求解点数目和这一次一模一样, 说明要死循环了
        if(unknown_edge_size == last_unknown_edge_size) break;
        last_unknown_edge_size = unknown_edge_size;
        // 遍历这一圈的求解点
        for(int u = 0; u < unknown_edge_size; ++u) {
            // 获取当前要求解的点的坐标, (pos / W, pos % W) 就是实际坐标
            const int pos = unknown_edge[u];
            // 在 pos 这个点为中心, 半径 radius 的局部窗口, 找到所有已知的前景点和背景点, 包括求解到的
            int valid_cnt = 0;
            float init_alpha = 0;
            std::vector<int> foreground_pixels, background_pixels; // 收集已知的前景点, 和对应的比重   
            std::vector<float> foreground_weights, background_weights;
            foreground_pixels.reserve(filter_size), background_pixels.reserve(filter_size); // 这些其实可以放最外层, 每次 clear 就行, 就是不太好看
            foreground_weights.reserve(filter_size), background_weights.reserve(filter_size);
            // 遍历局部窗口
            for(int i = 0;i < filter_size; ++i) {
                const int index = pos + gaussi_offset[i]; // 找到局部窗口这个点, 当前点坐标 + 偏移
                const float this_alpha = alpha_ptr[index];
                if(this_alpha >= 0) { // 当前点的 alpha 不是未知的
                    // 前景
                    float cur_f_weight = square(this_alpha) * gaussi_filter[i];
                    if(cur_f_weight > 0) { // > 0 说明这个点的 alpha 不为 0
                        foreground_pixels.emplace_back(index);
                        foreground_weights.emplace_back(cur_f_weight);
                    }
                    // 背景
                    float cur_b_weight = square(1 - this_alpha) * gaussi_filter[i];
                    if(cur_b_weight > 0) { // > 0 说明这个点的 alpha 不为 1
                        background_pixels.emplace_back(index);
                        background_weights.emplace_back(cur_b_weight);
                    }
                    init_alpha += this_alpha; // 记录不是空的 alpla, 为后面迭代优化的启动求局部 alpha 初始值做准备
                    ++valid_cnt; // 记录有多少个有效的 alpla
                }
            }
            init_alpha /= valid_cnt; 
            // 如果前景点或背景点太少, 放弃聚类, 等以后已知的多了, 再继续求解
            if(foreground_weights.size() < min_cluster_num or background_weights.size() < min_cluster_num)
                continue;
            // std::cout << pos / W << ", " << pos % W << " ==> " << foreground_weights.size() << ", " << background_weights.size() << std::endl;
            // 对前景点聚类
            const auto fore_clusters = make_clusters(foreground.data, foreground_pixels, foreground_weights, single);
            // 对背景点聚类
            const auto back_clusters = make_clusters(background.data, background_pixels, background_weights, single);
            // 当前像素值准备好, 做成向量
            Eigen::Vector3f C;
            C << norm(observation.data[3 * pos]), norm(observation.data[3 * pos + 1]), norm(observation.data[3 * pos + 2]);
            // 开始解 F, B, alpla
            // 记录最佳的结果
            float best_alpha = 1.0;
            Eigen::Vector3f best_F, best_B;
            // 准备一些中间变量
            const auto I = Eigen::MatrixXf::Identity(3, 3);
            const float sigma_c_inv = 1.f / (sigma_c * sigma_c);
            // 这个局部窗口内, 前景跟背景可能都有几个簇, 要两两计算, 找似然函数最大的结果
            float max_likelihood = -FLT_MAX;
            const int fore_cluster_num = fore_clusters.size();
            const int back_cluster_num = back_clusters.size();
            for(int i = 0;i < fore_cluster_num; ++i) {
                auto& fore_mean = fore_clusters[i].first;
                const auto fore_cov_inv = fore_clusters[i].second.inverse();
                for(int j = 0;j < back_cluster_num; ++j) {
                    auto& back_mean = back_clusters[j].first;
                    const auto back_cov_inv = back_clusters[i].second.inverse(); // 这里重复计算了, 但是矩阵很小, 可以忽略
                    float cur_alpha = init_alpha; // 每次前景的一个簇和背景的一个簇计算似然, 都给同一个初始化的 alpha
                    int iterations = 1;
                    float last_likelihood = -FLT_MAX; // 记录上一次迭代的似然是多少, 判断有没有收敛, 收敛提前退出
                    while(true) {
                        // 准备 A 矩阵
                        const auto A11 = fore_cov_inv + I * square(cur_alpha) * sigma_c_inv;
                        const auto A12 = I * cur_alpha * (1 - cur_alpha) * sigma_c_inv;
                        const auto A22 = back_cov_inv + I * square(1 - cur_alpha) * sigma_c_inv;
                        Eigen::MatrixXf A;
                        A.resize(6, 6);
                        A << A11, A12,
                             A12, A22; // Eigen 写法这里不能去掉换行, 奇特
                        // 准备 b 矩阵
                        const auto b1 = fore_cov_inv * fore_mean + C * cur_alpha * sigma_c_inv;
                        const auto b2 = back_cov_inv * back_mean + C * (1 - cur_alpha) * sigma_c_inv;
                        Eigen::MatrixXf b;
                        b.resize(6, 1);
                        b << b1,
                             b2; // 换行不能去掉
                        // 解方程组 Ax = b
                        Eigen::VectorXf X = A.inverse() * b; // auto X = A.ldlt().solve(b);  // Eigen3 在这方面很差劲
                        // 更新 F, B, 截断在 0-1 以内
                        Eigen::Vector3f F, B;
                        F << clip(X(0)), clip(X(1)), clip(X(2));
                        B << clip(X(3)), clip(X(4)), clip(X(5));
                        // 更新 alpha
                        cur_alpha = ((C - B).transpose() * (F - B) / (F - B).squaredNorm()).value();
                        // cur_alpha = ((C(0) - B(0)) * (F(0) - B(0)) + (C(1) - B(1)) * (F(1) - B(1)) + (C(2) - B(2)) * (F(2) - B(2))) / 
                        //             (square(F(0) - B(0)) + square(F(1) - B(1)) + square(F(2) - B(2)));
                        // 计算似然函数
                        const float LC = - (C - cur_alpha * F - (1 - cur_alpha) * B).squaredNorm() * sigma_c_inv;
                        const auto LF = -(F - fore_mean).transpose() * fore_cov_inv * (F - fore_mean) / 2;
                        const auto LB = -(B - back_mean).transpose() * back_cov_inv * (B - back_mean) / 2;
                        const float likelihood = LC + LF.value() + LB.value();
                        // 判断这次的似然概率有没有更大
                        if(likelihood > max_likelihood) {
                            max_likelihood = likelihood; 
                            best_alpha = cur_alpha; // 记录似然最大的结果
                            best_F = F;
                            best_B = B;
                        }
                        // 判断收敛条件
                        if(std::abs(likelihood - last_likelihood) < 1e-5 or ++iterations >= max_iterations)
                            break;
                        last_likelihood = likelihood;
                    }
                }
            }
            // 得到的 F, B, alpha 分别赋值给 foreground 和 background, alpha
            const int index = 3 * pos;
            for(int ch = 0;ch < 3; ++ch) {
                foreground.data[index + ch] = cv::saturate_cast<uchar>(255 * best_F(ch));
                background.data[index + ch] = cv::saturate_cast<uchar>(255 * best_B(ch));
            }
            alpha_ptr[pos] = best_alpha;
            // 标记这个点求解完毕 !
            um_ptr[pos] = 0;
            // 已求解数目 + 1
            ++obtained;
        }
    }
    alpha = alpha * 255;
    // for(int i = 0;i < length; ++i) if(alpha_ptr[i] < 60) alpha_ptr[i] = 0;
    alpha.convertTo(alpha, CV_8UC1);
    return alpha(cv::Rect(radius, radius, _observation.cols, _observation.rows)).clone();
}


int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    // 读取图像
    cv::Mat observation = cv::imread("./images/input/input_4.bmp");
    cv::Mat trimap = cv::imread("./images/input/mask_4.bmp", cv::IMREAD_GRAYSCALE);
    assert(not observation.empty() and not trimap.empty() and "读取的图像为空 !");

    cv::Mat alpha;
    run([&](){
        alpha = bayers_matting(observation, trimap, 12);
    }, "bayers_matting  ");

    cv_write(alpha, "./images/output/alpha_4.png");

    /*
    if(true) {
        cv::Mat background = cv::Mat::zeros(alpha.rows, alpha.cols, observation.type());
        std::vector<cv::Mat> alpla_stack({alpha, alpha, alpha});
        cv::merge(alpla_stack, alpha);
        alpha.convertTo(alpha, CV_32FC1);
        alpha /= 255;
        observation.convertTo(observation, CV_32FC1);
        background.convertTo(background, CV_32FC1);
        background = alpha.mul(observation) + (1 - alpha).mul(background);
        background.convertTo(background, CV_8UC3);
        cv_write(background, "./images/output/foreground.png");
    }
    */

    if(false) {
        // 和其他图像组合在一起
        cv::Mat background = cv::imread("./images/input/a0161-_DSC0022.png");
        assert(not background.empty() and "背景图像不能为空 !");
        assert(background.channels() == 3 and "目前只支持 BGR 图像");
        alpha.convertTo(alpha, CV_32FC1);
        alpha /= 255;
        // 选一个插入的位置
        const std::vector<int> pos({103, 20});
        // const std::vector<int> pos({544, 20});
        // 超出的部分舍弃
        cv::resize(background, background, cv::Size(background.cols - 150, background.rows - 100));
        const int H = observation.rows;
        const int W = observation.cols;
        const int W2 = background.cols;
        const int H3 = H - std::min(0, background.rows - pos[0] - alpha.rows);
        const int W3 = W - std::min(0, background.cols - pos[1] - alpha.cols);
        // 逐像素融合
        for(int i = 0; i < H; ++i) {
            const uchar* const src_ptr = observation.data + i * W * 3;
            const float* const alpha_ptr = alpha.ptr<float>() + i * W;
            uchar* const res_ptr = background.data + ((pos[0] + i) * W2 + pos[1]) * 3;
            for(int j = 0; j < W; ++j) {
                const int pos = 3 * j;
                const float this_alpha = alpha_ptr[j];
                // if(this_alpha < 0.5) continue;
                for(int ch = 0; ch < 3; ++ch)
                    res_ptr[pos + ch] = cv::saturate_cast<uchar>(
                        this_alpha * src_ptr[pos + ch] + (1 - this_alpha) * res_ptr[pos + ch]);
            }
        }
        cv_write(background, "./images/output/alpha_4_composed_3.png");
    }

    return 0;
}