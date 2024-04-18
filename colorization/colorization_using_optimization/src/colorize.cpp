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

    inline float square(const float x) {
        return x * x;
    }
}



cv::Mat colorize_using_optimization(const cv::Mat& gray_image, const cv::Mat& scrawl_image, const int kernel_size=7, const float min_var=1.f) {
    // 半径
    const int radius = (kernel_size - 1) >> 1;

	// BGR -> YUV, 提取出灰度图和涂鸦的颜色通道
	cv::Mat gray_image_yuv, scrawl_image_yuv;
	cvtColor(gray_image, gray_image_yuv, cv::COLOR_RGB2YUV);
	cvtColor(scrawl_image, scrawl_image_yuv, cv::COLOR_RGB2YUV);

	// 获取信息
	const int H = scrawl_image.rows;
	const int W = scrawl_image.cols;
	const int length = H * W;
	const int neighbor_num = kernel_size * kernel_size - 1; // 局部窗口的其它像素有多少个

	// 提取出 Y 通道, 因为后面计算局部窗口权重要根据亮度之差得到
	cv::Mat Y(H, W, CV_8UC1);
	uchar* const Y_ptr = Y.data;
	for(int i = 0;i < length; ++i) Y_ptr[i] = gray_image_yuv.data[3 * i] * 1.f; // 转成 float

	// 记录哪些位置是有颜色的, 做了涂鸦
	cv::Mat mask = cv::Mat::zeros(H, W, CV_8UC1);
	uchar* const mask_ptr = mask.data;
	for(int i = 0;i < length; ++i) {
        if(scrawl_image_yuv.data[3 * i + 1] != 128 or
	       scrawl_image_yuv.data[3 * i + 2] != 128) // U, V 任一通道 128 是没有颜色
	        mask_ptr[i] = 1; // U, V 中至少有一个通道的值不是 128, 有颜色
	}


	// 构造 A
	std::vector< Eigen::Triplet<float> > A_list;
	A_list.reserve(length * kernel_size * kernel_size); // 一共 length 个点要求解, 每个点最多有 kernel_size * kernel_size 个数参与方程
	for (int x = 0; x < H; ++x) {
	    for (int y = 0; y < W; ++y) {
	        // 获取局部窗口中心点的 id(位置)
	        const int center = x * W + y;
			// 每一个中心点在 A 中的系数都是 1.0
            A_list.emplace_back(center, center,1.0);
            // 如果这个点没有涂鸦, 说明可以构建约束方程
			if (mask_ptr[center] == 0) {
				// 找边界
				const int x_min = std::max(0, x - radius);
				const int x_max = std::min(H - 1, x + radius);
				const int y_min = std::max(0, y - radius);
				const int y_max = std::min(W - 1, y + radius);
                // 计算局部窗口均值
                int valid_cnt = 0, offset[neighbor_num];
                float window_mean = 0;
                for (int i = x_min; i <= x_max; ++i)
                    for (int j = y_min; j <= y_max; ++j) {
                        const int pos = i * W + j;
                        window_mean += Y_ptr[pos];  // 加上这个点的亮度值
                        offset[valid_cnt++] = pos;  // 记录有效点的偏移量, 一共 valid_cnt 个有效点
                    }
                window_mean /= valid_cnt;
                // 计算局部窗口方差
                float window_var = 0;
                for(int k = 0;k < valid_cnt; ++k)
                    window_var += square(Y_ptr[offset[k]] - window_mean);
                window_var /= valid_cnt;
                window_var = std::max(min_var, window_var);
                // 计算局部窗口的邻域点和中心点的差距, 算权重
                float weight_temp[valid_cnt];
                float weight_sum = 0;
                for(int k = 0;k < valid_cnt; ++k) {
                    const float weight = std::exp(-square(Y_ptr[offset[k]] - Y_ptr[center]) / (2 * window_var));
                    weight_sum += weight;
                    weight_temp[k] = weight;
                }
                // 局部窗口内归一化的权重 weight, 取负号填入 A 矩阵, 位置是第 center 行, 第 offset[k]
                for(int k = 0;k < valid_cnt; ++k)
                    A_list.emplace_back(center, offset[k], - weight_temp[k] / weight_sum);
			}
		}
	}
	// 从 list 填到稀疏矩阵中, 这个速度比 insert 快得多
	Eigen::SparseMatrix<float> A(length, length);
	A.setFromTriplets(A_list.begin(), A_list.end());
	A.makeCompressed();

    // 构造 b
	Eigen::VectorXf b_u(length);
	Eigen::VectorXf b_v(length);
	b_u.setZero(); // 初始化为 0
	b_v.setZero();
	for (int x = 0; x < H; ++x) {
	    for (int y = 0; y < W; ++y) {
	        const int pos = x * W + y;
            if (mask_ptr[pos] != 0) { // 如果这个点有涂鸦
                b_u(pos) = scrawl_image_yuv.data[3 * pos + 1]; // b 方程记住 U 颜色
                b_v(pos) = scrawl_image_yuv.data[3 * pos + 2]; // b 方程记住 V 颜色
            }
        }
	}

    // 稀疏矩阵解 Ax = b 线性方程组
    Eigen::SparseLU< Eigen::SparseMatrix<float> > solver;
	solver.compute(A);
	const Eigen::VectorXf u_result = solver.solve(b_u);
	const Eigen::VectorXf v_result = solver.solve(b_v);

    // 把解的 U, V 拷贝到结果上
	cv::Mat colored(H, W, CV_8UC3);
	for(int x = 0;x < H; ++x) {
	    uchar* const res_ptr = colored.data + x * W * 3;
	    uchar* const gray_ptr = gray_image_yuv.data + x * W * 3;
	    for(int y = 0;y < W; ++y) {
	        const int pos = 3 * y;
	        res_ptr[pos] = gray_ptr[pos];
	        res_ptr[pos + 1] = u_result(x * W + y);
	        res_ptr[pos + 2] = v_result(x * W + y);
	    }
	}
	return colored;
}





int main() {
	// 读取图像
	const std::string gray_path("./images/input/yellow_m.bmp");
	const std::string scrawl_path("./images/marked/yellow_m.bmp");
	const cv::Mat gray_image = cv::imread(gray_path);
	const cv::Mat scrawl_image = cv::imread(scrawl_path);
    assert(not gray_image.empty() and not scrawl_image.empty() and "读取图像失败 !");

    // 定义超参
	constexpr int kernel_size = 3;

	// 上色
	auto colored = colorize_using_optimization(gray_image, scrawl_image, kernel_size, 1.f);

    // YUV 结果转化成 BGR
	cv::cvtColor(colored, colored, cv::COLOR_YUV2RGB);

    // 保存与展示
    std::filesystem::path save_dir("./images/output/");
    if(not std::filesystem::exists(save_dir))
        std::filesystem::create_directories(save_dir);
    cv_write(colored,  save_dir.string() + "yellow_m.bmp");
    cv_show(colored);
    
    return 0;
}