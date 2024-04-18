// C++
#include <vector>
#include <iostream>
// 3rd party
#include <Eigen/Core>
#include <Eigen/Dense>
// self
#include "guided_filter.h"


namespace {

	void cv_show(const cv::Mat& one_image, const char* info="") {
		cv::imshow(info, one_image);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}

	void cv_show_double(const double* const one_image, const int H, const int W, const char* info="") {
		cv::Mat src(H, W,CV_8UC1);
		for(int i = 0;i < H; ++i) {
			const double* const row_ptr = one_image + i * W;
			for(int j = 0;j < W; ++j)
				src.at<uchar>(i, j) = cv::saturate_cast<uchar>(row_ptr[j] * 255);
		}
		cv_show(src);
	}

    // 均值滤波
    std::vector<double> box_filter(const double* const new_source, const int radius_h, const int radius_w, const int H, const int W) {
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
        // 填充四个角, 以后有时间再说 cv_show_double(padding_image.data(), new_H, new_W);
        // 复杂度分析, 彩色图的 guided_filter

        // 现在开始 box_filter, 注意有边界
        // 现在图像的高和宽分别是 new_H, new_W, 草稿画一下图就知道
        const int kernel_h = (radius_h << 1) + 1;
        const int kernel_w = (radius_w << 1) + 1;
        // 准备 buffer 和每一个点代表的 box 之和
        std::vector<double> buffer(new_W, 0.0);
        std::vector<double> sum(H * W, 0.0);
        double* const sum_ptr = sum.data();
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
		return sum;
	}

}

// 我可以先把它 padding, 之后在 Rect
cv::Mat guided_filter_with_gray(const cv::Mat& noise_image, const cv::Mat& guide_image, const int radius_h, const int radius_w, const double epsilon) {
    const int C = noise_image.channels();
    if(C != 1) {
        std::cout << "channels must be 1 !\n";
        return noise_image;
    }
    const int H = noise_image.rows;
	const int W = noise_image.cols;
	const int length = H * W;
	if(H != guide_image.rows or W != guide_image.cols) {
	    std::cout << "the size of guide image should be the same as input image\n";
	    return noise_image;
	}
	// 准备一些 double 数组存储中间结果
	std::vector<double> noise_double_image(H * W, 0);
	std::vector<double> guide_double_image(H * W, 0);
	std::vector<double> I_P(H * W, 0);
	std::vector<double> I_I(H * W, 0);
	std::vector<double> cov_IP(H * W, 0);
	std::vector<double> var_I(H * W, 0);
	std::vector<double> a(H * W, 0);
	std::vector<double> b(H * W, 0);
	// 将输入图片和引导图都除以 255
	const uchar* const noise_ptr = noise_image.data;
	const uchar* const guide_ptr = guide_image.data;
	for(int i = 0;i < length; ++i) noise_double_image[i] = (double)noise_ptr[i] / 255;
	for(int i = 0;i < length; ++i) guide_double_image[i] = (double)guide_ptr[i] / 255;
	// mean(P) 和 mean(I)
	const auto mean_P = box_filter(noise_double_image.data(), radius_h, radius_w, H, W);
	const auto mean_I = box_filter(guide_double_image.data(), radius_h, radius_w, H, W);
	// mean(P * I)
	for(int i = 0;i < length; ++i) I_P[i] = noise_double_image[i] * guide_double_image[i];
	const auto mean_IP = box_filter(I_P.data(), radius_h, radius_w, H, W);
	// mean(I * I)
	for(int i = 0;i < length; ++i) I_I[i] = guide_double_image[i] * guide_double_image[i];
	const auto mean_II = box_filter(I_I.data(), radius_h, radius_w, H, W);
	// 准备求 a 的分子 cov_IP  跟分母 var_I
	for(int i = 0;i < length; ++i) cov_IP[i] = mean_IP[i] - mean_I[i] * mean_P[i];
	for(int i = 0;i < length; ++i) var_I[i] = mean_II[i] - mean_I[i] * mean_I[i];
	// a = cov(I, P) / (var(I) + epsilon)
	for(int i = 0;i < length; ++i) a[i] = cov_IP[i] / (var_I[i] + epsilon);
	// b = mean(P) - a * mean(I)
	for(int i = 0;i < length; ++i) b[i] = mean_P[i] - a[i] * mean_I[i];
	// mean(a) 和 mean(b), 因为一个 q 点可能存在于多个窗口内, 多个窗口内都有 q 的一个值
	const auto mean_a = box_filter(a.data(), radius_h, radius_w, H, W);
	const auto mean_b = box_filter(b.data(), radius_h, radius_w, H, W);
	// q = a * I + b
	cv::Mat q = noise_image.clone();
	for(int i = 0;i < length; ++i) q.data[i] = cv::saturate_cast<uchar>(255 * (mean_a[i] * guide_double_image[i] + mean_b[i]));
	// 截取
	return q;
}




cv::Mat guided_filter_with_color(const cv::Mat& noise_image, const cv::Mat& guide_image, const int radius_h, const int radius_w, const double epsilon) {
    const int H = noise_image.rows;
    const int W = noise_image.cols;
    // -------- 【1】把引导图分离
    std::vector<cv::Mat> guide_bgr_images;
    cv::split(guide_image, guide_bgr_images);
    const uchar* const guide_B_ptr = guide_bgr_images[0].data;
    const uchar* const guide_G_ptr = guide_bgr_images[1].data;
    const uchar* const guide_R_ptr = guide_bgr_images[2].data;
    // -------- 【2】将输入图片和引导图都除以 255
	const int length = H * W;
	std::vector<double> noise_double_image(length, 0.0);
	std::vector<double> guide_double_image_B(length, 0.0), guide_double_image_G(length, 0.0), guide_double_image_R(length, 0.0);
	for(int i = 0;i < length; ++i) noise_double_image[i] = (double)noise_image.data[i] / 255;
	for(int i = 0;i < length; ++i) guide_double_image_B[i] = (double)guide_B_ptr[i] / 255;
	for(int i = 0;i < length; ++i) guide_double_image_G[i] = (double)guide_G_ptr[i] / 255;
	for(int i = 0;i < length; ++i) guide_double_image_R[i] = (double)guide_R_ptr[i] / 255;
    // -------- 【3】计算均值
    const auto mean_P = box_filter(noise_double_image.data(), radius_h, radius_w, H, W);
    const auto mean_I_B = box_filter(guide_double_image_B.data(), radius_h, radius_w, H, W);
    const auto mean_I_G = box_filter(guide_double_image_G.data(), radius_h, radius_w, H, W);
    const auto mean_I_R = box_filter(guide_double_image_R.data(), radius_h, radius_w, H, W);
    std::vector<double> IB_P(length, 0.0), IG_P(length, 0.0), IR_P(length, 0.0);
    for(int i = 0;i < length; ++i) IB_P[i] = guide_double_image_B[i] * noise_double_image[i];
    for(int i = 0;i < length; ++i) IG_P[i] = guide_double_image_G[i] * noise_double_image[i];
    for(int i = 0;i < length; ++i) IR_P[i] = guide_double_image_R[i] * noise_double_image[i];
    const auto mean_IB_P = box_filter(IB_P.data(), radius_h, radius_w, H, W);
    const auto mean_IG_P = box_filter(IG_P.data(), radius_h, radius_w, H, W);
    const auto mean_IR_P = box_filter(IR_P.data(), radius_h, radius_w, H, W);
    // -------- 【3】计算协方差
    std::vector<double> cov_IB_P(length, 0.0), cov_IG_P(length, 0.0), cov_IR_P(length, 0.0);
    for(int i = 0;i < length; ++i) cov_IB_P[i] = mean_IB_P[i] - mean_I_B[i] * mean_P[i];
    for(int i = 0;i < length; ++i) cov_IG_P[i] = mean_IG_P[i] - mean_I_G[i] * mean_P[i];
    for(int i = 0;i < length; ++i) cov_IR_P[i] = mean_IR_P[i] - mean_I_R[i] * mean_P[i];
    // -------- 【3】计算 I 三个通道的各种方差
    std::vector<double> I_BB(length, 0.0), I_BG(length, 0.0), I_BR(length, 0.0);
    std::vector<double> I_GG(length, 0.0), I_GR(length, 0.0), I_RR(length, 0.0);
    for(int i = 0;i < length; ++i) I_BB[i] = guide_double_image_B[i] * guide_double_image_B[i];
    for(int i = 0;i < length; ++i) I_BG[i] = guide_double_image_B[i] * guide_double_image_G[i];
    for(int i = 0;i < length; ++i) I_BR[i] = guide_double_image_B[i] * guide_double_image_R[i];
    for(int i = 0;i < length; ++i) I_GG[i] = guide_double_image_G[i] * guide_double_image_G[i];
    for(int i = 0;i < length; ++i) I_GR[i] = guide_double_image_G[i] * guide_double_image_R[i];
    for(int i = 0;i < length; ++i) I_RR[i] = guide_double_image_R[i] * guide_double_image_R[i];
    const auto mean_I_BB = box_filter(I_BB.data(), radius_h, radius_w, H, W);
    const auto mean_I_BG = box_filter(I_BG.data(), radius_h, radius_w, H, W);
    const auto mean_I_BR = box_filter(I_BR.data(), radius_h, radius_w, H, W);
    const auto mean_I_GG = box_filter(I_GG.data(), radius_h, radius_w, H, W);
    const auto mean_I_GR = box_filter(I_GR.data(), radius_h, radius_w, H, W);
    const auto mean_I_RR = box_filter(I_RR.data(), radius_h, radius_w, H, W);
    std::vector<double> var_I_BB(length, 0.0), var_I_BG(length, 0.0), var_I_BR(length, 0.0);
    std::vector<double> var_I_GG(length, 0.0), var_I_GR(length, 0.0), var_I_RR(length, 0.0);
    for(int i = 0;i < length; ++i) var_I_BB[i] = mean_I_BB[i] - mean_I_B[i] * mean_I_B[i];
    for(int i = 0;i < length; ++i) var_I_BG[i] = mean_I_BG[i] - mean_I_B[i] * mean_I_G[i];
    for(int i = 0;i < length; ++i) var_I_BR[i] = mean_I_BR[i] - mean_I_B[i] * mean_I_R[i];
    for(int i = 0;i < length; ++i) var_I_GG[i] = mean_I_GG[i] - mean_I_G[i] * mean_I_G[i];
    for(int i = 0;i < length; ++i) var_I_GR[i] = mean_I_GR[i] - mean_I_G[i] * mean_I_R[i];
    for(int i = 0;i < length; ++i) var_I_RR[i] = mean_I_RR[i] - mean_I_R[i] * mean_I_R[i];
    // 求解 a 和 b
    std::vector<std::vector<double> > a(3, std::vector<double>(length, 0.0));
    for(int i = 0;i < H; ++i) {
        for(int j = 0;j < W; ++j) {
            // 当前位置
            const int p = i * W + j;
            // 准备方差矩阵和 epsilon * eye(3)
            Eigen::Matrix<double, 3, 3> var_matrix;
            var_matrix << var_I_BB[p], var_I_BG[p], var_I_BR[p], var_I_BG[p], var_I_GG[p], var_I_GR[p], var_I_BR[p], var_I_GR[p], var_I_RR[p];
            var_matrix = var_matrix + epsilon * Eigen::MatrixXd::Identity(3, 3);
            // 求逆矩阵
            const auto inv_var_matrix = var_matrix.inverse();
            // 准备分子, cov_I_P
            Eigen::Matrix<double, 1, 3> cov_I_P;
            cov_I_P << cov_IB_P[p], cov_IG_P[p], cov_IR_P[p];
            // 乘法得到结果
            Eigen::Matrix<double, 1, 3> temp = cov_I_P * inv_var_matrix;
            // 更新到三通道分别的 a
            a[0][p] = temp(0, 0);
            a[1][p] = temp(0, 1);
            a[2][p] = temp(0, 2);
        }
    }
    // 求 b = mean(P) - a.* mean_I
    std::vector<double> b(length, 0.0);
    for(int i = 0;i < length; ++i) b[i] = mean_P[i] - a[0][i] * mean_I_B[i] - a[1][i] * mean_I_G[i] - a[2][i] * mean_I_R[i];
    // 求 mean(b) 和 mean(a)
    const auto mean_b = box_filter(b.data(), radius_h, radius_w, H, W);
    const auto mean_a_B = box_filter(a[0].data(), radius_h, radius_w, H, W);
    const auto mean_a_G = box_filter(a[1].data(), radius_h, radius_w, H, W);
    const auto mean_a_R = box_filter(a[2].data(), radius_h, radius_w, H, W);
    // 求滤波输出 q = mean_a .* I + mean_b
    cv::Mat q = noise_image.clone();
	for(int i = 0;i < length; ++i)
	    q.data[i] = cv::saturate_cast<uchar>(255 * (mean_a_B[i] * guide_double_image_B[i] + mean_a_G[i] * guide_double_image_G[i] + mean_a_R[i] * guide_double_image_R[i] + mean_b[i]));
    return q;
}




















