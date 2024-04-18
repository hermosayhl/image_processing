// C++
#include <cmath>
#include <vector>
#include <iostream>
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


    // https://www.cnblogs.com/lwl2015/p/4460711.html
    std::vector<double> box_filter(const double* const new_source, const int radius_h, const int radius_w, const int H, const int W) {
		const int kernel_w = 2 * radius_w + 1;
		const int kernel_h = 2 * radius_h + 1;
		const int H2 = H - kernel_h;
		const int W2 = W - kernel_w;
		std::vector<double> sum(W2 * H2, 0.0);
		std::vector<double> buffer(W, 0.0);

		// 先累加前 kernel_h 行的 box
		for(int i = 0; i< kernel_h; ++i){
			const double* const row_ptr = new_source + i * W;
			for(int j = 0; j < W; ++j) buffer[j] += row_ptr[j];
		}
		// 开始向右边和向下挪动, 计算每个点为中心的窗口累加值
		for(int i = 0; i < H2; ++i){
			// 当前 box 的累加值
			double cur_sum = 0;
			// 当前行第一个 box 的和
			for(int j = 0; j < kernel_w; ++j) cur_sum += buffer[j];
			const int _beg = i * W2;
			sum[_beg] = cur_sum;
			// 开始算这一行其它 box 的和, 减去左边去掉的一个, 加上右边加上的一个
			for(int j = 1; j < W2; ++j){
				cur_sum = cur_sum - buffer[j - 1] + buffer[j - 1 + kernel_w];
				sum[_beg + j] = cur_sum;
			}
			// 这一行求完了, 更新 buffer, 因为要向下挪, 这里其实是有错误的
			const double* const up_ptr = new_source + i * W;
			const double* const down_ptr = new_source + (i + kernel_h) * W;
			for(int j = 0; j < W; ++j)
				buffer[j] = buffer[j] - up_ptr[j] + down_ptr[j];
		}
		//遍历，得到每个点的和，传给矩阵result
		std::vector<double> result(H * W, 0.0);
		double* const result_ptr = result.data();
		// box 面积
		const int area = kernel_h * kernel_w;
		for(int i = radius_h + 1; i < H2; ++i){
			const int _beg = (i - radius_h) * W2;
			double * const row_ptr = result_ptr + i * H;
			for(int j = radius_w + 1; j < W - radius_w; ++j){
				const int pos = _beg + j - radius_w;
				row_ptr[j] = sum[pos] / area;
			}
		}
		return result;
	}

}

// 我可以先把它 padding, 之后在 Rect
cv::Mat guided_filter_channel(const cv::Mat& noise_image, const cv::Mat& guided_image, const int radius, const double epsilon) {
	const int H = noise_image.rows;
	const int W = noise_image.cols;
	const int length = H * W;
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
	const uchar* const guide_ptr = guided_image.data;
	for(int i = 0;i < length; ++i) noise_double_image[i] = (double)noise_ptr[i] / 255;
	for(int i = 0;i < length; ++i) guide_double_image[i] = (double)guide_ptr[i] / 255;
	// mean(P) 和 mean(I)
	const auto mean_P = box_filter(noise_double_image.data(), radius, radius, H, W);
	const auto mean_I = box_filter(guide_double_image.data(), radius, radius, H, W);
	// mean(P * I)
	for(int i = 0;i < length; ++i) I_P[i] = noise_double_image[i] * guide_double_image[i];
	const auto mean_IP = box_filter(I_P.data(), radius, radius, H, W);
	// mean(I * I)
	for(int i = 0;i < length; ++i) I_I[i] = guide_double_image[i] * guide_double_image[i];
	const auto mean_II = box_filter(I_I.data(), radius, radius, H, W);
	// 准备求 a 的分子 cov_IP  跟分母 var_I
	for(int i = 0;i < length; ++i) cov_IP[i] = mean_IP[i] - mean_I[i] * mean_P[i];
	for(int i = 0;i < length; ++i) var_I[i] = mean_II[i] - mean_I[i] * mean_I[i];
	// a = cov(I, P) / (var(I) + epsilon)
	for(int i = 0;i < length; ++i) a[i] = cov_IP[i] / (var_I[i] + epsilon);
	// b = mean(P) - a * mean(I)
	for(int i = 0;i < length; ++i) b[i] = mean_P[i] - a[i] * mean_I[i];
	// mean(a) 和 mean(b), 因为一个 q 点可能存在于多个窗口内, 多个窗口内都有 q 的一个值
	const auto mean_a = box_filter(a.data(), radius, radius, H, W);
	const auto mean_b = box_filter(b.data(), radius, radius, H, W);
	// q = a * I + b
	cv::Mat q = noise_image.clone();
	for(int i = 0;i < length; ++i) q.data[i] = cv::saturate_cast<uchar>(255 * (mean_a[i] * noise_double_image[i] + mean_b[i]));
	return q;
	// 截取, 如果前面有 padded 的话
	cv::Rect rect(radius, radius, noise_image.cols, noise_image.rows);
	return q(rect);
}



/*
 * // C++
#include <cmath>
#include <vector>
#include <iostream>
// self
#include "guided_filter.h"


namespace {

	void cv_show(const cv::Mat& one_image, const char* info="") {
		cv::imshow(info, one_image);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}

	cv::Mat box_filter(const cv::Mat& source, const int radius_h, const int radius_w) {
		// 首先我这里接收到的图片是 uchar, 但 type 是 1. / 255 的
		cv::Mat new_source;
		source.convertTo(new_source, CV_8UC1, 255);
		const int H = source.rows;
		const int W = source.cols;
		const int kernel_w = 2 * radius_w + 1;
		const int kernel_h = 2 * radius_h + 1;
		const int H2 = H - kernel_h;
		const int W2 = W - kernel_w;
		std::vector<double> sum(W2 * H2, 0.0);
		std::vector<double> buffer(W, 0.0);

		// 先累加前 kernel_h 行的 box
		for(int i = 0; i< kernel_h; ++i){
			uchar* const row_ptr = new_source.data + i * new_source.step;
			for(int j = 0; j < W; ++j) buffer[j] += row_ptr[j];
		}
		// 开始向右边和向下挪动, 计算每个点为中心的窗口累加值
		for(int i = 0; i < H2; ++i){
			// 当前 box 的累加值
			int cur_sum = 0;
			// 当前行第一个 box 的和
			for(int j = 0; j < kernel_w; ++j) cur_sum += buffer[j];
			const int _beg = i * W2;
			sum[_beg] = cur_sum;
			// 开始算这一行其它 box 的和, 减去左边去掉的一个, 加上右边加上的一个
			for(int j = 1; j < W2; ++j){
				cur_sum = cur_sum - buffer[j - 1] + buffer[j - 1 + kernel_w];
				sum[_beg + j] = cur_sum;
			}
			// 这一行求完了, 更新 buffer, 因为要向下挪
			uchar* const up_ptr = new_source.data + i * W;
			uchar* const down_ptr = new_source.data + (i + kernel_h) * W;
			for(int j = 0; j < W; ++j)
				buffer[j] = buffer[j] - up_ptr[j] + down_ptr[j];
		}
		//遍历，得到每个点的和，传给矩阵result
		cv::Mat result = new_source.clone();
		const int area = kernel_h * kernel_w;
		for(int i = radius_h + 1; i < H2; ++i){
			const int _beg = (i - radius_h) * W2;
			uchar* const row_ptr = result.data + i * H;
			for(int j = radius_w + 1; j < W - radius_w; ++j){
				const int pos = _beg + j - radius_w;
				row_ptr[j] = cv::saturate_cast<uchar>(sum[pos] / area);
			}
		}
		cv::Mat last_result;
		result.convertTo(last_result, CV_64FC1, 1. / 255);
		return last_result;
	}

	cv::Mat get_image_window_mean(const cv::Mat& source, const int radius) {
		cv::Mat mean_result = source.clone();
		// cv::boxFilter(source, mean_result, CV_64FC1, cv::Size(radius, radius));
		mean_result = box_filter(source, radius, radius);
		return mean_result;
	}

}


cv::Mat guided_filter(const cv::Mat& noise_image, const cv::Mat& guided_image, const int radius, const double epsilon) {
//    ------------【0】转换源图像信息，将输入扩展为64位浮点型，以便以后做乘法------------
	cv::Mat srcMat, guidedMat;
	noise_image.convertTo(srcMat, CV_64FC1, 1.0 / 255);
	guided_image.convertTo(guidedMat, CV_64FC1, 1.0 / 255);
	//--------------【1】各种均值计算----------------------------------
	const auto mean_p = get_image_window_mean(srcMat, radius);
	cv_show(mean_p);
	std::cout << mean_p.rows << "---" << mean_p.cols << std::endl;
	// 39 254 161 18 224 63 206 73 94 12
	const auto mean_I = get_image_window_mean(guidedMat, radius);
	const auto mean_Ip = get_image_window_mean(srcMat.mul(guidedMat), radius);
	const auto mean_II = get_image_window_mean(guidedMat.mul(guidedMat), radius);
	//--------------【2】计算相关系数，计算Ip的协方差cov和I的方差var------------------
	cv::Mat cov_Ip = mean_II.clone();
	const int length = noise_image.rows * noise_image.cols;
	std::cout << "length  " << length << std::endl;
//	for(int i = 0;i < length; ++i) {
//        cov_Ip.data[i] = mean_Ip.data[i] - mean_I.data[i] * mean_p.data[i];
//	}
	cov_Ip = mean_Ip - mean_I.mul(mean_p);
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);
	//---------------【3】计算参数系数a、b-------------------
	cv::Mat a = cov_Ip / (var_I + epsilon);
	cv::Mat b = mean_p - a.mul(mean_I);
	//--------------【4】计算系数a、b的均值-----------------
	const auto mean_a = get_image_window_mean(a, radius);
	const auto mean_b = get_image_window_mean(b, radius);
	//---------------【5】生成输出矩阵------------------
	cv::Mat dstImage = mean_a.mul(srcMat) + mean_b;
	cv::Mat result;
	dstImage.convertTo(result, CV_8UC1, 255);
	return result;
}

 */
