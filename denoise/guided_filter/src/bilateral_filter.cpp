
// self
#include "guided_filter.h"

namespace {

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }
}

// 双边滤波
cv::Mat bilateral_filter(
		const cv::Mat& noise_image, const int window_size,
		const double value_variance, const double space_variance) {
	// 检查合理性
	const int H = noise_image.rows;
	const int W = noise_image.cols;
	const int C = noise_image.channels();
	const int pad_size = (window_size - 1) >> 1;
	// std::cout << H << ", " << W << ", " << C << ", " << pad_size << "\n";
	// 对边缘填充
	const auto padded_image = make_pad(noise_image, pad_size, pad_size);
	// std::cout << "step  " << padded_image.step << std::endl;

	// 结果图像
	auto result = noise_image.clone();

	// 获取 range filter 模板
	const int value_range = C * 256;
	std::vector<double> value_table(value_range);
	const double value_variance_2 = 1.0 / (value_variance * value_variance);
	for(int i = 0;i < value_range; ++i)
		value_table[i] = std::exp(- 0.5 * value_variance_2 * i * i);

	// 获取 distance filter 模板
	const int half_size = (window_size - 1) >> 1;
	const int space_size = window_size * window_size;
	std::vector<double> space_table(space_size);
	std::vector<int> space_offset(space_size);
	int maxk = 0;
	const double space_variance_2 = - 0.5 / (space_variance * space_variance);
	// (以最中心的点为参照点, 计算其他点的空间权重, 以及在图像中的像素偏移值)
	for(int i = -half_size;i <= half_size; ++i) {
		for(int j = -half_size;j <= half_size; ++j) {
			space_table[maxk] = std::exp(space_variance_2 * (i * i + j * j));
			space_offset[maxk] = i * padded_image.step + j * C;
			++maxk;
		}
	}
	// 只是把 vector 换成指针的形式, 就这样 ? iterater 真的慢
	double* value_ptr = &value_table[0];
	double* space_ptr = &space_table[0];
	int *offset_ptr = &space_offset[0];

	// 遍历图像中每一个点, 然后滤波
	for(int i = 0;i < H; ++i) {
		// 从有效的滤波中心开始算, pad_size 行空的, 加上现在是有效图像第 i 行的数据, 当前第 i 行处在 pad_size 位置
		const uchar* row_ptr = padded_image.data + (i + pad_size) * padded_image.step + pad_size * C;
		uchar* result_row_ptr = result.data + i * result.step;
		for(int j = 0;j < W; ++j) {
			double sum_b = 0, sum_g = 0, sum_r = 0;
			double norm_b = 0, norm_g = 0, norm_r = 0;
			// 中心像素的下标是 j
			const int J = j * 3;
			const int b = row_ptr[J];
			const int g = row_ptr[J + 1];
			const int r = row_ptr[J + 2];
			// 遍历窗口(滤波核) 主要计算量就在这里
			for(int k = 0;k < maxk; ++k) {
				const uchar* cur_ptr = row_ptr + J + offset_ptr[k];

				const int __b = cur_ptr[0];
				const double w_b = space_ptr[k] * value_ptr[std::abs(b - __b)];
				sum_b += __b * w_b;
				norm_b += w_b;

				const int __g = cur_ptr[1];
				const double w_g = space_ptr[k] * value_ptr[std::abs(g - __g)];
				sum_g += __g * w_g;
				norm_g += w_g;

				const int __r = cur_ptr[2];
				const double w_r = space_ptr[k] * value_ptr[std::abs(r - __r)];
				sum_r += __r * w_r;
				norm_r += w_r;
			}
			// 更新到结果图像上
			result_row_ptr[J] = cv::saturate_cast<uchar>(sum_b / norm_b);
			result_row_ptr[J + 1] = cv::saturate_cast<uchar>(sum_g / norm_g);
			result_row_ptr[J + 2] = cv::saturate_cast<uchar>(sum_r / norm_r);
		}
	}
	return result;
}