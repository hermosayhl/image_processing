// C++
#include <cmath>
#include <vector>
#include <cstring>
#include <iostream>



namespace {

	template<typename T>
	void make_padding(T* const des_ptr, const T* const src_ptr, const int H, const int W, const int h_pad, const int w_pad) {
		// 获取目标图像一行的像素个数， 一列的像素个数
		const int W2 = W + 2 * h_pad;
		const int H2 = H + 2 * w_pad;
		// 拷贝一行的代价
		const int bytes_in_line = sizeof(T) * W;
		// 首先拷贝中间的内容
		for (int i = 0; i < H; ++i)
			std::memcpy(des_ptr + (h_pad + i) * W2 + w_pad, src_ptr + i * W, bytes_in_line);
		// 拷贝上面 padding 的内容, 镜像
		for (int i = 0; i < h_pad; ++i) 
			std::memcpy(des_ptr + (h_pad - 1 - i) * W2 + w_pad, src_ptr + i * W, bytes_in_line);
		// 拷贝下面 padding 的内容, 镜像
		for (int i = 0; i < h_pad; ++i) 
			std::memcpy(des_ptr + (H + h_pad + i) * W2 + w_pad, src_ptr + (H - 1 - i) * W, bytes_in_line);
		// 拷贝左边 padding 的内容
		for (int i = 0; i < H2; ++i) {
			int start = i * W2 + w_pad + w_pad - 1;
			for (int j = 0; j < w_pad; ++j) des_ptr[i * W2 + j] = des_ptr[start - j];
			start = i * W2 + W + w_pad;
			for (int j = 0; j < w_pad; ++j) des_ptr[start + j] = des_ptr[start - 1 - j];
		}
	}

	template<typename T>
	inline T square(const T x) {
		return x * x;
	}

	template<typename T, typename S>
	inline T clip(const S x, const T low, const T high) {
		if (x < low) return low;
		else if (x > high) return high;
		else return x;
	}
}



template<typename src_type, typename guide_type>
void joint_bilateral_upsampling_inplementation(
		src_type* result,
		src_type* source,
		guide_type* guide,
		float* extra_args) {
	// 首先解析各项参数
	int args_cnt         = 0;
	int h_small 	     = int(extra_args[args_cnt++]);
	int w_small          = int(extra_args[args_cnt++]);
	int channel_small    = int(extra_args[args_cnt++]);
	int h_large          = int(extra_args[args_cnt++]);
	int w_large 	     = int(extra_args[args_cnt++]);
	int channel_large    = int(extra_args[args_cnt++]);
	int h_small_radius   = int(extra_args[args_cnt++]);
	int w_small_radius   = int(extra_args[args_cnt++]);
	int h_large_radius   = int(extra_args[args_cnt++]);
	int w_large_radius   = int(extra_args[args_cnt++]);
	int h_small_pad      = int(extra_args[args_cnt++]);
	int w_small_pad      = int(extra_args[args_cnt++]);
	int h_large_pad      = int(extra_args[args_cnt++]);
	int w_large_pad      = int(extra_args[args_cnt++]);
	float h_scale        = extra_args[args_cnt++];
	float w_scale        = extra_args[args_cnt++];
	float spatial_sigma  = extra_args[args_cnt++];
	float range_sigma    = extra_args[args_cnt++];
	bool use_bilinear    = extra_args[args_cnt++] > 0;
	bool use_spatial_lut = extra_args[args_cnt++] > 0;
	bool use_range_lut   = extra_args[args_cnt++] > 0;
	// 生成一些辅助变量
	int w_small_padded   = w_small + 2 * w_small_pad;
	int w_large_padded   = w_large + 2 * w_large_pad;

	// printf("[%d %d %d] [%d %d %d] [%d %d %d %d] [%d %d %d %d] [%f %f] [%f %f] [%d %d %d] [%d %d]\n",
	// 	h_small, w_small, channel_small,
	// 	h_large, w_large, channel_large,
	// 	h_small_radius, w_small_radius, h_large_radius, w_large_radius,
	// 	h_small_pad, w_small_pad, h_large_pad, w_large_pad,
	// 	h_scale, w_scale,
	// 	spatial_sigma, range_sigma,
	// 	use_bilinear, use_spatial_lut, use_range_lut,
	// 	w_small_padded, w_large_padded);

	
	// 提前计算一个空间权重表
	int spatial_size = (2 * h_large_radius + 1) * (2 * w_large_radius + 1);
	std::vector<src_type> spatial_table;
	if (use_spatial_lut) {
		// 如果使用表优化, 就加速
		spatial_table.resize(spatial_size);
		int spatial_cnt{0};
		for (int x = -h_large_radius; x <= h_large_radius; ++x) {
			for (int y = -w_large_radius; y <= w_large_radius; ++y) {
				spatial_table[spatial_cnt++] = std::exp(-(square<src_type>(x / h_scale) + square<src_type>(w_scale)) / (2 * square<src_type>(spatial_sigma)));
			}
		}
	}

	// 提前计算一个值域权重表, 只有 guide 是 int 族时才能这么干
	constexpr int range_size{3 * 256 * 256};
	constexpr src_type range_norm{256.f};
	std::vector<src_type> range_table;
	if (use_range_lut) {
		// 如果对 range 做表优化
		src_type sigma_inv = 1.f / (square(range_norm) * 2 * square(range_sigma));
		range_table.resize(range_size);
		for (int i = 0; i < range_size; ++i) {
			range_table[i] = std::exp(-i * sigma_inv);
		}
	}
		
	// 生成高分辨率结果的每一个点的值
	for (int i = 0; i < h_large; ++i) {
		// 当前行的引导指针
		guide_type* guide_ptr = guide  + channel_large * ((h_large_pad + i) * w_large_padded + w_large_pad);
		// 当前行的结果指针
		src_type* res_ptr     = result + channel_small * i * w_large;
		// 当前行映射到小分辨率所在行
		int   i_small         = int(i / h_scale);
		src_type* src_ptr     = source + channel_small * ((h_small_pad + i_small) * w_small_padded + w_small_pad);
		// 遍历当前行的所有位置
		for (int j = 0; j < w_large; ++j) {
			// 首先找到当前位置在引导图上的值 P
			guide_type* P = guide_ptr + j * channel_large;
			// 初始化累加值
			std::vector<src_type> temp(channel_small, 0.f);
			// src_type temp[channel_small] = {0.f};
			src_type weight_sum{0.f};
			// 遍历邻域
			int spatial_cnt{0};
			for (int x = -h_large_radius; x <= h_large_radius; ++x) {
				for (int y = -w_large_radius; y <= w_large_radius; ++y) {
					
					// 查表获取这个位置的空间权重
					src_type spatial_weight;
					if (use_spatial_lut) 
						spatial_weight = spatial_table[spatial_cnt++];
					else
						spatial_weight = std::exp(-(square<src_type>(x / h_scale) + square<src_type>(w_scale)) / (2 * square<src_type>(spatial_sigma)));

					// 获取处在 (i + x, j + y) 的邻域像素
					guide_type* Q = P + channel_large * (x * w_large_padded + y);
					// 查表获取这个位置的值域差权重
					src_type range_weight;
					if (use_range_lut) {
						int diff_pos = 0;
						for (int c = 0; c < channel_large; ++c)
							diff_pos += square<int>(P[c] - Q[c]);
						range_weight = range_table[diff_pos];
					} else {
						// 这里还有点 bug
						src_type diff{0.f};
						for (int c = 0; c < channel_large; ++c)
							diff += square<src_type>(P[c] / range_norm - Q[c] / range_norm);
						range_weight = std::exp(-diff / (2 * square(range_sigma)));
					}
						
					// 获取邻域点 (i + x, j + y) 对中心点 (i, j) 的加权值
					src_type this_weight = spatial_weight * range_weight;
					weight_sum        += this_weight;
					// 找到 (i + x, j + y) 对应在小分辨率的值, 直接加权
					if (not use_bilinear) {
						// 如果不用插值, 直接最近邻获取小分辨率的值
						src_type* S = src_ptr + channel_small * (int(x / h_scale) * w_small_padded + int((j + y) / w_scale));
						for (int c = 0; c < channel_small; ++c)
							temp[c] += this_weight * S[c];
					}
					else {
						// 根据相对坐标 (x / h_scale, (j + y) / w_scale) 插值
						// src_type S[channel_small];
						std::vector<src_type> S(channel_small, 0.f);
						src_type x_offset = x / h_scale;
						src_type y_offset = (j + y) / w_scale;
						// 获取上下界
						int x_low    = std::floor(x_offset);
						int y_low    = std::floor(y_offset);
						// 获取四个坐标位置(用于插值的)
						src_type* Q1 = src_ptr + channel_small * (x_low * w_small_padded + y_low);
						src_type* Q2 = Q1 + channel_small;
						src_type* Q3 = Q1 + channel_small * w_small_padded;
						src_type* Q4 = Q3 + channel_small;
						// 计算加权值
						src_type x_high_weight = x_offset - x_low;
						src_type y_high_weight = y_offset - y_low;
						// 开始加权
						for (int c = 0; c < channel_small; ++c) {
							src_type up   = (1.f - y_high_weight) * Q1[c] + y_high_weight * Q2[c];
							src_type down = (1.f - y_high_weight) * Q3[c] + y_high_weight * Q4[c];
							src_type val  = (1.f - x_high_weight) * up + x_high_weight * down;
							temp[c] += this_weight * val;
						}
					}
				}
			}
			// 赋值
			for (int c = 0; c < channel_small; ++c) {
				res_ptr[channel_small * j + c] = temp[c] / weight_sum;
			}
		}
	}
}








namespace {

	template<typename src_type, typename res_type>
	void bilinear_interpolate(
			res_type* result, 
			src_type* source, 
			float x_offset, 
			float y_offset, 
			int width, 
			int channel) {
		// 得到下界
		int x_low = std::floor(x_offset);
		int y_low = std::floor(y_offset);
		// 得到权重
		float x_high_weight = x_offset - x_low;
		float y_high_weight = y_offset - y_low;
		// 找到四个顶点
		src_type* Q1 = source + (x_low * width + y_low) * channel;
		src_type* Q2 = Q1     + channel;
		src_type* Q3 = Q1     + width * channel;
		src_type* Q4 = Q3     + channel;
		// 开始加权
		for (int c = 0; c < channel; ++c) {
			float up   = (1.f - y_high_weight) * Q1[c] + y_high_weight * Q2[c];
			float down = (1.f - y_high_weight) * Q3[c] + y_high_weight * Q4[c];
			result[c]  = (1.f - x_high_weight) * up + x_high_weight * down;
		}
	}
}






void sparse_joint_bilateral_upsampling_inplementation(
		float* result,
		float* source,
		unsigned char* guide,
		float* extra_args) {
	// 首先解析参数
	// 首先解析各项参数
	int args_cnt         = 0;
	int h_small 	     = int(extra_args[args_cnt++]);
	int w_small          = int(extra_args[args_cnt++]);
	int channel_small    = int(extra_args[args_cnt++]);
	int h_large          = int(extra_args[args_cnt++]);
	int w_large 	     = int(extra_args[args_cnt++]);
	int channel_large    = int(extra_args[args_cnt++]);
	int h_small_radius   = int(extra_args[args_cnt++]);
	int w_small_radius   = int(extra_args[args_cnt++]);
	int h_large_radius   = int(extra_args[args_cnt++]);
	int w_large_radius   = int(extra_args[args_cnt++]);
	int h_small_pad      = int(extra_args[args_cnt++]);
	int w_small_pad      = int(extra_args[args_cnt++]);
	int h_large_pad      = int(extra_args[args_cnt++]);
	int w_large_pad      = int(extra_args[args_cnt++]);
	float h_scale        = extra_args[args_cnt++];
	float w_scale        = extra_args[args_cnt++];
	float spatial_sigma  = extra_args[args_cnt++];
	float range_sigma    = extra_args[args_cnt++];
	bool use_bilinear    = extra_args[args_cnt++] > 0;
	bool use_spatial_lut = extra_args[args_cnt++] > 0;
	bool use_range_lut   = extra_args[args_cnt++] > 0;
	// 生成一些辅助变量
	int w_small_padded   = w_small + 2 * w_small_pad;
	int w_large_padded   = w_large + 2 * w_large_pad;

	// printf("[%d %d %d] [%d %d %d] [%d %d %d %d] [%d %d %d %d] [%f %f] [%f %f] [%d %d %d] [%d %d]\n",
	// 	h_small, w_small, channel_small,
	// 	h_large, w_large, channel_large,
	// 	h_small_radius, w_small_radius, h_large_radius, w_large_radius,
	// 	h_small_pad, w_small_pad, h_large_pad, w_large_pad,
	// 	h_scale, w_scale,
	// 	spatial_sigma, range_sigma,
	// 	use_bilinear, use_spatial_lut, use_range_lut,
	// 	w_small_padded, w_large_padded);

	// 先把空间权重表算出来
	int   spatial_size = (2 * h_small_radius + 1) * (2 * w_small_radius + 1);
	std::vector<float> spatial_lut(spatial_size);
	int   spatial_cnt  = 0;
	for (int x = -h_small_radius; x <= h_small_radius; ++x) {
		for (int y = -w_small_radius; y <= w_small_radius; ++y) {
			spatial_lut[spatial_cnt++] = std::exp(-(square(x) + square(y)) / (2 * square(spatial_sigma)));
		}
	}
	// 先把值域权重表也算出来
	int   range_size = 3 * square(255);
	float range_inv  = 1.f / (square(255.f) * 2 * square(range_sigma));
	std::vector<float> range_lut;
	if (use_range_lut) {
		range_lut.resize(range_size);
		for (int diff = 0; diff < range_size; ++diff) {
			range_lut[diff] = std::exp(-diff * range_inv);
		}
	}


	// 开始联合滤波, 生成每一个点
	for (int i = 0; i < h_large; ++i) {
		// 当前行对应在 guide 引导图像中的指针
		unsigned char* guide_ptr = guide  + ((h_large_pad + i) * w_large_padded + w_large_pad) * channel_large;
		// 当前行对应结果 result 的指针
		float* res_ptr           = result + i * w_large * channel_small;
		// 当前行对应小分辨率图像上的指针
		float i_small_f          = i / h_scale;
		int   i_small            = std::floor(i_small_f);
		float* src_ptr           = source + ((h_small_pad + i_small) * w_small_padded + w_small_pad) * channel_small;
		// 对于每一个位置
		for (int j = 0; j < w_large; ++j) {
			// 当前列对应小分辨率上的值
			float j_small_f  = j / w_scale;
			float j_small    = std::floor(j_small_f);
			// 找到当前点在 guide 上的值
			unsigned char* P = guide_ptr + j * channel_large;
			// 初始化累计值
			float weight_sum{0.f};
			std::vector<float> temp(channel_small, 0.f);
			// 遍历当前点的邻域
			int cnt = 0;
			for (int x = -h_small_radius; x <= h_small_radius; ++x) {
				for (int y = -w_small_radius; y <= w_small_radius; ++y) {
					// 插值获取当前邻域点的值 Q
					std::vector<float> Q(channel_large, 0.f);
					{
						float x_offset = x * h_scale;
						float y_offset = y * w_scale + j;
						// 对 guide 做插值
						bilinear_interpolate<unsigned char, float>(Q.data(), guide_ptr, x_offset, y_offset, w_large_padded, channel_large);
					}
					// 获取邻域点 Q 对 P 的加权值
					float range_weight{0.f};
					if (not use_range_lut) {
						// P 和 Q 之间计算差异
						float diff = 0.f;
						for (int c = 0; c < channel_large; ++c)
							diff += square<float>(P[c] / 255.f - Q[c] / 255.f);
						// 根据 PQ 差异得到值域权重
						range_weight = std::exp(-diff / (2 * square(range_sigma)));
					}
					else {
						// 如果查表
						int diff_pos = 0;
						for (int c = 0; c < channel_large; ++c)
							diff_pos += square(int(P[c]) - int(Q[c]));
						range_weight = range_lut[diff_pos];
					}
						
					// 查表获取当前偏移 (x, y) 处的空间权重
					float spatial_weight = spatial_lut[cnt++];
					// 当前权重 = 值域权重 * 空间权重
					float this_weight = spatial_weight * range_weight;
					// 权重累积
					weight_sum += this_weight;
					// Q 对应到小分辨率上, 获取待加权的光流值
					std::vector<float> S(channel_small, 0.f);
					{
						float x_offset = x + i_small_f - i_small;
						float y_offset = y + j_small_f;
						bilinear_interpolate<float, float>(S.data(), src_ptr, x_offset, y_offset, w_small_padded, channel_small);
					}
					// 加权
					for (int c = 0; c < channel_small; ++c) {
						temp[c] += S[c] * this_weight;
					}
				}
			}
			// 赋值
			for (int c = 0; c < channel_small; ++c) {
				res_ptr[j * channel_small + c] = temp[c] / weight_sum;
			}
		}
	}
}













extern "C" {
	void joint_bilateral_upsampling(
			float* result,
			float* source,
			unsigned char* guide,
			float* extra_args) {
		joint_bilateral_upsampling_inplementation<float, unsigned char>(result, source, guide, extra_args);
	}


	void sparse_joint_bilateral_upsampling(
			float* result,
			float* source,
			unsigned char* guide,
			float* extra_args) {
		sparse_joint_bilateral_upsampling_inplementation(result, source, guide, extra_args);
	}
}