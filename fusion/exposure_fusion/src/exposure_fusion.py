# Python
import os
import sys
import math
import copy
# 3rd party
import cv2
import numpy


def visualize(sequence_pyramid):
	S2 = len(sequence_pyramid)
	layers_num_2 = len(sequence_pyramid[0])
	_size = sequence_pyramid[0][0].shape[1::-1]
	display = numpy.concatenate(
		[cv2.normalize(numpy.concatenate([cv2.resize(sequence_pyramid[s][i], _size) for i in range(layers_num_2)], axis=0), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) for s in range(S2)], axis=1)
	global save_dir
	cv2.imwrite(os.path.join(save_dir, "sequence_laplace_pyramids.png"), (display * 255).astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])


def cv_show(image, message="crane"):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



def exposure_fusion(sequence, alphas=(1.0, 1.0, 1.0), best_illumination=0.5, sigma=0.2, eps=1e-12, 
		use_lappyr=True, use_gaussi=False, layers_num=5, scale=2.0, visual=False):
	# 转化成 float 数据
	sequence = numpy.stack([it.astype("float32") / 255 for it in sequence], axis=0)
	S = len(sequence)
	H, W, C = sequence[0].shape
	# 准备一些中间变量
	laplace_kernel = numpy.array(([0,  1, 0], [1, -4, 1], [0,  1, 0]), dtype="float32")
	mse = lambda l, r: (l - r) * (l - r)
	best_illumination = numpy.full((H, W), best_illumination, dtype='float32')
	# 存放每张图像的权重图
	weights = []
	# 每张图像都求一个初始权重
	for s in range(S):
		# 从拉普拉斯求对比度
		gray = cv2.cvtColor(sequence[s], cv2.COLOR_BGR2GRAY)
		contrast = cv2.filter2D(gray, -1, laplace_kernel, borderType=cv2.BORDER_REPLICATE)
		contrast = numpy.abs(contrast)
		# 求饱和度
		mean = numpy.mean(sequence[s], axis=-1)
		saturation = numpy.sqrt(numpy.mean([mse(sequence[s][:, :, ch], mean) for ch in range(3)], axis=0))
		# 求亮度
		illumination = [numpy.exp(-0.5 * mse(sequence[s][:, :, ch], best_illumination) / (sigma * sigma)) for ch in range(3)]
		illumination = numpy.prod(illumination, axis=0)
		# 三者加权
		cur_weight = numpy.power(contrast, alphas[0]) * numpy.power(saturation, alphas[1]) * numpy.power(illumination, alphas[2])
		weights.append(cur_weight)
	# 归一化
	normalize = lambda x: x / numpy.expand_dims(numpy.sum(x, axis=0), axis=0)
	weights = numpy.stack(weights, axis=0)
	weights += eps
	weights = normalize(weights)
	# 这里要把 sequence 还原回来
	sequence *= 255;
	# 粗糙的融合
	origin_fusion = numpy.sum(numpy.stack([weights, weights, weights], axis=-1) * sequence, axis=0)
	origin_fusion = numpy.clip(origin_fusion, 0, 255).astype("uint8")
	results = {"naive": origin_fusion}
	# 是否用高斯
	if(use_gaussi == True):
		# 使用高斯模糊对 weights 做模糊
		smoothed_weights = numpy.stack([cv2.GaussianBlur(w, (49, 49), 8) for w in weights], axis=0)
		smoothed_weights = normalize(smoothed_weights)
		smoothed_fusion = numpy.sum(numpy.stack([smoothed_weights, smoothed_weights, smoothed_weights], axis=-1) * sequence, axis=0)
		smoothed_fusion = numpy.clip(smoothed_fusion, 0, 255).astype("uint8")
		results.update({"gaussi_smoothed": smoothed_fusion})
	# 是否用拉普拉斯
	if(use_lappyr == True):

		# 根据最高分辨率的图像 high_res, 得到高度 layers 的高斯金字塔
		def build_gaussi_pyramid(high_res, layers):
			this_flash = [high_res]
			for i in range(1, layers):
				# 先对当前权重做高斯模糊, 然后下采样
				blurred = cv2.GaussianBlur(this_flash[i - 1], (5, 5), 0.83)
				blurred = blurred[::2, ::2]
				this_flash.append(blurred)
			return this_flash

		# 根据已知的高斯金字塔, 从最底层开始上采样, 得到每一个尺度的 laplace 细节
		def build_laplace_pyramaid(gaussi_pyramid, layers):
			upsampled = gaussi_pyramid[layers - 1]
			pyramid = [upsampled]
			for i in range(layers - 1, 0, -1):
				size = (gaussi_pyramid[i - 1].shape[1], gaussi_pyramid[i - 1].shape[0])
				# upsampled = cv2.resize(upsampled, size) # 假如我一直上采样呢
				upsampled = cv2.resize(gaussi_pyramid[i], size)
				pyramid.append(gaussi_pyramid[i - 1] - upsampled)
			pyramid.reverse() # 目前分辨率都是从高到低排列的
			return pyramid

		# 求每张图的权重的高斯金字塔
		sequence_weights_pyramids = [build_gaussi_pyramid(weights[s], layers_num) for s in range(S)]
		# 求每张图的高斯金字塔, 以求 laplace
		sequence_gaussi_pyramids = [build_gaussi_pyramid(sequence[s], layers_num) for s in range(S)]
		# 求每张图的 laplace 金字塔
		sequence_laplace_pyramids = [build_laplace_pyramaid(sequence_gaussi_pyramids[s], layers_num) for s in range(S)]
		# 这里可以归一化, 做可视化
		if(visual): visualize(sequence_laplace_pyramids)
		# 每一个尺度, 融合一系列图像的的 laplace 细节, 得到一个融合的 laplace 金字塔
		fused_laplace_pyramid = [numpy.sum([sequence_laplace_pyramids[k][n] * 
				numpy.atleast_3d(sequence_weights_pyramids[k][n]) for k in range(S)], axis=0) for n in range(layers_num)]

		# 先从最底层的图像开始, 每次上采样都加上同等尺度的 laplace 细节
		start = fused_laplace_pyramid[layers_num - 1]
		for i in range(layers_num - 2, -1, -1):
			upsampled = cv2.resize(start, (fused_laplace_pyramid[i].shape[1], fused_laplace_pyramid[i].shape[0]))
			start = fused_laplace_pyramid[i] + upsampled
		# 灰度值截断在 0-255 之间
		start = numpy.clip(start, 0, 255).astype("uint8")
		# 放到结果列表中
		results.update({"laplace_pyramid": start})

	return results



images_dir = "../images/input/5"
save_dir = "../images/output/5"
os.makedirs(save_dir, exist_ok=True)

# 读取图片
images_list = [os.path.join(images_dir, it) for it in os.listdir(images_dir)]
sequence = numpy.stack([cv2.imread(name) for name in images_list])

# 曝光融合
fused_results = exposure_fusion(
	sequence, 
	use_lappyr=True, 
	use_gaussi=True, 
	layers_num=7,
	alphas=(1.0, 1.0, 1.0),
	visual=False)

# 展示与保存
for l, r in fused_results.items():
	cv_show(r, l)
	cv2.imwrite(os.path.join(save_dir, l + ".png"), r, [cv2.IMWRITE_PNG_COMPRESSION, 0])