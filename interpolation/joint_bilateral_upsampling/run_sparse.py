# Python
import os
import sys
import math
import time
import ctypes
# 3rd party
import cv2
import numpy
import onnxruntime
# self
import flow_viz



class Timer:
    def __init__(self, message=''):
        self.message = message
    def __enter__(self):
        self.start = time.process_time()
    def __exit__(self, type, value, trace):
        print(self.message + ' : {} s'.format(time.process_time() - self.start))


def cv_show(image, message="crane"):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

cv_write = lambda x, y: cv2.imwrite(x, y, [cv2.IMWRITE_PNG_COMPRESSION, 0])




opt = lambda: None
opt.images_dir     = "./images/sample_3"
opt.use_flow_cache = False
opt.flow_cache     = os.path.join(opt.images_dir, "forward_flow.npy")
opt.image1_path    = os.path.join(opt.images_dir, "image1.jpg")
opt.image2_path    = os.path.join(opt.images_dir, "image2.jpg")
opt.small_size     = (600, 480)
opt.onnx_file      = "./RAFT-sim.onnx"
os.makedirs(opt.images_dir, exist_ok=True)
add_to_save = lambda x, y: cv_write(os.path.join(opt.images_dir, x), y)



if (opt.use_flow_cache and os.path.exists(opt.flow_cache)):
	forward_flow  = numpy.load(opt.flow_cache)
	highres_input = cv2.imread(opt.image1_path)
	forward_flow_visualize = flow_viz.flow_to_image(forward_flow)[:, :, ::-1]
	print("直接从缓存中读取光流结果")

else:
	# 先读取两张图象
	image1 = cv2.imread(opt.image1_path)
	image2 = cv2.imread(opt.image2_path)

	# 计算从 image1 到 image2 的光流, 则光流和 image1 一致, 引导图像应该是 image1
	highres_input = image1

	# 把 image1, image2 下采样到小分辨率
	image1 = cv2.resize(image1, opt.small_size)
	image2 = cv2.resize(image2, opt.small_size)

	# 把两张图象转换成 onnx 模型支持的内存格式, 1xCxHxW, 同时是 RGB 序
	onnx_transform = lambda x: numpy.ascontiguousarray(numpy.expand_dims(numpy.transpose(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), [2, 0, 1]), axis=0)).astype("float32")
	image1 = onnx_transform(image1)
	image2 = onnx_transform(image2)
	print("image1  :  ", image1.shape)
	print("image2  :  ", image2.shape)

	# 读取 onnx 模型
	task      = onnxruntime.InferenceSession(opt.onnx_file, providers=['CPUExecutionProvider'])

	# 推理
	[forward_flow]    = task.run(["flow"], {"image1": image1, "image2": image2})
	forward_flow = numpy.ascontiguousarray(numpy.transpose(forward_flow[0], [1, 2, 0]))
	print("forward_flow  ", forward_flow.shape)

	# 可视化小分辨率光流
	forward_flow_visualize = flow_viz.flow_to_image(forward_flow)[:, :, ::-1]
	add_to_save("forward_flow_visualize.png", forward_flow_visualize)

	# 保存
	numpy.save(opt.flow_cache, forward_flow)


# 展示小分辨率光流
cv_show(forward_flow_visualize)





# 设定小分辨率的滤波半径
opt.h_small_radius    = 1
opt.w_small_radius    = 1
opt.upsample_lib_path = "./crane_upsample.so"
opt.upsample_lib_cmd  = "g++ -fPIC -shared -O2 ./crane_upsample.cpp -o {}".format(opt.upsample_lib_path)
opt.spatial_sigma     = 2.5
opt.range_sigma       = 0.9
opt.save_name         = "result_sparse_JBU_bilinear.png"
opt.use_bilinear      = 1
opt.use_spatial_lut   = 1
opt.use_range_lut     = 1

# 获取高宽的比例
h_small, w_small, channel_small = forward_flow.shape
h_large, w_large, channel_large = highres_input.shape
h_scale = float((h_large + 1) / h_small) # 比例这里设置更大了, 尽量避免那种 1080/3.60 = 300 越界的情况
w_scale = float((w_large + 1) / w_small)

# 根据比例得到大分辨率的 滤波半径
h_large_radius = math.ceil(opt.h_small_radius * h_scale)
w_large_radius = math.ceil(opt.w_small_radius * w_scale)

# 对小分辨率结果 和 高分辨率引导图做 padding(注意 pad 多加了 1, 防止后续滤波时溢出)
h_small_pad = opt.h_small_radius + 1
w_small_pad = opt.w_small_radius + 1
h_large_pad = h_large_radius + 1
w_large_pad = w_large_radius + 1
source = numpy.pad(forward_flow,  [(h_small_pad, h_small_pad), (w_small_pad, w_small_pad), (0, 0)], mode="reflect")
guide  = numpy.pad(highres_input, [(h_large_pad, h_large_pad), (w_large_pad, w_large_pad), (0, 0)], mode="reflect")

# 编译 C++ 代码
os.system(opt.upsample_lib_cmd)

# 加载动态库
upsample_lib = ctypes.cdll.LoadLibrary(opt.upsample_lib_path)

# 准备滤波参数
extra_args = []
extra_args += [h_small, w_small, channel_small, h_large, w_large, channel_large]
extra_args += [opt.h_small_radius, opt.w_small_radius, h_large_radius, w_large_radius]
extra_args += [h_small_pad, w_small_pad, h_large_pad, w_large_pad]
extra_args += [h_scale, w_scale]
# spatial sigma
extra_args += [opt.spatial_sigma]
# range sigma
extra_args += [opt.range_sigma]
# 是否用双线性插值获取小分辨率的光流值, 1 为 True, -1 为 False
extra_args += [opt.use_bilinear]
# 是否对 spatial 做查表优化
extra_args += [opt.use_spatial_lut]
# 是否对 range 做查表优化
extra_args += [opt.use_range_lut]
print("滤波参数  :  ", extra_args)
# 参数统一转成 float32!
extra_args = numpy.array(extra_args).astype("float32")

# 准备一个结果
result_JBU = numpy.zeros((h_large, w_large, channel_small), dtype=source.dtype)

# 执行
with Timer("JBU + bilinear + sparse") as scope:
	upsample_lib.sparse_joint_bilateral_upsampling(
		result_JBU.ctypes.data_as(ctypes.c_char_p),
		source.ctypes.data_as(ctypes.c_char_p),
		guide.ctypes.data_as(ctypes.c_char_p),
		extra_args.ctypes.data_as(ctypes.c_char_p)
	)
# 保存结果
add_to_save(opt.save_name, flow_viz.flow_to_image(result_JBU)[:, :, ::-1])



# 顺便保存下其它 resize 方法的结果, 用于对比
add_to_save("result_nearest.png",  flow_viz.flow_to_image(cv2.resize(forward_flow, (w_large, h_large), cv2.INTER_NEAREST))[:, :, ::-1])
add_to_save("result_bilinear.png", flow_viz.flow_to_image(cv2.resize(forward_flow, (w_large, h_large), cv2.INTER_LINEAR))[:, :, ::-1])
add_to_save("result_bicubic.png",  flow_viz.flow_to_image(cv2.resize(forward_flow, (w_large, h_large), cv2.INTER_CUBIC))[:, :, ::-1])
add_to_save("result_area.png",     flow_viz.flow_to_image(cv2.resize(forward_flow, (w_large, h_large), cv2.INTER_AREA))[:, :, ::-1])
add_to_save("result_lanzos4.png",  flow_viz.flow_to_image(cv2.resize(forward_flow, (w_large, h_large), cv2.INTER_LANCZOS4))[:, :, ::-1])



# 采用更密集的方法
opt.save_name = "result_JBU_bilinear.png"
result_JBU = numpy.zeros((h_large, w_large, channel_small), dtype=source.dtype)

# 执行(这里还有 bug, 亟需解决)
with Timer("JBU + bilinear") as scope:
	upsample_lib.joint_bilateral_upsampling(
		result_JBU.ctypes.data_as(ctypes.c_char_p),
		source.ctypes.data_as(ctypes.c_char_p),
		guide.ctypes.data_as(ctypes.c_char_p),
		extra_args.ctypes.data_as(ctypes.c_char_p)
	)
# 保存结果
add_to_save(opt.save_name, flow_viz.flow_to_image(result_JBU)[:, :, ::-1])
