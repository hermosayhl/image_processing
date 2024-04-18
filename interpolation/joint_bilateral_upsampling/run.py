# Python
import os
import sys
import math
import time
import ctypes
# 3rd party
import cv2
import numpy
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


# 读取小分辨率光流结果
backward_flow_small = numpy.load("backward_flow.npy")

# 读取高分辨率的输入图
highres_input = cv2.imread("./input.png")
print("highres_input  :  ", highres_input.shape)

# 获取高宽的比例
h_small, w_small, channel_small = backward_flow_small.shape
h_large, w_large, channel_large = highres_input.shape
h_scale = float((h_large + 1) / h_small)
w_scale = float((w_large + 1) / w_small)
print("small  :  ", h_small, w_small, channel_small)
print("large  :  ", h_large, w_large, channel_large)
print("scale  :  ", h_scale, w_scale)

# 设定小分辨率的滤波半径
h_small_radius = 1
w_small_radius = 1

# 根据比例得到大分辨率的 滤波半径
h_large_radius = math.ceil(h_small_radius * h_scale)
w_large_radius = math.ceil(w_small_radius * w_scale)
print("radius-small  :  ", h_small_radius, w_small_radius)
print("radius-large  :  ", h_large_radius, w_large_radius)

# 对小分辨率结果 和 高分辨率引导图做 padding
source = numpy.pad(backward_flow_small, [(h_small_radius, h_small_radius + 1), (w_small_radius, w_small_radius), (0, 0)], mode="reflect")
guide  = numpy.pad(highres_input,       [(h_large_radius, h_large_radius), (w_large_radius, w_large_radius), (0, 0)], mode="reflect")
print("source  :  ", source.shape)
print("guide   :  ", guide.shape)


# 编译 C++ 代码
upsample_lib_path = "./crane_upsample.so"
os.system("g++ -fPIC -shared -O2 ./crane_upsample.cpp -o ./crane_upsample.so")
# 加载动态库
upsample_lib = ctypes.cdll.LoadLibrary(upsample_lib_path)


# 准备滤波参数
extra_args = []
extra_args += [h_small, w_small, channel_small, h_large, w_large, channel_large]
extra_args += [h_small_radius, w_small_radius, h_large_radius, w_large_radius]
extra_args += [h_scale, w_scale]
# spatial sigma
extra_args += [20.0]
# range sigma
extra_args += [3.5]
# 是否用双线性插值获取小分辨率的光流值, 1 为 True, -1 为 False
extra_args += [1]
# 是否对 spatial 做查表优化
extra_args += [1]
# 是否对 range 做查表优化
extra_args += [1]
print(extra_args)
# 参数统一转成 float32
extra_args = numpy.array(extra_args).astype("float32")

# 准备一个结果
result_JBU = numpy.zeros((h_large, w_large, channel_small), dtype=source.dtype)
print("result  ", result_JBU.shape, result_JBU.dtype)

# 执行
with Timer("JBU + bilinear") as scope:
	upsample_lib.joint_bilateral_upsampling(
		result_JBU.ctypes.data_as(ctypes.c_char_p),
		source.ctypes.data_as(ctypes.c_char_p),
		guide.ctypes.data_as(ctypes.c_char_p),
		extra_args.ctypes.data_as(ctypes.c_char_p)
	)
# 保存结果
cv_write("./result_JBU_bilinear.png", flow_viz.flow_to_image(result_JBU)[:, :, ::-1])

# 不做双线性, 更改参数
extra_args[-3] = -1
with Timer("JBU") as scope:
	upsample_lib.joint_bilateral_upsampling(
		result_JBU.ctypes.data_as(ctypes.c_char_p),
		source.ctypes.data_as(ctypes.c_char_p),
		guide.ctypes.data_as(ctypes.c_char_p),
		extra_args.ctypes.data_as(ctypes.c_char_p)
	)
cv_write("./result_JBU.png", flow_viz.flow_to_image(result_JBU)[:, :, ::-1])


# 做双线性, 不用 spatial LUT 优化
extra_args[-3] = 1
extra_args[-2] = -1
with Timer("JBU + bilinear - spatial-LUT") as scope:
	upsample_lib.joint_bilateral_upsampling(
		result_JBU.ctypes.data_as(ctypes.c_char_p),
		source.ctypes.data_as(ctypes.c_char_p),
		guide.ctypes.data_as(ctypes.c_char_p),
		extra_args.ctypes.data_as(ctypes.c_char_p)
	)
cv_write("./result_JBU_nospatialLUT.png", flow_viz.flow_to_image(result_JBU)[:, :, ::-1])


# 做双线性, 不用 range LUT 优化
extra_args[-2] = 1
extra_args[-1] = -1
with Timer("JBU + bilinear - range-LUT") as scope:
	upsample_lib.joint_bilateral_upsampling(
		result_JBU.ctypes.data_as(ctypes.c_char_p),
		source.ctypes.data_as(ctypes.c_char_p),
		guide.ctypes.data_as(ctypes.c_char_p),
		extra_args.ctypes.data_as(ctypes.c_char_p)
	)
cv_write("./result_JBU_norangeLUT.png", flow_viz.flow_to_image(result_JBU)[:, :, ::-1])
