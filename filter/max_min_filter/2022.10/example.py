import cv2
import time
import numpy
import ctypes
from numpy.ctypeslib import ndpointer


# 计时
class Timer:
    def __init__(self, message=''):
        self.message = message

    def __enter__(self):
        self.start = time.process_time()

    def __exit__(self, type, value, trace):
        print(self.message + '耗时  :  {:.6f} s'.format(time.process_time() - self.start))


# 展示图像
def cv_show(one_image):
	cv2.imshow('crane', one_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# 加载动态库
lib = ctypes.cdll.LoadLibrary("./min_filter.so")
# 准备一个彩色图像
color_image = cv2.imread("./a0959-_DGW6327.png")
# 先在 rgb 通道上求一个最小
image = numpy.min(color_image, axis=-1).astype("uint8")



# 设定输入参数和返回结果
rows, cols = image.shape
# lib.plain_min_filter_uint8.restype = ndpointer(dtype=ctypes.c_ubyte, shape=(rows, cols))
radius     = 40
EXTREMUM   = 255


# 做最小值滤波
result_1 = image.copy()
with Timer("暴力") as scope:
	lib.plain_min_filter_uint8(
		result_1.ctypes.data_as(ctypes.c_char_p), 
		image.ctypes.data_as(ctypes.c_char_p),
		rows, 
		cols, 
		ctypes.c_int(radius), 
		ctypes.c_int(EXTREMUM)
	)


result_2 = image.copy()
with Timer("分离") as scope:
	lib.split_min_filter_uint8(
		result_2.ctypes.data_as(ctypes.c_char_p), 
		image.ctypes.data_as(ctypes.c_char_p),
		rows, 
		cols, 
		ctypes.c_int(radius), 
		ctypes.c_int(EXTREMUM)
	)
assert(numpy.allclose(result_1, result_2))



result_3 = image.copy()
with Timer("单调队列") as scope:
	lib.monotony_queue_min_filter_uint8(
		result_3.ctypes.data_as(ctypes.c_char_p), 
		image.ctypes.data_as(ctypes.c_char_p),
		rows, 
		cols, 
		ctypes.c_int(radius), 
		ctypes.c_int(EXTREMUM)
	)
assert(numpy.allclose(result_2, result_3))



result_4 = image.copy()
with Timer("加速单调队列") as scope:
	lib.faster_monotony_queue_min_filter_uint8(
		result_4.ctypes.data_as(ctypes.c_char_p), 
		image.ctypes.data_as(ctypes.c_char_p),
		rows, 
		cols, 
		ctypes.c_int(radius), 
		ctypes.c_int(EXTREMUM)
	)
assert(numpy.allclose(result_3, result_4))



result_5 = image.copy()
with Timer("动态规划") as scope:
	lib.dynamic_programming_min_filter_uint8(
		result_5.ctypes.data_as(ctypes.c_char_p), 
		image.ctypes.data_as(ctypes.c_char_p),
		rows, 
		cols, 
		ctypes.c_int(radius), 
		ctypes.c_int(EXTREMUM)
	)
assert(numpy.allclose(result_4, result_5))


# 保存结果
comparison = numpy.concatenate([
	color_image, numpy.stack([result_5, result_5, result_5], axis=-1)], axis=1)
# cv_show(comparison)
cv2.imwrite("./comparison.png", result_5, [cv2.IMWRITE_PNG_COMPRESSION, 0])