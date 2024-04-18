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
lib = ctypes.cdll.LoadLibrary("./example.so")
# 准备一个彩色图像
image = cv2.imread("../../images/input/a0959-_DGW6327.png")
# 现在 rgb 通道上求一个最小
image = numpy.min(image, axis=-1).astype("uint8")
input_ptr = image.ctypes.data_as(ctypes.c_char_p)
# cv_show(image)

# 设定返回结果类型
rows, cols = image.shape
# lib.fast_min_filtering.restype = ndpointer(dtype=ctypes.c_ubyte, shape=(rows, cols))
lib.fast_min_filtering_optimized.restype = ndpointer(dtype=ctypes.c_ubyte, shape=(rows, cols))
# 做最小值滤波
with Timer("快速最小值滤波") as scope:
	result = lib.fast_min_filtering_optimized(
		input_ptr,
		rows, 
		cols, 
		ctypes.c_int(81), 
		ctypes.c_int(255), 
		ctypes.c_bool(True)
	)

# 展示
# cv_show(result)


lib.dynamic_min_filtering.restype = ndpointer(dtype=ctypes.c_ubyte, shape=(rows, cols))
with Timer("第二种快速最小值滤波") as scope:
	result2 = lib.dynamic_min_filtering(
		input_ptr,
		rows, 
		cols, 
		ctypes.c_int(81), 
		ctypes.c_int(255), 
		ctypes.c_bool(True)
	)
# cv_show(result2)


print(cv2.PSNR(result, result2))

cv2.imwrite("./output.png", result, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# 检测一下这个结果对不对
one = cv2.imread("answer.png", 0)
print(cv2.PSNR(one, result))