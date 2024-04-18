import os
import cv2
import h5py
import numpy
import random


def cv_show(image, message='crane'):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def heatmap_show(image):
	heat_map = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	return cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)


random.seed(212)
numpy.random.seed(212)

# 清晰图片保存的文件夹
clear_dir = "../clear"

# 保存
npy_dir = "./npy"
os.makedirs(npy_dir, exist_ok=True)
haze_dir = '../hazy'
os.makedirs(haze_dir, exist_ok=True)

# 获取所有数据
images_list = os.listdir("./mat")


for file_name in images_list:
	depth_map = h5py.File("./mat/{}".format(file_name), "r")

	depth_map = numpy.transpose(depth_map['depth'])
	numpy.save(os.path.join(npy_dir, file_name + ".npy"), depth_map)

	# 读取图像
	clear_image = cv2.imread(os.path.join(clear_dir, "{}".format(file_name.replace(".mat", ".jpg"))))
	if(clear_image is None):
		continue

	# cv_show(heatmap_show(depth_map))

	beta = random.uniform(0.3, 0.5)
	t = numpy.exp(-beta * depth_map)
	t = numpy.atleast_3d(t)

	A = random.uniform(0.75, 0.9)
	J = clear_image.astype("float32") / 255
	I = J * t + A * (1 - t)
	I = numpy.clip(I * 255, 0, 255).astype("uint8")

	# cv_show(I)
	cv2.imwrite(os.path.join(haze_dir, file_name.replace(".mat", ".png")), I, [cv2.IMWRITE_PNG_COMPRESSION, 0])