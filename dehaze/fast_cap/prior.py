import os
import cv2
import math
import h5py
import numpy
import random
import matplotlib.pyplot as plt


def cv_show(image, message='crane'):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def heatmap_show(image):
	heat_map = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	return cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

# 获取所有文件 id
all_file_ids = [it.replace(".npy", "") for it in os.listdir("D:/data/datasets/RESIDE/OTS_beta/depth/npy/")]
# 选取一部分试试
selected_file_ids = random.sample(all_file_ids, k=30)

for cnt, file_id in enumerate(selected_file_ids):
	# 深度图
	depth = numpy.load("D:/data/datasets/RESIDE/OTS_beta/depth/npy/{}.npy".format(file_id))
	# hazy image
	hazy_image = cv2.imread("D:/data/datasets/RESIDE/OTS_beta/hazy/{}.png".format(file_id))
	# clear image
	clear_image = cv2.imread("D:/data/datasets/RESIDE/OTS_beta/clear/{}.jpg".format(file_id))

	def get_diff_depth(image, radius=7):
		# hsv
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		s = hsv[:, :, 1]
		v = hsv[:, :, 2]
		diff = []
		d = []
		H, W = hsv.shape[:2]
		for i in range(radius, H - radius, radius):
			for j in range(radius, W - radius, radius):
				# 直接统计局部均值, 看看 v - s 跟 depth 的关系
				d.append(depth[i - radius: i + radius, j - radius: j + radius].mean())
				diff.append((v - s)[i - radius: i + radius, j - radius: j + radius].mean())
		return d, diff

	plt.subplot(2, 2, 1)
	plt.imshow(clear_image[:, :, ::-1])
	plt.subplot(2, 2, 2)
	plt.scatter(*get_diff_depth(clear_image, 7), s=2, c='g')
	plt.subplot(2, 2, 3)
	plt.imshow(hazy_image[:, :, ::-1])
	plt.subplot(2, 2, 4)
	plt.scatter(*get_diff_depth(hazy_image, 7), s=2, c='r')
	plt.savefig("./images/prior/{}_comparison.png".format(file_id), dpi=600)
	# plt.show()
	plt.close()
	plt.clf()