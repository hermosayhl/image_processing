# Python
import os
import sys
import math
import random
# 3rd party
import cv2
import numpy

def cv_show(image, message='crane'):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def heatmap_show(image):
	heat_map = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	return cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)


# 定义一个数据读取器
class DataLoader():
	def __init__(self, haze_dir, depth_dir, shuffle=True):
		self.haze_dir = haze_dir
		self.depth_dir = depth_dir
		assert os.path.exists(haze_dir) and os.path.exists(depth_dir)
		self.files_list = os.listdir(haze_dir)
		self.shuffle = shuffle

	def __len__(self):
		return len(self.files_list)

	def __iter__(self):
		return self

	def __next__(self):
		while(True):
			if(self.shuffle): 
				random.shuffle(self.files_list)
			for file_name in self.files_list:
				hazy_image = cv2.imread(os.path.join(self.haze_dir, file_name))
				depth = numpy.load(os.path.join(self.depth_dir, file_name.replace(".png", ".npy")))
				if(hazy_image.shape[:2] != depth.shape[:2]):
					print("{} is invalid !".format(file_name))
					continue
				# 从 RGB 转成 
				hazy_image = hazy_image.astype("float32") / 255
				hsv_image = cv2.cvtColor(hazy_image, cv2.COLOR_BGR2HSV)
				return hsv_image[:, :, 1], hsv_image[:, :, 2], depth
				




# 定义随机种子
random.seed(212)
numpy.random.seed(212)

# 定义数据读取
train_loader = DataLoader(
	haze_dir="D:/data/datasets/RESIDE/OTS_beta/hazy", 
	depth_dir="D:/data/datasets/RESIDE/OTS_beta/depth/npy")

# 准备训练参数
total_iters = 500000
learning_rate = 1e-4

# 需要学习的参数
theta_0 = 0.0
theta_1 = 1.0
theta_2 = -1.0

# 遍历
mean_likehood = 0
likehood_count = 0
likehood_update = 1000

for cur_iter in range(1, total_iters + 1):
	# 先获取数据
	s, v, d = train_loader.__next__()
	# 计算梯度
	temp = d - theta_0 - theta_1 * v - theta_2 * s
	sigma = numpy.square(temp).mean()
	w_sum = temp.mean() / sigma
	v_sum = numpy.mean(v * temp) / sigma
	s_sum = numpy.mean(s * temp) / sigma
	# 更新参数
	theta_0 += learning_rate * w_sum
	theta_1 += learning_rate * v_sum	
	theta_2 += learning_rate * s_sum

	# theta_0 = 0.121779
	# theta_1 = 0.959710
	# theta_2 = -0.780245
	# sigma = 0.041337 ** 2
	# temp = d - theta_0 - theta_1 * v - theta_2 * s
	# 观察极大似然
	likehood = -numpy.square(temp).mean() / (2 * sigma) \
		- 0.5 * math.log(2 * 3.14 * sigma)
	mean_likehood += likehood.sum()
	likehood_count += 1

	sys.stdout.write("\r{}/{}==> [likehood  {}]".format(
		cur_iter, total_iters, mean_likehood / likehood_count))
	if(cur_iter % 10000 == 0):
		print("===>", theta_0, theta_1, theta_2, sigma, "\n")


	# 清空一次累计的最大似然
	if(likehood_count == likehood_update):
		mean_likehood = 0
		likehood_count = 0