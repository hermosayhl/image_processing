// C++
#include <iostream>
// self
#include "histogram_equalization.h"



cv::Mat plot_histogram_gray(const cv::Mat& source) {
    const int channels[] = { 0 };
	cv::Mat hist;//定义输出Mat类型
	int dims = 1;//设置直方图维度
	const int histSize[] = { 256 }; //直方图每一个维度划分的柱条的数目
	float pranges[] = { 0, 255 };//取值区间
	const float* ranges[] = { pranges };
	cv::calcHist(&source, 1, channels, cv::Mat(), hist, dims, histSize, ranges, true, false);

	int hist_height = 256;
	int scale = int(source.cols / hist_height);
	cv::Mat hist_img = cv::Mat::zeros(hist_height, source.cols, CV_8UC3); //创建一个黑底的8位的3通道图像，高256，宽256*2
	double max_val;
	minMaxLoc(hist, 0, &max_val, 0, 0); //计算直方图的最大像素值
	//遍历直方图得到的数据
	for (int i = 0; i < 256; i++) {
		float bin_val = hist.at<float>(i);   //遍历hist元素（注意hist中是float类型）
		int intensity = cvRound(bin_val * hist_height / max_val);  //绘制高度
		rectangle(hist_img, cv::Point(i * scale, hist_height - 1), cv::Point((i + 1) * scale - 1, hist_height - intensity), cv::Scalar(255, 255, 255));//绘制直方图
	}
	cv::cvtColor(hist_img, hist_img, cv::COLOR_BGR2GRAY);
	return hist_img;
}

//https://blog.csdn.net/didi_ya/article/details/113556309
cv::Mat plot_histogram(const cv::Mat& source) {
    const int C = source.channels();
    // 灰度图
    if(C == 1) return plot_histogram_gray(source);
    // 三通道一起画
    std::vector<cv::Mat> bgr_planes;
    cv::split(source, bgr_planes);
    int histSize = 256;
    float range[] = { 0, 255 };
    const float* Ranges = { range };
    cv::Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &Ranges, true, false);
    calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &Ranges, true, false);
    calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &Ranges, true, false);
    //归一化
	int hist_w = source.cols;//直方图的图像的宽
	int hist_h = 256; //直方图的图像的高
	int nHistSize = 256;
	int bin_w = cvRound((double)hist_w / nHistSize);	//区间
	cv::Mat hist_image(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));//绘制直方图显示的图像
	normalize(b_hist, b_hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());//归一化
	normalize(g_hist, g_hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(r_hist, r_hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 1; i < nHistSize; i++) {
		//绘制蓝色分量直方图
		line(hist_image, cv::Point((i - 1)*bin_w, hist_h - cvRound(b_hist.at<float>(i - 1))),
			cv::Point((i)*bin_w, hist_h - cvRound(b_hist.at<float>(i))), cv::Scalar(255, 0, 0),1);
		//绘制绿色分量直方图
		line(hist_image, cv::Point((i - 1)*bin_w, hist_h - cvRound(g_hist.at<float>(i - 1))),
			cv::Point((i)*bin_w, hist_h - cvRound(g_hist.at<float>(i))), cv::Scalar(0, 255, 0),1);
		//绘制红色分量直方图
		line(hist_image, cv::Point((i - 1)*bin_w, hist_h - cvRound(r_hist.at<float>(i - 1))),
			cv::Point((i)*bin_w, hist_h - cvRound(r_hist.at<float>(i))), cv::Scalar(0, 0, 255),1);
	}
	return hist_image;
}



// 直方图均衡化
cv::Mat histogram_equalization_gray(const cv::Mat& source) {
    // 首先统计每个像素值的出现次数
    std::vector<double> occurence(256, 0.0);
    uchar* const data_ptr = source.data;
    const int length = source.rows * source.cols;
    // 第一次计数
    for(int i = 0;i < length; ++i) ++occurence[data_ptr[i]];
    // 计算每个像素出现的相对概率
    for(int i = 0;i < 256; ++i) occurence[i] /= length;
    // 计算累积概率
    for(int i = 1;i < 256; ++i) occurence[i] += occurence[i - 1];
    // 计算每个点对应的值
    std::vector<uchar> lut(256, 0);
    for(int i = 0;i < 256; ++i) lut[i] = cv::saturate_cast<uchar>(occurence[i] * 255);
    // 映射
    auto result = source.clone();
    uchar* const res_ptr = result.data;
    for(int i = 0;i < length; ++i) res_ptr[i] = lut[data_ptr[i]];
    return result;
}

// 直方图均衡化
cv::Mat histogram_equalization(const cv::Mat& source) {
    const int C = source.channels();
    if(C == 1)
        return histogram_equalization_gray(source);
    std::vector<cv::Mat> bgr, des;
    cv::split(source, bgr);
    for(const auto& channel : bgr)
        des.emplace_back(histogram_equalization_gray(channel));
    cv::Mat result;
    cv::merge(des, result);
    return result;
}