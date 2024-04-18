#ifndef FAST_BILATERAL_UTILS_H
#define FAST_BILATERAL_UTILS_H


// C++
#include <assert.h>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <functional>
// OpenCV
#include <opencv2/opencv.hpp>


namespace {
    void run(const std::function<void()>& work=[]{}, const std::string message="") {
        auto start = std::chrono::steady_clock::now();
        work();
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << message << " " << duration.count() << " ms" <<  std::endl;
    }
    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    cv::Mat cv_concat(const std::vector<cv::Mat>& images){
        cv::Mat display;
        cv::hconcat(images, display);
        return display;
    }
    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    template<typename T>
    T min_in_array(const T* const data_ptr, const int length) {
        if(length < 1) return 0;
        T min_value = data_ptr[0];
        for(int i = 1;i < length; ++i)
            if(data_ptr[i] < min_value)
                min_value = data_ptr[i];
        return min_value;
    }
    template<typename T>
    T max_in_array(const T* const data_ptr, const int length) {
        if(length < 1) return 0;
        T max_value = data_ptr[0];
        for(int i = 1;i < length; ++i)
            if(data_ptr[i] > max_value)
                max_value = data_ptr[i];
        return max_value;
    }

    template<typename T>
    inline T clip(T x, T down, T up) {
        if(x < down) x = down;
        else if(x > up) x = up;
        return x;
    }
}

#endif //FAST_BILATERAL_UTILS_H
