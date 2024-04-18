#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

// a = crane.add_arrays_1d(np.array([1, 3, 5, 7, 9]), np.array([2, 4, 6, 8, 10]))
py::array_t<float> add_arrays_1d(py::array_t<float>& input1, py::array_t<float>& input2) {
    // 获取input1, input2的信息
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();
    if (buf1.ndim !=1 || buf2.ndim !=1) {
        throw std::runtime_error("Number of dimensions must be one");
    }
    if (buf1.size !=buf2.size) {
        throw std::runtime_error("Input shape must match");
    }
    //申请空间
    auto result = py::array_t<float>(buf1.size);
    py::buffer_info buf3 = result.request();
    //获取numpy.ndarray 数据指针
    float* ptr1 = (float*)buf1.ptr;
    float* ptr2 = (float*)buf2.ptr;
    float* ptr3 = (float*)buf3.ptr;
    //指针访问numpy.ndarray
    for (int i = 0; i < buf1.shape[0]; i++) {
        ptr3[i] = ptr1[i] + ptr2[i];
    }
    return result;
}


py::array_t<float> compute_dark(py::array_t<float>& padded, const int radius) {
    // 获取信息
    py::buffer_info padded_data = padded.request();
    const int H2 = padded_data.shape[0];
    const int W2 = padded_data.shape[1];
    float* pad_ptr = (float*)padded_data.ptr;
    // 算原来的高和宽
    const int H = H2 - 2 * radius;
    const int W = W2 - 2 * radius;
    // 准备一个结果
    auto result = py::array_t<float>(H * W);
    result.resize({H, W});
    py::buffer_info buf_result = result.request();
    float* res_ptr = (float*)buf_result.ptr;
    // 准备一个临时结果
    auto temp = py::array_t<float>(H2 * W);
    temp.resize({H2, W});
    py::buffer_info buf_temp = temp.request();
    float* temp_ptr = (float*)buf_temp.ptr;
    // 开始最小值滤波
    int cnt = 0;
    for(int i = 0;i < H2; ++i) {
        float* row_ptr = pad_ptr + i * W2 + radius;
        for(int j = 0;j < W; ++j) {
            float min_value = 1e7;
            for(int k = -radius; k <= radius; ++k)
                min_value = std::min(min_value, row_ptr[j + k]);
            temp_ptr[cnt++] = min_value;
        }
    }
    for(int j = 0;j < W; ++j) {
        for(int i = 0;i < H; ++i) {
            float min_value = 1e7;
            const int offset = (radius + i) * W + j;
            for(int k = -radius; k <= radius; ++k)
                min_value = std::min(min_value, temp_ptr[offset + k * W]);
            res_ptr[i * W + j] = min_value; 
        }
    }
    return result;
}


PYBIND11_MODULE(crane, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: crane

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add_arrays_1d", &add_arrays_1d);

    m.def("compute_dark", &compute_dark);

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
