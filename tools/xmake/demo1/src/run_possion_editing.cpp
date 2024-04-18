#include <filesystem>
#include "possion_editing.h"


int main() {

    // 查看当前目录
    std::cout << "working dir  :  " << std::filesystem::current_path().string() << std::endl;

    // 有缝合成
    seam_clone_demo();

    // 无缝隙合成图像
    seamless_cloning_demo();

    // 各种利用梯度的应用
    possion_edit_demo_1();

    // 纹理抹平
    possion_texture_flatten_demo();

    // 内容翻转
    possion_content_flip_demo();

    // 纹理去除
    possion_texture_transform_demo();

    // 局部亮度变换
    possion_illumination_change_demo();

    return 0;
}




