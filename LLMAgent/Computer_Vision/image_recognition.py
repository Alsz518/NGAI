from LLMAgent.Computer_Vision.Canny import byjc
from LLMAgent.Computer_Vision.Resize import resize_image
from LLMAgent.Computer_Vision.CornerDetection import jdjc
from LLMAgent.Computer_Vision.Draw import draw
import os
import zipfile
import shutil


def recognition(file, maxCorners=1000, qualityLevel=0.01, minDistance=30, blockSize=30, useHarrisDetector=False, k=0.02):
    # 返回生成的gmns格式压缩文件路径，以及识别出的路网示意图

    """
    maxCorners = 100            要检测的最大角点数目
    qualityLevel = 0.2         表示角点的质量阈值。通常，这个值越高，检测到的角点越稀疏，质量越高。
    minDistance = 30            控制检测到的角点之间的最小距离
    blockSize = 30              每个像素周围的窗口大小，用于计算角点的局部特性
    useHarrisDetector = False   如果设置为True，将使用Harris角点检测算法，否则将使用Shi-Tomasi角点检测算法。
    k = 0.02                    Harris角点检测器中的k值。只有在useHarrisDetector设置为True时才有效。
    """

    # 先删除之前生成的结果（result及其子文件夹）
    result_folder = 'result'
    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)

    # 检查需要创建的文件夹（result及其子文件夹）
    dir_name = ['result', 'result/gmns', 'result/temp', 'result/temp/success']
    for dir in dir_name:
        if not os.path.exists(dir):
            os.makedirs(dir)

    temp_save_name = "./result/temp_result.jpg"  # 保存临时参考图
    img_save_name = "./result/jdjc_result.jpg"  # 保存角点检测图名字
    best_img_name = "./result/best_result.jpg"  # 保存结果图名字

    # 1、修改图片尺寸到 800*800（file是一个路径，img是OpenCV图像对象）
    img = resize_image(file)

    # 2、边缘检测得到二值图, temp_save_name是边缘检测结果，更适合作为误差判断参考
    img_byjc = byjc(img, temp_save_name)

    # 3、角点检测得到角点corners.csv文件, 返回优化好的数组
    line_width = 3  # 角点检测时点的优化距离(建议取二值图的线宽的一半，2~5?）
    best_xy_list = jdjc(img_byjc, img_save_name, line_width
                        , maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k)

    # 4、画图，并生成link.csv文件
    # min_mse 指允许画线的最小误差阈值，意在防止重复连线。越大，对短线更严格，但更不容易出现重复连接
    # target 指线允许连接的距离的平方。越大允许连接的线越远，但同时计算速度越慢。当出现有断线时，可尝试增大该值。
    draw(best_xy_list, temp_save_name, best_img_name, min_mse=40, target=62500)

    # 5、将生成的gmns文件夹压缩为zip文件
    zip_file_path = './result/gmns.zip'
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        # 添加 corners.csv
        zipf.write('./result/gmns/corners.csv', 'gmns/corners.csv')
        # 添加 link.csv
        zipf.write('./result/gmns/link.csv', 'gmns/link.csv')

    return zip_file_path, best_img_name


# 测试用：
# file = '../data/inverted_binary_pred_0.png'
# recognition(file)
