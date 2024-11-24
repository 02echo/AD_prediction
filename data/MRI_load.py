import nibabel as nib
import matplotlib.pyplot as plt
import scipy, numpy, shutil, os, nibabel
import sys, getopt
import imageio
# import SimpleITK as sitk
import os
# import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import imageio
import torch
from PIL import Image

import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


def niito2D(filepath):
    inputfiles = os.listdir(filepath)  # 遍历文件夹数据
    outputfile = './data/'  # 输出文件夹
    print('Input file is ', inputfiles)
    print('Output folder is ', outputfile)

    for inputfile in inputfiles:
        image_array = nibabel.load(filepath + inputfile).get_data()  # 数据读取
        print(len(image_array.shape))

        # set destination folder
        if not os.path.exists(outputfile):
            os.makedirs(outputfile)  # 不存在输出文件夹则新建
            print("Created ouput directory: " + outputfile)
        print('Reading NIfTI file...')

        total_slices = 30  # 总切片数
        slice_counter = 70  # 从第几个切片开始

        # iterate through slices
        for current_slice in range(70, 70 + total_slices):
            # alternate slices
            if (slice_counter % 1) == 0:
                data = image_array[current_slice, :, :]  # 保存该切片，可以选择不同方向。

                # alternate slices and save as png
                if (slice_counter % 1) == 0:
                    print('Saving image...')
                    # 切片命名
                    image_name = inputfile[:-4] + "{:0>3}".format(str(current_slice + 1)) + ".png"
                    # 保存
                    imageio.imwrite(image_name, data)
                    print('Saved.')

                    # move images to folder
                    print('Moving image...')
                    src = image_name
                    shutil.move(src, outputfile)
                    slice_counter += 1
                    print('Moved.')

    print('Finished converting images')


def show_scan(root, output_folder, data_file):
    # 113, 137, 113
    # AC为 57， 81， 49

    for i in range(0, 818):
        df = open(data_file)
        lines = df.readlines()
        lst = lines[i].split()
        img_name1 = lst[0]
        img_name2 = lst[1]
        img_label = lst[2]

        filename = root + "{} {}.nii".format(img_name1, img_name2)
        data = nib.load(filename).get_fdata()
        # print(data.shape)  # 113, 137, 113

        # data1_1 = data[54, :, :][10:127, 10:103]  # 137, 113
        # data1_2 = data[57, :, :][10:127, 10:103]
        # data1_3 = data[60, :, :][10:127, 10:103]
        # plt.imshow(data1_1, cmap='gray')
        # plt.axis('off')
        # plt.savefig(output_folder + '/{}_1_1.jpg'.format(i), bbox_inches='tight', pad_inches=0)
        # # plt.show()
        # plt.imshow(data1_2, cmap='gray')
        # plt.axis('off')
        # plt.savefig(output_folder + '/{}_1_2.jpg'.format(i), bbox_inches='tight', pad_inches=0)
        #
        # plt.imshow(data1_3, cmap='gray')
        # plt.axis('off')
        # plt.savefig(output_folder + '/{}_1_3.jpg'.format(i), bbox_inches='tight', pad_inches=0)
        # # plt.show()
        #
        #
        #
        # # plt.subplot(1, 3, 2)
        # data2_1 = data[:, 80, :]  # 113, 113
        # data2_2 = data[:, 81, :]
        # data2_3 = data[:, 82, :]
        # data2_1 = data2_1[10:103, 10:103]
        # data2_2 = data2_2[10:103, 10:103]
        # data2_3 = data2_3[10:103, 10:103]
        # plt.imshow(data2_1, cmap='gray')
        # plt.axis('off')
        # plt.savefig(output_folder + '/{}_2_1.jpg'.format(i), bbox_inches='tight', pad_inches=0)
        # plt.imshow(data2_2, cmap='gray')
        # plt.axis('off')
        # plt.savefig(output_folder + '/{}_2_2.jpg'.format(i), bbox_inches='tight', pad_inches=0)
        # plt.imshow(data2_3, cmap='gray')
        # plt.axis('off')
        # plt.savefig(output_folder + '/{}_2_3.jpg'.format(i), bbox_inches='tight', pad_inches=0)

        # plt.show()

        # if img_label == 'MCI':
        #     for j in range(49, 63):  # (37, 76, 2) one person 26
        #         data_show = data[:, :, j][10:113, 10:137]
        #         plt.imshow(data_show, cmap='gray')
        #         plt.axis('off')
        #         plt.savefig(output_folder + '\\img{}_{}.jpg'.format(img_name2, j), bbox_inches='tight', pad_inches=0)
        #
        # else:
        #     for j in range(71, 86):  # (37, 76, 2) one person 26
        #         data_show = data[:, j, :][10:113, 10:113]
        #         plt.imshow(data_show, cmap='gray')
        #         plt.axis('off')
        #         plt.savefig(output_folder + '\\img{}_{}.jpg'.format(img_name2, j), bbox_inches='tight', pad_inches=0)

        for j in range(49, 63):
            data_show = data[:, :, j]
            plt.imshow(data_show, cmap='gray')
            plt.axis('off')
            plt.savefig(output_folder + '\\img{}_{}.jpg'.format(img_name2, j), bbox_inches='tight', pad_inches=0)




            # plt.show()



        # plt.show()
        plt.close()


def readData():
    image_dir = r"D:\\graduate\project\\AD_Prediction-master\\images\\1_17.jpg"

    imageGray2 = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    print(imageGray2.shape)

    image = Image.fromarray(imageGray2)  # 将之前的矩阵转换为图片
    image.show()  # 调用本地软件显示图片，win10是叫照片的工具


if __name__ == '__main__':
    KMP_DUPLICATE_LIB_OK = True
    path = 'D:\\graduate\\project\\DATA\\coregister-1000\\wm\\'
    output = 'D:\\graduate\\project\\DATA\\coregister-1000\\wm_image'
    data_file = "D:\\graduate\\project\\DATA\\coregister-1000\\y.txt"

    # readData()
    show_scan(path, output, data_file)
    # data = nib.load(path).get_fdata()
    # print(data[60].shape)
    # arr = data[29:109, 9:104, 29:109]
    # print(arr.shape)

    # readData()

    # print(arr)
    # for i in range(0,95):
    #     for j in range(0,80):
    #         print(arr[10][i][j])

    # 读取图片,灰度化，并转为数组
    # img = Image.fromarray(data[60, :, :])

    # # Image.ANTIALIAS等比例缩放
    # img.resize((137*4, 113*4), Image.ANTIALIAS).save(
    #     'D:\\graduate\\project\\Alzheimer-competition\\image\\temp.jpg')  # 放大四倍以便观察
    # plt.imshow(img, cmap='gray')
    # plt.show()  # 只有黑白

    # img.resize((480, 640)).save('D:\\graduate\\project\\Alzheimer-competition\\image\\temp.jpg')
