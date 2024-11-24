import cv2
import nibabel as nib
import os

import pydicom
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
# from skimage.transform import resize
from PIL import Image
import random

from torchvision.transforms import Resize
from torchvision import transforms


AX_SCETION = "[:, :, slice_i]"
COR_SCETION = "[:, slice_i, :]"
SAG_SCETION = "[slice_i, :, :]"
AX_INDEX = 78
COR_INDEX = 79
SAG_INDEX = 57

class AD_Standard_2DSlicesData(Dataset):
    """labeled Faces in the Wild dataset."""
    
    def __init__(self, root_dir, data_file, transform=None, slice = slice):
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_augmentation (boolean): Optional data augmentation.
        """
        self.root_dir = root_dir
        self.data_file = data_file
        self.transform = transform
    
    def __len__(self):
        return sum(1 for line in open(self.data_file))
    
    def __getitem__(self, idx):

        df = open(self.data_file)
        lines = df.readlines()
        lst = lines[idx].split()
        img_name1 = lst[0]
        img_name2 = lst[1]
        img_label = lst[2]
        samples = []
        if img_label == 'CN':
            label = 0
        if img_label == 'AD':
            label = 1
        if img_label == 'MCI':
            label = 2

        #     for j in range(49, 63):
        #         # image_path = os.path.join(self.root_dir, img_name1)
        #         image_path = self.root_dir + "\\img{}_{}.jpg".format(img_name2, j)
        #
        #         # 读图片
        #         # arr1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #         # arr1 = torch.Tensor(arr1)
        #
        #         arr1 = Image.open(image_path)
        #         test1 = transforms.Resize((256, 256))(arr1)
        #         arr1 = torch.from_numpy(np.array(test1)).permute(2, 0, 1).float() / 255.0
        #
        #         # x = arr1.resize(1, 256, 256)
        #         samples = []
        #         sample = {'image': arr1, 'label': label}
        #         samples.append(sample)
        #
        # else:
        #     for j in range(71, 86):
        #         # image_path = os.path.join(self.root_dir, img_name1)
        #         image_path = self.root_dir + "\\img{}_{}.jpg".format(img_name2, j)
        #
        #         # 读图片
        #         # arr1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #         # arr1 = torch.Tensor(arr1)
        #
        #         arr1 = Image.open(image_path)
        #         test1 = transforms.Resize((256, 256))(arr1)
        #         arr1 = torch.from_numpy(np.array(test1)).permute(2, 0, 1).float() / 255.0
        #
        #         # x = arr1.resize(1, 256, 256)
        #
        #         sample = {'image': arr1, 'label': label}
        #         samples.append(sample)

        # image
        for j in range(49, 63):
            # image_path = os.path.join(self.root_dir, img_name1)
            image_path = self.root_dir + "\\img{}_{}.jpg".format(img_name2, j)

            # 读图片
            # arr1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #
            # arr1 = torch.Tensor(arr1)

            arr1 = Image.open(image_path) #.convert('L')
            test1 = transforms.Resize((256, 256))(arr1)
            arr1 = torch.from_numpy(np.array(test1)).permute(2, 0, 1).float()/255.0
            # arr1 = torch.from_numpy(np.array(test1)).float() / 255.0
            x = arr1
            print(x.shape)
            # x = arr1.reshape(1, 256, 256)

            sample = {'image': x, 'label': label}
            samples.append(sample)
        #


        # dcm
        # file_path = os.path.join(self.root_dir, "1")
        # ds = pydicom.read_file(file_path + " ({}).dcm".format(int(img_name)+1))
        # img = ds.pixel_array  # 提取图像信息  256, 256
        # arr1 = torch.Tensor(img)
        # x = arr1.resize_(1, 256, 256)
        # samples = []
        # sample = {'image': x, 'label': label}
        # samples.append(sample)

        # nii
        # filename = os.path.join(self.root_dir, "{} {}.nii".format(img_name1, img_name2))
        # data = nib.load(filename).get_fdata()
        # for j in range(20, 91):
        #     data_small = data[:, :, j][10:103, 10:127]
        #     arr1 = np.array(data_small)
        #     arr1 = torch.Tensor(arr1)  # 80, 93, 117
        #     x = arr1.resize_([1, 256, 256])
        #
        #     samples = []
        #     sample = {'image': x, 'label': label}
        #     samples.append(sample)




        # AXimageList = axRandomSlice(image)
        # CORimageList = corRandomSlice(image)
        # SAGimageList = sagRandomSlice(image)
        # SAGimageList = np.reshape(image, (1,1, 1, 79, 95, 79))
        # print(np.array(SAGimageList).shape)  # 1 79 95 79
        # for img2DList in (SAGimageList):
        #     for image2D in img2DList:
        #         if self.transform:
        #             image2D = self.transform(image2D)
        #         sample = {'image': image2D, 'label': label}
        #         samples.append(sample)
        random.shuffle(samples)
        return samples


def getSlice(image_array, keyIndex, section, step = 1):
    slice_p = keyIndex
    slice_2Dimgs = []
    slice_select_0 = None
    slice_select_1 = None
    slice_select_2 = None
    # i = 0
    # for slice_i in range(slice_p-step, slice_p+step+1, step):
    #     slice_select = eval("image_array"+section)
    #     exec("slice_select_"+str(i)+"=slice_select")
    #     i += 1
    slice_i = slice_p - step
    slice_select = eval("image_array" + section)
    slice_select_0 = slice_select
    slice_i = slice_i + 1
    slice_select = eval("image_array" + section)
    slice_select_1 = slice_select
    slice_i = slice_i + 1
    slice_select = eval("image_array" + section)
    slice_select_2 = slice_select
    slice_2Dimg = np.stack((slice_select_0, slice_select_1, slice_select_2), axis = 2)
    slice_2Dimgs.append(slice_2Dimg)
    return slice_2Dimgs


def axKeySlice(image):
    image_array = np.array(image.get_data())
    return getSlice(image_array, AX_INDEX, AX_SCETION)


def corKeySlice(image):
    image_array = np.array(image.get_data())
    return getSlice(image_array, COR_INDEX, COR_SCETION)


def sagKeySlice(image):
    image_array = np.array(image.get_data())
    return getSlice(image_array, SAG_INDEX, SAG_SCETION)

