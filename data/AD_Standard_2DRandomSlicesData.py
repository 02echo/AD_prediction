import cv2
import nibabel as nib
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import random
import pydicom
import imageio

from torchvision.transforms import Resize
from torchvision import transforms


# AX_INDEX = 78
# COR_INDEX = 79
SAG_INDEX = 79
# AX_SCETION = "[:, :, slice_i]"
# COR_SCETION = "[:, slice_i, :]"
SAG_SCETION = "[slice_i, :, :]"


class AD_Standard_2DRandomSlicesData(Dataset):
    """labeled Faces in the Wild dataset."""
    
    def __init__(self, root_dir, data_file, transform=None, slice = slice): # image_path, train_path
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
        torch_resize = Resize([256, 256])

        # root = 'D:\\graduate\\project\\AD_Prediction-master\\nii\\'
        # filename = os.path.join(root, "nii ({}).nii".format(img_name))
        # data = nib.load(filename).get_fdata()
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
        #
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
        #         samples = []
        #         sample = {'image': arr1, 'label': label}
        #         samples.append(sample)

        # # nii
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


        # dcm
        # file_path = os.path.join(self.root_dir, "1")
        # ds = pydicom.read_file(file_path + " ({}).dcm".format(int(img_name)+1))
        # img = ds.pixel_array  # 提取图像信息  256, 256
        # arr1 = torch.Tensor(img)
        # x = arr1.resize_(1, 256, 256)
        # samples = []
        # sample = {'image': x, 'label': label}
        # samples.append(sample)



        # image
        for j in range(49, 63):
            # image_path = os.path.join(self.root_dir, img_name1)
            image_path = self.root_dir + "\\img{}_{}.jpg".format(img_name2, j)

            # 读图片
            # arr1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # arr1 = torch.Tensor(arr1)

            arr1 = Image.open(image_path)
            test1 = transforms.Resize((256, 256))(arr1)
            arr1 = torch.from_numpy(np.array(test1)).permute(2, 0, 1).float() / 255.0

            # x = arr1.resize(1, 256, 256)

            sample = {'image': arr1, 'label': label}
            samples.append(sample)









        # AXimageList = axRandomSlice(image)
        # CORimageList = corRandomSlice(image)
        # SAGimageList = sagRandomSlice(image)
        # SAGimageList = np.reshape(image, (1,1, 1, 79, 95, 79))
        # print(np.array(SAGimageList).shape)  # 1 79 95 79
        # for img2DList in (SAGimageList, AXimageList, CORimageList):
        #     for image2D in img2DList:
        #         if self.transform:
        #             image2D = self.transform(image2D)
        #         sample = {'image': image2D, 'label': label}
        #         samples.append(sample)
        random.shuffle(samples)
        # print(len(samples))  9
        return samples






def getAllData(data_file):
    x = []
    y = []
    df = open(data_file)
    lines = df.readlines()

    # 随机选择 行数据
    selected_lines = random.sample(lines, 818)
    for line in selected_lines:
        lst = line.split()
        img_name2 = lst[1]
        img_label = lst[2]
        x.append(img_name2)
        y.append(img_label)

    # for i in range(818):  #TEST:163
    #     lst = lines[i].split()
    #     img_name2 = lst[1]
    #     img_label = lst[2]
    #     x.append(img_name2)
    #     y.append(img_label)

    return x, y


def getGeneData(data_file):
    x = []
    y = []
    df = open(data_file)
    lines = df.readlines()
    for i in range(256):  #with gene:256  #with clinical:818
        lst = lines[i].split()
        img_name2 = lst[1]
        img_label = lst[2]
        x.append(img_name2)
        y.append(img_label)

    return x, y

def get3Data(data_file):
    x = []
    y = []
    df = open(data_file)
    lines = df.readlines()

    # 随机选择220行数据
    selected_lines = random.sample(lines, 220)
    for line in selected_lines:
        lst = line.split()
        img_name2 = lst[1]
        img_label = lst[2]
        x.append(img_name2)
        y.append(img_label)

    return x, y
    # #原始
    # for i in range(256):  #with gene:256  #with clinical:818
    #     lst = lines[i].split()
    #     img_name2 = lst[1]
    #     img_label = lst[2]
    #     x.append(img_name2)
    #     y.append(img_label)
    # return x, y

def get2Data_clinical(data_file):
    x = []
    y = []
    df = open(data_file)
    lines = df.readlines()
    for i in range(818):  #with gene:256  #with clinical:818
        lst = lines[i].split()
        img_name2 = lst[1]
        img_label = lst[2]
        x.append(img_name2)
        y.append(img_label)

    return x, y

def get2Data_gene(data_file):
    x = []
    y = []
    df = open(data_file)
    lines = df.readlines()
    for i in range(256):  #with gene:256  #with clinical:818
        lst = lines[i].split()
        img_name2 = lst[1]
        img_label = lst[2]
        x.append(img_name2)
        y.append(img_label)

    return x, y

class AD_2DSlicesDataFusion(Dataset):
    """labeled Faces in the Wild dataset."""

    def __init__(self, index, image_path, txt_all):  # image_path, train_path
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_augmentation (boolean): Optional data augmentation.
        """
        self.index = index
        self.image_path = image_path
        self.txt_all = txt_all

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):

        df = open(self.txt_all)
        lines = df.readlines()
        idx = self.index[index]

        lst = lines[idx].split()

        img_name2 = lst[1]
        img_label = lst[2]
        name = lst[3]

        if img_label == 'CN':
            label = 0
        if img_label == 'AD':
            label = 1
        if img_label == 'MCI':
            label = 2
        samples = []

        # append clinical information
        data = [float(i) for i in lst[4:]]
        clinical_data = torch.Tensor(data)

        # image
        for j in range(49, 63):
            # image_path = os.path.join(self.root_dir, img_name1)
            image_path = self.image_path + "\\img{}_{}.jpg".format(img_name2, j)

            # 读图片
            # arr1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # arr1 = torch.Tensor(arr1)

            arr1 = Image.open(image_path).convert('L')

            test1 = transforms.Resize((256, 256))(arr1)
            #arr1 = torch.from_numpy(np.array(arr1)).permute(2, 0, 1).float() / 255.0
            arr1 = torch.from_numpy(np.array(test1)).float() / 255.0

            #x = torch.reshape(arr1,(1, 256, 256))
            x = arr1.reshape((1, 256, 256))
            sample = {'image': x, 'label': label, 'clinical': clinical_data}

            samples.append(sample)
        #random.shuffle(samples)

        return samples


class AD_clinical(Dataset):
    """only for clinical data"""

    def __init__(self, image_path, txt_all):  # image_path, train_path
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_augmentation (boolean): Optional data augmentation.
        """
        self.image_path = image_path
        self.txt_all = txt_all

    def __len__(self):
        return sum(1 for line in open(self.txt_all))

    def __getitem__(self, index):

        df = open(self.txt_all)
        lines = df.readlines()

        lst = lines[index].split()

        img_name2 = lst[1]
        img_label = lst[2]
        name = lst[3]

        if img_label == 'CN':
            label = 0
        if img_label == 'AD':
            label = 1
        if img_label == 'MCI':
            label = 2
        samples = []

        # append clinical information
        data = [float(i) for i in lst[4:]]
        clinical_data = torch.Tensor(data)


        sample = { 'label': label, 'clinical': clinical_data}

        samples.append(sample)
        #random.shuffle(samples)

        return samples


class AD_2DSlicesData(Dataset):
    """labeled Faces in the Wild dataset."""

    def __init__(self, index, image_path, txt_all):  # image_path, train_path, (x):
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_augmentation (boolean): Optional data augmentation.
        """
        self.index = index
        self.image_path = image_path
        self.txt_all = txt_all

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):

        df = open(self.txt_all)
        lines = df.readlines()
        idx = self.index[index]

        lst = lines[idx].split()

        img_name2 = lst[1]
        img_label = lst[2]
        name = lst[3]

        if img_label == 'CN':
            label = 0
        if img_label == 'AD':
            label = 1
        if img_label == 'MCI':
            label = 2
        samples = []

        # image
        for j in range(49, 63):
            # image_path = os.path.join(self.root_dir, img_name1)
            image_path = self.image_path + "\\img{}_{}.jpg".format(img_name2, j)

            # 读图片
            # arr1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # arr1 = torch.Tensor(arr1)

            arr1 = Image.open(image_path).convert('L')

            test1 = transforms.Resize((256, 256))(arr1)
            #arr1 = torch.from_numpy(np.array(arr1)).permute(2, 0, 1).float() / 255.0
            arr1 = torch.from_numpy(np.array(test1)).float() / 255.0

            #x = torch.reshape(arr1,(1, 256, 256))
            x = arr1.reshape((1, 256, 256))
            sample = {'image': x, 'label': label}

            samples.append(sample)
        #random.shuffle(samples)


        return samples


class AD_2DSlicesData3Fusion(Dataset):
    """labeled Faces in the Wild dataset."""

    def __init__(self, index, image_path, txt_all):  # image_path, train_path
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_augmentation (boolean): Optional data augmentation.
        """
        self.index = index
        self.image_path = image_path
        self.txt_all = txt_all

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):

        df = open(self.txt_all)
        lines = df.readlines()
        idx = self.index[index]

        lst = lines[idx].split()

        img_name2 = lst[1]
        img_label = lst[2]
        name = lst[3]

        if img_label == 'CN':
            label = 0
        if img_label == 'MCI':
            label = 1
        if img_label == 'AD':
            label = 2
        samples = []

        # append clinical information
        data_clinical = [float(i.strip('[],')[1:-1]) for i in lst[3:105]]
        clinical_data = torch.Tensor(data_clinical)

        # gene
        gene_data = [float(i.strip('[],')[1:-1]) for i in lst[106:]]
        gene_data = torch.Tensor(gene_data)
        #print(clinical_data.shape)  # 102
        #print(gene_data.shape)  # 16380

        # image
        for j in range(49, 63):
            # image_path = os.path.join(self.root_dir, img_name1)
            image_path = self.image_path + "\\img{}_{}.jpg".format(img_name2, j)

            # 读图片
            # arr1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # arr1 = torch.Tensor(arr1)

            arr1 = Image.open(image_path).convert('L')

            test1 = transforms.Resize((256, 256))(arr1)
            #arr1 = torch.from_numpy(np.array(arr1)).permute(2, 0, 1).float() / 255.0
            arr1 = torch.from_numpy(np.array(test1)).float() / 255.0

            #x = torch.reshape(arr1,(1, 256, 256))
            x = arr1.reshape((1, 256, 256))
            sample = {'image': x, 'label': label, 'clinical': clinical_data, 'gene':gene_data}

            samples.append(sample)
        #random.shuffle(samples)


        return samples


class AD_2DSlicesData3Fusionv2(Dataset):
    """基因数据单独加载"""

    def __init__(self, index, image_path, txt_all, gene_txt):  # image_path, train_path
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_augmentation (boolean): Optional data augmentation.
        """
        self.index = index
        self.image_path = image_path
        self.txt_all = txt_all
        self.gene_txt = gene_txt

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):

        df = open(self.txt_all)
        lines = df.readlines()
        idx = self.index[index]

        df_gene = open(self.gene_txt)
        lines_gene = df_gene.readlines()
        lst_gene = lines_gene[idx].split()

        lst = lines[idx].split()

        img_name2 = lst[1]
        img_label = lst[2]
        name = lst[3]

        if img_label == 'CN':
            label = 0
        if img_label == 'MCI':
            label = 1
        if img_label == 'AD':
            label = 2
        samples = []

        # append clinical information
        data_clinical = [float(i.strip('[],')[1:-1]) for i in lst[3:105]]  # 102
        clinical_data = torch.Tensor(data_clinical)

        # gene
        gene_data = [float(i.strip('[],')[1:-1]) for i in lst_gene]  # 106：
        gene_data = torch.Tensor(gene_data)
        #print(clinical_data.shape)  # 102
        #print(gene_data.shape)  # 16380

        # image
        for j in range(49, 63):
            # image_path = os.path.join(self.root_dir, img_name1)
            image_path = self.image_path + "\\img{}_{}.jpg".format(img_name2, j)

            # 读图片
            # arr1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # arr1 = torch.Tensor(arr1)

            arr1 = Image.open(image_path).convert('L')

            test1 = transforms.Resize((256, 256))(arr1)
            #arr1 = torch.from_numpy(np.array(arr1)).permute(2, 0, 1).float() / 255.0
            arr1 = torch.from_numpy(np.array(test1)).float() / 255.0

            #x = torch.reshape(arr1,(1, 256, 256))
            x = arr1.reshape((1, 256, 256))
            sample = {'image': x, 'label': label, 'clinical': clinical_data, 'gene':gene_data}

            samples.append(sample)
        #random.shuffle(samples)


        return samples


class AD_FusionGene(Dataset):
    """labeled Faces in the Wild dataset."""

    def __init__(self, index, image_path, txt_all):  # image_path, train_path
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_augmentation (boolean): Optional data augmentation.
        """
        self.index = index
        self.image_path = image_path
        self.txt_all = txt_all

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):

        df = open(self.txt_all)
        lines = df.readlines()
        idx = self.index[index]

        lst = lines[idx].split()

        img_name2 = lst[1]
        img_label = lst[2]
        name = lst[3]

        if img_label == 'CN':
            label = 0
        if img_label == 'MCI':
            label = 1
        if img_label == 'AD':
            label = 2
        samples = []

        # append clinical information
        # data_clinical = [float(i.strip('[],')[1:-1]) for i in lst[3:105]]  # 102
        # clinical_data = torch.Tensor(data_clinical)

        # gene
        gene_data = [float(i.strip('[],')[1:-1]) for i in lst[106:]]  #take 40 pices
        gene_data = torch.Tensor(gene_data)
        #print(clinical_data.shape)  # 102
        #print(gene_data.shape)  # [16380]

        # image
        # for j in range(49, 63):
        #     # image_path = os.path.join(self.root_dir, img_name1)
        #     image_path = self.image_path + "\\img{}_{}.jpg".format(img_name2, j)
        #
        #     # 读图片
        #     # arr1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #     # arr1 = torch.Tensor(arr1)
        #
        #     arr1 = Image.open(image_path).convert('L')
        #
        #     test1 = transforms.Resize((256, 256))(arr1)
        #     #arr1 = torch.from_numpy(np.array(arr1)).permute(2, 0, 1).float() / 255.0
        #     arr1 = torch.from_numpy(np.array(test1)).float() / 255.0
        #
        #     #x = torch.reshape(arr1,(1, 256, 256))
        #     x = arr1.reshape((1, 256, 256))
        #     sample = {'image': x, 'label': label, 'gene':gene_data}
        #
        #     samples.append(sample)
        sample = {'label':label, 'gene':gene_data}
        samples.append(sample)
        random.shuffle(samples)
        return samples

class AD_FusionGenev2(Dataset):
    """少加载一个基因"""

    def __init__(self, index, image_path, txt_all):  # image_path, train_path
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_augmentation (boolean): Optional data augmentation.
        """
        self.index = index
        self.image_path = image_path
        self.txt_all = txt_all

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):

        df = open(self.txt_all)
        lines = df.readlines()
        idx = self.index[index]

        lst = lines[idx].split()

        img_name2 = lst[1]
        img_label = lst[2]
        name = lst[3]

        if img_label == 'CN':
            label = 0
        if img_label == 'MCI':
            label = 1
        if img_label == 'AD':
            label = 2
        samples = []

        # append clinical information
        # data_clinical = [float(i.strip('[],')[1:-1]) for i in lst[3:105]]  # 102
        # clinical_data = torch.Tensor(data_clinical)

        # gene
        gene_data = [float(i.strip('[],')[1:-1]) for i in lst[106:16466]]  #take 40 pices
        gene_data = torch.Tensor(gene_data)

        sample = {'label':label, 'gene':gene_data}
        samples.append(sample)
        random.shuffle(samples)
        return samples



class AD_Gene(Dataset):
    """labeled Faces in the Wild dataset."""

    def __init__(self, image_path, txt_all):  # image_path, train_path
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_augmentation (boolean): Optional data augmentation.
        """
        self.image_path = image_path
        self.txt_all = txt_all

    def __len__(self):
        #return len(self.index)
        return sum(1 for line in open(self.txt_all))

    def __getitem__(self, index):

        df = open(self.txt_all)
        lines = df.readlines()

        lst = lines[index].split()

        img_name2 = lst[1]
        img_label = lst[2]
        name = lst[3]

        if img_label == 'CN':
            label = 0
        if img_label == 'MCI':
            label = 1
        if img_label == 'AD':
            label = 2
        samples = []

        # gene
        gene_data = [float(i.strip('[],')[1:-1]) for i in lst[106:]]  #take 40 pices  # [106:]
        #gene_data = [float(i) for  i in lst[1:]]
        gene_data = torch.Tensor(gene_data)

        sample = {'label':label, 'gene':gene_data}
        samples.append(sample)
        random.shuffle(samples)
        return samples












