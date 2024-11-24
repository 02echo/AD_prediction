import argparse
import logging
import os.path

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from PIL import Image

import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import KFold


from data.AD_Standard_2DRandomSlicesData import AD_2DSlicesDataFusion, getAllData, AD_2DSlicesData, AD_2DSlicesData3Fusion, AD_FusionGene, getGeneData, AD_Gene, Gene
from data.AD_Standard_2DTestingSlices import AD_Standard_2DTestingSlices

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score, classification_report, confusion_matrix

import csv
from sklearn.feature_selection import VarianceThreshold

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Starter code for JHU CS661 Computer Vision HW3.")

parser.add_argument("--load",
                    help="Load saved network weights.")
parser.add_argument("--save", default="resnet18.pth",
                    help="Save network weights.")
parser.add_argument("--augmentation", default=True, type=bool,
                    help="Save network weights.")
parser.add_argument("--epochs", default=30, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=5e-5, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument("--batch_size", default=256, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpuid", default=0, nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")

# 这是尝试使用Logit Regression逻辑回归，根据P值进行筛选的代码

def data_loader(data, options):
    # Only shuffle the data when doing training
    loader = DataLoader(data,
                        batch_size=options.batch_size,
                        shuffle=True,
                        num_workers=0,
                        drop_last=False
                        )

    return loader





def main(options):
    # Path configuration
    global selector, decision_tree_classifier, label_traindata, label_testdata
    TXT_ALL_PATH = "./data/ref_all_gene.txt"

    IMAGE_PATH = "./data/wm_image"
    best_epoch = 0
    best_loss = 1000

    # Use argument load to distinguish training and testing

    use_cuda = 0
    if options.gpuid:
        cuda.set_device(options.gpuid[0])

    device = "cuda"
    #torch.nn.CrossEntropyLoss()  # 相当于先对从model中得到的概率做softmax，再取对数，再与真实值相乘。  # NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = float("-inf")

    x, y = getGeneData(TXT_ALL_PATH)  # x:(x)  y:AD/MCI/CN

    kf = KFold(n_splits=10, shuffle=True)  # 初始化KFold
    kf_i = 0


    train_data = AD_Gene(IMAGE_PATH, TXT_ALL_PATH)  # 读取全部
    valid_data = AD_Gene(IMAGE_PATH, TXT_ALL_PATH)
    train_loader = data_loader(train_data, options)
    valid_loader = data_loader(valid_data, options)


    use_gene = []

    for it, train_data in enumerate(train_loader):
        for data_dic in train_data:
            gene, labels = data_dic['gene'].to(device), data_dic['label'].to(device)

            # decisionTreeClsaaifier train
            decision_tree_classifier = DecisionTreeClassifier()

            # 对每个基因进行逻辑回归分析
            gene_np = gene.cpu().detach().numpy()
            labels_np = labels.cpu().detach().numpy()

            #
            gene_traindata = gene_np[:180, :]
            label_traindata = labels_np[:180]
            gene_testdata = gene_np[180:, :]
            label_testdata = labels_np[180:]
            decision_tree_classifier = DecisionTreeClassifier()
            decision_tree_classifier.fit(gene_traindata, label_traindata)
            # decisionTree predict
            y_pred_d = decision_tree_classifier.predict(gene_testdata)
            report = classification_report(label_testdata, y_pred_d)
            cm = confusion_matrix(label_testdata, y_pred_d)
            # 计算每个类别的准确率
            for class_label in set(label_testdata):
                class_mask = (label_testdata == class_label)
                class_accuracy = accuracy_score(label_testdata[class_mask], y_pred_d[class_mask])
                print(f"类别 {class_label} 的准确率: {class_accuracy}")
            print(cm)
            print(report)



            # for i in tqdm(range(819)):
            #     num = 20
            #     accuracy_d = 0
            #     recall_d = 0
            #     f1_d = 0
            #     precision_d = 0
            #
            #     gene_traindata = gene_np[:180, i*num:i*num+num]
            #     label_traindata = labels_np[:180]
            #     gene_testdata = gene_np[180:, i*num:i*num+num]
            #     label_testdata = labels_np[180:]
            #
            #     decision_tree_classifier.fit(gene_traindata, label_traindata)
            #
            #     # decisionTree predict
            #     y_pred_d = decision_tree_classifier.predict(gene_testdata)
            #     acc_discion = accuracy_score(label_testdata, y_pred_d)
            #     accuracy_d += acc_discion
            #     recall_decision = recall_score(label_testdata, y_pred_d)
            #     recall_d += recall_decision
            #     f1_decision = f1_score(label_testdata, y_pred_d)
            #     f1_d += f1_decision
            #     precision_decision = precision_score(label_testdata, y_pred_d)
            #     precision_d += precision_decision
            #

            # accuracy_d = 0
            # recall_d = 0
            # f1_d = 0
            # precision_d = 0
            # use_gene = np.array(use_gene)
            # # print(use_gene.shape)  （num， 256， 1）
            # use_gene = use_gene.reshape(-1, 256)
            # use_gene = use_gene.T
            # print(use_gene.shape)
            #
            # # 使用savetxt()函数将数组保存为文本文件
            # np.savetxt(file_name, use_gene, fmt='%f', delimiter='\t')
            #
            # gene_traindata = use_gene[:180, :]
            # gene_testdata = use_gene[180:, :]
            #
            # decision_tree_classifier = DecisionTreeClassifier(max_depth=500)
            # decision_tree_classifier.fit(gene_traindata, label_traindata)
            #
            # # decisionTree predict
            # y_pred_d = decision_tree_classifier.predict(gene_testdata)
            #
            # report = classification_report(label_testdata, y_pred_d)
            #
            # cm = confusion_matrix(label_testdata, y_pred_d)
            # # 计算每个类别的准确率
            # for class_label in set(label_testdata):
            #     class_mask = (label_testdata == class_label)
            #     class_accuracy = accuracy_score(label_testdata[class_mask], y_pred_d[class_mask])
            #     print(f"类别 {class_label} 的准确率: {class_accuracy}")
            # print(cm)
            # print(report)












if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)
