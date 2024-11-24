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


from data.AD_Standard_2DRandomSlicesData import AD_2DSlicesDataFusion, getAllData, AD_2DSlicesData, AD_2DSlicesData3Fusion, AD_FusionGene, getGeneData, AD_Gene, Gene, AD_clinical

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score, classification_report

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
parser.add_argument("--epochs", default=10, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=5e-5, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument("--batch_size", default=818, type=int,
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
    TXT_ALL_PATH = "D:\\graduate\\project\\DATA\\coregister-1000\\y_clinical.txt"

    IMAGE_PATH = "D:\\graduate\\project\\DATA\\coregister-1000\\wm_image"
    best_epoch = 0
    best_loss = 1000

    # Use argument load to distinguish training and testing

    use_cuda = 0
    if options.gpuid:
        cuda.set_device(options.gpuid[0])

    device = "cuda"#torch.nn.CrossEntropyLoss()  # 相当于先对从model中得到的概率做softmax，再取对数，再与真实值相乘。  # NLLLoss()


    train_data = AD_clinical(IMAGE_PATH, TXT_ALL_PATH)  # 读取全部
    train_loader = data_loader(train_data, options)


    use_gene = []

    for it, train_data in enumerate(train_loader):
        for data_dic in train_data:
            gene, labels = data_dic['clinical'].to(device), data_dic['label'].to(device)  #[1, 16800]
            print(gene.shape)

            # decisionTreeClsaaifier train
            decision_tree_classifier = KNeighborsClassifier(n_neighbors=30)

            # 对每个基因进行逻辑回归分析
            gene_np = gene.cpu().detach().numpy()
            labels_np = labels.cpu().detach().numpy()

            gene_traindata = gene_np[:572, :]
            label_traindata = labels_np[:572]
            gene_testdata = gene_np[572:, :]
            label_testdata = labels_np[572:]
            decision_tree_classifier.fit(gene_traindata, label_traindata)

            y_pred_d = decision_tree_classifier.predict(gene_testdata)
            report = classification_report(label_testdata, y_pred_d)
            print(report)





















if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)
