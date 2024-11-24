import argparse
import logging

import torch
import torch.nn as nn
from torch import cuda
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import KFold


from data.AD_Standard_2DRandomSlicesData import get3Data, AD_2DSlicesData3Fusion

from torch import optim

from model import GMU_clinical_gene

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

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
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpuid", default=0, nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")

# 改1.get2Data  2.'clinical' / 'gene'  3.改网络INPUT大小 和 num_class  4. 改数据文件地址  5.改加载数据(gene:AD_2DSlicesData3Fusion
# feel free to add more arguments as you need
def data_loader(data, options):
    # Only shuffle the data when doing training
    loader = DataLoader(data,
                        batch_size=options.batch_size,
                        shuffle=True,
                        num_workers=0,
                        drop_last=True
                        )

    return loader


# model = CAT_clinicalORgene.mobilevit_s_fusion()
model = GMU_clinical_gene.mobilevit_s_fusion()
def main(options):
    # Path configuration
    TXT_ALL_PATH = "./data/ref_all_gene.txt"  # y_rfe  ref_all_gene.txt
    IMAGE_PATH = "./data/wm_image"
    TXT_TEST_PATH = "./data/ref_test_gene.txt"
    x_test, y_test = get3Data(TXT_TEST_PATH)
    test_data = AD_2DSlicesData3Fusion(x_test, IMAGE_PATH, TXT_TEST_PATH)
    test_loader = data_loader(test_data, options)
    best_epoch = 0
    best_loss = 1000

    # Use argument load to distinguish training and testing

    use_cuda = 0
    if options.gpuid:
        cuda.set_device(options.gpuid[0])

    device = "cuda"
    criterion = torch.nn.CrossEntropyLoss()  # 相当于先对从model中得到的概率做softmax，再取对数，再与真实值相乘。  # NLLLoss()

    best_accuracy = float("-inf")

    x, y = get3Data(TXT_ALL_PATH)  # x:(x)  y:AD/MCI/CN
    kf = KFold(n_splits=5, shuffle=True)  # 初始化KFold
    i_kf = 0
    for train_index, test_index in kf.split(x):  # train_index是数组下标索引



        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)  # l2z正则化 权重0.001
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)  # ExponentialLR:next epoch LR is 0.9*LR

        train_data = AD_2DSlicesData3Fusion(train_index, IMAGE_PATH, TXT_ALL_PATH)
        valid_data = AD_2DSlicesData3Fusion(test_index, IMAGE_PATH, TXT_ALL_PATH)
        train_loader = data_loader(train_data, options)
        valid_loader = data_loader(valid_data, options)

        logging.info("K-Fold {} start:".format(i_kf))
        i_kf += 1
        # 将test集放epoch后
        for epoch_i in range(options.epochs):

            logging.info("At {0}-th epoch.".format(epoch_i))


            train_loss = 0.0
            train_correct_sum = 0
            train_simple_cnt = 0
            train_acc = 0
            model.train()

            for it, train_data in enumerate(train_loader):

                for data_dic in train_data:
                    gene, labels, clinical = data_dic['gene'].to(device), data_dic['label'].to(device), data_dic['clinical'].to(device)#, data_dic['clinical'].to(device)

                    optimizer.zero_grad()

                    train_output, train_feature = model(gene, clinical)

                    _, predict = torch.max(train_output.data, 1)

                    loss = criterion(train_output, labels)

                    # print(predict, labels)  # 16

                    if epoch_i == 0:
                        regularization_loss = 0
                        for param in model.parameters():
                            regularization_loss += torch.sum(abs(param))
                        loss = criterion(train_output, labels) + 1e-4 * regularization_loss  # L1正则化lamda = 0.001
                    #
                    # if i_epoch >= 0:
                    #     regularization_loss = 0
                    #     for param in model.parameters():
                    #         regularization_loss += torch.sum(abs(param))
                    #     loss = criterion(train_output, labels) + 1e-4 * regularization_loss  # L1正则化lamda = 0.001
                    # #
                    # if i_epoch >= 10:
                    #     regularization_loss = 0
                    #     for param in model.parameters():
                    #         regularization_loss += torch.sum(abs(param))
                    #     loss = criterion(train_output, labels) + 0.001 * regularization_loss  # L1正则化lamda = 0.001

                    train_loss += loss
                    # print(predict, labels.data)
                    correct_this_batch = (predict == labels.data).sum().float()
                    train_correct_sum += correct_this_batch
                    train_simple_cnt += labels.size(0)  # 第0维有几个数据

                    loss.backward()

                    # l2_regularization(model, 1e-2)
                    # l1_regularization(model, 1e-2)

                    optimizer.step()

                    # 分类器
                    train_feature = train_feature.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    # KNN train
                    knn_classifier = KNeighborsClassifier(n_neighbors=10)  # p:距离计算方式 , p=1
                    knn_classifier.fit(train_feature, labels)

                    # decisionTreeClsaaifier train
                    decision_tree_classifier = DecisionTreeClassifier()
                    decision_tree_classifier.fit(train_feature, labels)

                    # random_forest
                    random_forest_classifier = RandomForestClassifier(n_estimators=100)
                    random_forest_classifier.fit(train_feature, labels)

                    #支持向量机SVM
                    svm_classifier = SVC(kernel='poly', C=1, gamma=1,decision_function_shape='ovr')  # 三分类加ovr  , decision_function_shape='ovr'
                    svm_classifier.fit(train_feature, labels)

                    #best:
                    # if epoch_i == options.epochs - 1:
                    #     svm_classifier = GridSearchCV(SVC(decision_function_shape='ovr'), param_grid={"C": [0.1, 10, 1], "gamma": [1, 0.01, 1],
                    #                                                      "kernel": ['rbf', 'poly', 'linear',
                    #                                                                 'sigmoid']},
                    #                                   cv=4)
                    #     svm_classifier.fit(train_feature, labels)
                    #     print("The best parameters are %s with a score of %0.2f" % (
                    #         svm_classifier.best_params_, svm_classifier.best_score_))

            # print("lr = {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

            scheduler.step()
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct_sum / train_simple_cnt
            # train_loss_f.write("{0:.5f}\n".format(train_loss))

            logging.info(
                "Training loss is {0:.5f} Trian Accuracy:{1:.5f}..".format(
                    train_loss, train_acc))

            # if test_acc >= best_accuracy:
            #     if test_loss < best_loss:
            #         best_epoch = epoch_i
            #         best_loss = test_loss
            #         best_accuracy = test_acc
            #         torch.save(model.state_dict(), open(options.save, 'wb'))
        test_loss = 0.0
        test_correct_sum = 0
        test_simple_cnt = 0
        test_acc = 0
        count_test = 0

        # 分类器
        count = 0
        accuracy_knn = 0
        accuracy_d = 0
        accuracy_random = 0
        accuracy_svm = 0
        recall_k = 0
        recall_d = 0
        recall_r = 0
        recall_s = 0
        precision_k = 0
        precision_s = 0
        precision_r = 0
        precision_d = 0
        f1_k = 0
        f1_s = 0
        f1_r = 0
        f1_d = 0

        n = [[0] * 3 for i in range(3)]
        # test
        #model = load_checkpoint(model.state_dict(), model)

        # valid
        model.eval()
        with torch.no_grad():
            for it, test_data in enumerate(valid_loader):  #test_loader
                vote = []
                for data_dic in test_data:

                    gene, test_labels, clinical = data_dic['gene'].to(device), data_dic['label'].to(device), \
                         data_dic['clinical'].to(device)#data_dic['clinical'].to(device)
                    count_test += 1
                    test_output, test_feature = model(gene, clinical)#, gene)
                    _, test_predict = torch.max(test_output.data, 1)
                    correct_this_batch = (test_predict == test_labels.data).sum()
                    # print(test_predict, test_labels.data)
                    test_correct_sum += correct_this_batch
                    test_simple_cnt += test_labels.size(0)
                    loss = criterion(test_output, test_labels)
                    test_loss += loss

                    for i in range(options.batch_size):
                        if test_labels.data[i] == 0:
                            if test_predict[i] == 0:
                                n[0][0] += 1
                            elif test_predict[i] == 1:
                                n[0][1] += 1
                            else:
                                n[0][2] += 1
                        elif test_labels.data[i] == 1:
                            if test_predict[i] == 0:
                                n[1][0] += 1
                            elif test_predict[i] == 1:
                                n[1][1] += 1
                            else:
                                n[1][2] += 1
                        else:
                            if test_predict[i] == 0:
                                n[2][0] += 1
                            elif test_predict[i] == 1:
                                n[2][1] += 1
                            else:
                                n[2][2] += 1

                    # 分类器predict
                    test_feature = test_feature.cpu().detach().numpy()
                    test_labels = test_labels.cpu().detach().numpy()
                    # knn predict
                    y_pred_knn = knn_classifier.predict(test_feature)
                    acc_knn = accuracy_score(test_labels, y_pred_knn)
                    accuracy_knn += acc_knn
                    recall_knn = recall_score(test_labels, y_pred_knn, average='micro')
                    f1_knn = f1_score(test_labels, y_pred_knn, average='micro')
                    precision_knn = precision_score(test_labels, y_pred_knn, average='micro')

                    recall_k += recall_knn
                    f1_k += f1_knn
                    precision_k += precision_knn

                    count += 1

                    # decisionTree predict
                    y_pred_d = decision_tree_classifier.predict(test_feature)
                    acc_discion = accuracy_score(test_labels, y_pred_d)
                    accuracy_d += acc_discion
                    recall_decision = recall_score(test_labels, y_pred_d, average='micro')
                    precision_decision = precision_score(test_labels, y_pred_d, average='micro')
                    f1_decision = f1_score(test_labels, y_pred_d, average='micro')

                    recall_d += recall_decision
                    f1_d += f1_decision
                    precision_d += precision_decision


                    # randomforest predict
                    y_pred_r = random_forest_classifier.predict(test_feature)
                    acc_random = accuracy_score(test_labels, y_pred_r)
                    accuracy_random += acc_random
                    recall_random = recall_score(test_labels, y_pred_r, average='micro')
                    f1_random = f1_score(test_labels, y_pred_r, average='micro')
                    precision_random = precision_score(test_labels, y_pred_r, average='micro')

                    recall_r += recall_random
                    f1_r += f1_random
                    precision_r += precision_random

                    # svm
                    y_pred_svm = svm_classifier.predict(test_feature)
                    acc_svm = accuracy_score(test_labels, y_pred_svm)
                    accuracy_svm += acc_svm
                    recall_svm = recall_score(test_labels, y_pred_svm, average='micro')
                    f1_svm = f1_score(test_labels, y_pred_svm, average='micro')
                    precision_svm = precision_score(test_labels, y_pred_svm, average='micro')

                    recall_s += recall_svm
                    f1_s += f1_svm
                    precision_s += precision_svm





        # knn
        accuracy_knn = accuracy_knn / count
        precision_k = precision_k / count
        recall_k = recall_k / count
        f1_k = f1_k / count
        logging.info("knn准确率: {0:.3f}···precision：{1:.3f}···recall：{2:.3f}···f1：{3:.3f}".format(accuracy_knn, precision_k, recall_k, f1_k))
        # decisionTree
        accuracy_d = accuracy_d / count
        precision_d = precision_d / count
        recall_d = recall_d / count
        f1_d = f1_d / count
        logging.info("决策树准确率: {0:.3f}···precision：{1:.3f}···recall：{2:.3f}···f1：{3:.3f}".format(accuracy_d, precision_d, recall_d, f1_d))
        # randomForest
        accuracy_random = accuracy_random / count
        precision_r = precision_r / count
        recall_r = recall_r / count
        f1_r = f1_r / count
        logging.info("随机森林准确率: {0:.3f}···precision：{1:.3f}···recall：{2:.3f}···f1：{3:.3f}".format(accuracy_random, precision_r, recall_r, f1_r))
        # decisionTree
        accuracy_svm = accuracy_svm / count
        precision_s = precision_s / count
        recall_s = recall_s / count
        f1_s = f1_s / count
        logging.info("支持向量机准确率: {0:.3f}···precision：{1:.3f}···recall：{2:.3f}···f1：{3:.3f}".format(accuracy_svm, precision_s, recall_s, f1_s))


        test_acc = float(test_correct_sum) / test_simple_cnt
        test_loss = test_loss / len(valid_loader)

        precision = [0] * 3  # TP/TP+FP
        recall = [0] * 3  # TP/TP+FN
        F1 = [0] * 3  # 2*precision*recall/precision+recall
        if (n[0][0] + n[1][0] + n[2][0]) == 0:
            precision[0] = 0
        else:
            precision[0] = n[0][0] / (n[0][0] + n[1][0] + n[2][0])
        if (n[1][1] + n[0][1] + n[2][1]) == 0:
            precision[1] = 0
        else:
            precision[1] = n[1][1] / (n[1][1] + n[0][1] + n[2][1])
        if (n[2][2] + n[1][2] + n[0][2]) == 0:
            precision[2] = 0
        else:
            precision[2] = n[2][2] / (n[2][2] + n[1][2] + n[0][2])

        if (n[0][0] + n[0][1] + n[0][2]) == 0:
            recall[0] = 0
        else:
            recall[0] = n[0][0] / (n[0][0] + n[0][1] + n[0][2])
        if (n[1][1] + n[1][0] + n[1][2]) == 0:
            recall[1] = 0
        else:
            recall[1] = n[1][1] / (n[1][1] + n[1][0] + n[1][2])
        if (n[2][2] + n[2][0] + n[2][1]) == 0:
            recall[2] = 0
        else:
            recall[2] = n[2][2] / (n[2][2] + n[2][0] + n[2][1])

        if (precision[0] + recall[0]) == 0:
            F1[0] = 0
        else:
            F1[0] = 2 * precision[0] * recall[0] / (precision[0] + recall[0])
        if (precision[1] + recall[1]) == 0:
            F1[1] = 0
        else:
            F1[1] = 2 * precision[1] * recall[1] / (precision[1] + recall[1])
        if (precision[2] + recall[2]) == 0:
            F1[2] = 0
        else:
            F1[2] = 2 * precision[2] * recall[2] / (precision[2] + recall[2])

        logging.info(
            "Val Loss:{0:.5f}..Val Accuracy:{1:.3f}..".format(
                test_loss, test_acc))
        logging.info(
            "CN Precision:{0:.5f}..CN Recall:{1:.3f}..CN F1:{2:.3f}..MCI Precision:{3:.5f}..MCI Recall:{4:.3f}..MCI F1:{5:.3f}..AD Precision:{6:.5f}..AD Recall:{7:.3f}..AD F1:{8:.3f}..".format(
                precision[0], recall[0], F1[0], precision[1], recall[1], F1[1], precision[2], recall[2], F1[2]))

    logging.info("{}-th epoch, valid accuracy is : {}".format(best_epoch, best_accuracy))



def get_auc(y_labels, y_scores):
    auc = roc_auc_score(y_labels, y_scores)
    print('AUC calculated by sklearn tool is {}'.format(auc))
    return auc




def l2_regularization(model, l2_alpha):
    for module in model.modules():
        if type(module) is nn.Conv2d:
            module.weight.grad.data.add_(l2_alpha * module.weight.data)


def l1_regularization(model, l1_alpha):
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            module.weight.grad.data.add_(l1_alpha * torch.sign(module.weight.data))


def load_checkpoint(filepath, model, device='cuda'):
    checkpoint = torch.load(filepath, map_location=device)
    model = model

    model.load_state_dict(checkpoint)
    model.eval()

    return model


def test(model, test_loader, device):
    # validation -- this is a crude estimation because there might be some paddings at the end
    test_loss = 0.0
    correct_cnt = 0.0
    test_simple_cnt = 0
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for it, test_data in enumerate(test_loader):
            vote = []
            for data_dic in test_data:
                imgs, labels = data_dic['image'].to(device), data_dic['label'].to(device)

                test_output = model(imgs)
                _, test_predict = torch.max(test_output.data, 1)  # test_output.topk(1)
                correct_this_batch = (test_predict == labels.data).sum()

                correct_cnt += correct_this_batch
                test_simple_cnt += labels.size(0)

            acc = correct_cnt / test_simple_cnt

    return acc


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)
