import argparse
import logging
import os.path

import torch
import torch.nn as nn
from torch import cuda
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import KFold


from data.AD_Standard_2DRandomSlicesData import getAllData, AD_2DSlicesData

from torch import optim


from model.inceptionV4 import InceptionV4

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Starter code for JHU CS661 Computer Vision HW3.")

parser.add_argument("--load",
                    help="Load saved network weights.")
parser.add_argument("--save", default="Inception4_wm1.pth",
                    help="Save network weights.")
parser.add_argument("--augmentation", default=True, type=bool,
                    help="Save network weights.")
parser.add_argument("--epochs", default=50, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=5e-5, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument("--batch_size", default=8, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpuid", default=0, nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")



def data_loader(data, options):
    # Only shuffle the data when doing training
    loader = DataLoader(data,
                        batch_size=options.batch_size,
                        shuffle=True,
                        num_workers=0,
                        drop_last=True
                        )

    return loader


# model = ViT(
# image_size=256,
# patch_size=32,
# num_classes=3,
# dim=1024,
# depth=6,
# heads=16,
# mlp_dim=2048,
# dropout=0.1,
# emb_dropout=0.1)
# model = mobilevit_inception.mobilevit_s()
model = InceptionV4()


# model = Vgg16_net()
# model = mobilevit_s()
# model = mobileVit_inception.mobilevit_s()
# model = MobileNetV3_Small()
# model = VGG16.Vgg16_net()
# model = resnet.ResNet(Bottleneck, [3, 4, 6, 3], 3)

def main(options):
    # Path configuration
    TXT_ALL_PATH = "./data/y.txt"
    IMAGE_PATH = "./data/wm_image"
    best_epoch = 0
    best_loss = 1000

    # Use argument load to distinguish training and testing

    use_cuda = 0
    if options.gpuid:
        cuda.set_device(options.gpuid[0])

    device = "cuda"
    criterion = torch.nn.CrossEntropyLoss()  # 相当于先对从model中得到的概率做softmax，再取对数，再与真实值相乘。  # NLLLoss()

    best_accuracy = float("-inf")

    x, y = getAllData(TXT_ALL_PATH)  # x:(x)  y:AD/MCI/CN

    kf = KFold(n_splits=5, shuffle=True)  # 初始化KFold
    kf_i = 0

    total_acc = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_F1 = 0.0

    for train_index, test_index in kf.split(x):  # train_index是数组下标索引



        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)  # l2z正则化 权重0.001
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)  # ExponentialLR:next epoch LR is 0.9*LR

        train_data = AD_2DSlicesData(train_index, IMAGE_PATH, TXT_ALL_PATH)
        valid_data = AD_2DSlicesData(test_index, IMAGE_PATH, TXT_ALL_PATH)

        train_loader = data_loader(train_data, options)
        valid_loader = data_loader(valid_data, options)

        logging.info("K-Fold {} start:".format(kf_i))
        kf_i += 1
        for epoch_i in range(options.epochs):

            logging.info("At {0}-th epoch.".format(epoch_i))

            # train_acc, train_loss, test_acc, test_loss, precision, recall, F1 = train(epoch_i, model, train_loader,
            #                                                                       valid_loader, use_cuda,
            #                                                                       criterion, optimizer, scheduler, options.batch_size,
            #                                                                       device)
            train_loss = 0.0
            train_correct_sum = 0
            train_simple_cnt = 0
            train_acc = 0
            model.train()

            for it, train_data in enumerate(train_loader):

                for data_dic in train_data:
                    imgs, labels = data_dic['image'].to(device), data_dic['label'].to(device) # , data_dic['clinical'].to(device)

                    optimizer.zero_grad()

                    train_output = model(imgs)

                    _, predict = torch.max(train_output.data, 1)  # train_output.topk(1)

                    loss = criterion(train_output, labels)

                    train_loss += loss
                    # print(predict, labels.data)
                    correct_this_batch = (predict == labels.data).sum().float()  # predict.squeeze(1)
                    train_correct_sum += correct_this_batch
                    train_simple_cnt += labels.size(0)  # 第0维有几个数据

                    loss.backward()

                    # l2_regularization(model, 1e-2)
                    # l1_regularization(model, 1e-2)

                    optimizer.step()

            # print("lr = {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

            scheduler.step()
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct_sum / train_simple_cnt
            # train_loss_f.write("{0:.5f}\n".format(train_loss))

            logging.info(
                "Training loss is {0:.5f} Trian Accuracy:{1:.5f}..".format(
                    train_loss, train_acc))
            # if train_loss < best_loss:
            #     best_loss = train_loss
            #     torch.save(model.state_dict(), open(options.save, 'wb'))

        test_loss = 0.0
        test_correct_sum = 0
        test_simple_cnt = 0
        test_acc = 0
        count_test = 0
        n = [[0] * 3 for i in range(3)]

        model.eval()
        with torch.no_grad():
            for it, test_data in enumerate(valid_loader):
                vote = []
                for data_dic in test_data:

                    imgs, test_labels = data_dic['image'].to(device), data_dic['label'].to(device)  # data_dic['clinical'].to(device)
                    count_test += 1
                    test_output = model(imgs)  # , gene)
                    _, test_predict = torch.max(test_output.data, 1)  # test_output.topk(1)
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

        test_acc = float(test_correct_sum) / test_simple_cnt
        test_loss = test_loss / len(valid_loader)

        if test_acc >= best_accuracy:
            if test_loss < best_loss:
                best_loss = test_loss
                best_accuracy = test_acc
                torch.save(model.state_dict(), open(options.save, 'wb'))

        precision = [0] * 3  # TP/TP+FP
        recall = [0] * 3  # TP/TP+FN
        F1 = [0] * 3  # 2*precision*recall/precision+recall
        # 计算每个类别的精度、召回率和F1
        for i in range(3):
            # 计算精度：TP / (TP + FP)
            precision[i] = n[i][i] / sum(n[j][i] for j in range(3)) if (
                    sum(n[j][i] for j in range(3)) > 0) else 0
            # 计算召回率：TP / (TP + FN)
            recall[i] = n[i][i] / sum(n[i][j] for j in range(3)) if (
                    sum(n[i][j] for j in range(3)) > 0) else 0
            # 计算F1：2 * precision * recall / (precision + recall)
            if (precision[i] + recall[i]) == 0:
                F1[i] = 0
            else:
                F1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

        # 计算每个类别的平均精度、召回率和F1
        avg_precision = sum(precision) / 3
        avg_recall = sum(recall) / 3
        avg_F1 = sum(F1) / 3

        # 打印结果
        total_acc += test_acc
        total_recall += avg_recall
        total_precision += avg_precision
        total_F1 += avg_F1

        logging.info("Val Loss: {0:.5f} .. Val Accuracy: {1:.3f}".format(test_loss, test_acc))
        logging.info("Precision per class: {0}".format(precision))
        logging.info("Recall per class: {0}".format(recall))
        logging.info("F1 per class: {0}".format(F1))
        logging.info("Average Precision: {0:.3f}".format(avg_precision))
        logging.info("Average Recall: {0:.3f}".format(avg_recall))
        logging.info("Average F1: {0:.3f}".format(avg_F1))

    logging.info("Average total_precision: {0:.3f}".format(total_precision / 5))
    logging.info("Average total_recall: {0:.3f}".format(total_recall / 5))
    logging.info("Average total_F1: {0:.3f}".format(total_F1 / 5))
    logging.info("Average total_acc: {0:.3f}".format(total_acc / 5))

    root_pth = "D:/graduate/project/AD_Prediction-master"
    file_path = os.path.join(root_pth, options.save)

    acc = test(load_checkpoint(file_path, model), test_loader, "cuda")
    logging.info("{}-th epoch, dev accuracy is : {}".format(best_epoch, acc))



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
    test_correct_sum = 0
    test_simple_cnt = 0
    test_acc = 0
    count_test = 0
    n = [[0] * 3 for i in range(3)]

    model.eval()
    with torch.no_grad():
        for it, test_data in enumerate(test_loader):
            vote = []
            for data_dic in test_data:

                imgs, test_labels = data_dic['image'].to(device), data_dic['label'].to(
                    device)  # data_dic['clinical'].to(device)
                count_test += 1
                test_output = model(imgs)  # , gene)
                _, test_predict = torch.max(test_output.data, 1)  # test_output.topk(1)
                correct_this_batch = (test_predict == test_labels.data).sum()
                # print(test_predict, test_labels.data)
                test_correct_sum += correct_this_batch
                test_simple_cnt += test_labels.size(0)


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

    test_acc = float(test_correct_sum) / test_simple_cnt


    precision = [0] * 3  # TP/TP+FP
    recall = [0] * 3  # TP/TP+FN
    F1 = [0] * 3  # 2*precision*recall/precision+recall
    # 计算每个类别的精度、召回率和F1
    for i in range(3):
        # 计算精度：TP / (TP + FP)
        precision[i] = n[i][i] / sum(n[j][i] for j in range(3)) if (
                sum(n[j][i] for j in range(3)) > 0) else 0
        # 计算召回率：TP / (TP + FN)
        recall[i] = n[i][i] / sum(n[i][j] for j in range(3)) if (
                sum(n[i][j] for j in range(3)) > 0) else 0
        # 计算F1：2 * precision * recall / (precision + recall)
        if (precision[i] + recall[i]) == 0:
            F1[i] = 0
        else:
            F1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    avg_precision = sum(precision) / 3
    avg_recall = sum(recall) / 3
    avg_F1 = sum(F1) / 3



    logging.info("Val Accuracy: {1:.3f}".format(test_acc))
    logging.info("Precision per class: {0}".format(precision))
    logging.info("Recall per class: {0}".format(recall))
    logging.info("F1 per class: {0}".format(F1))
    logging.info("Average Precision: {0:.3f}".format(avg_precision))
    logging.info("Average Recall: {0:.3f}".format(avg_recall))
    logging.info("Average F1: {0:.3f}".format(avg_F1))


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
    TXT_TEST_PATH = "./data/y_test.txt"
    IMAGE_PATH = "./data/wm_image"
    x, y = getAllData(TXT_TEST_PATH)
    test_data = AD_2DSlicesData(x, IMAGE_PATH, TXT_TEST_PATH)
    test_loader = data_loader(test_data, options)
    test(model, test_loader, 'cuda')

