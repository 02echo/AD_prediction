import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torchsummary import summary

# Stem
class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.block1 = nn.Sequential(
            Conv1(1, 32, 3),
            nn.Conv2d(32, 32, 3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Conv2(32, 64, 3)
        )
        self.block2_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.block2_2 = nn.Sequential(
            Conv1(64, 96, 3)
        )
        self.block3_1 = nn.Sequential(
            Conv2(160, 64, 1),
            nn.Conv2d(64, 96, 3, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.block3_2 = nn.Sequential(
            Conv2(160, 64, 1),
            Conv3(64, 64, 7, 1),
            Conv3(64, 64, 1, 7),
            nn.Conv2d(64, 96, 3, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.block4_1 = Conv1(192, 192, 3)
        self.block4_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    def forward(self, x):
        x = self.block1(x)
        x1 = self.block2_1(x)
        x2 = self.block2_2(x)
        x = torch.cat([x1, x2], dim=1)
        x1 = self.block3_1(x)
        x2 = self.block3_2(x)
        x = torch.cat([x1, x2], dim=1)
        x1 = self.block4_1(x)
        x2 = self.block4_2(x)
        x = torch.cat([x1, x2], dim=1)
        return x

# 卷积核1：stride=2，padding=0
def Conv1(input_channel, output_channel, kernel_size):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size, stride=2, padding=0),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace = True)
    )
# 卷积核2：对称卷积核，stride=1, padding=kernel_size//2
def Conv2(input_channel, output_channel, kernel_size):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size, stride=1, padding=kernel_size//2),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True)
    )
# 卷积核3：非对称卷积核，kernel_size=(kernel_size1, kernel_size2),
#          stride=1，padding=(kernel_size1//2, kernel_size2//2)
def Conv3(input_channel, output_channel, kernel_size1, kernel_size2):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size=(kernel_size1, kernel_size2),
                  stride=1, padding=(kernel_size1//2, kernel_size2//2)),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace = True)
    )

# InceptionA
class InceptionA(nn.Module):
    def __init__(self, input_channel):
        super(InceptionA, self).__init__()
        self.block1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2(input_channel, 96, 1)
        )
        self.block2 = Conv2(input_channel, 96, 1)
        self.block3 = nn.Sequential(
            Conv2(input_channel, 64, 1),
            Conv2(64, 96, 3)
        )
        self.block4 = nn.Sequential(
            Conv2(input_channel, 64, 1),
            Conv2(64, 96, 3),
            Conv2(96, 96, 3)
        )
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x

# InceptionC
class InceptionC(nn.Module):
    def __init__(self, input_channel):
        super(InceptionC, self).__init__()
        self.block1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2(input_channel, 256, 1)
        )
        self.block2 = Conv2(input_channel, 256, 1)
        self.block3 = nn.Sequential(
            Conv2(input_channel, 384, 1),
            Conv3(384, 256, 1, 3)
        )
        self.block4 = nn.Sequential(
            Conv2(input_channel, 384, 1),
            Conv3(384, 256, 3, 1)
        )
        self.block5 = nn.Sequential(
            Conv2(input_channel, 384, 1),
            Conv3(384, 448, 1, 3),
            Conv3(448, 512, 3, 1),
            Conv3(512, 256, 3, 1)
        )
        self.block6 = nn.Sequential(
            Conv2(input_channel, 384, 1),
            Conv3(384, 448, 1, 3),
            Conv3(448, 512, 3, 1),
            Conv3(512, 256, 1, 3)
        )
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        x5 = self.block5(x)
        x6 = self.block6(x)
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        return x

# InceptionB
class InceptionB(nn.Module):
    def __init__(self, input_channel):
        super(InceptionB, self).__init__()
        self.block1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2(input_channel, 128, 1)
        )
        self.block2 = Conv2(input_channel, 384, 1)
        self.block3 = nn.Sequential(
            Conv2(input_channel, 192, 1),
            Conv3(192, 224, 1, 7),
            Conv3(224, 256, 1, 7)
        )
        self.block4 = nn.Sequential(
            Conv2(input_channel, 192, 1),
            Conv3(192, 192, 1, 7),
            Conv3(192, 224, 7, 1),
            Conv3(224, 224, 1, 7),
            Conv3(224, 256, 7, 1)
        )
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


# ReductionA:将35*35的图片降维至17*17
class ReductionA(nn.Module):
    def __init__(self, input_channel, k, l, m, n):
        super(ReductionA, self).__init__()
        self.block1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.block2 = Conv1(input_channel, n, 3)
        self.block3 = nn.Sequential(
            Conv2(input_channel, k, 1),
            Conv2(k, l, 3),
            Conv1(l, m, 3)
        )
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


# ReductionB:将17*17的图片降维至8*8
class ReductionB(nn.Module):
    def __init__(self, input_channel):
        super(ReductionB, self).__init__()
        self.block1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.block2 = nn.Sequential(
            Conv2(input_channel, 192, 1),
            Conv1(192, 192, 3)
        )
        self.block3 = nn.Sequential(
            Conv2(input_channel, 256, 1),
            Conv3(256, 256, 1, 7),
            Conv3(256, 320, 7, 1),
            Conv1(320, 320, 3)
        )
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


# InceptionV4 Network
class InceptionV4(nn.Module):
    def __init__(self):
        super(InceptionV4, self).__init__()
        self.block1 = Stem()
        self.block2 = nn.Sequential(
            InceptionA(384),
            InceptionA(384),
            InceptionA(384),
            InceptionA(384)
        )
        self.block3 = ReductionA(384, 192, 224, 256, 384)
        self.block4 = nn.Sequential(
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024)
        )
        self.block5 = ReductionB(1024)
        self.block6 = nn.Sequential(
            InceptionC(1536),
            InceptionC(1536),
            InceptionC(1536),
        )
        self.block7 = nn.Sequential(
            nn.AvgPool2d(kernel_size=6),
            nn.Dropout(p=0.8)
        )
        self.linear = nn.Linear(1536,3)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)


if __name__ == '__main__':
    img = torch.randn(1, 1, 256, 256)
    model = InceptionV4()
    summary(model, input_size=[(1, 256, 256)], batch_size=1, device="cpu")

    vit = InceptionV4()
    print("{}M".format(count_parameters(vit)))
    out = model(img)
    print(out.shape)
    #print(count_parameters(vit))