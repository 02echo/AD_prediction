import torch
import torch.nn as nn

from einops import rearrange
from torchsummary import summary
import torch.nn.functional as F


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# MV3 BLOCK
class linear_hardswish(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x * F.relu6(x + 3) / 6


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.add_(3.).clamp_(0., 6.).div_(6.)
        else:
            return F.relu6(x + 3.) / 6.



class SELayer(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            HardSigmoid()  # 使用Sigmoid函数将输出限制在0到1之间
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 对输入进行全局平均池化
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层计算通道权重
        return x * y.expand_as(x)  # 将通道权重应用到输入特征图上


class MV3Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(), #nn.SiLU(), # linear_hardswish(),#
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),

                #SE module
                SELayer(hidden_dim),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, size_in1, size_in2, size_in3, expansion=4, kernel_size=3, patch_size=(2, 2), size_out=128):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(1, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV3Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV3Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV3Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV3Block(channels[2], channels[3], 1, expansion))  # Repeat
        self.mv2.append(MV3Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV3Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV3Block(channels[7], channels[8], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

        self.descend = nn.Sequential()
        self.descend.add_module("fc1", nn.Linear(640 * 64, 64 * 16))

        self.descend_gene = nn.Sequential()
        self.descend_gene.add_module("fc1", nn.Linear(16380, 64 * 16))
        #self.classifer.add_module("fc2", nn.Linear(64, num_classes))

        # super(GatedMultimodal, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        self.hidden_sigmoid = nn.Linear(size_out*3, 1, bias=False)  # size_out * 3

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

        # add classifer
        self.classify2 = nn.Sequential()
        self.classify2.add_module("bn", nn.BatchNorm1d(size_out))
        self.classify2.add_module("dp", nn.Dropout(0.6))
        self.classify2.add_module("fc1", nn.Linear(size_out, 32))
        self.classify2.add_module("relu", nn.ReLU())
        self.classify2.add_module("fc2", nn.Linear(32, num_classes))

    def forward(self, x1, x2, x3):
        x = self.conv1(x1)  # 16, 128, 128
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)  #64, 64, 64 # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)
        x1 = x.view(x.shape[0], -1)
        x1 = self.descend(x1)

        # x3 = self.descend_gene(x3)

        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden2(x2))
        h3 = self.tanh_f(self.hidden3(x3))

        x1 = torch.cat((h1, h2, h3), dim=1)
        x2 = torch.cat((h2, h1, h3), dim=1)
        x3 = torch.cat((h3, h1, h2), dim=1)
        z1 = self.sigmoid_f(self.hidden_sigmoid(x1))
        z2 = self.sigmoid_f(self.hidden_sigmoid(x2))
        z3 = self.sigmoid_f(self.hidden_sigmoid(x3))

        #print(z1, z2, z3)

        z = z1.view(z1.size()[0], 1) * h1 + z2.view(z2.size()[0], 1) * h2 + z3.view(z3.size()[0], 1) * h3
        y = self.classify2(z)

        return y


def changed_mobilevit_xxs(img_size=(256, 256), num_classes=3):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((img_size[0], img_size[1]), dims, channels, num_classes=num_classes, expansion=2)


def changed_mobilevit_xs(img_size=(256, 256), num_classes=3):
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((img_size[0], img_size[1]), dims, channels, num_classes=num_classes)


def mobilevit_s(img_size=(256, 256), num_classes=3):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    size_in1 = 1024
    size_in2 = 102  # 323  102
    size_in3 = 16380  #16380
    return MobileViT((img_size[0], img_size[1]), dims, channels, num_classes=num_classes, size_in1=size_in1, size_in2=size_in2, size_in3=size_in3, expansion=2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




if __name__ == '__main__':
    img = torch.randn(5, 1, 256, 256)
    clinical = torch.randn(5, 102)  # 323  ref:102
    geneData = torch.randn(5, 16380)
    model = mobilevit_s()

    y = model(img, clinical, geneData)

