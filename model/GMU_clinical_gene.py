import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchsummary import summary

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        # kernel_size =  self.conv.kernel_size
        # print("卷积核大小:", kernel_size)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA_2(nn.Module):
    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA_2, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 128, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = conv_block(64, 128, kernel_size=3, padding=1)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 128, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(128, 256, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)  # 64 8 8

        branch5x5 = self.branch3x3_1(x)
        branch5x5 = self.branch3x3_2(branch5x5)  # 64 8 8

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)  # 96 8 8

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)  # 64 8 8

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

'''---InceptionA---'''
class InceptionA_1(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA_1, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 16, kernel_size=1, stride=2)

        self.branch3x3_1 = conv_block(in_channels, in_channels*4, kernel_size=1)
        self.branch3x3_2 = conv_block(in_channels*4, 32, kernel_size=3, stride=2, padding=1)

        self.branch3x3dbl_1 = conv_block(in_channels, in_channels*2, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(in_channels*2, in_channels*4, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(in_channels*4, 64, kernel_size=3, stride=2, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)  # 64 8 8

        branch5x5 = self.branch3x3_1(x)
        branch5x5 = self.branch3x3_2(branch5x5)  # 64 8 8

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)  # 96 8 8

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        branch_pool = self.branch_pool(branch_pool)  # 64 8 8

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

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
            nn.ReLU6(inplace=True),
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
                nn.ReLU6(), #nn.SiLU(), linear_hardswish()
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
                linear_hardswish(),# nn.SiLU(),
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


class Gene_clinical(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, size_in1, size_in2, expansion=4, kernel_size=3, patch_size=(2, 2), size_out=32):
        super().__init__()


        # add classifer
        self.classify2 = nn.Sequential()
        self.classify2.add_module("bn", nn.BatchNorm1d(size_out))
        self.classify2.add_module("dp", nn.Dropout(0.6))
        self.classify2.add_module("fc1", nn.Linear(size_out, 16))
        self.classify2.add_module("relu", nn.ReLU())
        self.classify2.add_module("fc2", nn.Linear(16, num_classes))

        # add Inception block
        self.conv_inception_1 = InceptionA_1(96, 16)
        self.conv_inception_2 = InceptionA_2(160, 128)

    def forward(self, x1, x2):

        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden2(x2))
        x = torch.cat((h1, h2), dim=1)
        z1 = self.sigmoid_f(self.hidden_sigmoid(x))

        z = z1.view(z1.size()[0], 1) * h1 + (1 - z1).view(z1.size()[0], 1) * h2
        y = self.classify2(z)
        return y, x


def mobilevit_xxs(img_size=(256, 256), num_classes=3):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return Gene_clinical((img_size[0], img_size[1]), dims, channels, num_classes=num_classes, expansion=2)


def mobilevit_xs(img_size=(256, 256), num_classes=3):
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return Gene_clinical((img_size[0], img_size[1]), dims, channels, num_classes=num_classes)


def mobilevit_s_fusion(img_size=(256, 256), num_classes=3, size_in2=102):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    size_in1 = 16380  #640 *8*8
    size_in2 = size_in2  # gene:16380  # 323  ref clinical:102
    return Gene_clinical((img_size[0], img_size[1]), dims, channels, num_classes=num_classes, size_in1=size_in1, size_in2=size_in2, expansion=2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    gene = torch.randn(5, 16380)
    clinical = torch.randn(5, 102)  # 323  102  16380
    model = mobilevit_s_fusion()
    # summary(model, input_size=[(1, 256, 256)], batch_size=5, device="cpu")

    # vit = mobilevit_xxs(img_size=(256, 256))
    # out = vit(img)
    # print(out.shape)
    # print(count_parameters(vit))
    #
    # vit = mobilevit_xs()
    # out = vit(img)
    # print(out.shape)
    # print(count_parameters(vit))

    out = model(gene, clinical)
    print(out.shape)
    #print(count_parameters(vit))
