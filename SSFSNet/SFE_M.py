import torch
import torch.nn as nn

from timm.models.layers import DropPath
from einops.layers.torch import Rearrange
from torch.nn import init

class AGCA(nn.Module):
    def __init__(self, in_channel, op,ratio,activation):
        super(AGCA, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)
        # Choose to deploy A0 on GPU or CPU according to your needs
        self.A0 = torch.eye(hide_channel).to('cuda')
        # self.A0 = torch.eye(hide_channel)
        # A2 is initialized to 1e-6
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu =activation
        self.conv4 = nn.Conv2d(hide_channel, op, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)

        y = self.conv1(y)
        B, C, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)
        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C)
        # print(self.A0.device,A1.device,self.A2.device)
        A = (self.A0 * A1) + self.A2
        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1)
        y = self.sigmoid(self.conv4(y))


        return  y

d = 5,
g = int(1)
class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=  d, groups=g, bias=False, theta=1.0):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.k = kernel_size
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], self.k * self.k).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=d, groups=g, bias=False, theta=1.5):

        super(Conv2d_ad, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.k = kernel_size
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation= d, groups=g, bias=False, theta=1.0):

        super(Conv2d_hd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.k =kernel_size
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], self.k * self.k).fill_(0)

        conv_weight=conv_weight.to(torch.device("cuda"))

        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=d, groups=g, bias=False):

        super(Conv2d_vd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.k =kernel_size
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], self.k * self.k).fill_(0)

        conv_weight = conv_weight.to(torch.device("cuda"))
        # print(conv_weight.device)

        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class DEConv(nn.Module):
    def __init__(self, dim, k,gg=1):
        super(DEConv, self).__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, kernel_size=k, padding=k // 2, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, kernel_size=k, padding=k // 2, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, kernel_size=k, padding=k // 2, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, kernel_size=k, padding=k // 2, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, bias=True)

        self.g=gg
        # 初始化权重参数为0.2
        self.alpha = nn.Parameter(torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float32))
        self.alpha2 = nn.Parameter(torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float32))

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → (batch, dim*5, 1, 1)
            nn.Conv2d(dim * 5, 5, 1),  # 映射到5个分支权重 → (batch, 5, 1, 1)
            nn.Softmax(dim=1)  # 归一化为概率分布
        )
    def forward(self, x):
        # 获取各卷积分支的权重和偏置
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        # 将权重参数α通过Softmax归一化
        alphas = torch.softmax(self.alpha, dim=0)  # shape: (5,)
        alphas2 = torch.softmax(self.alpha2, dim=0)

        # 构建权重融合公式
        weight_fusion = (
                # w1 + w2 + w3 + w4 + w5

                alphas[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * w1 +
                alphas[1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * w2 +
                alphas[2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * w3 +
                alphas[3].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * w4 +
                alphas[4].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * w5
        )

        # 构建偏置融合公式
        bias_fusion = (
                # b1 + b2 + b3 + b4 + b5
                alphas2[0].unsqueeze(-1) * b1 +
                alphas2[1].unsqueeze(-1) * b2 +
                alphas2[2].unsqueeze(-1) * b3 +
                alphas2[3].unsqueeze(-1) * b4 +
                alphas2[4].unsqueeze(-1) * b5
        )

        # 执行融合后的卷积操作
        res = nn.functional.conv2d(
            input=x,
            weight=weight_fusion,
            bias=bias_fusion,
            stride=1,
            padding=1,
            groups=self.g
        )
        return res

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


#
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU(),  # nn.ReLU,#,#,nn.ReLU()swish()
                 drop=0.1, ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

        self.conv =nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        #
        self.conv1 =nn.Conv2d(out_features, out_features, 1, 1, 0)
        # self.norm2 = nn.BatchNorm2d(in_features)

    def forward(self, x):

        x = self.fc1(x)

        x = self.norm1(x)

        x = self.act1(x)
        x = self.drop(x)


        x = self.fc2(x)

        x = self.norm2(x)
        x = self.act1(x)

        x = self.drop(x)
        return x


class SpatialFeatureEnhancement(nn.Module):
    def __init__(self, conv_dim, dim, mlp_ratio=1/8,  drop=0.1, drop_path=0., act_layer=nn.GELU()):
        super().__init__()
        self.activation = act_layer

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=conv_dim, out_channels=dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(dim),
            self.activation, )

        # self.deconv=DEConv(conv_dim, 3)
        self.norm1 =LayerNorm2d(conv_dim)# ,DyT(conv_dim,0.5)

        self.norm2 = nn.BatchNorm2d(conv_dim)
        self.activation = act_layer
        self.conv1 = DEConv(conv_dim, 3)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(conv_dim)
        self.mlp2 = Mlp(in_features=conv_dim, hidden_features=int(conv_dim * mlp_ratio), out_features=conv_dim, act_layer=act_layer,
                        drop=drop)

        self.activation = act_layer

    def forward(self, x):
        x1=self.norm1(x)
        x1 = self.conv1(x1)
        x1 = self.norm2(x1)
        x1 = self.activation(x1)

        x = x   + self.drop_path(x1)
        x = x + self.drop_path(self.mlp2(self.norm2(x)))

        x = self.norm2(x)
        x = self.activation(x)

        x = self.conv(x)

        return x


if __name__ == '__main__':
    in_nc = 3
    out_nc = 3# 输入通道数
    block = SpatialFeatureEnhancement(in_nc,out_nc).to('cuda')
    input = torch.rand(4, in_nc, 64, 64).to('cuda')
    output = block(input)
    print(input.size())
    print(output.size())

