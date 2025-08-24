import torch
from einops import rearrange
from torch import nn
from activation import mish
from Transformer import SCSFormer

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from SFE_M import AGCA

from SFE_M import SpatialFeatureEnhancement




class SSFSNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=16, Conv=False, dim=32, depth=1, heads=8, size=11,
                 mlp_dim=1/8, attn_drop=0.1, drop=0.2, drop_path=0.1, dataset_name='pu'):
        super(SSFSNet, self).__init__()
        # self.L = num_tokens
        self.activation = mish()  ## # gelu()gelu_new()nn.GELU()nn.Sigmoid()nn.ReLU()
        # ！#swish()
        self.Conv = Conv

        # self.cT = dim
        in_channels = 1
        if self.Conv:
            in_channels = 1
            conv_dim = 64
        else:
            conv_dim = 64
        conv3dc = 8
        conv3dc1 = 8


        if dataset_name == 'pu':
            dim1 = 48  # ,  120 128
        elif dataset_name == 'sa':
            dim1 = 104  # 104264 248 256
        elif dataset_name == 'in':
            dim1 = 256
        elif dataset_name == 'whulk':
            dim1 = 144
        elif dataset_name == 'wh_hc':
            dim1 =144
        elif dataset_name == 'hs':
            dim1 = 72  # 176
        elif dataset_name == 'bw':
            dim1 = 72  # 176
        elif dataset_name == 'ksc':
            dim1 = 88  # 24


        self.conv3d_features = nn.Sequential(

            nn.Conv3d(in_channels, out_channels=conv3dc, kernel_size=(5, 1, 1), stride=(5, 1, 1), padding=(0, 0, 0)),
            # nn.Conv3d(in_channels,out_channels=conv3dc,kernel_size=1),
            nn.BatchNorm3d(conv3dc),

            self.activation,
        )

        self.conv3d_features1 = nn.Sequential(
            nn.Conv3d(conv3dc, out_channels=conv3dc1, kernel_size=(3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0), ),
            # nn.Conv3d(conv3dc1,conv3dc1,kernel_size=1),
            nn.BatchNorm3d(conv3dc1),
            self.activation,
        )


        self.conv3d1 = nn.Sequential(
            nn.Conv3d(1, out_channels=1, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2)),
            # nn.Conv3d(conv3dc1,conv3dc1,kernel_size=1),
            nn.BatchNorm3d(1),
            self.activation,
        )


        self.sfe = SpatialFeatureEnhancement(conv_dim=conv_dim, dim=dim,act_layer=self.activation)  # in_nc

        self.conv2d1 = nn.Sequential(
            nn.Conv2d(in_channels=dim1, out_channels=conv_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(conv_dim),
            self.activation, )


        # self.agca1 = AGCA(dim1, dim1,16, activation=self.activation)
        self.agca = AGCA(conv_dim, conv_dim, 8, activation=self.activation)
        #
        # self.eca = ECAAttention(dim1)
        self.transformer = SCSFormer(
                    dim=dim,
                    num_heads=heads, mlp_ratio=mlp_dim,
                    drop=drop,  drop_path=drop_path,
                    act_layer=self.activation,  # ,#,nn.ReLU,,,mishnn.GELU
                    layerscale=False, )


        self.nn1 = nn.Linear(dim, num_classes)


        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.norm = nn.BatchNorm2d(dim)
        self.norm1 = nn.BatchNorm2d(dim1)


        self.norm2 = nn.BatchNorm2d(conv_dim)


    def forward(self, x):
        _, _, _, _, size = x.shape

        if self.Conv:

            x = x+self.conv3d1(x)

        x = self.conv3d_features(x)
        x = self.conv3d_features1(x)

        x = rearrange(x, 'b c h w y -> b (c h) w y')

        x = self.conv2d1(x)
        x = x*self.agca(x)

        x = self.sfe(x)

        x = self.transformer(x)

        x = self.avgpool(x).flatten(1)

        x = self.nn1(x)

        return x


if __name__ == '__main__':
    dataset = 'pu'
    size = 11
    if dataset == 'pu':
        input = torch.randn(64, 1, 103, size, size).to('cuda')
    if dataset == 'sa':
        input = torch.randn(64, 1, 204, size, size).to('cuda')
    if dataset == 'ksc':
        input = torch.randn(64, 1, 176, size, size).to('cuda')

    if dataset == 'ksc':
        conv = True
    else:
        conv = False
    model = SSFSNet(dataset_name=dataset, Conv=conv,size=size).to('cuda')

    total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    flops = FlopCountAnalysis(model, input)
    print(flop_count_table(flops))
    print("Number of parameter: %.5fM" % (total / 1e6))

