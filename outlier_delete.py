import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from torch import Tensor
from torchvision.transforms import Compose,Resize,ToTensor
from einops import rearrange,reduce,repeat
from einops.layers.torch import Rearrange,Reduce
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import data_manage_OD
import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd

from einops import rearrange
from einops.layers.torch import Rearrange


def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=1000, block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)
        
        self.s0t = self._make_layer(conv_3x3_bn, 1, 32, num_blocks[0], (ih // 2, iw // 2))
        self.s0rgb = self._make_layer(conv_3x3_bn, 3, 96, num_blocks[0], (ih // 2, iw // 2))        

    def forward(self, x):
        r,g,b,thermal = torch.chunk(x,4,dim=1)
        ipcam = torch.cat((r,g,b),dim=1)
        x = torch.cat((ipcam, thermal),dim=1)

        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        
        x = self.pool(x).view(-1, x.shape[1])
        # x = self.fc(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)

def coatnet_0():
    num_blocks = [2, 2, 3, 5, 2]            # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((224, 224), 4, num_blocks, channels, num_classes=5)
def coatnet_1():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((224, 224), 4, num_blocks, channels, num_classes=5)

def coatnet_2():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [128, 128, 256, 512, 1026]   # D
    return CoAtNet((224, 224), 4, num_blocks, channels, num_classes=5)


def coatnet_3():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((224, 224), 4, num_blocks, channels, num_classes=5)


def coatnet_4():
    num_blocks = [2, 2, 12, 28, 2]          # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((224, 224), 4, num_blocks, channels, num_classes=5)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))

    starts_from_zeros = x - np.min(x)

    return starts_from_zeros / value_range

from collections import OrderedDict

from torch.utils.data import DataLoader
model = coatnet_2()

#state_dict = torch.load('C:/Users/y/PycharmProjects/pythonProject/checkpoint54_acc_0.7959183673469388.pth')
## create new OrderedDict that does not contain `module.`
#from collections import OrderedDict
#new_state_dict = OrderedDict()
#for k, v in state_dict.items():
#    name = k[7:] # remove `module.`
#    new_state_dict[name] = v
# load params
#model.load_state_dict(new_state_dict)


model.load_state_dict(torch.load("",map_location='cuda:0'))
#model = torch.nn.DataParallel(model, device_ids=[0, 1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model = model.to(device)

batch = 32
PATH = ''
csv_path = '22_21_trainset.csv'
test_dataset = data_manage_OD.CustomDataset(PATH + csv_path, PATH, mode='test')
testloader = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=0)

condition = ['None','DRY','WET','SNOW','ICE','BI']
import math
import torchvision
import os
tot=0
cor=0

import time
count_cls = np.zeros(5)

histo = np.zeros(10)
class_acc = np.zeros(5)
with torch.no_grad():
    # for debug
    arr_d = np.array([])
    features = np.array([])
    labels = np.array([])

    for i,data in enumerate(testloader,0):
        
        org_image = data['image'].type(torch.FloatTensor)
        
        
        org_image = org_image.to(device)
        label = data['landmarks'].type(torch.FloatTensor)

        start_time = time.time()
        output = model(org_image)
        if i == 0:
            features = output.cpu().numpy()
            labels = label.cpu().numpy()
        else:
            features = np.concatenate((features, output.cpu().numpy()))
            labels = np.concatenate((labels, label))
        
        # gt_label = torch.argmax(gt, dim=1).cpu().detach().tolist()


tsne = TSNE(n_components=2, random_state=36).fit_transform(features)

tx = tsne[:, 0]
ty = tsne[:, 1]
# normalzation
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)


## get JSON_file combining "index, "class", "image name", "2d features"
### 
pass
###

# get upper fence and under fence to detect outlier
colors_per_class = [0, 1, 2, 3, 4]
indices_per_class = []
tx_per_class = []
ty_per_class = []
Q1_x_per_class = []
Q3_x_per_class = []
Q1_y_per_class = []
Q3_y_per_class = []

for iter, label in enumerate(colors_per_class):
    indices = [i for i, l in enumerate(labels) if l == label]
    tx_per_label = np.take(tx, indices)
    ty_per_label = np.take(ty, indices) 
    if iter == 0:
        tx_per_class = [tx_per_label]
        ty_per_class = [ty_per_label]
        indices_per_class = [indices]
    else: 
        tx_per_class.append(tx_per_label)
        ty_per_class.append(ty_per_label)
        indices_per_class.append(indices)
    
    Q1_x_per_class.append(np.percentile(tx_per_class[label], 25))
    Q3_x_per_class.append(np.percentile(tx_per_class[label], 75))
    Q1_y_per_class.append(np.percentile(ty_per_class[label], 25))
    Q3_y_per_class.append(np.percentile(ty_per_class[label], 75))

IQR_x_per_class = np.array(Q3_x_per_class) - np.array(Q1_x_per_class)
under_x_fence = Q1_x_per_class - 1.5 * IQR_x_per_class
upper_x_fence = Q3_x_per_class + 1.5 * IQR_x_per_class
IQR_y_per_class = np.array(Q3_y_per_class) - np.array(Q1_y_per_class)
under_y_fence = Q1_y_per_class - 1.5 * IQR_y_per_class
upper_y_fence = Q3_y_per_class + 1.5 * IQR_y_per_class

# outlier identification
## get outlier indices per class: outlier_indices_whole[0:5] (index means class)
outlier_idx_pc = []
outlier_indices_whole = []
for iter, label in enumerate(colors_per_class):
    indices = [i for i, value in enumerate(zip(tx_per_class[label], ty_per_class[label])) if value[0] < under_x_fence[label] or 
                value[0] > upper_x_fence[label] or value[1] < under_y_fence[label] or value[1] > upper_y_fence[label]]
    outlier_idx_pc.append(indices)
    
    if iter == 0:
        outlier_indices_whole = [np.take(indices_per_class[label], outlier_idx_pc[iter])]
    else:
        outlier_indices_whole.append(np.take(indices_per_class[label], outlier_idx_pc[iter]))

## get outlier image names per class: outlier_image_name_pc[0:5] (index means class). therm is about thermal images.
outlier_image_name_pc = []
outlier_therm_name_pc = []

for iter, label in enumerate(colors_per_class):
    if iter == 0:
        outlier_image_name_pc = [[test_dataset[idx]['file_name'][0] for iter, idx in enumerate(outlier_idx_pc[label])]]
        outlier_therm_name_pc = [[test_dataset[idx]['file_name'][1] for iter, idx in enumerate(outlier_idx_pc[label])]]
    else:
        outlier_image_name_pc.append([test_dataset[idx]['file_name'][0] for iter, idx in enumerate(outlier_idx_pc[label])])
        outlier_therm_name_pc.append([test_dataset[idx]['file_name'][1] for iter, idx in enumerate(outlier_idx_pc[label])])

# remove outlier
csv_df = pd.read_csv(csv_path, encoding='euc-kr')
csv_df = csv_df.drop([idx for idx in outlier_idx_pc[label]])


# t-SNE visualization
tx_01 = scale_to_01_range(tx)
ty_01 = scale_to_01_range(ty)

fig = plt.figure()
ax = fig.add_subplot(111)


for label in colors_per_class:
    indices = [i for i, l in enumerate(labels) if l == label]

    current_tx_01 = np.take(tx_01, indices)
    current_ty_01 = np.take(ty_01, indices)

    color_comp = (colors_per_class[label] + 1) * 255 / 5
    color = np.array([color_comp, color_comp, color_comp], dtype=np.float)
    ax.scatter(current_tx_01, current_ty_01, label=label)

ax.legend(loc='best')
plt.show()

print('for debug')

