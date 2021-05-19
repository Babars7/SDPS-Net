import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
from utils import eval_utils
import torch.nn.functional as F
import argparse
from time import time
import math
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.losses import LabelSmoothingCrossEntropy
from options  import stage1_opts
#from src import cct as cct_models
import numpy as np
from .transformers import TransformerEncoderLayer #Added from cct
args = stage1_opts.TrainOpts().parse()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if (not args.no_cuda) and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

def getshape(d):
    if isinstance(d, dict):
        return {k:getshape(d[k]) for k in d}
    else:
        # Replace all non-dict values with None.
        return None

def dumpTranfo(self, out):
    out = out.reshape(out.shape[0], out.shape[1], 1, 1)
    #out = out.clone().detach() #uncomented
    out = torch.tensor(out, dtype=torch.float,  device='cuda:0')
    return out

#######################FROM CTTT#############
class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=4,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=False),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=4, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TransformerClassifier(nn.Module):
    num_heads=4

    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 positional_embedding='sine',
                 sequence_length=None,
                 *args, **kwargs):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim),
                                          requires_grad=True)
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                   requires_grad=True)
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = nn.Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                   requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        print("embedding_dim", embedding_dim, "num_heads", num_heads, "attention", attention_dropout )
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_dim)

        self.fc_dirx = nn.Linear(embedding_dim, 36) #it is here I have to modify MLP Head
        self.apply(self.init_weight)
        self.fc_diry = nn.Linear(embedding_dim, 36)
        self.apply(self.init_weight)
        self.fc_intens = nn.Linear(embedding_dim, 20)
        self.apply(self.init_weight)

    def forward(self, x):
        #print('classifierinput', x.shape)
        if self.positional_emb is None and x.size(1) < self.sequence_length: #note done
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1) #not done
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None: #this is done
            x += self.positional_emb
        #print('afterpositionalembeding', x.shape, self.positional_emb)
        x = self.dropout(x)

        #print('beforetransformer', x.shape)
        for blk in self.blocks: #Transformer encoder layer
            x = blk(x)
        x = self.norm(x)
        #print('aftertransformer', x.shape)
        
        if self.seq_pool: #Sequence Pooling
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]
        ##################
        #print('afterseqpool', x.shape)
        x_out = self.fc_dirx(x)
        #print('afterMLP1', x_out.shape)
        y_out = self.fc_diry(x)
        #print('afterMLP2', y_out.shape)
        ints_out = self.fc_intens(x)

        return x_out, y_out, ints_out
        ######
        #return x
        
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)





                
#############################################
class LCCCTNet(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}, img_size=224,
                 embedding_dim=768,
                 n_input_channels=4,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 *args, **kwargs):
        super(LCCCTNet, self).__init__()
        self.c_in      = c_in
        img_size = 128
        positional_embedding = 'learnable'
        conv_layers = 2#7
        conv_size = 3
        patch_size = 4
        num_layers=4
        num_heads=2#4
        mlp_ratio=1#2
        embedding_dim=128#256
        kernel_size=7
        stride=None 
        padding=None
        stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
        padding = padding if padding is not None else max(1, (kernel_size // 2))

        self.other     = other

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            num_heads=num_heads,#Test
            num_layers=num_layers,#Test
            #mlp_ratio=mlp_ratio,#Test
            *args, **kwargs)


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        

        #self.dir_x_est = nn.Sequential( # 128 -> cct_2 256 -> cct_7 (embeding size)
        #            model_utils.conv(batchNorm, 128, 64,  k=1, stride=1, pad=0), 
        #            model_utils.outputConv(64, other['dirs_cls'], k=1, stride=1, pad=0))

        #self.dir_y_est = nn.Sequential(
        #            model_utils.conv(batchNorm, 128, 64,  k=1, stride=1, pad=0),
        #            model_utils.outputConv(64, other['dirs_cls'], k=1, stride=1, pad=0))

        #self.int_est = nn.Sequential(
        #            model_utils.conv(batchNorm, 128, 64,  k=1, stride=1, pad=0),
        #            model_utils.outputConv(64, other['ints_cls'], k=1, stride=1, pad=0))
    

    def prepareInputs(self, x):
        n, c, h, w = x[0].shape
        t_h, t_w = self.other['test_h'], self.other['test_w']
        if (h == t_h and w == t_w):
            imgs = x[0] 
        else:
            print('Rescaling images: from %dX%d to %dX%d' % (h, w, t_h, t_w))
            imgs = torch.nn.functional.upsample(x[0], size=(t_h, t_w), mode='bilinear')

        inputs = list(torch.split(imgs, 3, 1)) #split imgs in every 3 in dimension 1 (32,96,128,128)->(32,3,128,128)
        idx = 1
        if self.other['in_light']:
            light = torch.split(x[idx], 3, 1)
            for i in range(len(inputs)):
                inputs[i] = torch.cat([inputs[i], light[i]], 1)
            idx += 1
        if self.other['in_mask']:
            mask = x[idx]
            if mask.shape[2] != inputs[0].shape[2] or mask.shape[3] != inputs[0].shape[3]:
                mask = torch.nn.functional.upsample(mask, size=(t_h, t_w), mode='bilinear')
            for i in range(len(inputs)):
                inputs[i] = torch.cat([inputs[i], mask], 1)#(32,3,128,128)->(32,3,128,128)
            idx += 1
        return inputs


    def convertMidDirs(self, pred):
        _, x_idx = pred['dirs_x'].data.max(1)
        _, y_idx = pred['dirs_y'].data.max(1)
        dirs = eval_utils.SphericalClassToDirs(x_idx, y_idx, self.other['dirs_cls'])
        return dirs

    def convertMidIntens(self, pred, img_num):
        _, idx = pred['ints'].data.max(1)
        ints = eval_utils.ClassToLightInts(idx, self.other['ints_cls'])
        ints = ints.view(-1, 1).repeat(1, 3)
        ints = torch.cat(torch.split(ints, ints.shape[0] // img_num, 0), 1)
        return ints



    def forward(self, x):

        inputs = self.prepareInputs(x)

        

        #global best_acc1

        l_dirs_x, l_dirs_y, l_ints = [], [], []
        outputs= []
        for i in range(len(inputs)):

            #print("plotting input image", type(inputs[i]))
            #img_1 = inputs[20].detach().cpu()
            #img_1 = img_1[0, 0, :, :].numpy()
            #img_2 = inputs[20].detach().cpu()
            #img_2 = img_2[0, 1, :, :].numpy()
            #img_3 = inputs[20].detach().cpu()
            #img_3 = img_3[0, 3, :, :].numpy()
            #plt.imsave('img_1.png', img_1)
            #plt.imsave('img_2.png', img_2)
            #plt.imsave('img_3.png', img_2)
            
            #if (not args.no_cuda) and torch.cuda.is_available():
            #    inp = inputs[i].cuda(args.gpu_id, non_blocking=True)
            #print('before token:', inputs[i].shape, type(inputs[i]))
            #shape here is 2,256, 8, 8
            inp = self.tokenizer(inputs[i])
            ##print('before classifier:', inp.shape, type(inp))
            #out = self.classifier(inp)
            ##print('after classifier', type(out))
            #out = out.reshape(out.shape[0], out.shape[1], 1, 1)
            ##out = out.clone().detach().requires_grad_(True)
            #out = torch.tensor(out, dtype=torch.float,  device='cuda:0')

            x_out, y_out, ints_out = self.classifier(inp)
            x_out = dumpTranfo(self, x_out)
            y_out = dumpTranfo(self, y_out)
            ints_out = dumpTranfo(self, ints_out)

            outputs = {}
            #if self.other['s1_est_d']:
            #    outputs['dir_x'] = x_out
            #    outputs['dir_y'] = y_out
            #if self.other['s1_est_i']:
            #    outputs['ints'] = ints_out

            #x_out = self.dir_x_est(out)
            #y_out = self.dir_y_est(out)
            #ints_out = self.int_est(out)
            #print(x_out[1,:,0,0], torch.sum(x_out[1,:,0,0]))

            if self.other['s1_est_d']:
                l_dirs_x.append(x_out)
                l_dirs_y.append(y_out)
            if self.other['s1_est_i']:
                l_ints.append(ints_out)

            #print('prediction', torch.sum(x_out/torch.sum(x_out)))


        pred = {}
        if self.other['s1_est_d']:
            pred['dirs_x'] = torch.cat(l_dirs_x, 0).squeeze()
            pred['dirs_y'] = torch.cat(l_dirs_y, 0).squeeze()
            #print('dict', getshape(pred), pred['dirs_x'].shape)
            pred['dirs']   = self.convertMidDirs(pred)
            #print('dir_x', pred['dirs_x'].shape, pred['dirs_x'][1,:])
        if self.other['s1_est_i']:
            pred['ints'] = torch.cat(l_ints, 0).squeeze()
            if pred['ints'].ndimension() == 1:
                pred['ints'] = pred['ints'].view(1, -1)
            pred['intens'] = self.convertMidIntens(pred, len(inputs))
            #print('ints', (pred['ints'][1,:]).shape )
        #print('pred', getshape(pred))
        return pred