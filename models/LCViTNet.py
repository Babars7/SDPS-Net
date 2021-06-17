import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
from utils import eval_utils
import torch.nn.functional as F
#from .transformers import TransformerEncoderLayer
import argparse
from time import time
import math
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from options  import stage1_opts
from .transformers import TransformerEncoderLayer #Added from cct
args = stage1_opts.TrainOpts().parse()


def getshape(d):
    if isinstance(d, dict):
        return {k:getshape(d[k]) for k in d}
    else:
        # Replace all non-dict values with None.
        return None



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
        #self.Lin = nn.Sequential(nn.Linear(2048, 1024),
        #                        nn.Linear(1024, 512),
        #                        nn.Linear(512, 256))

    def sequence_length(self, n_channels=4, height=224, width=224):#n_channels=3
        print('sequence_length', self.forward(torch.zeros((1, n_channels, height, width))).shape[1])
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):

#Conv + Flatten correspond to the process flattening image and multiply with Embeding matrix and then this will go into encoder
        x = self.conv_layers(x)

        flat = self.flattener(x)

        return flat.transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TransformerClassifier(nn.Module):


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
            #self.attention_pool = nn.Sequential(nn.Linear(embedding_dim, 128),
            #                    nn.Linear(128, 64),
            #                    nn.Linear(64, 1))

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
        #self.fc_dirx = nn.Sequential(nn.Linear(embedding_dim, 128),
        #                        nn.Linear(128, 64),
        #                        nn.Linear(64, 36))
        self.apply(self.init_weight)
        self.fc_diry = nn.Linear(embedding_dim, 36)
        #self.fc_diry = nn.Sequential(nn.Linear(embedding_dim, 128),
        #                        nn.Linear(128, 64),
        #                        nn.Linear(64, 36))
        self.fc_intens = nn.Linear(embedding_dim, 20)
        self.apply(self.init_weight)
        #self.fc_intens = nn.Sequential(nn.Linear(embedding_dim, 128),
        #                        nn.Linear(128, 64),
        #                        nn.Linear(64, 20))
        self.apply(self.init_weight)

    def forward(self, x):
        
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb
        
      
        x = self.dropout(x)
   
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
       
        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]
       
        out_x = self.fc_dirx(x)
        out_y = self.fc_diry(x)
        out_ints = self.fc_intens(x)
        
        return out_x, out_y, out_ints
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




class LCViTNet(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={},img_size=224,
                 embedding_dim=768,
                 n_input_channels=4,
                 patch_size=16,
                 *args, **kwargs):

        super(LCViTNet, self).__init__()
        assert img_size % patch_size == 0, f"Image size ({img_size}) has to be" \
                                           f"divisible by patch size ({patch_size})"

        self.c_in      = c_in

        self.other     = other
        img_size = 128
        positional_embedding = 'learnable'
        conv_layers = 1
        conv_size = 3
        patch_size = 4
        num_layers=7#7 
        num_heads=4#4 
        mlp_ratio=2#2 
        embedding_dim=256#256
        kernel_size=128

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

#using ViT Lite 7 with 1 layer of transformer encoder
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=128,#patch_size,
                                   stride=128,#patch_size,
                                   padding=0,
                                   max_pool=False,
                                   activation=None,
                                   n_conv_layers=1)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=False,
            dropout=0.1,
            attention_dropout=0.,
            stochastic_depth=0.,
            num_heads=num_heads,
            num_layers=num_layers,
            *args, **kwargs)


    def prepareInputs(self, x):
        n, c, h, w = x[0].shape
        t_h, t_w = self.other['test_h'], self.other['test_w']
        if (h == t_h and w == t_w):
            imgs = x[0] 
        else:
            print('Rescaling images: from %dX%d to %dX%d' % (h, w, t_h, t_w))
            imgs = torch.nn.functional.upsample(x[0], size=(t_h, t_w), mode='bilinear')

        #split imgs in every 3 in dimension 1 (32,96,128,128)->(32,3,128,128)
        inputs = list(torch.split(imgs, 3, 1)) 
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
                inputs[i] = torch.cat([inputs[i], mask], 1)
                #(32,3,128,128)->(32,3,128,128)
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

        trans=0
        

        l_dirs_x, l_dirs_y, l_ints = [], [], []
        for i in range(len(inputs)):
            
            #out_feat = self.featExtractor(inputs[i])
            x = self.tokenizer(inputs[i])
            
            out_x, out_y, out_ints = self.classifier(x)
            #out = self.classifier(x)
            
            #out = out.reshape(out.shape[0], out.shape[1], 1, 1)
            ##out = out.clone().detach().requires_grad_(True)
            #out = torch.tensor(out, dtype=torch.float,  device='cuda:0')
            
            #out_x = self.dir_x_est(out)
            #out_y = self.dir_y_est(out)
            #out_ints = self.int_est(out)

            
            if self.other['s1_est_d']:
                l_dirs_x.append(out_x)
                l_dirs_y.append(out_y)
            if self.other['s1_est_i']:
                l_ints.append(out_ints)

        pred = {}
        if self.other['s1_est_d']:
            
            pred['dirs_x'] = torch.cat(l_dirs_x, 0).squeeze()
            pred['dirs_y'] = torch.cat(l_dirs_y, 0).squeeze()
            
            pred['dirs']   = self.convertMidDirs(pred)
            
        if self.other['s1_est_i']:
            pred['ints'] = torch.cat(l_ints, 0).squeeze()
            if pred['ints'].ndimension() == 1:
                pred['ints'] = pred['ints'].view(1, -1)
            pred['intens'] = self.convertMidIntens(pred, len(inputs))
            
        return pred
