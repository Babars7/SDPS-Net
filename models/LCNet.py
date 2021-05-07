import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
from utils import eval_utils
import torch.nn.functional as F

###########added_import##############
import argparse
from time import time
import math

import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src import cct as cct_models
from utils.losses import LabelSmoothingCrossEntropy

from options  import stage1_opts

args = stage1_opts.TrainOpts().parse()

def getshape(d):
    if isinstance(d, dict):
        return {k:getshape(d[k]) for k in d}
    else:
        # Replace all non-dict values with None.
        return None

def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 quick training script')

    # Data args


    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')



    # Optimization hyperparams
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup', default=5, type=int, metavar='N',
                        help='number of warmup epochs')
    parser.add_argument('-b', '--batch-size', default=1, type=int, #default=128
                        metavar='N',
                        help='mini-batch size (default: 128)', dest='batch_size')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=3e-2, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--clip-grad-norm', default=0., type=float,
                        help='gradient norm clipping (default: 0 (disabled))')

    parser.add_argument('-m', '--model',
                        type=str.lower,
                        default='cct_2', dest='model')

    parser.add_argument('-p', '--positional-embedding',
                        type=str.lower,
                        choices=['learnable', 'sine', 'none'],
                        default='learnable', dest='positional_embedding')

    parser.add_argument('--conv-layers', default=2, type=int,
                        help='number of convolutional layers (cct only)')

    parser.add_argument('--conv-size', default=3, type=int,
                        help='convolution kernel size (cct only)')

    parser.add_argument('--patch-size', default=4, type=int,
                        help='image patch size (vit and cvt only)')

    parser.add_argument('--disable-cos', action='store_true',
                        help='disable cosine lr schedule')

    parser.add_argument('--disable-aug', action='store_true',
                        help='disable augmentation policies for training')

    parser.add_argument('--gpu-id', default=0, type=int)

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable cuda')

    parser.add_argument('--in_img_num',  default=32,    type=int)#32

    return parser

#####################################

# Classification
class FeatExtractor(nn.Module):
    def __init__(self, batchNorm, c_in, c_out=256):
        super(FeatExtractor, self).__init__()
        self.conv1 = model_utils.conv(batchNorm, c_in, 64,    k=3, stride=2, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64,   128,   k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128,  128,   k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 128,  128,   k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 128,  128,   k=3, stride=1, pad=1)
        self.conv6 = model_utils.conv(batchNorm, 128,  256,   k=3, stride=2, pad=1)
        self.conv7 = model_utils.conv(batchNorm, 256,  256,   k=3, stride=1, pad=1)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        return out

class Classifier(nn.Module):
    def __init__(self, batchNorm, c_in, other):
        super(Classifier, self).__init__()
        self.conv1 = model_utils.conv(batchNorm, 512,  256, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 256,  256, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 256,  256, k=3, stride=2, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 256,  256, k=3, stride=2, pad=1)
        self.other = other
        
        self.dir_x_est = nn.Sequential(
                    model_utils.conv(batchNorm, 256, 64,  k=1, stride=1, pad=0),
                    model_utils.outputConv(64, other['dirs_cls'], k=1, stride=1, pad=0))

        self.dir_y_est = nn.Sequential(
                    model_utils.conv(batchNorm, 256, 64,  k=1, stride=1, pad=0),
                    model_utils.outputConv(64, other['dirs_cls'], k=1, stride=1, pad=0))

        self.int_est = nn.Sequential(
                    model_utils.conv(batchNorm, 256, 64,  k=1, stride=1, pad=0),
                    model_utils.outputConv(64, other['ints_cls'], k=1, stride=1, pad=0))

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)#torch.Size([8, 256, 1, 1])
        #print('out', out.shape)
        outputs = {}
        if self.other['s1_est_d']:
            outputs['dir_x'] = self.dir_x_est(out) #torch.Size([32, 36, 1, 1])
            #print('dir_x', outputs['dir_x'], outputs['dir_x'].shape)
            outputs['dir_y'] = self.dir_y_est(out) #torch.Size([32, 36, 1, 1])
            #print('dir_y', outputs['dir_y'], outputs['dir_y'].shape)
        if self.other['s1_est_i']:
            outputs['ints'] = self.int_est(out) #torch.Size([32, 20, 1, 1])
            #print('ints', outputs['ints'], outputs['ints'].shape)
        return outputs

class LCNet(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(LCNet, self).__init__()
        self.featExtractor = FeatExtractor(batchNorm, c_in, 128)
        self.classifier = Classifier(batchNorm, 256, other)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other     = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    

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

    def fuseFeatures(self, feats, fuse_type): #It is here the max pooling 
        if fuse_type == 'mean':
            feat_fused = torch.stack(feats, 1).mean(1)
        elif fuse_type == 'max':
            feat_fused, _ = torch.stack(feats, 1).max(1)
        return feat_fused

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


        

        ###################

        #print('x:', len(x))
        #print('x[0]:',list(x[0].shape))
        #print('x[1]:',list(x[1].shape))
        inputs = self.prepareInputs(x)
        #print('length:', len(inputs))
        #print('input[0]:',list(inputs[0].shape))
        #print('input[1]:',list(inputs[1].shape))
        #print('input[2]:',list(inputs[2].shape))

        ##############APROACH_1################

        
        
        global best_acc1


        img_size = 128
        num_classes = 92 #36+36+20 (x,y,ints)
        positional_embedding = 'learnable'
        conv_layers = 2
        conv_size = 3
        patch_size = 4


        parser = init_parser()
        args_cct = parser.parse_args()

        self.model = cct_models.__dict__[args_cct.model](img_size=img_size,
                                            num_classes=num_classes,
                                            positional_embedding=args_cct.positional_embedding,
                                            n_conv_layers=args_cct.conv_layers,
                                            kernel_size=args_cct.conv_size,
                                            patch_size=args_cct.patch_size)
        ####################
        if (not args_cct.no_cuda) and torch.cuda.is_available():
            torch.cuda.set_device(args_cct.gpu_id)
            self.model.cuda(args_cct.gpu_id)
            #print('xxxxxxxxxxxxxxxxxx')
        


        l_dirs_x, l_dirs_y, l_ints = [], [], []
        outputs= []
        for i in range(len(inputs)):
            #print('types', type(inputs[i]))
            if (not args_cct.no_cuda) and torch.cuda.is_available():
                inp = inputs[i].cuda(args_cct.gpu_id, non_blocking=True)
                #print('yyyyyyyyyyyyyyy')
            #print(type(inp), inp.size(), type(inputs[i]))
            out = self.model(inp)
            out = out.reshape(out.shape[0], out.shape[1], 1, 1)
            #print('outputs', type(out), out[:,0:36].shape)
            if self.other['s1_est_d']:
                l_dirs_x.append(out[:,0:36])
                l_dirs_y.append(out[:,36:72])
            if self.other['s1_est_i']:
                l_ints.append(out[:,72:92])
            #outputs.append(out)
            

        #outputs=outputs.reshape(outputs.shape[0], outputs.shape[0], 1, 1)
        #print('outputs', type(outputs))
        #if self.other['s1_est_d']:
        #    l_dirs_x = outputs[:,0:36]
        #    l_dirs_y = outputs[:,36:72]
        #if self.other['s1_est_i']:
        #    l_ints = outputs[:,72:92]

        pred = {}
        if self.other['s1_est_d']:
            pred['dirs_x'] = torch.cat(l_dirs_x, 0).squeeze()
            pred['dirs_y'] = torch.cat(l_dirs_y, 0).squeeze()
            #print('dict', getshape(pred), pred['dirs_x'].shape)
            pred['dirs']   = self.convertMidDirs(pred)
        if self.other['s1_est_i']:
            pred['ints'] = torch.cat(l_ints, 0).squeeze()
            if pred['ints'].ndimension() == 1:
                pred['ints'] = pred['ints'].view(1, -1)
            pred['intens'] = self.convertMidIntens(pred, len(inputs))
        #print('pred', getshape(pred))
        return pred