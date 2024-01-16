
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import clear_output
import numpy as np
from PIL import Image

import glob
import random
import os
import sys

torch.manual_seed(0)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        '''No change in dimension'''
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        
        self.instancenorm = nn.InstanceNorm2d(input_channels)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # we dont need to add AdaIN as we did in StyleGAN 
        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        
        return original_x + x # Residual connections
    
    

class ContractingBlock(nn.Module):
    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        '''Encoder of the U-Net architecture'''
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size, padding=1,
                               stride=2, padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        
        if use_bn: # batch normalization
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn
        # use instance norm if only BN
    
    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class ExpandingBlock(nn.Module): # upsample
    def __init__(self, input_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        '''Decoder of the U-Net architecture'''
        self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3,
                                        stride=2, padding=1, output_padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x        


class FeatureMapBlock(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, 
                              padding=3, padding_mode='reflect')

    def forward(self, x):
        x = self.conv(x)
        return x
    

'''CycleGAN Generator'''

class Generator(nn.Module):
    '''
    Feature Map
    2 Downsampling blocks
    9 Residual blocks
    2 Upsampling blocks
    Feature Map 
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(Generator, self).__init__()
        # feature map
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        # Downsampling the latent space
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        
        # Upsampling the feature learnt from the downsampling
        res_mult = 4
        self.res0 = ResidualBlock(hidden_channels * res_mult)
        self.res1 = ResidualBlock(hidden_channels * res_mult)
        self.res2 = ResidualBlock(hidden_channels * res_mult)
        self.res3 = ResidualBlock(hidden_channels * res_mult)
        self.res4 = ResidualBlock(hidden_channels * res_mult)
        self.res5 = ResidualBlock(hidden_channels * res_mult)
        self.res6 = ResidualBlock(hidden_channels * res_mult)
        self.res7 = ResidualBlock(hidden_channels * res_mult)
        self.res8 = ResidualBlock(hidden_channels * res_mult)
        
        # expand
        self.expand2 = ExpandingBlock(hidden_channels * 4)
        self.expand3 = ExpandingBlock(hidden_channels * 2)
        # feature map 
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        
        # Activation [-1, 1]
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x0 = self.upfeature(x)
        
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        
        x12 = self.expand2(x11)
        x13 = self.expand3(x12)
        
        xn = self.downfeature(x13)
        return self.tanh(xn)


'''The discriminator of the CycleGAN is discriminator of the PatchGAN '''
class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels=64):
        super().__init__()
        
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        # In disc, we use Leaky Relu, and we dont use batch norm as we are just dicriminating it
        
        '''Downsampling the Image obtained from the Generator'''
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=4, activation='lrelu')
        
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1) # getting single output 
    
    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn        


class Main:
    # config
    dim_A = 3
    dim_B = 3
    load_shape = 286
    target_shape = 256
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    checkpoint_path = "/home/jerlshin/Documents/My_Work/Generative Adversarial Networks Specialization/C3 Apply GAN - 26 hrs/cycleGAN_9800.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device=device))

    def preprocess_input(self, image):
        transform = transforms.Compose([
            transforms.Resize(self.load_shape),
            transforms.ToTensor(),
        ])
        return transform(image).to(self.device)
    
    def postprocess_output(self, tensor):
        image_transformed = (tensor + 1) / 2
        image_unflat = image_transformed.detach().cpu()
        image = image_unflat.permute(1, 2, 0).squeeze()
        image_flipped = np.fliplr(image) 
        return image_flipped

    def translate_image_H_Z(self, image):
        gen_AB = Generator(self.dim_A, self.dim_B).to(self.device)
        gen_AB.load_state_dict(self.checkpoint['gen_AB'])
        gen_AB.eval()
        image = self.preprocess_input(image)
        
        with torch.no_grad():
            trans = gen_AB(image)
        
        return self.postprocess_output(trans)

    def translate_image_Z_H(self, image):
        gen_BA = Generator(self.dim_B, self.dim_A).to(self.device)
        gen_BA.load_state_dict(self.checkpoint['gen_BA'])
        gen_BA.eval()
        image = self.preprocess_input(image)
        
        with torch.no_grad():
            trans = gen_BA(image)
        
        return self.postprocess_output(trans)
        

# Generator from domain A to B and vice versa

