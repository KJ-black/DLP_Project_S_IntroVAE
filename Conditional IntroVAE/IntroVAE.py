import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import math
import time
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import data_parallel


class _Residual_Block(nn.Module): 
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()
        
        midc=int(outc*scale)
        
        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
          self.conv_expand = None
          
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x): 
        if self.conv_expand is not None:
          identity_data = self.conv_expand(x)
        else:
          identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output,identity_data)))
        return output 

class Encoder(nn.Module):
    def __init__(self, cdim=3, hdim=512, emb_dim=1024, channels=[64, 128, 256, 512, 512, 512], image_size=256, conditional=False):
        super(Encoder, self).__init__() 

        assert (2 ** len(channels)) * 4 == image_size
        
        self.hdim = hdim
        self.cdim = cdim
        self.image_size = image_size
        self.conditional = conditional
        self.cond_dim = emb_dim
        cc = channels[0]
        self.main = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))   
        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]

        if self.conditional:
            self.fc = nn.Linear(num_fc_features + self.cond_dim, 2 * hdim)
        else:
            self.fc = nn.Linear(num_fc_features, 2 * hdim) 

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape
    
    def forward(self, x, o_cond=None):
        y = self.main(x).view(x.size(0), -1)
        if self.conditional and o_cond is not None:
            y = torch.cat([y, o_cond], dim=1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, cdim=3, hdim=512, emb_dim=1024, channels=[64, 128, 256, 512, 512, 512], image_size=256, conditional=False, 
                 conv_input_size=None):
        super(Decoder, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size

        super(Decoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.conditional = conditional
        self.cond_dim = emb_dim
        self.conv_input_size = conv_input_size
        
        cc = channels[-1]
        if conv_input_size is None:
            num_fc_features = cc * 4 * 4
        else:
            num_fc_features = torch.zeros(self.conv_input_size).view(-1).shape[0]
        if self.conditional:
            self.fc = nn.Sequential(
                nn.Linear(hdim + self.cond_dim, num_fc_features),
                nn.ReLU(True),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(hdim, num_fc_features),
                nn.ReLU(True),
            )
                  
        sz = 4
        
        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz*2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz*2
       
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))
                    
    def forward(self, z, y_cond=None):
        z = z.view(z.size(0), -1)
        if self.conditional and y_cond is not None:
            y_cond = y_cond.view(y_cond.size(0), -1)
            z = torch.cat([z, y_cond], dim=1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
        y = self.main(y)
        return y
  
        
class IntroVAE(nn.Module):
    def __init__(self, cdim=3, hdim=512, emb_dim=1024, channels=[64, 128, 256, 512, 512, 512], image_size=256, conditional=False):
        super(IntroVAE, self).__init__()         
        
        self.hdim = hdim
        self.conditional = conditional
        self.cond_dim = emb_dim
        
        self.encoder = Encoder(cdim, hdim, emb_dim, channels, image_size, conditional=conditional)
        
        self.decoder = Decoder(cdim, hdim, emb_dim, channels, image_size, conditional=conditional, conv_input_size=self.encoder.conv_output_size)
        
      
    def forward(self, x, o_cond=None, deterministic=False):
        if self.conditional and o_cond is not None:
            mu, logvar = self.encode(x, o_cond=o_cond)
            if deterministic:
                z = mu
            else:
                z = self.reparameterize(mu, logvar)
            y = self.decode(z, y_cond=o_cond)
        else:
            mu, logvar = self.encode(x)
            if deterministic:
                z = mu
            else:
                z = self.reparameterize(mu, logvar)
            y = self.decode(z)
        return mu, logvar, z, y
        
    def sample(self, z, y_cond=None):
        y = self.decode(z, y_cond=y_cond)
        return y
    
    def encode(self, x, o_cond=None):  
        if self.conditional and o_cond is not None:
            mu, logvar = self.encoder(x, o_cond=o_cond)
        else:
            mu, logvar = self.encoder(x)
        return mu, logvar
        
    def decode(self, z, y_cond=None):        
        if self.conditional and y_cond is not None:
            y = self.decoder(z, y_cond=y_cond)
        else:
            y = self.decoder(z)
        return y

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).cuda()
        return mu + eps * std
    
    def kl_loss(self, mu, logvar, prior_mu=0):
        v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        v_kl = v_kl.sum(dim=-1).mul_(-0.5) # (batch, 2)
        return v_kl
    
    def reconstruction_loss(self, prediction, target, size_average=False):        
        error = (prediction - target).view(prediction.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=-1)
        
        if size_average:
            error = error.mean()
        else:
            error = error.sum()
               
        return error
        

