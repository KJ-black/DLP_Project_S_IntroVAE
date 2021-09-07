from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from dataset import *
import time
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
from IntroVAE import *
from math import log10
import pickle
import torchvision
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from utils import CA_NET
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter

img_size = 32
channels = "64, 128, 256"
data_root = 'Data/flowers/'
batchSize = 256
epoch = 500
conditional = True
emb_dim = 128
test_iter = 500

parser = argparse.ArgumentParser()
parser.add_argument('--lr_e', type=float, default=0.0002, help='learning rate of the encoder, default=0.0002')
parser.add_argument('--lr_g', type=float, default=0.0002, help='learning rate of the generator, default=0.0002')
parser.add_argument("--num_vae", type=int, default=0, help="the epochs of pretraining a VAE, Default=0")
parser.add_argument("--weight_neg", type=float, default=1.0, help="Default=1.0")
parser.add_argument("--weight_rec", type=float, default=1.0, help="Default=1.0")
parser.add_argument("--weight_kl", type=float, default=1.0, help="Default=1.0")
parser.add_argument("--m_plus", type=float, default=100.0, help="the margin in the adversarial part, Default=100.0")
parser.add_argument('--channels', default=channels, type=str, help='the list of channel numbers')
parser.add_argument("--hdim", type=int, default=128, help="dim of the latent code, Default=512")
parser.add_argument("--save_iter", type=int, default=1, help="Default=1")
parser.add_argument("--test_iter", type=int, default=test_iter, help="Default=1000")
parser.add_argument('--nrow', type=int, help='the number of images in each row', default=8)
parser.add_argument('--dataroot', default=data_root, type=str, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=batchSize, help='input batch size')
parser.add_argument('--input_height', type=int, default=128, help='the height  of the input image to network')
parser.add_argument('--input_width', type=int, default=None, help='the width  of the input image to network')
parser.add_argument('--output_height', type=int, default=img_size, help='the height  of the output image to network')
parser.add_argument('--output_width', type=int, default=None, help='the width  of the output image to network')
parser.add_argument('--crop_height', type=int, default=None, help='the width  of the output image to network')
parser.add_argument('--crop_width', type=int, default=None, help='the width  of the output image to network')
parser.add_argument("--nEpochs", type=int, default=epoch, help="number of epochs to train for")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument('--clip', type=float, default=100, help='the threshod for clipping gradient')
parser.add_argument("--step", type=int, default=500, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument('--cuda', action='store_false', help='enables cuda')
parser.add_argument('--outf', default='results/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--tensorboard', action='store_true', help='enables tensorboard')
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

str_to_list = lambda x: [int(xi) for xi in x.split(',')]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg",".bmp"])
    
def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)

def record_image(writer, image_list, cur_iter):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(image_to_show, nrow=opt.nrow), cur_iter)
    

def main():
    
    global opt, model
    opt = parser.parse_args()
    # print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        
    is_scale_back = False
    
    #--------------build models -------------------------
    model = IntroVAE(cdim=3, hdim=opt.hdim, emb_dim=emb_dim, channels=str_to_list(opt.channels), image_size=opt.output_height, conditional=conditional).cuda()
    ca_net = CA_NET().cuda()

    if opt.pretrained:
        load_model(model, opt.pretrained)
    # print(model)
            
    optimizerE = optim.Adam(model.encoder.parameters(), lr=opt.lr_e)
    optimizerG = optim.Adam(model.decoder.parameters(), lr=opt.lr_g)
    
    #-----------------load dataset--------------------------
    with open(data_root+'train/filenames.pickle', 'rb') as f:
        train_list = pickle.load(f)

    with open(data_root+'train/char-CNN-RNN-embeddings.pickle', 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            embeddings = u.load()

    assert len(train_list) > 0
    
    train_set = ImageDatasetFromFile(train_list, embeddings, opt.dataroot, input_height=None, crop_height=None, output_height=opt.output_height, is_mirror=True)     
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=opt.outf)
    
    start_time = time.time()
            
    cur_iter = 0
    
    def train(epoch, iteration, batch, cur_iter):  
        emb = batch[1].cuda()
        emb, mu, logvar = ca_net(emb)
        batch = batch[0]
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)
            
        batch_size = batch.size(0)
        
        noise = Variable(torch.zeros(batch_size, opt.hdim).normal_(0, 1)).cuda() 
               
        real= Variable(batch).cuda() 
        
        info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration, len(train_data_loader), time.time()-start_time)
        
        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'
            
        #=========== Update E ================ 
        fake = model.sample(noise, y_cond=emb)           
        real_mu, real_logvar, z, rec = model(real, o_cond=emb)

        rec_mu, rec_logvar = model.encode(rec.detach(), o_cond=emb.detach())
        fake_mu, fake_logvar = model.encode(fake.detach(), o_cond=emb.detach())
        
        loss_rec =  model.reconstruction_loss(rec, real, True)
        
        lossE_real_kl = model.kl_loss(real_mu, real_logvar).mean()
        lossE_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean()
        lossE_fake_kl = model.kl_loss(fake_mu, fake_logvar).mean()            
        loss_margin = lossE_real_kl + \
                      (F.relu(opt.m_plus-lossE_rec_kl) + \
                      F.relu(opt.m_plus-lossE_fake_kl)) * 0.5 * opt.weight_neg
        
                    
        lossE = loss_rec  * opt.weight_rec + loss_margin * opt.weight_kl
        optimizerG.zero_grad()
        optimizerE.zero_grad()       
        lossE.backward(retain_graph=True)
        # nn.utils.clip_grad_norm(model.encoder.parameters(), 1.0)            
        
        
        #========= Update G ==================   
        rec_mu, rec_logvar = model.encode(rec, o_cond=emb)
        fake_mu, fake_logvar = model.encode(fake, o_cond=emb)
        
        lossG_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean()
        lossG_fake_kl = model.kl_loss(fake_mu, fake_logvar).mean()
        
        lossG = (lossG_rec_kl + lossG_fake_kl)* 0.5 * opt.weight_kl      
        # lossG = (lossG_fake_kl)* 0.5 * opt.weight_kl      
                    
        # optimizerG.zero_grad()
        lossG.backward()
        # nn.utils.clip_grad_norm(model.decoder.parameters(), 1.0)
        optimizerE.step()
        optimizerG.step()
        
        info += 'Rec: {:.4f}, '.format(loss_rec.item())
        info += 'Kl_E: {:.4f}, {:.4f}, {:.4f}, '.format(lossE_real_kl.item(), 
                                lossE_rec_kl.item(), lossE_fake_kl.item())
        info += 'Kl_G: {:.4f}, {:.4f}, '.format(lossG_rec_kl.item(), lossG_fake_kl.item())
       
        
        if cur_iter % opt.test_iter == 0:            
            if opt.tensorboard:
                record_scalar(writer, eval(loss_info), loss_info, cur_iter)
                if cur_iter % 1000 == 0:
                    record_image(writer, [real, rec, fake], cur_iter)   
            else:
                max_imgs = min(batch.size(0), 16)
                vutils.save_image(torch.cat([real[:max_imgs], rec[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(), '{}/image_{}.jpg'.format(opt.outf, cur_iter),nrow=opt.nrow)

        if iteration == 0: 
            print(info)

        fake_kl = (lossE_fake_kl.item()+lossG_fake_kl.item())/2
        rec_kl = (lossE_rec_kl.item()+lossG_rec_kl.item())/2
        return lossE_real_kl.item(), fake_kl, rec_kl, loss_rec.item()
            
    
    #----------------Train by epochs--------------------------
    kls_real = []
    kls_fake = []
    kls_rec = []
    rec_errs = []
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):  
        #save models
        save_epoch = (epoch//opt.save_iter)*opt.save_iter   
        save_checkpoint(model, save_epoch, 0, '')
        model.train()
        
        batch_kls_real = []
        batch_kls_fake = []
        batch_kls_rec = []
        batch_rec_errs = []

        for iteration, batch in enumerate(train_data_loader, 0):
            #--------------train------------
            kl_real, kl_fake, kl_rec, rec_err = train(epoch, iteration, batch, cur_iter)
            cur_iter += 1

            batch_kls_real.append(kl_real)
            batch_kls_fake.append(kl_fake)
            batch_kls_rec.append(kl_rec)
            batch_rec_errs.append(rec_err)

        kls_real.append(np.mean(batch_kls_real))
        kls_fake.append(np.mean(batch_kls_fake))
        kls_rec.append(np.mean(batch_kls_rec))
        rec_errs.append(np.mean(batch_rec_errs))

        if epoch == opt.nEpochs:
            with torch.no_grad():

                emb = batch[1].cuda()
                emb, mu, logvar = ca_net(emb)
                batch = batch[0]

                real= Variable(batch).cuda() 
                real_mu, real_logvar, z, rec = model(real, o_cond=emb)

                batch_size = batch.size(0)

                noise = Variable(torch.zeros(batch_size, opt.hdim).normal_(0, 1)).cuda() 
                fake = model.sample(noise, y_cond=emb)
                max_imgs = min(batch.size(0), 16)
                vutils.save_image(torch.cat([real[:max_imgs], rec[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(), '{}/image_{}.jpg'.format(opt.outf, cur_iter),nrow=opt.nrow)
                    
            # plot graphs
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(np.arange(len(kls_real)), kls_real, label="kl_real")
            ax.plot(np.arange(len(kls_fake)), kls_fake, label="kl_fake")
            ax.plot(np.arange(len(kls_rec)), kls_rec, label="kl_rec")
            ax.plot(np.arange(len(rec_errs)), rec_errs, label="rec_err")
            ax.set_ylim([0, 200])
            ax.legend()
            plt.savefig(opt.outf+'/intro_vae_train_graphs.jpg')
            plt.show()
            
def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights['model'].state_dict()  
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
            
def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = "model/" + prefix +"model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()    