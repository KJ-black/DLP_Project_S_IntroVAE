# imports for the tutorial
import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.utils as vutils

# other Functions
from soft_intro_vae import SoftIntroVAE, calc_kl, calc_reconstruction_loss, save_checkpoint, load_model, reparameterize
from dataset import ImageDatasetFromFile
from metrics.fid_score import calculate_fid_given_dataset

def train_soft_intro_vae(dataset='cifar10', z_dim=128, lr_e=2e-4, lr_d=2e-4, batch_size=128, num_workers=4, start_epoch=0,
                       num_epochs=250, num_vae=0, save_interval=5000, recon_loss_type="mse",
                       beta_kl=1.0, beta_rec=1.0, beta_neg=1.0, img_size=32, with_fid = False, test_iter=1000, seed=-1, 
                       pretrained=None, device=torch.device("cpu"), num_row=8, gamma_r=1e-8):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)

    # --------------build models -------------------------
    if dataset == '102flowers':
        image_size = img_size
        channels = [64, 128, 256]
        data_root = 'Data/flowers/'
        output_height = img_size
        with open(data_root+'train/filenames.pickle', 'rb') as f:
            train_list = pickle.load(f)
        train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True)
        ch = 3

    else:
        raise NotImplementedError("dataset is not supported")

    model = SoftIntroVAE(cdim=ch, zdim=z_dim, channels=channels, image_size=image_size).to(device)
    if pretrained is not None:
        load_model(model, pretrained, device)
    # print(model)

    optimizer_e = optim.Adam(model.encoder.parameters(), lr=lr_e)
    optimizer_d = optim.Adam(model.decoder.parameters(), lr=lr_d)

    e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=(350,), gamma=0.1)
    d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=(350,), gamma=0.1)

    scale = 1 / (ch * image_size ** 2)  # normalizing constant, 's' in the paper

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    start_time = time.time()

    cur_iter = 0
    kls_real = []
    kls_fake = []
    kls_rec = []
    rec_errs = []
    best_fid = None
    
    for epoch in range(start_epoch, num_epochs):

        if with_fid and ((epoch == 0) or (epoch >= 100 and epoch % 20 == 0) or epoch == num_epochs - 1):
            with torch.no_grad():
                print("calculating fid...")
                fid = calculate_fid_given_dataset(train_data_loader, model, batch_size, cuda=True, dims=2048,
                                                  device=device, num_images=50000)
                print("fid:", fid)
                if best_fid is None:
                    best_fid = fid
                elif best_fid > fid:
                    print("best fid updated: {} -> {}".format(best_fid, fid))
                    best_fid = fid
                    # save
                    save_epoch = epoch
                    prefix = dataset + "_soft_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
                        beta_rec) + "_" + "fid_" + str(fid) + "_"
                    save_checkpoint(model, save_epoch, cur_iter, prefix)


        diff_kls = []
        # save models
        if epoch % save_interval == 0 and epoch > 0:
            save_epoch = (epoch // save_interval) * save_interval
            prefix = dataset + "_soft_intro_vae" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
                beta_rec) + "_"
            save_checkpoint(model, save_epoch, cur_iter, prefix)

        model.train()
        batch_kls_real = []
        batch_kls_fake = []
        batch_kls_rec = []
        batch_rec_errs = []

        for iteration, batch in enumerate(train_data_loader, 0):
            # --------------train------------
            # soft-intro-vae training
            if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)

            b_size = batch.size(0)
            
            # generate random noise to produce 'fake' later
            noise_batch = torch.randn(size=(b_size, z_dim)).to(device)
            real_batch = batch.to(device)

            # =========== Update E ================
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.decoder.parameters():
                param.requires_grad = False
            
            # generate 'fake' data
            fake = model.sample(noise_batch)
            
            # ELBO for real data
            real_mu, real_logvar = model.encode(real_batch)
            z = reparameterize(real_mu, real_logvar)
            rec = model.decoder(z)

            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
            lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")

            # prepare 'fake' data for expELBO
            rec_mu, rec_logvar, z_rec, rec_rec = model(rec.detach())
            fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())
            
            # KLD loss for the fake data
            fake_kl_e = calc_kl(fake_logvar, fake_mu, reduce="none")
            rec_kl_e = calc_kl(rec_logvar, rec_mu, reduce="none")

            # reconstruction loss for the fake data
            loss_fake_rec = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction="none")
            loss_rec_rec = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction="none")

            # expELBO
            exp_elbo_fake = (-2 * scale * (beta_rec * loss_fake_rec + beta_neg * fake_kl_e)).exp().mean()
            exp_elbo_rec = (-2 * scale * (beta_rec * loss_rec_rec + beta_neg * rec_kl_e)).exp().mean()

            # total loss
            lossE = scale * (beta_rec * loss_rec + beta_kl * lossE_real_kl) + 0.25 * (exp_elbo_fake + exp_elbo_rec)
            
            # backprop
            optimizer_e.zero_grad()
            lossE.backward()
            optimizer_e.step()

            # ========= Update D ==================
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = True

            # generate 'fake' data
            fake = model.sample(noise_batch)
            rec = model.decoder(z.detach())
            
                # ELBO loss for real -- just the reconstruction, KLD for real doesn't affect the decoder
            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")

            # prepare 'fake' data for the ELBO
            rec_mu, rec_logvar = model.encode(rec)
            z_rec = reparameterize(rec_mu, rec_logvar)

            fake_mu, fake_logvar = model.encode(fake)
            z_fake = reparameterize(fake_mu, fake_logvar)

            rec_rec = model.decode(z_rec.detach())
            rec_fake = model.decode(z_fake.detach())

            loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type=recon_loss_type,
                                                    reduction="mean")
            loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type,
                                                        reduction="mean")

            rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
            fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")

            lossD = scale * (loss_rec * beta_rec + (rec_kl + fake_kl) * 0.5 * beta_kl + \
                                            gamma_r * 0.5 * beta_rec * (loss_rec_rec + loss_fake_rec))

            optimizer_d.zero_grad()
            lossD.backward()
            optimizer_d.step()
            if torch.isnan(lossD) or torch.isnan(lossE):
                raise SystemError

            # statistics for plotting later
            diff_kls.append(-lossE_real_kl.data.cpu().item() + fake_kl.data.cpu().item())
            batch_kls_real.append(lossE_real_kl.data.cpu().item())
            batch_kls_fake.append(fake_kl.cpu().item())
            batch_kls_rec.append(rec_kl.data.cpu().item())
            batch_rec_errs.append(loss_rec.data.cpu().item())
            
            if cur_iter % test_iter == 0:
                info = "\nEpoch[{}]({}/{}): time: {:4.4f}: ".format(epoch, iteration, len(train_data_loader), 
                                                                    time.time() - start_time)
                info += 'Rec: {:.4f}, '.format(loss_rec.data.cpu())
                info += 'Kl_E: {:.4f}, expELBO_R: {:.4e}, expELBO_F: {:.4e}, '.format(lossE_real_kl.data.cpu(),
                                                                                exp_elbo_rec.data.cpu(),
                                                                                exp_elbo_fake.cpu())
                info += 'Kl_F: {:.4f}, KL_R: {:.4f}'.format(rec_kl.data.cpu(), fake_kl.data.cpu())
                info += ' DIFF_Kl_F: {:.4f}'.format(-lossE_real_kl.data.cpu() + fake_kl.data.cpu())
                print(info)

                _, _, _, rec_det = model(real_batch, deterministic=True)
                max_imgs = min(batch.size(0), 16)
                vutils.save_image(
                        torch.cat([real_batch[:max_imgs], rec_det[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(),
                        '{}/image_{}.jpg'.format("train_record/", cur_iter), nrow=num_row)                 
            cur_iter += 1
        e_scheduler.step()
        d_scheduler.step()
        
        if epoch > num_vae - 1:
            kls_real.append(np.mean(batch_kls_real))
            kls_fake.append(np.mean(batch_kls_fake))
            kls_rec.append(np.mean(batch_kls_rec))
            rec_errs.append(np.mean(batch_rec_errs))

        if epoch == num_epochs - 1:
            with torch.no_grad():
                _, _, _, rec_det = model(real_batch, deterministic=True)
                noise_batch = torch.randn(size=(b_size, z_dim)).to(device)
                fake = model.sample(noise_batch)
                max_imgs = min(batch.size(0), 16)
                vutils.save_image(
                        torch.cat([real_batch[:max_imgs], rec_det[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(),
                        '{}/image_{}.jpg'.format("train_record/", cur_iter), nrow=num_row)
                    
            # plot graphs
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(np.arange(len(kls_real)), kls_real, label="kl_real")
            ax.plot(np.arange(len(kls_fake)), kls_fake, label="kl_fake")
            ax.plot(np.arange(len(kls_rec)), kls_rec, label="kl_rec")
            ax.plot(np.arange(len(rec_errs)), rec_errs, label="rec_err")
            ax.set_ylim([0, 200])
            ax.legend()
            plt.savefig('./soft_intro_vae_train_graphs.jpg')
            # save models
            prefix = dataset + "_soft_intro_vae" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
                beta_rec) + "_"
            save_checkpoint(model, epoch, cur_iter, prefix)
            plt.show()
    return model


