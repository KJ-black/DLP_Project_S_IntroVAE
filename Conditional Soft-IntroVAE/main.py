import torch
from train_soft_intro_vae import train_soft_intro_vae

if __name__ == "__main__":
    # hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    num_epochs = 200
    lr = 2e-4
    batch_size = 32
    beta_kl = 1.0
    beta_rec = 1.0
    beta_neg = 256
    img_size = 64
    model = train_soft_intro_vae(dataset='102flowers', z_dim=128, lr_e=2e-4, lr_d=2e-4, batch_size=batch_size,
                                num_workers=0, start_epoch=0, num_epochs=num_epochs, num_vae=0, save_interval=5000,
                                recon_loss_type="mse", beta_kl=beta_kl, beta_rec=beta_rec, beta_neg=beta_neg,
                                img_size=img_size, with_fid=True, test_iter=1000, seed=-1, pretrained=None, 
                                device=device)