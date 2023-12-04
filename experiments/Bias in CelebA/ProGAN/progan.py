# General Imports
import os
import numpy as np
import pandas as pd
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Pytorch and Torchvision Imports
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Augmentare Imports
import augmentare
from augmentare.methods.gan import *

#Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_df = pd.read_csv('dataframe_dataset/df_female_gray.csv')

image_path = "/home/vuong.nguyen/vuong/augmentare/Bias Celeba/dataset/img_align_celeba/img_align_celeba"

class CelebDataset(Dataset):
    def __init__(self, df, image_path, transform=None, mode='train'):
        super().__init__()
        self.sen = df['Sens']
        self.label = df['Male']
        self.path = image_path
        self.image_id = df['image_id']
        self.transform=transform
        self.mode=mode

    def __len__(self):
        return self.image_id.shape[0]

    def __getitem__(self, idx:int):
        image_name = self.image_id.iloc[idx]
        image = Image.open(os.path.join(image_path, image_name))
        labels = np.asarray(self.label.iloc[idx].T)
        sens = np.asarray(self.sen.iloc[idx].T)

        if self.transform:
            image=self.transform(image)
        return image, labels

image_size = 256
transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])  

dataset=CelebDataset(train_df,image_path,transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

net_gen = PROGANGenerator(
    latent_size=256,
    in_channels=256,
    img_channels=3,
    alpha=1e-5,
    steps=4
)

net_dis = PROGANDiscriminator(
    in_channels=256,
    img_channels=3,
    alpha=1e-5,
    steps=4
)

optimizer_gen = Adam(net_gen.parameters(), lr=1e-3, betas=(0.0, 0.999))
optimizer_dis = Adam(net_dis.parameters(), lr=1e-3, betas=(0.0, 0.999))
loss_fn_gen =  torch.cuda.amp.GradScaler()
loss_fn_dis =  torch.cuda.amp.GradScaler()

# Create GAN network
gan = PROGAN(
    net_gen,
    net_dis,
    optimizer_gen,
    optimizer_dis,
    loss_fn_gen,
    loss_fn_dis,
    device,
    latent_size=256
)

gen_losses, dis_losses = gan.train(
    subset_a=dataloader,
    num_epochs=200,
    num_decay_epochs=None,
    num_classes = None,
    batch_size = [32, 32, 32, 16, 16, 16, 16, 8, 4],
    subset_b = None
)

torch.save(gan, f"/home/vuong.nguyen/vuong/augmentare/Bias Celeba/Weights_GAN/pro_gan_female_gray_200.pth")
torch.cuda.empty_cache()

img_list, img = gan.generate_samples(
    nb_samples = 30,
    num_classes = None,
    real_image_a = None,
    real_image_b = None
)

torch.save(img_list, f"/home/vuong.nguyen/vuong/augmentare/Bias Celeba/Outputs_GAN/ProGAN/female_gray_200.pt")
