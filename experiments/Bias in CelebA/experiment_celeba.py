import os
import random
import numpy as np
import pandas as pd
import torch
from model import ResNet18, Vgg16
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import transforms 
from utils import CelebDataset

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

image_path = "/home/vuong.nguyen/vuong/augmentare/Bias Celeba/dataset/img_align_celeba"

train_df=pd.read_csv('/home/vuong.nguyen/vuong/augmentare/Bias Celeba/dataframe_dataset/train_df.csv')
valid_df=pd.read_csv('/home/vuong.nguyen/vuong/augmentare/Bias Celeba/dataframe_dataset/valid_df.csv')
test_df=pd.read_csv('/home/vuong.nguyen/vuong/augmentare/Bias Celeba/dataframe_dataset/test_df.csv')

# Apply Data augmentation and different type of transforms on train data.
train_transform = transforms.Compose([transforms.Resize((224,224)),
                                    #transforms.RandomVerticalFlip(p=0.5),
                                    #transforms.RandomHorizontalFlip(p=0.5),
                                    #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.5063, 0.4258, 0.3832],std=[0.2644, 0.2436, 0.2397])
                                    ])

# Apply Data augmentation and different type of transforms on test and validation data.
valid_transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.5063, 0.4258, 0.3832],std=[0.2644, 0.2436, 0.2397])
                                    ])

valid_data=CelebDataset(valid_df,image_path,valid_transform)
valid_dataloader=DataLoader(valid_data,batch_size=16,num_workers=4)

test_data=CelebDataset(test_df,image_path,valid_transform)
test_dataloader=DataLoader(test_data,batch_size=16,num_workers=4)

def augmentare(train_df, new_df, per, mode="all", new_df_2=None):

    if mode == "male":
        df_male_blond = train_df[(train_df['Male']==1) & (train_df['Sens']==2)]
        df_male = train_df[train_df['Male']==1]

        num_male_blond = len(df_male_blond)
        num_male = len(df_male)

        num_male_aug = round(num_male*per/100)
        num_add = num_male_aug - num_male_blond

        if num_add < len(new_df):
            new_frame = new_df.sample(num_add)

        elif num_add > len(new_df):
            nb_male = num_add//len(new_df)
            new_frame = new_df

            for _ in range(nb_male-1):
                data = [new_frame, new_df]
                new_frame = pd.concat(data)

            if num_add - len(new_frame) != 0:
                data = [new_frame, new_df.sample(num_add - len(new_frame))]
                new_frame = pd.concat(data)

    elif mode == "female":
        df_female_gray = train_df[(train_df['Male']==0) & (train_df['Sens']==4)]
        df_female = train_df[train_df['Male']==0]

        num_female_gray = len(df_female_gray)
        num_female = len(df_female)

        num_female_aug = round(num_female*per/100)
        num_add = num_female_aug - num_female_gray

        if num_add < len(new_df):
            new_frame = new_df.sample(num_add)

        elif num_add > len(new_df):
            nb_female = num_add//len(new_df)
            new_frame = new_df

            for _ in range(nb_female-1):
                data = [new_frame, new_df]
                new_frame = pd.concat(data)

            if num_add - len(new_frame) != 0:
                data = [new_frame, new_df.sample(num_add - len(new_frame))]
                new_frame = pd.concat(data)
    
    elif mode == "all":
        df_male_blond = train_df[(train_df['Male']==1) & (train_df['Sens']==2)]
        df_male = train_df[train_df['Male']==1]

        num_male_blond = len(df_male_blond)
        num_male = len(df_male)

        num_male_aug = round(num_male*per/100)
        num_male_add = num_male_aug - num_male_blond

        df_female_gray = train_df[(train_df['Male']==0) & (train_df['Sens']==4)]
        df_female = train_df[train_df['Male']==0]

        num_female_gray = len(df_female_gray)
        num_female = len(df_female)

        num_female_aug = round(num_female*per/100)
        num_female_add = num_female_aug - num_female_gray
        
        ### Male
        if num_male_add < len(new_df):
            new_frame_male = new_df.sample(num_male_add)

        elif num_male_add > len(new_df):
            nb_male = num_male_add//len(new_df)
            new_frame_male = new_df

            for _ in range(nb_male-1):
                data = [new_frame_male, new_df]
                new_frame_male = pd.concat(data)

            if num_male_add - len(new_frame_male) != 0:
                data = [new_frame_male, new_df.sample(num_male_add - len(new_frame_male))]
                new_frame_male = pd.concat(data)

        ### Female
        if num_female_add < len(new_df_2):
            new_frame_female = new_df_2.sample(num_female_add)

        elif num_female_add > len(new_df_2):
            nb_female = num_female_add//len(new_df_2)
            new_frame_female = new_df_2

            for _ in range(nb_female-1):
                data = [new_frame_female, new_df_2]
                new_frame_female = pd.concat(data)

            if num_female_add - len(new_frame_female) != 0:
                data = [new_frame_female, new_df_2.sample(num_female_add - len(new_frame_female))]
                new_frame_female = pd.concat(data)

        new_frame = pd.concat([new_frame_male, new_frame_female])
    
    return new_frame

def main(num_epochs, device, save_folder_path):
    list_per = [5, 10, 15, 20]
    for per in list_per:
        seed = 42
        # (Re)set the seeds
        os.environ['PYTHONHASHSEED'] = str(seed)
        # Torch RNG
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Python RNG
        np.random.seed(seed)
        random.seed(seed)

        mode = "female"
        method = "CycleGAN_hair_color" 

        if mode == "male":
            new_path = f"/home/vuong.nguyen/vuong/augmentare/Bias Celeba/new_dataset/{method}/male_blond"
            new_df = pd.read_csv(f"/home/vuong.nguyen/vuong/augmentare/Bias Celeba/new_dataset/{method}/new_male_blond.csv")
            
            new_frame = augmentare(train_df, new_df, per, mode="male", new_df_2=None)
            print(len(new_frame))
            new_dataset = CelebDataset(new_frame, new_path, train_transform)

        elif mode == "female":
            new_path = f"/home/vuong.nguyen/vuong/augmentare/Bias Celeba/new_dataset/{method}/female_gray"
            new_df = pd.read_csv(f"/home/vuong.nguyen/vuong/augmentare/Bias Celeba/new_dataset/{method}/new_female_gray.csv")

            new_frame = augmentare(train_df, new_df, per, mode="female", new_df_2=None)
            print(len(new_frame))
            new_dataset = CelebDataset(new_frame, new_path, train_transform)

        elif mode == "all":
            male_path = f"/home/vuong.nguyen/vuong/augmentare/Bias Celeba/new_dataset/{method}/male_blond"
            male_df = pd.read_csv(f"/home/vuong.nguyen/vuong/augmentare/Bias Celeba/new_dataset/{method}/new_male_blond.csv")

            female_path = f"/home/vuong.nguyen/vuong/augmentare/Bias Celeba/new_dataset/{method}/female_gray"
            female_df = pd.read_csv(f"/home/vuong.nguyen/vuong/augmentare/Bias Celeba/new_dataset/{method}/new_female_gray.csv")

            male_frame = augmentare(train_df, male_df, per, mode="male", new_df_2=None)
            print(len(male_frame))
            male_dataset = CelebDataset(male_frame, male_path, train_transform)

            female_frame = augmentare(train_df, female_df, per, mode="female", new_df_2=None)
            print(len(female_frame))
            female_dataset = CelebDataset(female_frame, female_path, train_transform)

            new_dataset = ConcatDataset((male_dataset, female_dataset))
        

        old_dataset = CelebDataset(train_df, image_path, train_transform)

        dataset = ConcatDataset((old_dataset, new_dataset))

        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

        model = ResNet18(device=device)
        train_acc, train_losses, val_acc, val_losses = model.train(
            train_loader=dataloader, valid_loader=valid_dataloader,
            learning_rate=0.00008, num_epochs=num_epochs,
            save_path=f"{save_folder_path}/{method}/{per}/{mode}_{method}.pt"
        )
        
        # _, acc, _, ratio_wrong = model.test(test_loader=test_dataloader, test_gobal=True)
        his = [train_acc, train_losses, val_acc, val_losses]

        print(f"=======>>>   ======>>>   ======>>>    ======>>>   ======>>>   =======>>>")
        torch.cuda.empty_cache()

        # np.save(f"{save_folder_path}/{per}/accuracy_seed_total.npy", acc)
        # np.save(f"{save_folder_path}/{per}/error_rate_seed_total.npy", ratio_wrong)
        np.save(f"{save_folder_path}/{method}/{per}/history_seed_{mode}.npy", his)

    print("Doneeeeee!")

save_path = '/home/vuong.nguyen/vuong/augmentare/Bias Celeba/Outputs_ResNet18'
if __name__ == '__main__':
    main(num_epochs=100, device=device, save_folder_path=save_path)
