import os
import random
import numpy as np
import pandas as pd
import torch
from model import ResNet18, Vgg16
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils import CelebDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

image_path = "/home/vuong.nguyen/vuong/augmentare/Bias Celeba/dataset/img_align_celeba/img_align_celeba"

train_df=pd.read_csv('/home/vuong.nguyen/vuong/augmentare/Bias Celeba/train_df.csv')
valid_df=pd.read_csv('/home/vuong.nguyen/vuong/augmentare/Bias Celeba/valid_df.csv')
test_df=pd.read_csv('/home/vuong.nguyen/vuong/augmentare/Bias Celeba/test_df.csv')

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

def undersampling(train_df, per_under):
    df_male_blond = train_df[(train_df['Male']==1) & (train_df['Sens']==2)]
    df_male = train_df[train_df['Male']==1]

    num_male_blond = len(df_male_blond)
    num_male = len(df_male)

    df_female_gray = train_df[(train_df['Male']==0) & (train_df['Sens']==4)]
    df_female = train_df[train_df['Male']==0]

    num_female_gray = len(df_female_gray)
    num_female = len(df_female)

    num_female_under = round(num_female_gray*100/per_under)
    num_male_under = round(num_male_blond*100/per_under)

    df_total = train_df
    df_male_under = train_df
    df_female_under = train_df

    # Under female
    df_female_under = df_female_under.drop(df_female_under[df_female_under['Male']==0].sample(n=num_female-num_female_under).index)

    # Under male
    df_male_under = df_male_under.drop(df_male_under[df_male_under['Male']==1].sample(n=num_male-num_male_under).index)

    # Under total
    df_total = df_total.drop(df_total[(df_total['Male']==0) & (~df_total['Sens'].isin([4]))].sample(n=num_female-num_female_under).index)
    df_total = df_total.drop(df_total[(df_total['Male']==1) & (~df_total['Sens'].isin([2]))].sample(n=num_male-num_male_under).index)

    return df_female_under, df_male_under, df_total

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

        df_female_under, df_male_under, df_total = undersampling(train_df, per)

        # female_data=CelebDataset(df_female_under, image_path, train_transform)
        # female_dataloader=DataLoader(female_data, batch_size=16, shuffle=True, num_workers=4)

        # female_model = ResNet18(device=device)
        # female_train_acc, female_train_losses, female_val_acc, female_val_losses = female_model.train(
        #     train_loader=female_dataloader, valid_loader=valid_dataloader,
        #     learning_rate=0.00008, num_epochs=num_epochs,
        #     save_path=f"{save_folder_path}/{per}/female_under.pt"
        # )

        # _, female_acc, _, female_ratio_wrong = female_model.test(test_loader=test_dataloader, test_gobal=True)

        # female_his = [female_train_acc, female_train_losses, female_val_acc, female_val_losses]

        # print(f"=======>>>   ======>>>   ======>>>    ======>>>   ======>>>   =======>>>")
        # torch.cuda.empty_cache()

        # np.save(f"{save_folder_path}/{per}/accuracy_seed_female.npy", female_acc)
        # np.save(f"{save_folder_path}/{per}/error_rate_seed_female.npy", female_ratio_wrong)
        # np.save(f"{save_folder_path}/{per}/history_seed_female.npy", female_his)

        # male_data=CelebDataset(df_male_under, image_path, train_transform)
        # male_dataloader=DataLoader(male_data, batch_size=16, shuffle=True, num_workers=4)

        # male_model = ResNet18(device=device)
        # male_train_acc, male_train_losses, male_val_acc, male_val_losses = male_model.train(
        #     train_loader=male_dataloader, valid_loader=valid_dataloader,
        #     learning_rate=0.00008, num_epochs=num_epochs,
        #     save_path=f"{save_folder_path}/{per}/male_under.pt"
        # )

        # _, male_acc, _, male_ratio_wrong = male_model.test(test_loader=test_dataloader, test_gobal=True)

        # male_his = [male_train_acc, male_train_losses, male_val_acc, male_val_losses]

        # print(f"=======>>>   ======>>>   ======>>>    ======>>>   ======>>>   =======>>>")
        # torch.cuda.empty_cache()

        # np.save(f"{save_folder_path}/{per}/accuracy_seed_male.npy", male_acc)
        # np.save(f"{save_folder_path}/{per}/error_rate_seed_male.npy", male_ratio_wrong)
        # np.save(f"{save_folder_path}/{per}/history_seed_male.npy", male_his)

        total_data=CelebDataset(df_total, image_path, train_transform)
        total_dataloader=DataLoader(total_data, batch_size=16, shuffle=True, num_workers=4)

        total_model = ResNet18(device=device)
        total_train_acc, total_train_losses, total_val_acc, total_val_losses = total_model.train(
            train_loader=total_dataloader, valid_loader=valid_dataloader,
            learning_rate=0.00008, num_epochs=num_epochs,
            save_path=f"{save_folder_path}/{per}/total_under.pt"
        )

        # _, total_acc, _, total_ratio_wrong = total_model.test(test_loader=test_dataloader, test_gobal=True)
        total_his = [total_train_acc, total_train_losses, total_val_acc, total_val_losses]

        print(f"=======>>>   ======>>>   ======>>>    ======>>>   ======>>>   =======>>>")
        torch.cuda.empty_cache()

        # np.save(f"{save_folder_path}/{per}/accuracy_seed_total.npy", total_acc)
        # np.save(f"{save_folder_path}/{per}/error_rate_seed_total.npy", total_ratio_wrong)
        np.save(f"{save_folder_path}/{per}/history_seed_total.npy", total_his)
    print("Doneeeeee!")

save_path = '/home/vuong.nguyen/vuong/augmentare/Bias Celeba/Outputs_ResNet18/Undersampling'
if __name__ == '__main__':
    main(num_epochs=100, device=device, save_folder_path=save_path)
