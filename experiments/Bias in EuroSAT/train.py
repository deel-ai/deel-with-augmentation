import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from cnn_model import ResNet18, Vgg16, CNN, CNN1
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset_filtered_eurosat

device = "cuda" if torch.cuda.is_available() else "cpu"

ds_path = "/home/vuong.nguyen/vuong/augmentare/Bias in EuroSAT/data/EuroSAT"

X_train, S_train, y_train, X_test, S_test, y_test = load_dataset_filtered_eurosat(path=ds_path)

X_train = torch.from_numpy(X_train)
S_train = torch.from_numpy(S_train)
y_train = torch.from_numpy(y_train)

X_test = torch.from_numpy(X_test)
S_test = torch.from_numpy(S_test)
y_test = torch.from_numpy(y_test)

test_dataset = TensorDataset(X_test, y_test, S_test)
test_dataloader = DataLoader(test_dataset, batch_size=56, shuffle=False)

save_path = "/home/vuong.nguyen/vuong/augmentare/Bias in EuroSAT/NST_output/CCPL/ResNet18"

gen_blue_highway = torch.load("/home/vuong.nguyen/vuong/augmentare/Bias in EuroSAT/NST_output/CCPL/ccpl_bl_hw.pt")
gen_blue_river = torch.load("/home/vuong.nguyen/vuong/augmentare/Bias in EuroSAT/NST_output/CCPL/ccpl_bl_rv.pt")

trans = transforms.Compose([
    transforms.Resize(64)
])

gen_blue_highway = torch.stack(gen_blue_highway).float()
gen_blue_river = torch.stack(gen_blue_river).float()

img_list_blue_highway = trans(gen_blue_highway)
img_list_blue_river = trans(gen_blue_river)

nbs_highway = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
nbs_river = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

def main(test_set, num_epochs,
        X_train, S_train, y_train,
        new_images_1, nb_new_images_1, 
        new_images_2, nb_new_images_2,
        device, save_folder_path,
        type_1=['blue_highway', 'blue_river'],
        type_2=['blue_highway', 'blue_river']):
    
    seeds = [21, 42, 84, 168, 336, 745, 913, 1245, 1674, 2134]
    for k in range(len(seeds)):
        # (Re)set the seeds
        seed = seeds[k]
        os.environ['PYTHONHASHSEED'] = str(seed)
        # Torch RNG
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Python RNG
        np.random.seed(seed)
        random.seed(seed)

        accuracy = []
        error_rate = []
        history = []
        
        for i in range(len(nb_new_images_1)):
            new_acc = []
            new_err_rate = []
            new_history = []

            for j in range(len(nb_new_images_2)):
                if (type_1 == 'blue_highway') & (type_2 == 'blue_river'):
                    new_X_train = torch.cat((X_train, new_images_1[torch.randint(len(new_images_1), (nb_new_images_1[i],))], new_images_2[torch.randint(len(new_images_2), (nb_new_images_2[j],))]), 0)
                    new_S_train = torch.cat((S_train, torch.zeros(nb_new_images_1[i]), torch.zeros(nb_new_images_2[j])), 0)
                    new_y_train = torch.cat((y_train, torch.zeros(nb_new_images_1[i]).long(), torch.ones(nb_new_images_2[j]).long()), 0)
                elif (type_1 == 'blue_river') & (type_2 == 'blue_highway'):
                    new_X_train = torch.cat((X_train, new_images_1[torch.randint(len(new_images_1), (nb_new_images_1[i],))], new_images_2[torch.randint(len(new_images_2), (nb_new_images_2[j],))]), 0)
                    new_S_train = torch.cat((S_train, torch.zeros(nb_new_images_1[i]), torch.zeros(nb_new_images_2[j])), 0)
                    new_y_train = torch.cat((y_train, torch.ones(nb_new_images_1[i]).long(), torch.zeros(nb_new_images_2[j]).long()), 0)

                train_dataset = TensorDataset(new_X_train, new_y_train, new_S_train)
                train_dataloader = DataLoader(train_dataset, batch_size=56, shuffle=True)

                #new_path = f"{save_folder_path}/checkpoint_{type_1}_{nb_new_images_1[i]}_{type_2}_{nb_new_images_2[j]}_seed_{k+1}.pt"
                new_path = f"{save_folder_path}/checkpoint.pt"

                cnn_model = ResNet18(device=device)
                
                train_acc, train_losses, val_acc, val_losses = cnn_model.train(train_loader=train_dataloader, valid_loader=test_set, learning_rate=0.00008, num_epochs=num_epochs, save_path=new_path)
                _, acc, _, ratio_wrong = cnn_model.test(test_loader=test_set, test_gobal=True)
                
                his = [train_acc, train_losses, val_acc, val_losses]

                new_acc.append(acc)
                new_err_rate.append(ratio_wrong)
                new_history.append(his)
                print(f"=======>>>   ======>>>   ======>>>    ======>>>   ======>>>   =======>>>")
                torch.cuda.empty_cache()
                
            accuracy.append(new_acc)
            error_rate.append(new_err_rate)
            history.append(new_history)

        np.save(f"{save_folder_path}/accuracy_seed_{k+1}.npy", accuracy)
        np.save(f"{save_folder_path}/error_rate_seed_{k+1}.npy", error_rate)
        np.save(f"{save_folder_path}/history_seed_{k+1}.npy", history)
    print("Doneeeeee!")
    
if __name__ == '__main__':
    main(test_set=test_dataloader, num_epochs=500,
        X_train=X_train, S_train=S_train, y_train=y_train,
        new_images_1=img_list_blue_highway, nb_new_images_1=nbs_highway, 
        new_images_2=img_list_blue_river, nb_new_images_2=nbs_river,
        device=device, save_folder_path=save_path,
        type_1='blue_highway',
        type_2='blue_river')
