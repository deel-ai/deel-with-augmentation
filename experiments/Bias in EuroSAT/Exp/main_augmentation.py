import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from model import ResNet18, Vgg16
from torch.utils.data import TensorDataset, DataLoader
from utils import EarlyStopping
from dataset import load_dataset_filtered_eurosat

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

def train(model, train_loader, valid_loader, learning_rate, num_epochs, save_path):
        # Set loss function with criterion
        criterion = nn.CrossEntropyLoss()
        # Optimizer Adam
        #optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        # Optimizer SGD
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        # List to store the train and validation accuracy
        train_acc = []
        train_losses = []
        val_acc = []
        val_losses = []

        early_stopping = EarlyStopping(patience=100, verbose=True, path=save_path)
        for epoch in range(num_epochs):
            # Forward pass
            model.train()
            train_loss, correct, cnt_acc, cnt = 0, 0, 0, 0
            for i, (images, labels, sensibilities) in enumerate(train_loader):
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)
                sensibilities = sensibilities.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                with torch.no_grad():
                    train_loss += loss.item()
                    correct += (outputs.argmax(1) == labels).sum().item()
                    cnt_acc += outputs.shape[0]
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cnt = i+1
            
            exp_lr_scheduler.step()
            # Calculate the average loss and accuracy train
            avg_loss_train = train_loss/cnt
            per_crr_train = (correct/cnt_acc)*100
            train_acc.append(per_crr_train)
            train_losses.append(avg_loss_train)
            
            model.eval()
            val_loss, val_correct, val_cnt_acc, val_cnt = 0, 0, 0, 0
            for i, (img, lab, sen) in enumerate(valid_loader):
                img = img.to(device)
                lab = lab.to(device)
                sen = sen.to(device)
                val_pred = model(img)
                loss = criterion(val_pred, lab)
                val_correct += (val_pred.argmax(1) == lab).sum().item()
                val_cnt_acc += val_pred.shape[0]
                val_loss += loss.item()
                val_cnt = i+1

            # Calculate the average loss and accuracy validation
            avg_loss_val = val_loss/val_cnt
            per_crr_val = (val_correct/val_cnt_acc)*100
            val_acc.append(per_crr_val)
            val_losses.append(avg_loss_val)

            # Print the epoch and it's accuracy and loss
            print('Epoch [{}/{}], Train loss: {:.4f}, Val loss: {:.4f}, Train accuracy: {:.4f}%, Val accuracy: {:.4f}%'.format(epoch+1, num_epochs, avg_loss_train, avg_loss_val, per_crr_train, per_crr_val))
        
            early_stopping(avg_loss_val, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        return train_acc, train_losses, val_acc, val_losses

def test(model, model_path, test_loader, test_gobal=True):
        # Load the last checkpoint with the best model
        model.load_state_dict(torch.load(model_path))
        model.eval()
        criterion = nn.CrossEntropyLoss()
        test_target = []
        test_predict = []

        wrong_pred_hw = []
        wrong_pred_rv = []
        wrong_blue_pred_hw = []
        wrong_blue_pred_rv = []

        nb_bl_hw = 0
        nb_bl_rv  = 0
        nb_nor_hw = 0
        nb_nor_rv  = 0
        nb_bl = 0
        nb_nor = 0

        nb_wrong = 0
        nb_wrong_blue_hw = 0
        nb_wrong_blue_rv = 0
        nb_wrong_blue = 0
        nb_wrong_normal_hw = 0
        nb_wrong_normal_rv = 0

        with torch.no_grad():
            cnt, val_loss, correct, cnt_acc = 0, 0, 0, 0
            for i, (img, lab, sen) in enumerate(test_loader):
                test_target.extend(list(lab))
                img = img.to(device)
                lab = lab.to(device)
                sen = sen.to(device)
                val_pred, _ = model(img)
                pred_test = torch.argmax(val_pred, dim = 1).to(torch.device("cpu")).numpy()
                loss = criterion(val_pred, lab)
                correct += (val_pred.argmax(1) == lab).sum().item()
                cnt_acc += val_pred.shape[0]
                val_loss += loss.item()
                cnt = i+1
                test_predict.extend(list(pred_test))

                # Calculating size of image groups
                for i in range(len(lab)):
                    if sen[i] == 0:
                        nb_bl += 1
                        if lab[i] == 0:
                            nb_bl_hw += 1
                        elif lab[i] == 1:
                            nb_bl_rv += 1
                    elif sen[i] == 1:
                        nb_nor += 1
                        if lab[i] == 0:
                            nb_nor_hw += 1
                        elif lab[i] == 1:
                            nb_nor_rv += 1

                # Checking for different predicted images
                for i in range(pred_test.shape[0]):
                    if pred_test[i] != lab[i]:
                        # Passing the actual image, predicted image, labels, max of val_predict
                        nb_wrong += 1
                        if sen[i] == 0:
                            nb_wrong_blue += 1
                            if lab[i] == 0:
                                wrong_blue_pred_hw.append([pred_test[i],img[i],lab[i],torch.max(val_pred)])
                                nb_wrong_blue_hw += 1
                            elif lab[i] == 1:
                                wrong_blue_pred_rv.append([pred_test[i],img[i],lab[i],torch.max(val_pred)])
                                nb_wrong_blue_rv += 1
                        elif sen[i] == 1:
                            if lab[i] == 0:
                                wrong_pred_hw.append([pred_test[i],img[i],lab[i],torch.max(val_pred)])
                                nb_wrong_normal_hw += 1
                            elif lab[i] == 1:
                                wrong_pred_rv.append([pred_test[i],img[i],lab[i],torch.max(val_pred)])
                                nb_wrong_normal_rv += 1
            nb_hw = nb_bl_hw + nb_nor_hw
            nb_rv = nb_bl_rv + nb_nor_rv

            nb_wrong_hw = nb_wrong_blue_hw + nb_wrong_normal_hw
            nb_wrong_rv = nb_wrong_blue_rv + nb_wrong_normal_rv
            nb_wrong_nor = nb_wrong - nb_wrong_blue
            # Calculate the average loss and accuracy
            avg_loss = val_loss/cnt
            per_crr = (correct/cnt_acc)*100
            if test_gobal is True:
                err_bl_hw = (nb_wrong_blue_hw/nb_bl_hw)*100
                err_bl_rv = (nb_wrong_blue_rv/nb_bl_rv)*100

                err_bl = (nb_wrong_blue/nb_bl)*100
                err_nor = (nb_wrong_nor/nb_nor)*100

                err_nor_hw = (nb_wrong_normal_hw/nb_nor_hw)*100
                err_nor_rv = (nb_wrong_normal_rv/nb_nor_rv)*100

                err_hw = (nb_wrong_hw/nb_hw)*100
                err_rv = (nb_wrong_rv/nb_rv)*100

                err_rate = [err_bl_hw, err_bl_rv, err_bl, err_nor, err_nor_hw, err_nor_rv, err_hw, err_rv]
                # Print loss and accuracy
                print('Test loss: {:.4f}, Accuracy: {:.4f}%, Error rate of blue images: {:.4f}%'.format(avg_loss, per_crr, err_bl))
            else:
                # Print loss and accuracy
                print('Test loss: {:.4f}, Accuracy: {:.4f}%'.format(avg_loss, per_crr))
        
        if test_gobal is True:
            dict = {"test_target" : test_target, "test_pred" : test_predict,
                    "wrong_pred_hw" : wrong_pred_hw, "wrong_pred_rv" : wrong_pred_rv, 
                    "wrong_blue_pred_hw" : wrong_blue_pred_hw, "wrong_blue_pred_rv" : wrong_blue_pred_rv}
            return avg_loss, per_crr, dict, err_rate
        return avg_loss, per_crr

gen_blue_highway = torch.load("/home/vuong.nguyen/vuong/augmentare/Bias in EuroSAT/NST_output/Flow/flow_bl_hw.pt")
gen_blue_river = torch.load("/home/vuong.nguyen/vuong/augmentare/Bias in EuroSAT/NST_output/Flow/flow_bl_rv.pt")

trans = transforms.Compose([
    transforms.Resize(64)
])

gen_blue_highway = torch.stack(gen_blue_highway).float()
gen_blue_river = torch.stack(gen_blue_river).float()

img_list_blue_highway = trans(gen_blue_highway)
img_list_blue_river = trans(gen_blue_river)

nbs_highway = 400
nbs_river = 1000

def main(test_set, num_epochs,
        X_train, S_train, y_train,
        new_images_1, nb_new_images_1, 
        new_images_2, nb_new_images_2,
        device, save_folder_path,
        type_1=['blue_highway', 'blue_river'],
        type_2=['blue_highway', 'blue_river']):
    
    #seeds = [21, 42, 84, 168, 336, 745, 913, 1245, 1674, 2134]
    seeds = [21, 84]
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
        
        if (type_1 == 'blue_highway') & (type_2 == 'blue_river'):
            new_X_train = torch.cat((X_train, new_images_1[torch.randint(len(new_images_1), (nb_new_images_1,))], new_images_2[torch.randint(len(new_images_2), (nb_new_images_2,))]), 0)
            new_S_train = torch.cat((S_train, torch.zeros(nb_new_images_1), torch.zeros(nb_new_images_2)), 0)
            new_y_train = torch.cat((y_train, torch.zeros(nb_new_images_1).long(), torch.ones(nb_new_images_2).long()), 0)
        elif (type_1 == 'blue_river') & (type_2 == 'blue_highway'):
            new_X_train = torch.cat((X_train, new_images_1[torch.randint(len(new_images_1), (nb_new_images_1,))], new_images_2[torch.randint(len(new_images_2), (nb_new_images_2,))]), 0)
            new_S_train = torch.cat((S_train, torch.zeros(nb_new_images_1), torch.zeros(nb_new_images_2)), 0)
            new_y_train = torch.cat((y_train, torch.ones(nb_new_images_1).long(), torch.zeros(nb_new_images_2).long()), 0)

        train_dataset = TensorDataset(new_X_train, new_y_train, new_S_train)
        train_dataloader = DataLoader(train_dataset, batch_size=56, shuffle=True)

        new_path = f"{save_folder_path}/xai_checkpoint_augmentation_{k+1}.pt"

        cnn_model = Vgg16(num_classes=2)
        cnn_model.to(device)
        
        train_acc, train_losses, val_acc, val_losses = train(model=cnn_model, train_loader=train_dataloader, valid_loader=test_set, learning_rate=0.001, num_epochs=num_epochs, save_path=new_path)
        #_, acc, _, ratio_wrong = test(model=cnn_model, model_path=new_path, test_loader=test_set, test_gobal=True)
        
        #his = [train_acc, train_losses, val_acc, val_losses]

        #accuracy.append(acc)
        #error_rate.append(ratio_wrong)
        #history.append(his)
        print(f"=======>>>   ======>>>   ======>>>    ======>>>   ======>>>   =======>>>")
        torch.cuda.empty_cache()

        #np.save(f"{save_folder_path}/accuracy_seed_{k+1}.npy", accuracy)
        #np.save(f"{save_folder_path}/error_rate_seed_{k+1}.npy", error_rate)
        #np.save(f"{save_folder_path}/history_seed_{k+1}.npy", history)
    print("Doneeeeee!")

save_path = '/home/vuong.nguyen/vuong/augmentare/Bias in EuroSAT/Exp/Output_Vgg16/Flow/Aug'  
if __name__ == '__main__':
    main(test_set=test_dataloader, num_epochs=500,
        X_train=X_train, S_train=S_train, y_train=y_train,
        new_images_1=img_list_blue_highway, nb_new_images_1=nbs_highway, 
        new_images_2=img_list_blue_river, nb_new_images_2=nbs_river,
        device=device, save_folder_path=save_path,
        type_1='blue_highway',
        type_2='blue_river')
