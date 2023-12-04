import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from model import ResNet18, Vgg16
from torch.utils.data import TensorDataset, DataLoader
from utils import EarlyStopping, get_confusion_matrix, clip_max_ratio
from torch.utils.data.sampler import WeightedRandomSampler
from bias_contrastive import BiasContrastiveLoss
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

def train(model, train_loader, cont_loader, confusion_matrix, learning_rate, num_epochs, weight, save_path):
        # Set loss function with criterion
        criterion = BiasContrastiveLoss(confusion_matrix=confusion_matrix, bb=1)
        # Optimizer Adam
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        # Optimizer SGD
        #optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        # List to store the train and validation accuracy
        train_acc = []
        train_losses = []

        early_stopping = EarlyStopping(patience=100, verbose=True, path=save_path)
        for epoch in range(num_epochs):
            # Forward pass
            model.train()
            train_loss, correct, cnt_acc, cnt = 0, 0, 0, 0

            train_iter = iter(train_loader)
            cont_train_iter = iter(cont_loader)
            for i, (images, labels, biases) in enumerate(train_iter):
                try:
                    cont_images_1, cont_images_2, cont_labels, cont_biases = next(cont_train_iter)
                except:
                    cont_train_iter = iter(cont_loader)
                    cont_images_1, cont_images_2, cont_labels, cont_biases = next(cont_train_iter)

                bsz = labels.shape[0]
                cont_bsz = cont_labels.shape[0]

                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)
                biases = biases.to(device)
                outputs, _ = model(images)

                total_images = torch.cat([cont_images_1, cont_images_2], dim=0)

                total_images, cont_labels, cont_biases = total_images.to(device), cont_labels.to(device), cont_biases.to(device)
                _, cont_features = model(total_images)

                f1, f2 = torch.split(cont_features, [cont_bsz, cont_bsz], dim=0)
                cont_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                ce_loss, con_loss = criterion(outputs, labels, biases, cont_features, cont_labels, cont_biases)

                loss = ce_loss * weight + con_loss

                with torch.no_grad():
                    train_loss += loss.item()
                    correct += (outputs.argmax(1) == labels).sum().item()
                    cnt_acc += outputs.shape[0]
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cnt = i+1

            #exp_lr_scheduler.step()
            # Calculate the average loss and accuracy train
            avg_loss_train = train_loss/cnt
            per_crr_train = (correct/cnt_acc)*100
            train_acc.append(per_crr_train)
            train_losses.append(avg_loss_train)

            # Print the epoch and it's accuracy and loss
            print('Epoch [{}/{}], Train loss: {:.4f}, Train accuracy: {:.4f}%'.format(epoch+1, num_epochs, avg_loss_train, per_crr_train))
        
            early_stopping(avg_loss_train, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        return train_acc, train_losses

def test(model, model_path, test_loader, test_gobal=True):
        # Load the last checkpoint with the best model
        model.load_state_dict(torch.load(model_path))
        model.eval()
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
            correct, cnt_acc = 0, 0
            for i, (img, lab, sen) in enumerate(test_loader):
                test_target.extend(list(lab))
                img = img.to(device)
                lab = lab.to(device)
                sen = sen.to(device)
                val_pred, _ = model(img)
                pred_test = torch.argmax(val_pred, dim = 1).to(torch.device("cpu")).numpy()
                
                correct += (val_pred.argmax(1) == lab).sum().item()
                cnt_acc += val_pred.shape[0]

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
            per_crr = (correct/cnt_acc)*100
            err_rate_bl = (nb_wrong_blue/nb_wrong)*100
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
                print('Accuracy: {:.4f}%, Error rate of blue images: {:.4f}%'.format(per_crr, err_rate_bl))
            else:
                # Print loss and accuracy
                print('Accuracy: {:.4f}%'.format(per_crr))
        
        if test_gobal is True:
            dict = {"test_target" : test_target, "test_pred" : test_predict,
                    "wrong_pred_hw" : wrong_pred_hw, "wrong_pred_rv" : wrong_pred_rv, 
                    "wrong_blue_pred_hw" : wrong_blue_pred_hw, "wrong_blue_pred_rv" : wrong_blue_pred_rv}
            return per_crr, dict, err_rate
        return per_crr

gen_blue_highway = torch.load("/home/vuong.nguyen/vuong/augmentare/Bias in EuroSAT/NST_output/FDA/fda_bl_hw.pt")
gen_blue_river = torch.load("/home/vuong.nguyen/vuong/augmentare/Bias in EuroSAT/NST_output/FDA/fda_bl_rv.pt")

trans = transforms.Compose([
    transforms.Resize(64)
])

gen_blue_highway = torch.stack(gen_blue_highway).float()
gen_blue_river = torch.stack(gen_blue_river).float()

img_list_blue_highway = trans(gen_blue_highway)
img_list_blue_river = trans(gen_blue_river)

nbs_highway = [64, 158, 250, 600]
nbs_river = [14, 108, 200, 1100]

def main(num_epochs,
        X_train, S_train, y_train,
        X_test, S_test, y_test,
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
                
            if (type_1 == 'blue_highway') & (type_2 == 'blue_river'):
                new_X_train = torch.cat((X_train, new_images_1[torch.randint(len(new_images_1), (nb_new_images_1[i],))], new_images_2[torch.randint(len(new_images_2), (nb_new_images_2[i],))]), 0)
                new_S_train = torch.cat((S_train, torch.zeros(nb_new_images_1[i]), torch.zeros(nb_new_images_2[i])), 0)
                new_y_train = torch.cat((y_train, torch.zeros(nb_new_images_1[i]).long(), torch.ones(nb_new_images_2[i]).long()), 0)
            elif (type_1 == 'blue_river') & (type_2 == 'blue_highway'):
                new_X_train = torch.cat((X_train, new_images_1[torch.randint(len(new_images_1), (nb_new_images_1[i],))], new_images_2[torch.randint(len(new_images_2), (nb_new_images_2[i],))]), 0)
                new_S_train = torch.cat((S_train, torch.zeros(nb_new_images_1[i]), torch.zeros(nb_new_images_2[i])), 0)
                new_y_train = torch.cat((y_train, torch.ones(nb_new_images_1[i]).long(), torch.zeros(nb_new_images_2[i]).long()), 0)

            confusion_matrix = get_confusion_matrix(num_classes=2, targets=new_y_train, biases=new_S_train)

            transform = transforms.Compose([
                    transforms.RandomResizedCrop(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            
            cont_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=64, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
        
            X_train_cont_1 = cont_transform(new_X_train)
            X_train_cont_2 = cont_transform(new_X_train)

            test_transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            
            X_train_nor = transform(new_X_train)
            X_test_test = test_transform(X_test)

            train_dataset = TensorDataset(X_train_nor, new_y_train, new_S_train)
            train_dataloader = DataLoader(train_dataset, batch_size=56, shuffle=True,
                                        sampler=None, num_workers=8, pin_memory=True, drop_last=False)

            cont_train_dataset = TensorDataset(X_train_cont_1, X_train_cont_2, new_y_train, new_S_train)

            weights = [1 / confusion_matrix[c.long(), b.long()] for c, b in zip(new_y_train, new_S_train)]
            weights = clip_max_ratio(np.array(weights), 10)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

            cont_train_dataloader = DataLoader(cont_train_dataset, batch_size=28, shuffle=False,
                                        sampler=sampler, pin_memory=True, drop_last=True)

            test_dataset = TensorDataset(X_test_test, y_test, S_test)
            test_dataloader = DataLoader(test_dataset, batch_size=56, shuffle=False,
                                        sampler=None, num_workers=8, pin_memory=True, drop_last=False)

            new_path = f"{save_folder_path}/checkpoint_aug_bc_bb_seed_{k+1}.pt"

            cnn_model = ResNet18(num_classes=2)
            cnn_model.to(device)
            
            train_acc, train_losses = train(model=cnn_model, train_loader=train_dataloader, 
                                            cont_loader=cont_train_dataloader, confusion_matrix=confusion_matrix,
                                            learning_rate=0.00008, num_epochs=num_epochs, weight=0.01, save_path=new_path)
            
            acc, _, ratio_wrong = test(model=cnn_model, model_path=new_path, test_loader=test_dataloader, test_gobal=True)
            
            his = [train_acc, train_losses]

            accuracy.append(acc)
            error_rate.append(ratio_wrong)
            history.append(his)
            print(f"=======>>>   ======>>>   ======>>>    ======>>>   ======>>>   =======>>>")
            torch.cuda.empty_cache()

        np.save(f"{save_folder_path}/accuracy_seed_{k+1}.npy", accuracy)
        np.save(f"{save_folder_path}/error_rate_seed_{k+1}.npy", error_rate)
        np.save(f"{save_folder_path}/history_seed_{k+1}.npy", history)
    print("Doneeeeee!")

save_path = '/home/vuong.nguyen/vuong/augmentare/Bias in EuroSAT/Exp/Output_ResNet18/FDA/Aug_BC_BB'

if __name__ == '__main__':
    main(num_epochs=500,
        X_train=X_train, S_train=S_train, y_train=y_train,
        X_test=X_test, S_test=S_test, y_test=y_test,
        new_images_1=img_list_blue_highway, nb_new_images_1=nbs_highway, 
        new_images_2=img_list_blue_river, nb_new_images_2=nbs_river,
        device=device, save_folder_path=save_path,
        type_1='blue_highway',
        type_2='blue_river')
