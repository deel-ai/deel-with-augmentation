import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Parameters:
    -----------
    patience
        How long to wait after last time validation loss improved
    verbose
        If True, prints a message for each validation loss improvement (Default: False)
    delta
        Minimum change in the monitored quantity to qualify as an improvement (Default: 0)
    path
        Path for the checkpoint to be saved to (Default: 'checkpoint.pt')
    trace_func
        trace print function (Default: print)
    """
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ===>>>')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class ResNet18(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Making the parameter pretrained True in Resnet 18
        model = models.resnet18(weights = 'DEFAULT')
        features = model.fc.in_features
        # Add an output dense layer with 2 nodes
        model.fc = nn.Linear(features,2)
        
        # for param in model.parameters():
        #     param.requires_grad=False
        # for param in model.fc.parameters():
        #     param.requires_grad=True
        model.to(device)
        self.model = model
        self.device = device

    def train(self, train_loader, valid_loader, learning_rate, num_epochs, save_path):
        # Set loss function with criterion
        criterion = nn.CrossEntropyLoss()
        # Optimizer Adam
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        # List to store the train and validation accuracy
        train_acc = []
        train_losses = []
        val_acc = []
        val_losses = []

        early_stopping = EarlyStopping(patience=100, verbose=True, path=save_path)
        for epoch in range(num_epochs):
            # Forward pass
            self.model.train()
            train_loss, correct, cnt_acc, cnt = 0, 0, 0, 0
            for i, (images, labels, sensibilities) in enumerate(train_loader):
                # Move tensors to the configured device
                images = images.to(self.device)
                labels = labels.to(self.device)
                sensibilities = sensibilities.to(self.device)
                outputs = self.model(images)
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
      
            # Calculate the average loss and accuracy train
            avg_loss_train = train_loss/cnt
            per_crr_train = (correct/cnt_acc)*100
            train_acc.append(per_crr_train)
            train_losses.append(avg_loss_train)
            
            self.model.eval()
            val_loss, val_correct, val_cnt_acc, val_cnt = 0, 0, 0, 0
            for i, (img, lab, sen) in enumerate(valid_loader):
                img = img.to(self.device)
                lab = lab.to(self.device)
                sen = sen.to(self.device)
                val_pred = self.model(img)
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
        
            early_stopping(avg_loss_val, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(save_path))
        return train_acc, train_losses, val_acc, val_losses

    def test(self, test_loader, test_gobal=True):
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
                img = img.to(self.device)
                lab = lab.to(self.device)
                sen = sen.to(self.device)
                val_pred = self.model(img)
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

    def evaluate(self, X_train, y_train, S_train):
        highway_images = X_train[torch.where((S_train==1) & (y_train==0))]
        highway_sen    = S_train[torch.where((S_train==1)&(y_train==0))]
        highway_labels = y_train[torch.where((S_train==1) & (y_train==0))]

        blue_highway_images = X_train[torch.where((S_train==0) & (y_train==0))]
        blue_highway_sen    = S_train[torch.where((S_train==0)&(y_train==0))]
        blue_highway_labels = y_train[torch.where((S_train==0) & (y_train==0))]

        river_images = X_train[torch.where((S_train==1) & (y_train==1))]
        river_sen    = S_train[torch.where((S_train==1)&(y_train==1))]
        river_labels = y_train[torch.where((S_train==1) & (y_train==1))]

        blue_river_images = X_train[torch.where((S_train==0) & (y_train==1))]
        blue_river_sen    = S_train[torch.where((S_train==0)&(y_train==1))]
        blue_river_labels = y_train[torch.where((S_train==0) & (y_train==1))]

        dataset = TensorDataset(X_train, y_train, S_train)
        dataloader = DataLoader(dataset, batch_size=56, shuffle=False)

        hw_set = TensorDataset(highway_images, highway_labels, highway_sen)
        hw_loader = DataLoader(hw_set, batch_size=56, shuffle=False)

        bhw_set = TensorDataset(blue_highway_images, blue_highway_labels, blue_highway_sen)
        bhw_loader = DataLoader(bhw_set, batch_size=56, shuffle=False)

        rv_set = TensorDataset(river_images, river_labels, river_sen)
        rv_loader = DataLoader(rv_set, batch_size=56, shuffle=False)

        brv_set = TensorDataset(blue_river_images, blue_river_labels, blue_river_sen)
        brv_loader = DataLoader(brv_set, batch_size=56, shuffle=False)

        _, glo_acc, _, _ = self.test(test_loader=dataloader, test_gobal=True)
        _, hw_acc = self.test(test_loader=hw_loader, test_gobal=False)
        _, bhw_acc = self.test(test_loader=bhw_loader, test_gobal=False)
        _, rv_acc = self.test(test_loader=rv_loader, test_gobal=False)
        _, brv_acc = self.test(test_loader=brv_loader, test_gobal=False)
        return glo_acc, hw_acc, bhw_acc, rv_acc, brv_acc

class Vgg16(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Making the parameter pretrained True in Vgg 16
        model = models.vgg16(weights = 'DEFAULT')
        features = model.classifier[6].in_features
        # Add an output dense layer with 2 nodes
        model.classifier[6] = nn.Linear(features,2)
        model.to(device)
        self.model = model
        self.device = device

    def train(self, train_loader, valid_loader, learning_rate, num_epochs, save_path):
        # Set loss function with criterion
        criterion = nn.CrossEntropyLoss()
        # Optimizer SGD
        optimizer = torch.optim.SGD(self.model.parameters(), lr = learning_rate, momentum=0.9)
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
            self.model.train()
            train_loss, correct, cnt_acc, cnt = 0, 0, 0, 0
            for i, (images, labels, sensibilities) in enumerate(train_loader):
                # Move tensors to the configured device
                images = images.to(self.device)
                labels = labels.to(self.device)
                sensibilities = sensibilities.to(self.device)
                outputs = self.model(images)
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

            self.model.eval()
            val_loss, val_correct, val_cnt_acc, val_cnt = 0, 0, 0, 0
            for i, (img, lab, sen) in enumerate(valid_loader):
                img = img.to(self.device)
                lab = lab.to(self.device)
                sen = sen.to(self.device)
                val_pred = self.model(img)
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
        
            early_stopping(avg_loss_val, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(save_path))
        return train_acc, train_losses, val_acc, val_losses

    def test(self, test_loader, test_gobal=True):
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
                img = img.to(self.device)
                lab = lab.to(self.device)
                sen = sen.to(self.device)
                val_pred = self.model(img)
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

    def evaluate(self, X_train, y_train, S_train):
        highway_images = X_train[torch.where((S_train==1) & (y_train==0))]
        highway_sen    = S_train[torch.where((S_train==1)&(y_train==0))]
        highway_labels = y_train[torch.where((S_train==1) & (y_train==0))]

        blue_highway_images = X_train[torch.where((S_train==0) & (y_train==0))]
        blue_highway_sen    = S_train[torch.where((S_train==0)&(y_train==0))]
        blue_highway_labels = y_train[torch.where((S_train==0) & (y_train==0))]

        river_images = X_train[torch.where((S_train==1) & (y_train==1))]
        river_sen    = S_train[torch.where((S_train==1)&(y_train==1))]
        river_labels = y_train[torch.where((S_train==1) & (y_train==1))]

        blue_river_images = X_train[torch.where((S_train==0) & (y_train==1))]
        blue_river_sen    = S_train[torch.where((S_train==0)&(y_train==1))]
        blue_river_labels = y_train[torch.where((S_train==0) & (y_train==1))]

        dataset = TensorDataset(X_train, y_train, S_train)
        dataloader = DataLoader(dataset, batch_size=56, shuffle=False)

        hw_set = TensorDataset(highway_images, highway_labels, highway_sen)
        hw_loader = DataLoader(hw_set, batch_size=56, shuffle=False)

        bhw_set = TensorDataset(blue_highway_images, blue_highway_labels, blue_highway_sen)
        bhw_loader = DataLoader(bhw_set, batch_size=56, shuffle=False)

        rv_set = TensorDataset(river_images, river_labels, river_sen)
        rv_loader = DataLoader(rv_set, batch_size=56, shuffle=False)

        brv_set = TensorDataset(blue_river_images, blue_river_labels, blue_river_sen)
        brv_loader = DataLoader(brv_set, batch_size=56, shuffle=False)

        _, glo_acc, _, _ = self.test(test_loader=dataloader, test_gobal=True)
        _, hw_acc = self.test(test_loader=hw_loader, test_gobal=False)
        _, bhw_acc = self.test(test_loader=bhw_loader, test_gobal=False)
        _, rv_acc = self.test(test_loader=rv_loader, test_gobal=False)
        _, brv_acc = self.test(test_loader=brv_loader, test_gobal=False)
        return glo_acc, hw_acc, bhw_acc, rv_acc, brv_acc

class Net(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        # Input is [B, 3, 64, 64]
        self.convolutions = nn.Sequential( 
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1), #[B, 32, 64, 64]
            nn.ReLU(),  #[B, 32, 64, 64]
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),  #[B, 64, 64, 64]
            nn.ReLU(),  #[B, 64, 64, 64]
            nn.MaxPool2d(2,2),  #[B, 64, 32, 32]
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1), #[B, 128, 32, 32]
            nn.ReLU(), #[B, 64, 32, 32]
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1), #[B, 128, 32, 32]
            nn.ReLU(), #[B, 128, 32, 32]
            nn.MaxPool2d(2,2), #[B, 128, 16, 16]
            nn.Dropout(0.2), #To Prevent Overfitting

            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1), #[B, 128256, 16, 16]
            nn.ReLU(), #[B, 256, 16, 16]
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1), #[B, 256, 16, 16]
            nn.ReLU(), #[B, 256, 16, 16]
            nn.MaxPool2d(2,2), #[B, 256, 8, 8]
            nn.Dropout(0.15) #To Prevent Overfitting
        )
        # Input will be reshaped from [B, 256, 8, 8] to [B, 256*8*8] for fully connected layers
        self.fully_connected = nn.Sequential(
            nn.Linear(256*8*8, 16), # [B, 16]
            nn.ReLU(inplace=True), # [B, 16]
            nn.Linear(16, n_classes), # [B, n_classes]
        )

    def forward(self, img):
        # Apply convolution operations
        x = self.convolutions(img)
        # Reshape
        x = x.view(x.size(0), -1)
        # Apply fully connected operations
        x = self.fully_connected(x)
        return x

class Net1(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.mp3 = nn.MaxPool2d(kernel_size=2)
        
        self.f = nn.Flatten()
        self.linear = nn.Linear(in_features=192*4*4, out_features=1024)
        self.relu4 = nn.ReLU()
        
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(in_features=1024, out_features=512)
        self.relu5 = nn.ReLU()
        
        self.o = nn.Linear(in_features=512, out_features=num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.mp1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.mp2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.mp3(x)
        
        x = self.f(x)
        x = self.linear(x)
        x = self.relu4(x)
        
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu5(x)
        
        x = self.o(x)
        return x

class CNN(nn.Module):
    def __init__(self, device):
        super().__init__()
        model = Net()
        model.to(device)
        self.model = model
        self.device = device

    def train(self, train_loader, valid_loader, learning_rate, num_epochs, save_path):
        # Set loss function with criterion
        criterion = nn.CrossEntropyLoss()
        # Optimizer SGD
        optimizer = torch.optim.SGD(self.model.parameters(), lr = learning_rate, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=0.0001)
        # List to store the train and validation accuracy
        train_acc = []
        train_losses = []
        val_acc = []
        val_losses = []

        early_stopping = EarlyStopping(patience=100, verbose=True, path=save_path)
        for epoch in range(num_epochs):
            # Forward pass
            self.model.train()
            train_loss, correct, cnt_acc, cnt = 0, 0, 0, 0
            for i, (images, labels, sensibilities) in enumerate(train_loader):
                # Move tensors to the configured device
                images = images.to(self.device)
                labels = labels.to(self.device)
                sensibilities = sensibilities.to(self.device)
                outputs = self.model(images)
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

            self.model.eval()
            val_loss, val_correct, val_cnt_acc, val_cnt = 0, 0, 0, 0
            for i, (img, lab, sen) in enumerate(valid_loader):
                img = img.to(self.device)
                lab = lab.to(self.device)
                sen = sen.to(self.device)
                val_pred = self.model(img)
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
        
            early_stopping(avg_loss_val, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(save_path))
        return train_acc, train_losses, val_acc, val_losses

    def test(self, test_loader, test_gobal=True):
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
                img = img.to(self.device)
                lab = lab.to(self.device)
                sen = sen.to(self.device)
                val_pred = self.model(img)
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

    def evaluate(self, X_train, y_train, S_train):
        highway_images = X_train[torch.where((S_train==1) & (y_train==0))]
        highway_sen    = S_train[torch.where((S_train==1)&(y_train==0))]
        highway_labels = y_train[torch.where((S_train==1) & (y_train==0))]

        blue_highway_images = X_train[torch.where((S_train==0) & (y_train==0))]
        blue_highway_sen    = S_train[torch.where((S_train==0)&(y_train==0))]
        blue_highway_labels = y_train[torch.where((S_train==0) & (y_train==0))]

        river_images = X_train[torch.where((S_train==1) & (y_train==1))]
        river_sen    = S_train[torch.where((S_train==1)&(y_train==1))]
        river_labels = y_train[torch.where((S_train==1) & (y_train==1))]

        blue_river_images = X_train[torch.where((S_train==0) & (y_train==1))]
        blue_river_sen    = S_train[torch.where((S_train==0)&(y_train==1))]
        blue_river_labels = y_train[torch.where((S_train==0) & (y_train==1))]

        dataset = TensorDataset(X_train, y_train, S_train)
        dataloader = DataLoader(dataset, batch_size=56, shuffle=False)

        hw_set = TensorDataset(highway_images, highway_labels, highway_sen)
        hw_loader = DataLoader(hw_set, batch_size=56, shuffle=False)

        bhw_set = TensorDataset(blue_highway_images, blue_highway_labels, blue_highway_sen)
        bhw_loader = DataLoader(bhw_set, batch_size=56, shuffle=False)

        rv_set = TensorDataset(river_images, river_labels, river_sen)
        rv_loader = DataLoader(rv_set, batch_size=56, shuffle=False)

        brv_set = TensorDataset(blue_river_images, blue_river_labels, blue_river_sen)
        brv_loader = DataLoader(brv_set, batch_size=56, shuffle=False)

        _, glo_acc, _, _ = self.test(test_loader=dataloader, test_gobal=True)
        _, hw_acc = self.test(test_loader=hw_loader, test_gobal=False)
        _, bhw_acc = self.test(test_loader=bhw_loader, test_gobal=False)
        _, rv_acc = self.test(test_loader=rv_loader, test_gobal=False)
        _, brv_acc = self.test(test_loader=brv_loader, test_gobal=False)
        return glo_acc, hw_acc, bhw_acc, rv_acc, brv_acc

class CNN1(nn.Module):
    def __init__(self, device):
        super().__init__()
        model = Net1()
        model.to(device)
        self.model = model
        self.device = device

    def train(self, train_loader, valid_loader, learning_rate, num_epochs, save_path):
        # Set loss function with criterion
        criterion = nn.CrossEntropyLoss()
        # Optimizer Adam
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        # List to store the train and validation accuracy
        train_acc = []
        train_losses = []
        val_acc = []
        val_losses = []

        early_stopping = EarlyStopping(patience=100, verbose=True, path=save_path)
        for epoch in range(num_epochs):
            # Forward pass
            self.model.train()
            train_loss, correct, cnt_acc, cnt = 0, 0, 0, 0
            for i, (images, labels, sensibilities) in enumerate(train_loader):
                # Move tensors to the configured device
                images = images.to(self.device)
                labels = labels.to(self.device)
                sensibilities = sensibilities.to(self.device)
                outputs = self.model(images)
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
            
            # Calculate the average loss and accuracy train
            avg_loss_train = train_loss/cnt
            per_crr_train = (correct/cnt_acc)*100
            train_acc.append(per_crr_train)
            train_losses.append(avg_loss_train)

            self.model.eval()
            val_loss, val_correct, val_cnt_acc, val_cnt = 0, 0, 0, 0
            for i, (img, lab, sen) in enumerate(valid_loader):
                img = img.to(self.device)
                lab = lab.to(self.device)
                sen = sen.to(self.device)
                val_pred = self.model(img)
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
        
            early_stopping(avg_loss_val, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(save_path))
        return train_acc, train_losses, val_acc, val_losses

    def test(self, test_loader, test_gobal=True):
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
                img = img.to(self.device)
                lab = lab.to(self.device)
                sen = sen.to(self.device)
                val_pred = self.model(img)
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

    def evaluate(self, X_train, y_train, S_train):
        highway_images = X_train[torch.where((S_train==1) & (y_train==0))]
        highway_sen    = S_train[torch.where((S_train==1)&(y_train==0))]
        highway_labels = y_train[torch.where((S_train==1) & (y_train==0))]

        blue_highway_images = X_train[torch.where((S_train==0) & (y_train==0))]
        blue_highway_sen    = S_train[torch.where((S_train==0)&(y_train==0))]
        blue_highway_labels = y_train[torch.where((S_train==0) & (y_train==0))]

        river_images = X_train[torch.where((S_train==1) & (y_train==1))]
        river_sen    = S_train[torch.where((S_train==1)&(y_train==1))]
        river_labels = y_train[torch.where((S_train==1) & (y_train==1))]

        blue_river_images = X_train[torch.where((S_train==0) & (y_train==1))]
        blue_river_sen    = S_train[torch.where((S_train==0)&(y_train==1))]
        blue_river_labels = y_train[torch.where((S_train==0) & (y_train==1))]

        dataset = TensorDataset(X_train, y_train, S_train)
        dataloader = DataLoader(dataset, batch_size=56, shuffle=False)

        hw_set = TensorDataset(highway_images, highway_labels, highway_sen)
        hw_loader = DataLoader(hw_set, batch_size=56, shuffle=False)

        bhw_set = TensorDataset(blue_highway_images, blue_highway_labels, blue_highway_sen)
        bhw_loader = DataLoader(bhw_set, batch_size=56, shuffle=False)

        rv_set = TensorDataset(river_images, river_labels, river_sen)
        rv_loader = DataLoader(rv_set, batch_size=56, shuffle=False)

        brv_set = TensorDataset(blue_river_images, blue_river_labels, blue_river_sen)
        brv_loader = DataLoader(brv_set, batch_size=56, shuffle=False)

        _, glo_acc, _, _ = self.test(test_loader=dataloader, test_gobal=True)
        _, hw_acc = self.test(test_loader=hw_loader, test_gobal=False)
        _, bhw_acc = self.test(test_loader=bhw_loader, test_gobal=False)
        _, rv_acc = self.test(test_loader=rv_loader, test_gobal=False)
        _, brv_acc = self.test(test_loader=brv_loader, test_gobal=False)
        return glo_acc, hw_acc, bhw_acc, rv_acc, brv_acc
