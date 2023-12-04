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
    
        model.to(device)
        #model.load_state_dict(torch.load(init_weights))
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

        early_stopping = EarlyStopping(patience=50, verbose=True, path=save_path)
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

        wrong_pred_male_black = []
        wrong_pred_male_blond = []
        wrong_pred_male_brown = []
        wrong_pred_male_gray = []

        wrong_pred_female_black = []
        wrong_pred_female_blond = []
        wrong_pred_female_brown = []
        wrong_pred_female_gray = []

        nb_male_black = 0
        nb_male_blond = 0
        nb_male_brown = 0
        nb_male_gray = 0

        nb_female_black = 0
        nb_female_blond = 0
        nb_female_brown = 0
        nb_female_gray = 0

        nb_wrong = 0
        nb_wrong_male_black = 0
        nb_wrong_male_blond = 0
        nb_wrong_male_brown = 0
        nb_wrong_male_gray = 0

        nb_wrong_female_black = 0
        nb_wrong_female_blond = 0
        nb_wrong_female_brown = 0
        nb_wrong_female_gray = 0

        nb_black = 0
        nb_blond = 0
        nb_brown = 0
        nb_gray = 0

        nb_wrong_black = 0
        nb_wrong_blond = 0
        nb_wrong_brown = 0
        nb_wrong_gray = 0

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
                    if sen[i] == 1:
                        nb_black += 1
                        if lab[i] == 0:
                            nb_female_black += 1
                        elif lab[i] == 1:
                            nb_male_black += 1

                    elif sen[i] == 2:
                        nb_blond += 1
                        if lab[i] == 0:
                            nb_female_blond += 1
                        elif lab[i] == 1:
                            nb_male_blond += 1

                    elif sen[i] == 3:
                        nb_brown += 1
                        if lab[i] == 0:
                            nb_female_brown += 1
                        elif lab[i] == 1:
                            nb_male_brown += 1

                    elif sen[i] == 4:
                        nb_gray += 1
                        if lab[i] == 0:
                            nb_female_gray += 1
                        elif lab[i] == 1:
                            nb_male_gray += 1

                # Checking for different predicted images
                for i in range(pred_test.shape[0]):
                    if pred_test[i] != lab[i]:
                        # Passing the actual image, predicted image, labels, max of val_predict
                        nb_wrong += 1
                        if sen[i] == 1:
                            nb_wrong_black += 1
                            if lab[i] == 0:
                                wrong_pred_female_black.append([pred_test[i],img[i],lab[i]])
                                nb_wrong_female_black += 1
                            elif lab[i] == 1:
                                wrong_pred_male_black.append([pred_test[i],img[i],lab[i]])
                                nb_wrong_male_black += 1

                        if sen[i] == 2:
                            nb_wrong_blond += 1
                            if lab[i] == 0:
                                wrong_pred_female_blond.append([pred_test[i],img[i],lab[i]])
                                nb_wrong_female_blond += 1
                            elif lab[i] == 1:
                                wrong_pred_male_blond.append([pred_test[i],img[i],lab[i]])
                                nb_wrong_male_blond += 1

                        if sen[i] == 3:
                            nb_wrong_brown += 1
                            if lab[i] == 0:
                                wrong_pred_female_brown.append([pred_test[i],img[i],lab[i]])
                                nb_wrong_female_brown += 1
                            elif lab[i] == 1:
                                wrong_pred_male_brown.append([pred_test[i],img[i],lab[i]])
                                nb_wrong_male_brown += 1

                        if sen[i] == 4:
                            nb_wrong_gray += 1
                            if lab[i] == 0:
                                wrong_pred_female_gray.append([pred_test[i],img[i],lab[i]])
                                nb_wrong_female_gray += 1
                            elif lab[i] == 1:
                                wrong_pred_male_gray.append([pred_test[i],img[i],lab[i]])
                                nb_wrong_male_gray += 1

            nb_male = nb_male_black + nb_male_blond + nb_male_brown + nb_male_gray
            nb_female = nb_female_black + nb_female_blond + nb_female_brown + nb_female_gray

            nb_wrong_male = nb_wrong_male_black + nb_wrong_male_blond + nb_wrong_male_brown + nb_wrong_male_gray
            nb_wrong_female = nb_wrong_female_black + nb_wrong_female_blond + nb_wrong_female_brown + nb_wrong_female_gray

            # Calculate the average loss and accuracy
            avg_loss = val_loss/cnt
            per_crr = (correct/cnt_acc)*100

            if test_gobal is True:
                err_male_black = (nb_wrong_male_black/nb_male_black)*100
                err_male_blond = (nb_wrong_male_blond/nb_male_blond)*100
                err_male_brown = (nb_wrong_male_brown/nb_male_brown)*100
                err_male_gray = (nb_wrong_male_gray/nb_male_gray)*100

                err_female_black = (nb_wrong_female_black/nb_female_black)*100
                err_female_blond = (nb_wrong_female_blond/nb_female_blond)*100
                err_female_brown = (nb_wrong_female_brown/nb_female_brown)*100
                err_female_gray = (nb_wrong_female_gray/nb_female_gray)*100


                err_black = (nb_wrong_black/nb_black)*100
                err_blond = (nb_wrong_blond/nb_blond)*100
                err_brown = (nb_wrong_brown/nb_brown)*100
                err_gray = (nb_wrong_gray/nb_gray)*100

                err_male = (nb_wrong_male/nb_male)*100
                err_female = (nb_wrong_female/nb_female)*100

                err_rate = [err_male_black, err_male_blond, err_male_brown, err_male_gray,
                            err_female_black, err_female_blond, err_female_brown, err_female_gray,
                            err_black, err_blond, err_brown, err_gray, err_male, err_female]
                # Print loss and accuracy
                print('Test loss: {:.4f}, Accuracy: {:.4f}%'.format(avg_loss, per_crr))
            else:
                # Print loss and accuracy
                print('Test loss: {:.4f}, Accuracy: {:.4f}%'.format(avg_loss, per_crr))
        
        if test_gobal is True:
            dict = {"test_target" : test_target, "test_pred" : test_predict,
                    "wrong_pred_male_black" : wrong_pred_male_black, "wrong_pred_male_blond" : wrong_pred_male_blond,
                    "wrong_pred_male_brown" : wrong_pred_male_brown, "wrong_pred_male_gray" : wrong_pred_male_gray,
                    "wrong_pred_female_black" : wrong_pred_female_black, "wrong_pred_female_blond" : wrong_pred_female_blond,
                    "wrong_pred_female_brown" : wrong_pred_female_brown, "wrong_pred_female_gray" : wrong_pred_female_gray}
            return avg_loss, per_crr, dict, err_rate
        return avg_loss, per_crr
