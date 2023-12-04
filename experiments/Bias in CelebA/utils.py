import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

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

class CelebDataset(Dataset):
    def __init__(self, df, image_path, transform=None):
        super().__init__()
        self.sen = df['Sens']
        self.label = df['Male']
        self.path = image_path
        self.image_id = df['image_id']
        self.transform=transform

    def __len__(self):
        return self.image_id.shape[0]

    def __getitem__(self, idx:int):
        image_name = self.image_id.iloc[idx]
        image = Image.open(os.path.join(self.path, image_name))
        labels = np.asarray(self.label.iloc[idx].T)
        sens = np.asarray(self.sen.iloc[idx].T)

        labels = torch.from_numpy(labels)
        sens = torch.from_numpy(sens)
        
        if self.transform:
            image=self.transform(image)
        return image, labels, sens
    
def get_celeb(df, image_path, transform=None):
    sen = df['Sens']
    label = df['Male']
    path = image_path
    image_id = df['image_id']
    transform=transform
    
    celeb_images = []
    celeb_labels = []
    celeb_sens = []

    _len_ = image_id.shape[0]
    for i in range(_len_):
        image_name = image_id.iloc[i]
        image = Image.open(os.path.join(path, image_name))
        labels = np.asarray(label.iloc[i].T)
        sens = np.asarray(sen.iloc[i].T)

        if transform:
            image=transform(image)

        celeb_images.append(image)
        celeb_labels.append(labels)
        celeb_sens.append(sens)

    celeb_images = torch.stack(celeb_images)
    celeb_labels = torch.stack(celeb_labels)
    celeb_sens = torch.stack(celeb_sens)
    return celeb_images, celeb_labels, celeb_sens
