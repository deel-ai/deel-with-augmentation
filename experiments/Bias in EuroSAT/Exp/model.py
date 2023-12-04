import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model = models.resnet18(weights = 'DEFAULT')
        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.num_classes = num_classes
        self.fc = nn.Linear(512, num_classes)
        for param in self.extractor.parameters(): # Chi true moi hoat dong
            param.requires_grad=True
        for param in self.fc.parameters(): # False hay True deu cao
            param.requires_grad=True

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        logits = self.fc(out)
        feat = F.normalize(out, dim=1)
        return logits

class Vgg16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Making the parameter pretrained True in Vgg 16
        model = models.vgg16(weights = 'DEFAULT')
        feat = model.classifier[6].in_features
        # Add an output dense layer with 2 nodes
        model.classifier[6] = nn.Linear(feat, num_classes)
        self.extrac_feat = model.features
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = model

    def forward(self, x):
        logits = self.classifier(x)
        feat = self.extrac_feat(x)
        feat = self.avg(feat)
        feat = feat.squeeze(-1).squeeze(-1)
        feat = F.normalize(feat, dim=1)
        return logits

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
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, img):
        # Apply convolution operations
        out = self.convolutions(img)
        feat = self.avg(out)
        feat = feat.squeeze(-1).squeeze(-1)
        #feat = F.normalize(feat, dim=1)
        # Reshape
        out = out.view(out.size(0), -1)
        # Apply fully connected operations
        logits = self.fully_connected(out)
        return logits, feat
