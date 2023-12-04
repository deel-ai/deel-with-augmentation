import torch
import torch.nn as nn
import torch.nn.functional as F

class BBLoss(nn.Module):
    def __init__(self, confusion_matrix, device):
        super().__init__()
        self.confusion_matrix = confusion_matrix.to(device)
        self.min_prob = 1e-9

    def forward(self, logits, labels, biases):
        prior = self.confusion_matrix[biases.long()]
        logits += torch.log(prior + self.min_prob)
        label_loss = F.cross_entropy(logits, labels)
        return label_loss
