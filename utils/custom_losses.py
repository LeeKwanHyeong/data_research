import torch.nn as nn
import torch.nn.functional as F
import torch

class CustomLoss(nn.Module):
    def __init__(self, pred, target):
        super(CustomLoss, self).__init__()
        pred = self.pred
        target = self.target

    def smooth_l1(self, alpha = 0.7):
        base_loss = F.smooth_l1_loss(self.pred, self.target)
        rel_error = torch.abs(self.pred - self.target) / (self.target + 1e-6)
        rel_loss = rel_error.mean()
        return alpha * base_loss + (1 - alpha) * rel_loss

    def combined_loss(self, beta = 0.001):
        base = self.smooth_l1()
        smoothness = torch.mean(torch.diff(self.pred, n = 2, dim = 1) ** 2)
        return base + beta * smoothness

