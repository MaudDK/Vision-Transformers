import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        probs = probs.view(-1)
        targets = targets.view(-1).float()

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()

    def forward(self, probs, targets):
        return self.bce(probs, targets) + self.dice(probs, targets)