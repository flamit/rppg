import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    """Defines the negative pearson correlation loss"""

    def __init__(self, T):
        """
        Initializes the loss
        :param T: Length of the signal (number of frames in the video).
        """
        super(PearsonLoss, self).__init__()
        self.T = T

    def forward(self, logits, target):
        """
        Calculates the parson loss
        :param logits: Network predictions (batch x signal_length)
        :param target: The ground truth of size (batch x signal_length)
        :return: The negative pearson loss
        """
        num = (self.T * torch.diag(torch.mm(logits, target.t()))) - (logits.sum(dim=1) * target.sum(dim=1))
        denom = (self.T * torch.pow(logits, 2).sum(dim=1) - torch.pow(logits.sum(dim=1), 2)) * (
                    self.T * torch.pow(target, 2).sum(dim=1) - torch.pow(target.sum(dim=1), 2))
        loss = (num / torch.sqrt(denom)).mean()
        return 1.0 - loss
