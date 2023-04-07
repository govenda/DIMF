import torch
import torch.nn as nn
class Dice_Loss(nn.Module):
    def __init__(self):
        super(Dice_Loss, self).__init__()
    def _dice_loss(self, predict, target):
        """
        Compute the dice loss of the prediction decision map and ground-truth label
        :param predict: tensor, the prediction decision map
        :param target: tensor, ground-truth label
        :return:
        """
        target = target.float()
        intersect = torch.sum(predict * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(predict * predict)
        loss = (2 * intersect + 1) / (z_sum + y_sum + 1
                                      )
        loss = 1 - loss
        return loss

    def forward(self,labels,predictions):
        dice_loss=self._dice_loss(predictions,labels)
        return dice_loss


