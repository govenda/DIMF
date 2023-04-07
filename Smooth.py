import torch
import torch.nn as nn
class Smooth_Loss(nn.Module):
    def __init__(self):
        super(Smooth_Loss, self).__init__()
    def _smooth_loss(self,labels,predictions):
        device = labels.device
        diff = torch.abs(labels - predictions)
        less_than_one = torch.le(diff, 1.0).type(torch.FloatTensor).cuda(device)  # Bool to float32
        smooth_l1_loss = ((less_than_one * 0.5 * diff ** 2) + (1.0 - less_than_one) * (diff - 0.5))  # 同上图公式
        return torch.mean(smooth_l1_loss)
    def forward(self,labels,predictions):
        smooth_loss=self._smooth_loss(labels,predictions)
        return smooth_loss