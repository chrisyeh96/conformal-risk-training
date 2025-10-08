import torch
import torch.nn.functional as F
from torch import Tensor

from polyps.PraNet_Res2Net import PraNet


def structure_loss(pred: Tensor, mask: Tensor) -> Tensor:
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def pranet_loss(model: PraNet, images: Tensor, masks: Tensor) -> Tensor:
    lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
    # ---- loss function ----
    loss5 = structure_loss(lateral_map_5, masks)
    loss4 = structure_loss(lateral_map_4, masks)
    loss3 = structure_loss(lateral_map_3, masks)
    loss2 = structure_loss(lateral_map_2, masks)
    return loss2 + loss3 + loss4 + loss5
