import torch
import math
from torch import nn
import torch.nn.functional as F


def distillation_loss(y, labels, teacher_scores, T, alpha, reduction_kd='mean', reduction_nll='mean'):
    #if teacher_scores is not None and y.dtype != teacher_scores.dtype:
    #    teacher_scores = teacher_scores.half()

    if teacher_scores is not None:
        d_loss = nn.KLDivLoss(reduction=reduction_kd)(F.log_softmax(y / T, dim=1),
                                                      F.softmax(teacher_scores / T, dim=1)) * T * T
    else:
        assert alpha == 0, 'alpha cannot be {} when teacher scores are not provided'.format(alpha)
        d_loss = 0.0
    nll_loss = F.cross_entropy(y, labels, reduction=reduction_nll)
    # print(d_loss.shape, d_loss)
    # print('\n', nll_loss.shape, nll_loss)
    tol_loss = alpha * d_loss + (1.0 - alpha) * nll_loss
    # print('in func:', d_loss.item(), nll_loss.item(), alpha, tol_loss.item())
    return tol_loss, d_loss, nll_loss


def patience_loss(teacher_patience, student_patience, normalized_patience=False):
    # n_batch = teacher_patience.shape[0]
    if normalized_patience:
        teacher_patience = F.normalize(teacher_patience, p=2, dim=2)
        student_patience = F.normalize(student_patience, p=2, dim=2)
    return F.mse_loss(teacher_patience.float(), student_patience.float()).half()

    # diff = (teacher_patience - student_patience).pow(2).sum()
    # const = math.sqrt(teacher_patience.numel())
    # return diff / const / const
