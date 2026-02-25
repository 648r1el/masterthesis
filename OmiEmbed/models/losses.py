import math

import numpy as np

import torch
import torch.nn as nn


def get_loss_func(loss_name, reduction='mean'):
    """
    Return the loss function.
    Parameters:
        loss_name (str)    -- the name of the loss function: BCE | MSE | L1 | CE
        reduction (str)    -- the reduction method applied to the loss function: sum | mean
    """
    if loss_name == 'BCE':
        return nn.BCEWithLogitsLoss(reduction=reduction)
    elif loss_name == 'MSE':
        return nn.MSELoss(reduction=reduction)
    elif loss_name == 'L1':
        return nn.L1Loss(reduction=reduction)
    elif loss_name == 'CE':
        return nn.CrossEntropyLoss(reduction=reduction)
    elif loss_name == 'NLL':
        return nn.NLLLoss(reduction=reduction)
    else:
        raise NotImplementedError('Loss function %s is not found' % loss_name)


def kl_loss(mean, log_var, reduction='mean'):
    part_loss = 1 + log_var - mean.pow(2) - log_var.exp()
    if reduction == 'mean':
        loss = -0.5 * torch.mean(part_loss)
    else:
        loss = -0.5 * torch.sum(part_loss)
    return loss


def mmd_loss(x: torch.Tensor, y: torch.Tensor, var: torch.Tensor):
    xx = torch.mm(x, x.t())
    xy = torch.mm(x, y.t())
    yy = torch.mm(y, y.t())

    x_diag = xx.diag().unsqueeze(0).expand_as(xx)
    y_diag = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = x_diag.t() - 2 * xx + x_diag
    dxy = x_diag.t() - 2 * xy + y_diag
    dyy = y_diag.t() - 2 * yy + y_diag

    XX = torch.exp(-0.5 * dxx / var)
    XY = torch.exp(-0.5 * dxy / var)
    YY = torch.exp(-0.5 * dyy / var)

    return torch.mean(XX + YY - 2 * XY)


def MTLR_survival_loss(y_pred, y_true, E, tri_matrix, reduction='mean'):
    """
    Compute the MTLR survival loss
    """
    # Get censored index and uncensored index
    censor_idx = []
    uncensor_idx = []
    for i in range(len(E)):
        # If this is a uncensored data point
        if E[i] == 1:
            # Add to uncensored index list
            uncensor_idx.append(i)
        else:
            # Add to censored index list
            censor_idx.append(i)

    # Separate y_true and y_pred
    y_pred_censor = y_pred[censor_idx]
    y_true_censor = y_true[censor_idx]
    y_pred_uncensor = y_pred[uncensor_idx]
    y_true_uncensor = y_true[uncensor_idx]

    # Calculate likelihood for censored datapoint
    phi_censor = torch.exp(torch.mm(y_pred_censor, tri_matrix))
    reduc_phi_censor = torch.sum(phi_censor * y_true_censor, dim=1)

    # Calculate likelihood for uncensored datapoint
    phi_uncensor = torch.exp(torch.mm(y_pred_uncensor, tri_matrix))
    reduc_phi_uncensor = torch.sum(phi_uncensor * y_true_uncensor, dim=1)

    # Likelihood normalisation
    z_censor = torch.exp(torch.mm(y_pred_censor, tri_matrix))
    reduc_z_censor = torch.sum(z_censor, dim=1)
    z_uncensor = torch.exp(torch.mm(y_pred_uncensor, tri_matrix))
    reduc_z_uncensor = torch.sum(z_uncensor, dim=1)

    # MTLR loss
    loss = - (torch.sum(torch.log(reduc_phi_censor)) + torch.sum(torch.log(reduc_phi_uncensor)) - torch.sum(torch.log(reduc_z_censor)) - torch.sum(torch.log(reduc_z_uncensor)))

    if reduction == 'mean':
        loss = loss / E.shape[0]

    return loss
