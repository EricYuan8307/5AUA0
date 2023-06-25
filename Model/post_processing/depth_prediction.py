import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb

def depth_act(input, MaxDepth = 88): # Depth = MaxDepth X Sigmoid(f_d)
    depth = MaxDepth * torch.sigmoid(input)
    return depth

def log_mse(self, output, gt):  # the first loss
        # output dimension: (B,H,W)
        ignore_label = 0.0
        label_mask = gt != ignore_label
        # print(label_mask)

        masked_output = torch.masked_select(output, label_mask)  # Apply mask to output
        masked_gt = torch.masked_select(gt, label_mask)  # Apply mask to gt

        log_output = torch.log(masked_output)  # log(di_hat)
        log_gt = torch.log(masked_gt)  # log(di)
        diff = log_gt - log_output
        diff = torch.pow(diff, 2)
        mse = torch.mean(diff)

        return mse.item()

def n2_log_mse(self, output, gt):  # the second loss
    # output dimension: (B,H,W)
    ignore_label = 0
    label_mask = gt != ignore_label

    masked_output = torch.masked_select(output, label_mask)  # Apply mask to output
    masked_gt = torch.masked_select(gt, label_mask)  # Apply mask to gt

    log_output = torch.log(masked_output)  # log(di_hat)
    log_gt = torch.log(masked_gt)  # log(di)
    diff = log_gt - log_output
    diff = torch.mean(diff)
    n2_mse = torch.pow(diff, 2)

    return n2_mse.item()

def rmse(self, output, gt):  # the third loss
    ignore_label = 0
    label_mask = gt != ignore_label

    masked_output = torch.masked_select(output, label_mask)  # Apply mask to output
    masked_gt = torch.masked_select(gt, label_mask)  # Apply mask to gt

    diff = (masked_gt - masked_output) / masked_gt
    diff = torch.pow(diff, 2)
    mse = torch.mean(diff)
    mse = torch.sqrt(mse)

    return mse.item()

def loss_depth(self, output, gt):  # loss for depth
    loss = 0
    iter = 0

    output = nn.functional.normalize(output) + 1
    for m, n in zip(output, gt):
        if torch.sum(n) == 0:
            continue


        output_norm = self.depth_act(m)
        gt_norm = n/255

        los_depth = self.log_mse(output_norm, gt_norm) - self.n2_log_mse(output_norm, gt_norm) + self.rmse(output_norm, gt_norm)
        loss += los_depth
        iter += 1

    return loss / iter

# def log_mse(output, target): #the first loss
#     # output dimension: (2, 1, 1026, 1026)
#     log_output = torch.log(output[1]) #log(di_hat)
#     log_target = torch.log(target[1]) # log(di)
#     diff = log_target - log_output
#     diff2 = torch.pow(diff, 2)
#     mse = torch.sum(diff2)/(output.shape[2] * output.shape[3])

#     return mse

# def n2_log_mse(output, target): # the second loss
#     # output dimension: (2, 1, 1026, 1026)
#     log_output = torch.log(output[1])  # log(di_hat)
#     log_target = torch.log(target[1])  # log(di)
#     diff = log_target - log_output
#     diff2 = torch.sum(diff)
#     n2_mse = torch.pow(diff2, 2) / (output.shape[2] * output.shape[3])**2

#     return n2_mse

# def rmse(output, target): # the third loss
#     log_output = torch.log(output[1])  # log(di_hat)
#     log_target = torch.log(target[1])  # log(di)
#     diff = (log_target - log_output)/log_target
#     diff2 = torch.pow(diff, 2)
#     mse = torch.sum(diff2) / (output.shape[2] * output.shape[3])
#     r_mse = torch.sqrt(mse)
#     return r_mse

# def loss_depth(output, target): # loss for depth
#     output_norm = depth_act(output)
#     target_norm = depth_act(target)
#     loss = log_mse(output_norm, target_norm) - n2_log_mse(output_norm, target_norm) + rmse(output_norm, target_norm)
#     return loss

# x = torch.randn(2, 1, 1026, 1026)
# x_positive = torch.abs(x)
# x_output = logits(x_positive)
#
# y = torch.randn(2, 1, 1026, 1026)
# y_positive = torch.abs(y)
# y_output = logits(y_positive)
#
# z1 = log_mse(x_output,y_output)
# z2 = n2_log_mse(x_output,y_output)
# z3 = rmse(x_output,y_output)
# z = loss_depth(x_output, y_output)
#
# print(z)


