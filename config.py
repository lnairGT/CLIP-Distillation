# Details of training and model specified in this file

import torch

batch_size = 64
num_workers = 0
lr = 1e-4
weight_decay = 1e-1
epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# image size
size = 32

# Student model details
patch_sz = 4
width = 256
layers = 6
heads = 8

# Cluster details
clip_loss_wt = 0.5  # Weight to use with L_clip term of loss
