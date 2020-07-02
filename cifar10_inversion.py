# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Pavlo Molchanov and Hongxu Yin
# --------------------------------------------------------

from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse

import torch
from torch import distributed, nn

import random
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from torchvision import datasets, transforms

import numpy as np

# from apex import amp

import os
import torch.multiprocessing as mp

import torchvision.models as models
from utils import load_model_pytorch, distributed_is_initialized
from deepinversion import validate_one
from resnet import *

import wandb

random.seed(0)

def run(coefficients):
    torch.backends.cudnn.benchmark = True
    use_fp16 = False
    torch.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_cifar10_half = False
    if train_cifar10_half:
        net = resnet34(num_classes=5)
        net.load_state_dict(torch.load('models/cifar10_resnet34_classifier_half.pth', map_location=torch.device(device)))
        net_verifier = resnet18(num_classes=5)
        net_verifier.load_state_dict(torch.load('models/cifar10_resnet18_verifier_half.pth', map_location=torch.device(device)))

    else:
        net = resnet34(pretrained=True)
        net_verifier = resnet18(pretrained=True)

    resnet = True
    update_generator = False
    net = net.to(device)

    ### load feature statistics
    print('==> Getting BN params as feature statistics')
    feature_statistics = list()
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            mean, std = module.running_mean.data, module.running_var.data
            if use_fp16:
                feature_statistics.append((std.type(torch.float16), mean.type(torch.float16)))
            else:
                feature_statistics.append((std, mean))
    net.eval()
    net_verifier.to(device)
    net_verifier.eval()


    from deepinversion import DeepInversionClass

    exp_name = "dream_cifar10"
    # final images will be stored here:
    adi_data_path = "%s/final_images/%s"%(coefficients.output_folder,exp_name)
    # temporal data and generations will be stored here
    exp_name = "%s/generations/%s"%(coefficients.output_folder,exp_name)

    iterations = 10000
    start_noise = True
    # args.detach_student = False

    resolution = 32
    bs = 100
    jitter = 3 #random shift, 30,3

    parameters = dict()
    parameters["resolution"] = 32#64
    parameters["random_label"] = False
    parameters["start_noise"] = True
    parameters["detach_student"] = False
    parameters["channels"] = 3
    parameters["do_flip"] = True
    parameters["store_best_images"] = True

    
    criterion = nn.CrossEntropyLoss()
    wandb.init(project="cifar10_inversion")
    wandb.config.r_feature = coefficients.r_feature
    wandb.config.tv_l2 = coefficients.tv_l2
    wandb.config.l2 = coefficients.l2
    wandb.config.lr = coefficients.lr
    wandb.config.main_loss_multiplier = coefficients.main_loss_multiplier
    wandb.config.adi_scale = coefficients.adi_scale


    network_output_function = lambda x: x

    # check accuracy of verifier
    verifier = True
    if verifier:
        hook_for_display = lambda x,y: validate_one(x, y, net_verifier)
    else:
        hook_for_display = None

    torch.backends.cudnn.benchmark = True

    DeepInversionEngine = DeepInversionClass(wandb, rank,net_teacher=net,
                                              final_data_path=adi_data_path,
                                              path=exp_name,
                                              parameters=parameters,
                                              setting_id=2,
                                              bs = bs,
                                              use_fp16 = False,
                                              jitter = jitter,
                                              criterion=criterion,
                                              coefficients = coefficients,
                                              network_output_function = network_output_function,
                                              hook_for_display = hook_for_display)
    net_student=None
    if coefficients.adi_scale != 0: 
        net_student = net_verifier
    DeepInversionEngine.generate_batch(net_student=net_student)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--r_feature', default=5, type=float)
    parser.add_argument('--tv_l1', default=0, type=float)
    parser.add_argument('--tv_l2', default=2.5e-5, type=float)
    parser.add_argument('--l2', default=3e-8, type=float)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--main_loss_multiplier', default=1, type=float)
    parser.add_argument('--adi_scale', default=0.001, type=float)
    parser.add_argument('--resolution', default=32, type=int)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--gan_location', type=str)
    parser.add_argument('--checkpoint_folder', type=str)
    parser.add_argument('--decay',default=1e-5, type=float)
    args = parser.parse_args()
    
    run(args)

