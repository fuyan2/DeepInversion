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
from multiprocessing import Process

import torchvision.models as models
from utils import load_model_pytorch, distributed_is_initialized
from resnet import *

import wandb
wandb.init(project="deepinversion")
random.seed(0)


def validate_one(input, target, model):
    """Perform validation on the validation set"""

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())
    #wandb.log({"Verifier accuracy:", prec1})

def run(net, net_verifier, coefficients=dict()):
    torch.backends.cudnn.benchmark = True
    use_fp16 = False
    torch.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    adi_data_path = "./final_images/%s"%exp_name
    # temporal data and generations will be stored here
    exp_name = "generations/%s"%exp_name

    iterations = 10000
    start_noise = True
    # args.detach_student = False

    resolution = 32
    bs = 50
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
    # coefficients = dict()
    # coefficients["r_feature"] = wandb.config.r_feature
    # coefficients["tv_l1"] = wandb.config.tv_l2 
    # coefficients["tv_l2"] = wandb.config.l2
    # coefficients["l2"] = wandb.config.lr
    # coefficients["lr"] = wandb.config.lr
    # coefficients["main_loss_multiplier"] = wandb.config.main_loss_multiplier 
    # coefficients["adi_scale"] = wandb.config.adi_scale


    network_output_function = lambda x: x

    # check accuracy of verifier
    verifier = True
    if verifier:
        hook_for_display = lambda x,y: validate_one(x, y, net_verifier)
    else:
        hook_for_display = None

    torch.backends.cudnn.benchmark = True

    DeepInversionEngine = DeepInversionClass(wandb, net_teacher=net,
                                              final_data_path=adi_data_path,
                                              path=exp_name,
                                              parameters=parameters,
                                              setting_id=0,
                                              bs = bs,
                                              use_fp16 = False,
                                              jitter = jitter,
                                              criterion=criterion,
                                              coefficients = coefficients,
                                              network_output_function = network_output_function,
                                              hook_for_display = hook_for_display)
    net_student=None
    if wandb.config.adi_scale != 0: 
        net_student = net_verifier
    DeepInversionEngine.generate_batch(net_student=net_student)


def main():

    wandb.config.r_feature = 1.
    wandb.config.tv_l2 = 128  
    wandb.config.l2 = 3e-8
    wandb.config.lr = [0.1, 1e-2, 1e-3]
    wandb.config.main_loss_multiplier = 1.
    wandb.config.adi_scale = [10,1,0.1]

    train_cifar10_half = False
    if train_cifar10_half:
        net = resnet34(num_classes=5)
        net.load_state_dict(torch.load('models/cifar10_resnet34_classifier_half.pth', map_location=torch.device(device)))
    else:
        net = resnet34(pretrained=True)

    # net.share_memory()

    train_cifar10_half = False
    if train_cifar10_half:
        net_verifier = resnet18(num_classes=5)
        net_verifier.load_state_dict(torch.load('models/cifar10_resnet18_verifier_half.pth', map_location=torch.device(device)))
    else:
        net_verifier = resnet18(pretrained=True)

    # num_processes = 9
    processes = []
    for lr  in wandb.config.lr:
        for adi_scale in wandb.config.adi_scale:
            coefficients = dict()
            coefficients["r_feature"] = wandb.config.r_feature
            coefficients["tv_l1"] = wandb.config.tv_l2 
            coefficients["tv_l2"] = wandb.config.l2
            coefficients["l2"] = wandb.config.lr
            coefficients["lr"] = lr
            coefficients["main_loss_multiplier"] = wandb.config.main_loss_multiplier 
            coefficients["adi_scale"] = adi_scale
            p = mp.Process(target=run, args=(net,net_verifier,coefficients,))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()





if __name__ == '__main__':
    main()

