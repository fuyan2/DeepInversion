# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Pavlo Molchanov and Hongxu Yin
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import collections
# from apex import amp
import random
import torch
import torchvision.utils as vutils
from PIL import Image
import os 

import numpy as np

from utils import lr_cosine_policy, lr_policy, beta_policy, mom_cosine_policy, clip, denormalize, create_folder
from PIL import Image

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

    return prec1


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module, r):
        self.r = r
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature * self.r
        # must have no output

    def close(self):
        self.hook.remove()


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2

class DeepInversionClass(object):
    def __init__(self, wandb, rank,bs=84,
                 use_fp16=True, net_teacher=None, path="./gen_images/",
                 final_data_path="/gen_images_final/",
                 fid_images_path='.',
                 parameters=dict(),
                 setting_id=0,
                 jitter=30,
                 do_clip=False,
                 criterion=None,
                 coefficients=dict(),
                 network_output_function=lambda x: x,
                 hook_for_display = None):
        '''
        :param bs: batch size per GPU for image generation
        :param use_fp16: use FP16 (or APEX AMP) for model inversion, uses less memory and is faster for GPUs with Tensor Cores
        :parameter net_teacher: Pytorch model to be inverted
        :param path: path where to write temporal images and data
        :param final_data_path: path to write final images into
        :param parameters: a dictionary of control parameters:
            "resolution": input image resolution, single value, assumed to be a square, 224
            "random_label" : for classification initialize target to be random values
            "start_noise" : start from noise, def True, other options are not supported at this time
            "detach_student": if computing Adaptive DI, should we detach student?
        :param setting_id: predefined settings for optimization:
            0 - will run low resolution optimization for 1k and then full resolution for 1k;
            1 - will run optimization on high resolution for 2k
            2 - will run optimization on high resolution for 20k

        :param jitter: amount of random shift applied to image at every iteration
        :param coefficients: dictionary with parameters and coefficients for optimization.
            keys:
            "r_feature" - coefficient for feature distribution regularization
            "tv_l1" - coefficient for total variation L1 loss
            "tv_l2" - coefficient for total variation L1 loss
            "l2" - l2 penalization weight
            "lr" - learning rate for optimization
            "main_loss_multiplier" - coefficient for the main loss optimization
            "adi_scale" - coefficient for Adaptive DeepInversion, competition, def =0 means no competition

        network_output_function: function to be applied to the output of the network to get the output

        hook_for_display: function to be executed at every print/save call, useful to check accuracy of verifier
        wandb: print output
        '''

        print("Deep inversion class generation")
        print("current process id: ", os.getpid())
        # for reproducibility
        torch.manual_seed(torch.cuda.current_device())
        self.fid_images_path = fid_images_path
        self.net_teacher = net_teacher
        self.wandb = wandb
        self.rank = rank
        self.device = torch.device("cuda:{}".format(rank))
        if "resolution" in parameters.keys():
            self.image_resolution = parameters["resolution"]
            self.channels = parameters["channels"]
            self.random_label = parameters["random_label"]
            self.start_noise = parameters["start_noise"]
            self.detach_student = parameters["detach_student"]
            self.do_flip = parameters["do_flip"]
            self.store_best_images = parameters["store_best_images"]
        else:
            self.image_resolution = 224
            self.random_label = False
            self.start_noise = True
            self.detach_student = False
            self.do_flip = True
            self.store_best_images = False

        self.setting_id = setting_id

        self.bs = bs  # batch size
        self.use_fp16 = use_fp16
        self.do_clip = do_clip

        self.save_every = 100

        self.jitter = jitter
        self.criterion = criterion

        self.network_output_function = network_output_function

        if "r_feature" in coefficients:
            self.bn_reg_scale = coefficients["r_feature"]
            self.var_scale_l1 = coefficients["tv_l1"]
            self.var_scale_l2 = coefficients["tv_l2"]
            self.l2_scale = coefficients["l2"]
            self.lr = coefficients["lr"]
            self.main_loss_multiplier = coefficients["main_loss_multiplier"]
            self.adi_scale = coefficients["adi_scale"]
        else:
            print("Provide a dictionary with ")

        self.num_generations = 0

        self.final_data_path = final_data_path

        ## Create folders for images and logs
        prefix = path
        self.prefix = prefix

        local_rank = torch.cuda.current_device()
        if local_rank==0:
            create_folder(prefix)
            create_folder(prefix + "/best_images/")
            create_folder(self.final_data_path)
            # save images to folders
            # for m in range(1000):
            #     create_folder(self.final_data_path + "/s{:03d}".format(m))

        ## Create hooks for feature statistics
        self.loss_r_feature_layers = []
        cifar10_resnet = True
        r = 500.
        for module in  self.net_teacher.modules():
            if cifar10_resnet:
                if module in self.net_teacher.layer1.modules():
                    r = self.bn_reg_scale**3#100.
                elif module in self.net_teacher.layer2.modules():
                    r = self.bn_reg_scale**2#10.
                elif module in self.net_teacher.layer3.modules():
                    r = self.bn_reg_scale#5.
                elif module in self.net_teacher.layer4.modules():
                    r = 1.
                else:
                    r = 1.0
            else:
                r = 1.
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(module, r))

        self.hook_for_display = None
        if hook_for_display is not None:
            self.hook_for_display = hook_for_display

    def get_images(self, net_student=None, targets=None):
        print("get_images call")
        net_teacher = self.net_teacher

        use_fp16 = self.use_fp16
        save_every = self.save_every

        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        local_rank = torch.cuda.current_device()

        best_cost = 1e4

        criterion = self.criterion

        # setup target labels
        if targets is None:
            #only works for classification now, for other tasks need to provide target vector
            targets = torch.LongTensor([random.randint(0, 9) for _ in range(self.bs)]).to(self.device)
            if not self.random_label:
                # preselected classes, good for ResNet50v1.5
                # targets = [1, 933, 946, 980, 25, 63, 92, 94, 107, 985, 151, 154, 207, 250, 270, 277, 283, 292, 294, 309,
                #            311,
                #            325, 340, 360, 386, 402, 403, 409, 530, 440, 468, 417, 590, 670, 817, 762, 920, 949, 963,
                #            967, 574, 487]
                # targets = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9]
                # targets = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4]
                targets = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9]
                targets = torch.LongTensor(targets * (int(self.bs / len(targets)))).to(self.device)

        img_original = self.image_resolution

        data_type = torch.half if use_fp16 else torch.float
        inputs = torch.randn((self.bs, self.channels, img_original, img_original), requires_grad=True, device=self.device,
                             dtype=data_type)

        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)

        if self.setting_id==0:
            skipfirst = False
        else:
            skipfirst = True

        iteration = 0
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it==0:
                iterations_per_layer = 2000
            else:
                iterations_per_layer = 1000 if not skipfirst else 2000
                if self.setting_id == 2:
                    iterations_per_layer = 20000

            if lr_it==0 and skipfirst:
                continue

            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res

            if self.setting_id == 0:
                #multi resolution, 2k iterations with low resolution, 1k at normal, ResNet50v1.5 works the best, ResNet50 is ok
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
            elif self.setting_id == 1:
                #2k normal resolultion, for ResNet50v1.5; Resnet50 works as well
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
            elif self.setting_id == 2:
                #20k normal resolution the closes to the paper experiments for ResNet50
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.9, 0.999], eps = 1e-8)

            if use_fp16:
                static_loss_scale = 256
                static_loss_scale = "dynamic"
                _, optimizer = amp.initialize([], optimizer, opt_level="O2", loss_scale=static_loss_scale)

            # lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                iteration += 1

                # learning rate scheduling
                # lr_scheduler(optimizer, iteration_loc, iteration_loc)

                # perform downsampling if needed
                if lower_res!=1:
                    inputs_jit = pooling_function(inputs)
                else:
                    inputs_jit = inputs

                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                # Flipping
                flip = random.random() > 0.5
                if flip and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))

                # forward pass
                optimizer.zero_grad()
                net_teacher.zero_grad()
                net_student.zero_grad()
                outputs = net_teacher(inputs_jit)
                outputs = self.network_output_function(outputs)

                # R_cross classification loss
                class_loss = criterion(outputs, targets)

                # R_prior losses
                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

                # R_feature loss
                loss_r_feature = sum([mod.r_feature for mod in self.loss_r_feature_layers])

                # R_ADI
                loss_verifier_cig = torch.zeros(1)
                if self.adi_scale!=0.0:
                    if self.detach_student:
                        outputs_student = net_student(inputs_jit).detach()
                    else:
                        outputs_student = net_student(inputs_jit)

                    T = 3.0
                    if 1:
                        T = 3.0
                        # Jensen Shanon divergence:
                        # another way to force KL between negative probabilities
                        P = nn.functional.softmax(outputs_student / T, dim=1)
                        Q = nn.functional.softmax(outputs / T, dim=1)
                        M = 0.5 * (P + Q)

                        P = torch.clamp(P, 0.01, 0.99)
                        Q = torch.clamp(Q, 0.01, 0.99)
                        M = torch.clamp(M, 0.01, 0.99)
                        eps = 0.0
                        loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
                         # JS criteria - 0 means full correlation, 1 - means completely different
                        loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)

                    if local_rank==0:
                        if iteration % save_every==0:
                            print('loss_verifier_cig', loss_verifier_cig.item())

                # l2 loss on images
                loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()


                # combining losses
                loss_aux = self.var_scale_l2 * loss_var_l2 + \
                           self.var_scale_l1 * loss_var_l1 + \
                           self.bn_reg_scale * loss_r_feature + \
                           self.l2_scale * loss_l2

                if self.adi_scale!=0.0:
                    loss_aux += self.adi_scale * loss_verifier_cig

                loss = self.main_loss_multiplier * class_loss + loss_aux
                Verifier_acc = validate_one(inputs_jit, targets, net_student)
                
                # example_images = []
                # for i in range(10):
                #     example_images.append(self.wandb.Image(inputs_jit[i], caption="Truth: {}".format(targets[i])))

                self.wandb.log({
                    "total Loss": loss, 
                    "feature loss": loss_r_feature, 
                    "class_loss":class_loss, 
                    "Verifier accuracy":Verifier_acc,
                    "Examples" : [self.wandb.Image(i) for i in inputs]})

                if local_rank==0:
                    if iteration % save_every==0:
                        print("------------iteration {}----------".format(iteration))
                        print("total loss", loss.item())
                        print("loss_r_feature", loss_r_feature.item())
                        print("main criterion", class_loss.item())
                        print("verifier acc", Verifier_acc.item())

                # do image update
                if use_fp16:
                    # optimizer.backward(loss)
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()

                # clip color outlayers
                if self.do_clip:
                    inputs.data = clip(inputs.data, use_fp16=use_fp16)

                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.data.clone()

                if iteration % save_every==0 and (save_every > 0):
                    vutils.save_image(inputs,
                                      '{}/output_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                       iteration // save_every,
                                                                                       self.rank),
                                      normalize=True, scale_each=True, nrow=int(10))

        for i in range(self.ny*10):
            save_image(inputs[i], '%s/image%d.png'%(self.final_data_path,i),normalize=True)

        if self.store_best_images:
            best_inputs = denormalize(best_inputs)
            self.save_images(best_inputs, targets)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
        
    def save_images(self, images, targets):
        # method to store generated images locally
        local_rank = torch.cuda.current_device()
        for id in range(images.shape[0]):
            class_id = targets[id].item()
            if 0:
                #save into separate folders
                place_to_store = '{}/s{:03d}/img_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(self.final_data_path, class_id,
                                                                                          self.num_generations, id,
                                                                                          local_rank)
            else:
                place_to_store = '{}/img_s{:03d}_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(self.final_data_path, class_id,
                                                                                          self.num_generations, id,
                                                                                          local_rank)
            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)

    def generate_batch(self, net_student=None, targets=None):
        # for ADI detach student and add put to eval mode
        net_teacher = self.net_teacher

        use_fp16 = self.use_fp16

        # fix net_student
        if not (net_student is None):
            net_student = net_student.eval()

        if targets is not None:
            targets = torch.from_numpy(np.array(targets).squeeze()).cuda()
            if use_fp16:
                targets = targets.half()

        self.get_images(net_student=net_student, targets=targets)

        net_teacher.eval()

        self.num_generations += 1
