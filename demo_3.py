#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:09:31 2018

@author: xie
"""




import numpy as np
import fire
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from Dataset_folder import Dataset_floder as data_prepare
#from CAISA_CNN import CAISA_cnn
from conditional_VGG16 import conditional_VGG16
import torch.utils.model_zoo as model_zoo

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, loader, optimizer, epoch, n_epochs, device, lemta1, lemta2, print_freq=1):

    batch_time = AverageMeter()
    male_black_softmax_losses=AverageMeter()
    male_white_softmax_losses=AverageMeter()
    female_black_softmax_losses=AverageMeter()
    female_white_softmax_losses=AverageMeter()   
    male_black_mean_losses=AverageMeter()
    male_white_mean_losses=AverageMeter()
    female_black_mean_losses=AverageMeter()
    female_white_mean_losses=AverageMeter()   
    male_black_deviation_losses=AverageMeter()
    male_white_deviation_losses=AverageMeter()
    female_black_deviation_losses=AverageMeter()
    female_white_deviation_losses=AverageMeter()
    age_losses=AverageMeter()
    race_losses=AverageMeter()
    gender_losses=AverageMeter()
    total_losses=AverageMeter()
    race_prec=AverageMeter()
    gender_prec=AverageMeter()  
    train_MAE=AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # Create vaiables
        input_var, target_var = input.to(device), target.to(device)     

        # compute output
        output = model(input_var)
        
        batch_size=target.size(0)
        race_prec1, gender_prec1, MAE=accuracy_MAE(output.cpu(), target)
#        age_class_prec1, MAE=accuracy_MAE(output.data.cpu(), target)


        age_range=np.arange(101).reshape((1,101))
        age_list=np.tile(age_range,(batch_size,1))
        age_list_var=torch.from_numpy(age_list).to(device).float()   
        
        pred_age_male_black_prob=F.softmax(output[:,0:101],1)
        pred_age_male_white_prob=F.softmax(output[:,101:202],1)
        pred_age_female_black_prob=F.softmax(output[:,202:303],1)
        pred_age_female_white_prob=F.softmax(output[:,303:404],1)
        pred_race_prob=F.softmax(output[:,404:406],1)
        pred_gender_prob=F.softmax(output[:,406:408],1)
        batches_predict_age_male_black=torch.sum(age_list_var*pred_age_male_black_prob,dim=1)  
        batches_predict_age_male_white=torch.sum(age_list_var*pred_age_male_white_prob,dim=1) 
        batches_predict_age_female_black=torch.sum(age_list_var*pred_age_female_black_prob,dim=1) 
        batches_predict_age_female_white=torch.sum(age_list_var*pred_age_female_white_prob,dim=1)
        
        batches_predict_age=(pred_gender_prob[:,0]*pred_race_prob[:,0]*batches_predict_age_male_black+
                             pred_gender_prob[:,0]*pred_race_prob[:,1]*batches_predict_age_male_white+
                             pred_gender_prob[:,1]*pred_race_prob[:,0]*batches_predict_age_female_black+
                             pred_gender_prob[:,1]*pred_race_prob[:,1]*batches_predict_age_female_white) 
        
        
#        print(target_var)
        target_age_list_var=target_var[:,0].expand(101,-1).float().t()
        
        
        batch_deviations_male_black=(target_age_list_var-age_list_var).pow(2).mul(pred_age_male_black_prob)
        batch_deviations_male_white=(target_age_list_var-age_list_var).pow(2).mul(pred_age_male_white_prob)
        batch_deviations_female_black=(target_age_list_var-age_list_var).pow(2).mul(pred_age_female_black_prob)
        batch_deviations_female_white=(target_age_list_var-age_list_var).pow(2).mul(pred_age_female_white_prob)

        loss_mean_age_male_black=F.l1_loss(batches_predict_age_male_black, target_var[:,0].float())
        loss_mean_age_male_white=F.l1_loss(batches_predict_age_male_white, target_var[:,0].float())
        loss_mean_age_female_black=F.l1_loss(batches_predict_age_female_black, target_var[:,0].float())
        loss_mean_age_female_white=F.l1_loss(batches_predict_age_female_white, target_var[:,0].float())

        
        loss_deviation_male_black=batch_deviations_male_black.sum(dim=1).sum()/batch_size
        loss_deviation_male_white=batch_deviations_male_white.sum(dim=1).sum()/batch_size
        loss_deviation_female_black=batch_deviations_female_black.sum(dim=1).sum()/batch_size
        loss_deviation_female_white=batch_deviations_female_white.sum(dim=1).sum()/batch_size
        
        loss_softmax_male_black=F.cross_entropy(output[:,0:101], target_var[:,0])    
        loss_softmax_male_white=F.cross_entropy(output[:,101:202], target_var[:,0])  
        loss_softmax_female_black=F.cross_entropy(output[:,202:303], target_var[:,0])  
        loss_softmax_female_white=F.cross_entropy(output[:,303:404], target_var[:,0])  
        loss_age=F.l1_loss(batches_predict_age, target_var[:,0].float())
        loss_race=F.cross_entropy(output[:,404:406], target_var[:,1])
        loss_gender=F.cross_entropy(output[:,406:408], target_var[:,2])

        loss=((loss_softmax_male_black+loss_softmax_male_white+loss_softmax_female_black+loss_softmax_female_white)+
              (loss_deviation_male_black+loss_deviation_male_white+loss_deviation_female_black+loss_deviation_female_white)*lemta2+
              loss_age*lemta1+loss_race+loss_gender)        


        male_black_softmax_losses.update(loss_softmax_male_black.item(), batch_size)
        male_white_softmax_losses.update(loss_softmax_male_white.item(), batch_size)
        female_black_softmax_losses.update(loss_softmax_female_black.item(), batch_size)
        female_white_softmax_losses.update(loss_softmax_female_white.item(), batch_size)
        male_black_mean_losses.update(loss_mean_age_male_black.item(), batch_size)
        male_white_mean_losses.update(loss_mean_age_male_white.item(), batch_size)
        female_black_mean_losses.update(loss_mean_age_female_black.item(), batch_size)
        female_white_mean_losses.update(loss_mean_age_female_white.item(), batch_size)
        male_black_deviation_losses.update(loss_deviation_male_black.item(), batch_size)
        male_white_deviation_losses.update(loss_deviation_male_white.item(), batch_size)
        female_black_deviation_losses.update(loss_deviation_female_black.item(), batch_size)
        female_white_deviation_losses.update(loss_deviation_female_white.item(), batch_size)
        age_losses.update(loss_age.item(), batch_size)
        race_losses.update(loss_race.item(), batch_size)
        gender_losses.update(loss_gender.item(), batch_size)
        total_losses.update(loss.item(), batch_size)
        race_prec.update(race_prec1.item(), batch_size)
        gender_prec.update(gender_prec1.item(), batch_size)
        train_MAE.update(MAE.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'mbs_loss %.3f (%.3f)' % (male_black_softmax_losses.val, male_black_softmax_losses.avg),
                'mws_loss %.3f (%.3f)' % (male_white_softmax_losses.val, male_white_softmax_losses.avg),
                'fbs_loss %.3f (%.3f)' % (female_black_softmax_losses.val, female_black_softmax_losses.avg),
                'fws_loss %.3f (%.3f)' % (female_white_softmax_losses.val, female_white_softmax_losses.avg),
                'mbm_loss %.3f (%.3f)' % (male_black_mean_losses.val, male_black_mean_losses.avg),
                'mwm_loss %.3f (%.3f)' % (male_white_mean_losses.val, male_white_mean_losses.avg),
                'fbm_loss %.3f (%.3f)' % (female_black_mean_losses.val, female_black_mean_losses.avg),
                'fwm_loss %.3f (%.3f)' % (female_white_mean_losses.val, female_white_mean_losses.avg),
                'mbd_loss %.3f (%.3f)' % (male_black_deviation_losses.val, male_black_deviation_losses.avg),
                'mwd_loss %.3f (%.3f)' % (male_white_deviation_losses.val, male_white_deviation_losses.avg),
                'fbd_loss %.3f (%.3f)' % (female_black_deviation_losses.val, female_black_deviation_losses.avg),
                'fwd_loss %.3f (%.3f)' % (female_white_deviation_losses.val, female_white_deviation_losses.avg),
                'age_loss %.3f (%.3f)' % (age_losses.val, age_losses.avg),
                'race_loss %.3f (%.3f)' % (race_losses.val, race_losses.avg),
                'gender_loss %.3f (%.3f)' % (gender_losses.val, gender_losses.avg),
                'total_loss %.3f (%.3f)' % (total_losses.val, total_losses.avg),
                'race_prec %.3f (%.3f)' % (race_prec.val, race_prec.avg),
                'gender_prec %.3f (%.3f)' % (gender_prec.val, gender_prec.avg),
                'train_MAE %.3f (%.3f)' % (train_MAE.val, train_MAE.avg),
            ])
#            print(res)

    # Return summary statistics
    return (batch_time.avg, 
            male_black_softmax_losses.avg, male_black_mean_losses.avg, male_black_deviation_losses.avg,
            male_white_softmax_losses.avg, male_white_mean_losses.avg, male_white_deviation_losses.avg,
            female_black_softmax_losses.avg, female_black_mean_losses.avg, female_black_deviation_losses.avg,
            female_white_softmax_losses.avg, female_white_mean_losses.avg, female_white_deviation_losses.avg,
            age_losses.avg, race_losses.avg, gender_losses.avg, total_losses.avg, race_prec.avg, gender_prec.avg, train_MAE.avg)




def test(model, loader, device, model_state_dir, lemta1, lemta2, print_freq=1):

    batch_time = AverageMeter()
    race_prec=AverageMeter()
    gender_prec=AverageMeter()  
    test_MAE=AverageMeter()    
    
    # Model on eval mode
    model.eval()
    
    absolute_error_all=0
    count=0

    AE_list=[]
    predict_age_list=[]
    real_age_list=[]

    end = time.time()
    with torch.no_grad():  
        for batch_idx, (input, target) in enumerate(loader):

            # Create vaiables
            input_var, target_var = input.to(device), target.to(device)     
    
            # compute output
            model.load_state_dict(torch.load(model_state_dir))
            output = model(input_var)
            
            batch_size=target.size(0)
            race_prec1, gender_prec1, MAE=accuracy_MAE(output.detach().cpu(), target)
    #        age_class_prec1, MAE=accuracy_MAE(output.data.cpu(), target)
    
    
            age_range=np.arange(101).reshape((1,101))
            age_list=np.tile(age_range,(batch_size,1))
            age_list_var=torch.from_numpy(age_list).to(device).float()   
            
            pred_age_male_black_prob=F.softmax(output[:,0:101],1)
            pred_age_male_white_prob=F.softmax(output[:,101:202],1)
            pred_age_female_black_prob=F.softmax(output[:,202:303],1)
            pred_age_female_white_prob=F.softmax(output[:,303:404],1)
            pred_race_prob=F.softmax(output[:,404:406],1)
            pred_gender_prob=F.softmax(output[:,406:408],1)
            batches_predict_age_male_black=torch.sum(age_list_var*pred_age_male_black_prob,dim=1)  
            batches_predict_age_male_white=torch.sum(age_list_var*pred_age_male_white_prob,dim=1) 
            batches_predict_age_female_black=torch.sum(age_list_var*pred_age_female_black_prob,dim=1) 
            batches_predict_age_female_white=torch.sum(age_list_var*pred_age_female_white_prob,dim=1)
            
            batches_predict_age=(pred_gender_prob[:,0]*pred_race_prob[:,0]*batches_predict_age_male_black+
                                 pred_gender_prob[:,0]*pred_race_prob[:,1]*batches_predict_age_male_white+
                                 pred_gender_prob[:,1]*pred_race_prob[:,0]*batches_predict_age_female_black+
                                 pred_gender_prob[:,1]*pred_race_prob[:,1]*batches_predict_age_female_white) 
            

            race_prec.update(race_prec1.item(), batch_size)
            gender_prec.update(gender_prec1.item(), batch_size)
            test_MAE.update(MAE.item(), batch_size)
    
            batch_AE=torch.abs(batches_predict_age-target_var[:,0].float())
            AE_list.extend(batch_AE.cpu())
            predict_age_list.extend(batches_predict_age.cpu())
            real_age_list.extend(target[:,0])

            absolute_error_all+=torch.sum(torch.abs(batches_predict_age-target_var[:,0].float()))
            count+=batch_size     
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'valid'
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'race_prec %.3f (%.3f)' % (race_prec.val, race_prec.avg),
                    'gender_prec %.3f (%.3f)' % (gender_prec.val, gender_prec.avg),
                    'test_MAE %.3f (%.3f)' % (test_MAE.val, test_MAE.avg),
                ])
#                print(res)  
                
        calculated_MAE=absolute_error_all.cpu()/count
        print('absolute_error_all: %d' % absolute_error_all.cpu())
        print('count: %d' % count)
        print('calculated_MAE: %.4f' % calculated_MAE) 

        return batch_time.avg, race_prec.avg, gender_prec.avg, test_MAE.avg, AE_list, predict_age_list, real_age_list



def accuracy_MAE(output, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        _,pred_race=output[:,404:406].topk(1, 1, True, True)  
        _,pred_gender=output[:,406:408].topk(1, 1, True, True)
        
        correct_race=pred_race.squeeze().eq(target[:,1])
        correct_race=correct_race.view(-1).float().sum(0)
        prec_race=correct_race.mul(100.0/batch_size)
        
        correct_gender=pred_gender.squeeze().eq(target[:,2])
        correct_gender=correct_gender.view(-1).float().sum(0)
        prec_gender=correct_gender.mul(100.0/batch_size)
         

        age_range=np.arange(101).reshape((1,101))
        age_list=np.tile(age_range,(batch_size,1))
        age_list_var=torch.from_numpy(age_list).float()   
        
        pred_age_male_black_prob=F.softmax(output[:,0:101],1)
        pred_age_male_white_prob=F.softmax(output[:,101:202],1)
        pred_age_female_black_prob=F.softmax(output[:,202:303],1)
        pred_age_female_white_prob=F.softmax(output[:,303:404],1)
        pred_race_prob=F.softmax(output[:,404:406],1)
        pred_gender_prob=F.softmax(output[:,406:408],1)
        batches_predict_age_male_black=torch.sum(age_list_var*pred_age_male_black_prob,dim=1)  
        batches_predict_age_male_white=torch.sum(age_list_var*pred_age_male_white_prob,dim=1) 
        batches_predict_age_female_black=torch.sum(age_list_var*pred_age_female_black_prob,dim=1) 
        batches_predict_age_female_white=torch.sum(age_list_var*pred_age_female_white_prob,dim=1)
        
        batches_predict_age=(pred_gender_prob[:,0]*pred_race_prob[:,0]*batches_predict_age_male_black+
                             pred_gender_prob[:,0]*pred_race_prob[:,1]*batches_predict_age_male_white+
                             pred_gender_prob[:,1]*pred_race_prob[:,0]*batches_predict_age_female_black+
                             pred_gender_prob[:,1]*pred_race_prob[:,1]*batches_predict_age_female_white) 
       
#        batches_predict_age=torch.sum(age_list_var*pred_age_prob,dim=1)
#        batches_predict_age=torch.sum(age_list_var*pred_age_prob,dim=1).round()
        
        absolute_error_all=torch.sum(torch.abs(batches_predict_age-target[:,0].float()))
        MAE=absolute_error_all/batch_size
        
        return prec_race, prec_gender, MAE




def demo(data_root, train_list, test_list, save, n_epochs=1,
         batch_size=64, lr=0.01, wd=0.0005, momentum=0.9, lemta1=0.1, lemta2=0.004, seed=None):
#def demo(data_root, train_list, validation_list, test_list, save, n_epochs=1,
#      batch_size=64, lr=0.001, wd=0.0005, seed=None):
    """
    A demo to show off training and testing of :
    "Deep facial age estimation using conditional multitask learning with weak label esxpansion."
    Trains and evaluates a mean-variance loss on MOPPH Album2 dataset.

    Args:
        data_root (str) - path to directory where data exist
        train_list (str) - path to directory where train_data_list exist
        validation_list (str) - path to directory where validation_data_list exist
        test_list (str) - path to directory where test_data_list exist
        save (str) - path to save the model and results to 

        n_epochs (int) - number of epochs for training (default 3)
        batch_size (int) - size of minibatch (default 64)
        lr (float) - base lerning rate (default 0.001)
        wd (float) -weight deday (default 0.0001)
        momentum (float) momentum (default 0.9)
        seed (int) - manually set the random seed (default None)
    """

    # Mean and std value from Imagenet 
    mean=[0.485, 0.456, 0.406]
    stdv=[0.229, 0.224, 0.225]
#    mean=[0.5, 0.5, 0.5]
#    stdv=[0.5, 0.5, 0.5]
    train_transforms = transforms.Compose([
#        transforms.Resize(146),
#        transforms.RandomCrop(128),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5), 
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
#        transforms.Resize(146),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    # Datasets
    train_set = data_prepare(data_root=data_root, data_list=train_list, transform=train_transforms)
    test_set = data_prepare(data_root=data_root, data_list=test_list, transform=test_transforms)    
    

    pretrained_dict=model_zoo.load_url(model_urls['vgg16'])
#    print(pretrained_dict.keys())
    model=conditional_VGG16()
#    print(model)
    
    model_dict = model.state_dict()
#    print(model_dict.keys())
#    os._exit(0)
#    del model_dict[]
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#    print(pretrained_dict.keys())
#    os._exit(0)
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


    
    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    # Model on cuda
    use_cuda=torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # prepare data
    if seed is not None:
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed_all(seed)

            
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=4)  
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=4)
      
    # Wrap model for multi-GPUs, if necessary
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model.features = torch.nn.DataParallel(model.features)
    model_wrapper = model.to(device)
    # Optimizer
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
#    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
#                                                     gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,70,90],
                                                     gamma=0.1)
#    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=lr, weight_decay=wd, amsgrad=True)

    # Start log
    if os.path.exists(os.path.join(save, 'conditional_VGG16_nesterov_results_3.csv')):
        os.remove(os.path.join(save, 'conditional_VGG16_nesterov_results_3.csv'))
    with open(os.path.join(save, 'conditional_VGG16_nesterov_results_3.csv'), 'w') as f:
        f.write('epoch, '
                'train_softmax_loss_male_black, train_mean_loss_male_black, train_deviation_loss_male_black, '
                'train_softmax_loss_male_white, train_mean_loss_male_white, train_deviation_loss_male_white, '
                'train_softmax_loss_female_black, train_mean_loss_female_black, train_deviation_loss_female_black, '
                'train_softmax_loss_female_white, train_mean_loss_female_white, train_deviation_loss_female_white, '
                'train_age_loss, train_race_loss, train_gender_loss, train_total_loss, train_race_accuracy, train_gender_accuracy, train_MAE\n')
    # Train and validate model
    best_MAE = 100
    model_state_dir=os.path.join(save, 'conditional_VGG16_nesterov_model_3.dat')
    for epoch in range(n_epochs):

        scheduler.step()
        _, tra_slmb, tra_mlmb, tra_dlmb, tra_slmw, tra_mlmw, tra_dlmw, tra_slfb, tra_mlfb, tra_dlvb, tra_slfw, tra_mlfw, tra_dlfw, tra_al, tra_rl, tra_gl, tra_tl, tra_ra, tra_ga, train_MAE = train(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            device=device,
            lemta1=lemta1,
            lemta2=lemta2
        )
        print('*********************************')
#         Determine if model is the best

        if train_MAE < best_MAE:
            best_MAE = train_MAE
            print('New best MAE: %.4f' % best_MAE)
            if os.path.exists(model_state_dir):
                os.remove(model_state_dir)
            torch.save(model_wrapper.state_dict(), model_state_dir)
        
        with open(os.path.join(save, 'conditional_VGG16_nesterov_results_3.csv'), 'a') as f:
            f.write('%03d, '
                    '%0.3f, %0.3f, %0.3f, '
                    '%0.3f, %0.3f, %0.3f, '
                    '%0.3f, %0.3f, %0.3f, '
                    '%0.3f, %0.3f, %0.3f, '
                    '%0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f\n' % (
                    (epoch + 1),
                    tra_slmb, tra_mlmb, tra_dlmb,
                    tra_slmw, tra_mlmw, tra_dlmw,
                    tra_slfb, tra_mlfb, tra_dlvb,
                    tra_slfw, tra_mlfw, tra_dlfw,
                    tra_al, tra_rl, tra_gl, tra_tl, tra_ra, tra_ga, train_MAE
                ))


   # Test model       
    _, tes_ra, tes_ga, test_MAE, AE_list, predict_age_list, real_age_list = test(
        model=model_wrapper,
        loader=test_loader,
        device=device,
        model_state_dir=model_state_dir,
        lemta1=lemta1,
        lemta2=lemta2
    )
    CS_1_numerator=CS_2_numerator=CS_3_numerator=CS_4_numerator=CS_5_numerator=CS_6_numerator=CS_7_numerator=CS_8_numerator=CS_9_numerator=CS_10_numerator=0
    for i in range(len(AE_list)):
        if AE_list[i]<=1:
            CS_1_numerator+=1
        if AE_list[i]<=2:
            CS_2_numerator+=1
        if AE_list[i]<=3:
            CS_3_numerator+=1
        if AE_list[i]<=4:
            CS_4_numerator+=1
        if AE_list[i]<=5:
            CS_5_numerator+=1
        if AE_list[i]<=6:
            CS_6_numerator+=1
        if AE_list[i]<=7:
            CS_7_numerator+=1
        if AE_list[i]<=8:
            CS_8_numerator+=1
        if AE_list[i]<=9:
            CS_9_numerator+=1
        if AE_list[i]<=10:
            CS_10_numerator+=1
            
    CS_1=CS_1_numerator/len(AE_list)
    CS_2=CS_2_numerator/len(AE_list)
    CS_3=CS_3_numerator/len(AE_list)
    CS_4=CS_4_numerator/len(AE_list)
    CS_5=CS_5_numerator/len(AE_list)
    CS_6=CS_6_numerator/len(AE_list)
    CS_7=CS_7_numerator/len(AE_list)
    CS_8=CS_8_numerator/len(AE_list)
    CS_9=CS_9_numerator/len(AE_list)
    CS_10=CS_10_numerator/len(AE_list)    
    with open(test_list) as f:
        test_lines=f.readlines()
    
    index=0
    with open(os.path.join(save, 'conditional_VGG16_nesterov_results_3.csv'), 'a') as f:
        f.write('******************************\n')
        f.write('\n')
        f.write('******************************\n')
        f.write('test_img_name, real_age, predict_age, absolute_error(AE):\n')
    for test_line in test_lines:
        img_name=test_line.split()[0]
        img_predict_age=predict_age_list[index].item()
        img_real_age=real_age_list[index].item()
        img_AE=AE_list[index].item()
        record_line=img_name+'  '+str(round(img_real_age,2))+'  '+str(round(img_predict_age,2))+'  '+str(round(img_AE,2))+'\n'
        index+=1
        with open(os.path.join(save, 'conditional_VGG16_nesterov_results_3.csv'), 'a') as f:
            f.write(record_line)
    with open(os.path.join(save, 'conditional_VGG16_nesterov_results_3.csv'), 'a') as f:
        f.write('******************************\n')
        f.write('test_race_accuracy, test_gender_accuracy, test_MAE\n')
        f.write('%0.3f, %0.3f, %0.3f\n'%(tes_ra, tes_ga, test_MAE))
        f.write('CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10:\n')
        f.write('%0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f\n'
                % (CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10))


"""
A demo to train and testv MORPH Album2 dataset with protocol S1-S2-S3.

usage:
python demo.py --data_root <path_to_data_dir> --data_list <path_to_data_list_dir> --save <path_to_save_dir>


"""
if __name__ == '__main__':
    fire.Fire(demo)