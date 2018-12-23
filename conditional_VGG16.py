#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 17:40:54 2018

@author: xie
"""



import torch
import torch.nn as nn
#import os




class CONDITIONAL_VGG16(nn.Module):

    def __init__(self, features, init_weights=True):
        super(CONDITIONAL_VGG16, self).__init__()
        self.features = features
        self.race_classifier=nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 2),
            )
        self.gender_classifier=nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 2),
            )

        self.fc_320_age=nn.Linear(512 * 7 * 7, 512)
        self.fc_101_age_male_black_1=nn.Linear(512, 512)
        self.fc_101_age_male_black_2=nn.Linear(512, 101)
        self.fc_101_age_male_white_1=nn.Linear(512, 512)
        self.fc_101_age_male_white_2=nn.Linear(512, 101)
        self.fc_101_age_female_black_1=nn.Linear(512, 512) 
        self.fc_101_age_female_black_2=nn.Linear(512, 101) 
        self.fc_101_age_female_white_1=nn.Linear(512, 512) 
        self.fc_101_age_female_white_2=nn.Linear(512, 101) 
        self.Relu_male_black_1=nn.ReLU(inplace=True)
        self.Relu_male_white_1=nn.ReLU(inplace=True)
        self.Relu_age=nn.ReLU(inplace=True)
        self.Relu_female_black_1=nn.ReLU(inplace=True)
        self.Relu_female_white_1=nn.ReLU(inplace=True)
        self.Dropout_male_black_1=nn.Dropout()
        self.Dropout_male_white_1=nn.Dropout()
        self.Dropout_age=nn.Dropout()
        self.Dropout_female_black_1=nn.Dropout()
        self.Dropout_female_white_1=nn.Dropout()

        if init_weights:
            self._initialize_weights()
            
            
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x_race=self.race_classifier(x)
        x_gender=self.gender_classifier(x)
        x_age_tmp=self.fc_320_age(x)
        x_age_tmp=self.Relu_age(x_age_tmp)
        x_age_tmp=self.Dropout_age(x_age_tmp)
        
        x_age_male_black=self.fc_101_age_male_black_1(x_age_tmp)
        x_age_male_black=self.Relu_male_black_1(x_age_male_black)
        x_age_male_black=self.Dropout_male_black_1(x_age_male_black)
        x_age_male_black=self.fc_101_age_male_black_2(x_age_male_black)
        
        x_age_male_white=self.fc_101_age_male_white_1(x_age_tmp)
        x_age_male_white=self.Relu_male_white_1(x_age_male_white)
        x_age_male_white=self.Dropout_male_white_1(x_age_male_white)
        x_age_male_white=self.fc_101_age_male_white_2(x_age_male_white)
        
        x_age_female_black=self.fc_101_age_female_black_1(x_age_tmp)
        x_age_female_black=self.Relu_female_black_1(x_age_female_black)
        x_age_female_black=self.Dropout_female_black_1(x_age_female_black)
        x_age_female_black=self.fc_101_age_female_black_2(x_age_female_black)
        
        x_age_female_white=self.fc_101_age_female_white_1(x_age_tmp)
        x_age_female_white=self.Relu_female_white_1(x_age_female_white)
        x_age_female_white=self.Dropout_female_white_1(x_age_female_white)
        x_age_female_white=self.fc_101_age_female_white_2(x_age_female_white)
        
        out=torch.cat((x_age_male_black, x_age_male_white,  
                       x_age_female_black, x_age_female_white,
                       x_race, x_gender), 1)
        return out
    



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}




def conditional_VGG16():
    model=CONDITIONAL_VGG16(features=make_layers(cfg['D']))
    return model