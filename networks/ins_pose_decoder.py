# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *


"""
same structure like pose_decoder
ins_pose_decoder input: [bs, 1024, 3, 3] 
instead of [bs, 512, 6, 20] in pose_decoder
"""
class InsPoseDecoder(nn.Module):
    def __init__(self, num_RoI_cat_features, num_input_features, num_frames_to_predict_for=None, stride=1, num_output_channels=1, use_skips=False, predict_delta=False):
        super(InsPoseDecoder, self).__init__()
        self.predict_delta = predict_delta

        self.num_ch_enc = num_RoI_cat_features # 1024
        self.num_input_features = num_input_features # 1
        self.num_frames_to_predict_for = num_frames_to_predict_for # 2
        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(num_RoI_cat_features, 256, 1)    
        # Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))

        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        # Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))

        self.relu = nn.ReLU()

        # delta decoder
        if self.predict_delta == True:
            #self.num_ch_dec = np.array([16, 32, 64, 128, 256])
            self.num_ch_dec = np.array([32, 64, 128, 256, 512])
            self.num_output_channels = num_output_channels
            self.use_skips = use_skips
            self.scales = range(4)
            self.sigmoid = nn.Sigmoid()
            for i in range(4, -1, -1):
                # upconv_0
                #num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
                num_ch_in = self.num_ch_enc if i == 4 else self.num_ch_dec[i + 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

                # upconv_1
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips and i > 0:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

            '''
            for s in self.scales:
                self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            '''
            # define delta_x
            self.convs[("dispconv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
            # define delta_y
            self.convs[("dispconv", 1)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
            # define delta_z
            self.convs[("dispconv", 2)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, RoI_feature):
        cat_features = self.relu(self.convs["squeeze"](RoI_feature))
        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            # [bs, 256, 3, 3] or [bs, 256, 6, 20]
            # [bs, 256, 3, 3] or [bs, 256, 6, 20]
            # [bs, 12, 3, 3] or [bs, 12, 6, 20]
            if i != 2:
                out = self.relu(out)
        
        out = out.mean(3).mean(2) # [bs, 12]
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)  # [bs, 2, 1, 6]
        axisangle = out[..., :3] # [12, 2, 1, 3]
        translation = out[..., 3:] # [12, 2, 1, 3]
        
        # delta decoder
        if self.predict_delta == True:
            x = RoI_feature # bs, 1024, 3, 3 or bs, 1024, 6, 20
            #cat_features = torch.cat(last_features, 0)
            #x = cat_features # bs, 512, 6, 20
            self.outputs = {}
            for i in range(4, -1, -1):
                #upconv	 torch.Size([bs, 256, 12, 40])
                #upconv	 torch.Size([bs, 128, 24, 80])
                #upconv	 torch.Size([bs, 64, 48, 160])
                #upconv	 torch.Size([bs, 32, 96, 320])
                #upconv	 torch.Size([bs, 16, 192, 640])
                #4 	 torch.Size([1, 512, 3, 3]) or torch.Size([1, 512, 6, 20])
                #3 	 torch.Size([1, 256, 6, 6]) or torch.Size([1, 256, 12, 40])
                #2 	 torch.Size([1, 128, 12, 12]) or torch.Size([1, 128, 24, 80])
                #1 	 torch.Size([1, 64, 24, 24]) or torch.Size([1, 64, 48, 160])
                #0 	 torch.Size([1, 32, 48, 48]) or torch.Size([1, 32, 96, 320])
                
                x = self.convs[("upconv", i, 0)](x)
                x = [upsample(x)]
                
                if self.use_skips and i > 0:
                    x += [input_features[i - 1]]
                
                x = torch.cat(x, 1)
                x = self.convs[("upconv", i, 1)](x)
                
                '''
                if i in self.scales:
                    self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                    print('i\t', self.outputs[("disp", i)].shape, self.convs[("dispconv", i)](x).shape)
                '''
            
            delta_x_inv = self.sigmoid(self.convs[("dispconv", 0)](x)) # bs, 1, 192, 640
            delta_y_inv = self.sigmoid(self.convs[("dispconv", 1)](x)) # bs, 1, 192, 640
            delta_z_inv = self.sigmoid(self.convs[("dispconv", 2)](x)) # bs, 1, 192, 640

            delta_x_inv = (delta_x_inv - 0.5) * 2 # rescale to [-1, 1]
            delta_y_inv = (delta_y_inv - 0.5) * 2 # rescale to [-1, 1]
            delta_z_inv = (delta_z_inv - 0.5) * 2 # rescale to [-1, 1]
            return axisangle, translation, delta_x_inv, delta_y_inv, delta_z_inv
        else:
            return axisangle, translation