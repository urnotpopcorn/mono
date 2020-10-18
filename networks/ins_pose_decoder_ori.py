# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


"""
same structure like pose_decoder
ins_pose_decoder input: [bs, 1024, 3, 3] 
instead of [bs, 512, 6, 20] in pose_decoder
"""
class InsPoseDecoder(nn.Module):
    def __init__(self, num_RoI_cat_features, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(InsPoseDecoder, self).__init__()

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

        self.net = nn.ModuleList(list(self.convs.values()))
        self.relu = nn.ReLU()

    def forward(self, RoI_feature):
       
        cat_features = self.relu(self.convs["squeeze"](RoI_feature))
        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            # [12, 256, 6, 20]
            # [12, 256, 6, 20]
            # [12, 12, 6, 20]
            if i != 2:
                out = self.relu(out)
        
        out = out.mean(3).mean(2) # [12, 12]
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)  # [12, 2, 1, 6]
        axisangle = out[..., :3] # [12, 2, 1, 3]
        translation = out[..., 3:] # [12, 2, 1, 3]

        return axisangle, translation