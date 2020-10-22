# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import time
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageCms
import PIL.Image as pil

import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision

class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 opt=None,
                 mode=None):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = self.pil_loader
        self.to_tensor = transforms.ToTensor()

        self.opt = opt
        self.mode = mode

        # project_dir = "/userhome/34/h3567721/monodepth-project" or "/local/xjqi/monodepth-project"
        if self.opt.instance_pose:
            #self.data_bbox_path = os.path.join(self.opt.project_dir, "dataset", "kitti_data_bbox_eigen_zhou", self.mode)
            self.data_bbox_path = os.path.join("kitti_data_bbox_eigen_zhou", self.mode)
            #self.data_ins_path = os.path.join(self.opt.project_dir, "dataset", "kitti_data_ins_eigen_zhou", self.mode)
            self.data_ins_path = os.path.join("kitti_data_ins_eigen_zhou", self.mode)

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                # PILLOW image: [W,H], [0,255] -> [C, H, W], [0,1] torch.FloadTensor；
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
    
        if self.opt.instance_pose:
            K_num = 5 # only K obj instances are included, K=4, K+1(bg)=K_num
            for k in list(inputs):
                if "ins_id_seg" in k:
                    f = inputs[k]
                    n, im, i = k
                    max_num = int(np.max(f))
                    ins_tensor = torch.Tensor(np.asarray(f))
                    if max_num+1 <= K_num:
                        ins_tensor_one_hot = torch.nn.functional.one_hot(ins_tensor.to(torch.int64), K_num).type(torch.bool)
                    else:
                        ins_tensor_one_hot = torch.nn.functional.one_hot(ins_tensor.to(torch.int64), max_num+1).type(torch.bool)

                    ins_tensor_one_hot = ins_tensor_one_hot[:,:,:K_num]

                    inputs[(n, im, 0)] = ins_tensor_one_hot.permute(2,0,1)

                    # shape: [K+1, 192, 640], 
                    # background mask : [0, :, :]
                    # instance 1 mask : [1, :, :]
                    # instance 2 mask : [2, :, :] ...... K in total, dtype in boolean

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        # only do augmentation in train mode
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
    
        # ['2011_09_26/2011_09_2..._0028_sync', '268', 'l']
        line = self.filenames[index].split() 
        folder = line[0]  # '2011_09_26/2011_09_26_drive_0028_sync'
        if len(line) == 3:
            frame_index = int(line[1]) # 268
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2] # l
        else:
            side = None

        for i in self.frame_idxs: # [0,1,-1]
            try:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
            except FileNotFoundError as fnf_error:
                # if the frame_index = 0, there is no -1 file, so pick the 0
                # out the rame_index+i is out of range, also pick the frame_index
                # print(fnf_error)
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, side, do_flip)

        if self.opt.instance_pose:
            for i in self.frame_idxs: # [0,1,-1]
                try:
                    ins_seg = self.get_sem_ins(self.data_ins_path, folder, frame_index + i, side, do_flip)
                except FileNotFoundError as fnf_error:
                    # print(fnf_error)
                    ins_seg = self.get_sem_ins(self.data_ins_path, folder, frame_index, side, do_flip)

                sig_ins_id_seg = Image.fromarray(np.uint8(ins_seg[:,:,1])).convert("L")
                ins_width, ins_height = sig_ins_id_seg.size
                
                sig_ins_id_seg = sig_ins_id_seg.resize((self.opt.width, self.opt.height), Image.NEAREST)
                inputs[("ins_id_seg", i, -1)] = sig_ins_id_seg

                ratio_w = self.opt.width / ins_width
                ratio_h = self.opt.height / ins_height

                try:
                    ins_RoI_bbox = self.get_ins_bbox(self.data_bbox_path, folder, frame_index + i, side, 
                        ratio_w, ratio_h, self.opt.width, self.opt.height, do_flip)
                except FileNotFoundError as fnf_error:
                    # if the frame_index = 0, there is no -1 file, so pick the frame_index
                    # out the rame_index+i is out of range, also pick the frame_index
                    # print(fnf_error)
                    ins_RoI_bbox = self.get_ins_bbox(self.data_bbox_path, folder, frame_index, side, 
                        ratio_w, ratio_h, self.opt.width, self.opt.height, do_flip)   

                inputs[("ins_RoI_bbox", i, 0)] = torch.Tensor(ins_RoI_bbox)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K).float()
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K).float()

        # https://pytorch.org/docs/stable/torchvision/transforms.html
        if do_color_aug:
            # error in Lab mode, works in RGB mode
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs: # [0,1,-1]
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

            if self.opt.instance_pose:
                del inputs[("ins_id_seg", i, -1)]
        
        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
