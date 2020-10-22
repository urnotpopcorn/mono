# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

import warnings

class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0,    0.5, 0],
                           [0,    1.92, 0.5, 0],
                           [0,    0,    1,   0],
                           [0,    0,    0,   1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0] # 979
        frame_index = int(line[1]) # 979

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


    def get_sem_ins(self, sem_ins_path, folder, frame_index, side, do_flip):
        sem_ins = np.load(self.get_sem_ins_path(sem_ins_path, folder, frame_index, side))

        if do_flip:
            sem_ins = np.fliplr(sem_ins)

        return sem_ins

    def omit_small_RoI_pad(self, x_1, y_1, x_2, y_2, width, height):
        RoI_width = x_2 - x_1
        RoI_height = y_2 - y_1

        # pad the RoI with ratio 1.3
        RoI_width_pad = RoI_width * 0.15
        RoI_height_pad = RoI_height * 0.15
        
        # 
        # (x1, y1) ----------------------------
        # |                |                  |
        # |                |                  |
        # |---------RoI: bbox of the Mask ----|
        # |                |                  |
        # |                |                  |
        # ------------------------------(x2, y2)

        if RoI_width * RoI_height < 10*10:
            # if the obj is too small, use the entire img
            x_1 = 0
            y_1 = 0
            x_2 = width
            y_2 = height
        else:
            x_1 = 0 if x_1 - RoI_width_pad <= 0 else x_1 - RoI_width_pad
            y_1 = 0 if y_1 - RoI_height_pad <= 0 else y_1 - RoI_height_pad
            x_2 = width if x_2 + RoI_width_pad >= width else x_2 + RoI_width_pad
            y_2 = height if y_2 + RoI_height_pad >= height else y_2 + RoI_height_pad
        
        return x_1, y_1, x_2, y_2
    
    def get_ins_bbox(self, data_bbox_path, folder, frame_index, side, 
                        ratio_w, ratio_h, width, height, do_flip):

        # method 1: extract bbox from ins data
        # method 2: load bbox from local disk
        with warnings.catch_warnings():
            # if there is no bbox, the txt file is empty.
            warnings.simplefilter("ignore")
            ins_bbox_mat = np.loadtxt(self.get_ins_txt_path(data_bbox_path, folder, frame_index, side))
        
        K_num = 5 # assume the maximum k+1=4+1=5 (including the bg)
        if len(ins_bbox_mat) > 0:
            if len(ins_bbox_mat.shape) == 1:
                # if there is only one obj
                ins_bbox_mat = np.expand_dims(ins_bbox_mat, 0)
                # (4,) -> (1,4)

            RoI_bbox = []
            if len(ins_bbox_mat) >= K_num-1: # e.g. 4 >= 4 or 5 >= 4
                select_K = K_num-1 # select_K = 4
            else: # 3 < 4
                select_K = len(ins_bbox_mat)

            for i in range(select_K): # only K obj instances are included, K=4
                x_1 = int(ins_bbox_mat[i, 0] * ratio_w)
                y_1 = int(ins_bbox_mat[i, 1] * ratio_h)
                x_2 = int(ins_bbox_mat[i, 2] * ratio_w)
                y_2 = int(ins_bbox_mat[i, 3] * ratio_h)

                x_1, y_1, x_2, y_2 = self.omit_small_RoI_pad(x_1, y_1, x_2, y_2, width, height)
                if do_flip:
                    RoI_bbox.append([(width - x_2)/32, y_1/32, (width - x_1)/32, y_2/32])
                else:
                    RoI_bbox.append([x_1/32, y_1/32, x_2/32, y_2/32])

            if len(ins_bbox_mat) < K_num-1: 
                x_1 = 0
                y_1 = 0
                x_2 = width/32
                y_2 = height/32
                for i in range(K_num-1-len(ins_bbox_mat)):
                    RoI_bbox.append([x_1, y_1, x_2, y_2])
        else:
            RoI_bbox= []
            x_1 = 0
            y_1 = 0
            x_2 = width/32
            y_2 = height/32

            for i in range(K_num-1):
                RoI_bbox.append([x_1, y_1, x_2, y_2])

        # (4, 4)
        return np.asarray(RoI_bbox)
    
        
class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_ins_txt_path(self, ins_bbox_path, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, ".txt") # '0000000268.txt'

        ins_txt_path = os.path.join(
                ins_bbox_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        
        return ins_txt_path


    def get_sem_ins_path(self, sem_ins_path, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, ".npy") # '0000000268.npy'

        sem_ins_path = os.path.join(
                sem_ins_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        
        return sem_ins_path


    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext) # '0000000268.jpg'

        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
