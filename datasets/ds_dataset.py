from __future__ import absolute_import, division, print_function

import os
import cv2
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .ds_mono_dataset import DSMonoDataset

import warnings

class DSDataset(DSMonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(DSDataset, self).__init__(*args, **kwargs)

        # TODO: depend on sequences 
        self.K = np.array([[1.14, 0,     0.518, 0],
                           [0,    2.509, 0.494, 0],
                           [0,    0,     1,     0],
                           [0,    0,     0,     1]], dtype=np.float32)

        self.full_res_shape = (880, 400)

    def check_ds_depth(self):
        line = self.filenames[0]

        file_name = os.path.join(self.data_path, self.depth_dir, line+".png")

        return os.path.isfile(file_name)

    def get_ds_color(self, frame_path, do_flip):
        color = self.loader(frame_path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_sem_ins(self, frame_path, do_flip):
        sem_ins = np.load(frame_path)

        if do_flip:
            sem_ins = np.fliplr(sem_ins)

        return sem_ins

    def RoI_pad(self, x_1, y_1, x_2, y_2, width, height):
        RoI_width = x_2 - x_1
        RoI_height = y_2 - y_1
        
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
            # pad the RoI with ratio 1.3
            RoI_width_pad = RoI_width * 0.15
            RoI_height_pad = RoI_height * 0.15

            x_1 = 0 if x_1 - RoI_width_pad <= 0 else x_1 - RoI_width_pad
            y_1 = 0 if y_1 - RoI_height_pad <= 0 else y_1 - RoI_height_pad
            x_2 = width if x_2 + RoI_width_pad >= width else x_2 + RoI_width_pad
            y_2 = height if y_2 + RoI_height_pad >= height else y_2 + RoI_height_pad
        
        return x_1, y_1, x_2, y_2

    def mask_area(self, box_list):
        x_1,y_1,x_2,y_2 = box_list
                
        return (y_2-y_1)*(x_2-x_1)

    def get_ins_bbox(self, frame_path, ratio_w, ratio_h, width, height, do_flip):
        with warnings.catch_warnings():
            # if there is no bbox, the txt file is empty.
            warnings.simplefilter("ignore")
            ins_bbox_mat = np.loadtxt(frame_path)

        K_num = 5
        if len(ins_bbox_mat) > 0:
            if len(ins_bbox_mat.shape) == 1:
                # if there is only one obj, (4,) -> (1,4)
                ins_bbox_mat = np.expand_dims(ins_bbox_mat, 0)
            
            box_area_list = []
            for i in range(len(ins_bbox_mat)):
                box_area_list.append(self.mask_area(ins_bbox_mat[i]))
        
            sort_idx = np.argsort(box_area_list)[::-1][:K_num-1] # descending

            RoI_bbox = []
            for i in sort_idx:
                x_1 = int(ins_bbox_mat[i, 0] * ratio_w)
                y_1 = int(ins_bbox_mat[i, 1] * ratio_h)
                x_2 = int(ins_bbox_mat[i, 2] * ratio_w)
                y_2 = int(ins_bbox_mat[i, 3] * ratio_h)

                x_1, y_1, x_2, y_2 = self.RoI_pad(x_1, y_1, x_2, y_2, width, height)

                if do_flip:
                    RoI_bbox.append([(width - x_2)/32, y_1/32, (width - x_1)/32, y_2/32])
                else:
                    RoI_bbox.append([x_1/32, y_1/32, x_2/32, y_2/32])

            if len(sort_idx) < K_num-1: 
                x_1 = 0
                y_1 = 0
                x_2 = width/32
                y_2 = height/32
                for i in range(K_num-1-len(sort_idx)):
                    RoI_bbox.append([x_1, y_1, x_2, y_2])

            r_sort_idx = [i+1 for i in sort_idx]    
            
            return np.asarray(RoI_bbox), r_sort_idx

        else:
            RoI_bbox= []
            x_1 = 0
            y_1 = 0
            x_2 = width/32
            y_2 = height/32

            for i in range(K_num-1):
                RoI_bbox.append([x_1, y_1, x_2, y_2])

            # (4, 4)
            return np.asarray(RoI_bbox), None


class DSRAWDataset(DSDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(DSRAWDataset, self).__init__(*args, **kwargs)

    def get_ds_depth(self, frame_name, do_flip):
        file_name = os.path.join(self.data_path, self.depth_dir, frame_name+".png")
        depth_gt = cv2.imread(file_name, -1)
        depth_gt = cv2.resize(depth_gt, self.full_res_shape, cv2.INTER_NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
