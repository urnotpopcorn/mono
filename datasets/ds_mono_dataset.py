from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import skimage.color as color

import torch
import torch.utils.data as data
from torchvision import transforms

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class DSMonoDataset(data.Dataset):
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
        super(DSMonoDataset, self).__init__()

        self.mode = mode
        if self.mode in ["train", "val"]:
            self.mode = "train"
        self.data_path = os.path.join(data_path, self.mode)

        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        self.opt = opt

        if self.mode in ["train", "val"]:
            self.mode = "train"
            self.img_dir = "train-left-image"
            self.depth_dir = "train-depth-map"
        else:
            self.img_dir = "left-image-half-size"
            self.depth_dir = "depth-map-half-size"

        if self.opt.SIG or self.opt.instance_pose :
            self.data_sem_path = os.path.join(self.opt.project_dir, "dataset", "ds_data_sem", self.mode)
            self.data_ins_path = os.path.join(self.opt.project_dir, "dataset", "ds_data_ins", self.mode)
            self.data_bbox_path = os.path.join(self.opt.project_dir, "dataset", "ds_data_bbox", self.mode)

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

        self.load_depth = self.check_ds_depth()

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
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

        if self.opt.SIG:
            for k in list(inputs):
                f = inputs[k]
                if "sem_seg" in k:
                    n, im, i = k
                    sem_tensor = torch.Tensor(np.asarray(f))

                    max_num = int(np.max(f))
                    if max_num+1 <= 19:
                        sem_tensor_one_hot = torch.nn.functional.one_hot(sem_tensor.to(torch.int64), 19).type(torch.FloatTensor)
                    else:
                        sem_tensor_one_hot = torch.nn.functional.one_hot(sem_tensor.to(torch.int64), max_num+1).type(torch.FloatTensor)
                        sem_tensor_one_hot = sem_tensor_one_hot[:,:,:19]
                    
                    # sem_seg_one_hot
                    inputs[(n + "_one_hot", im, 0)] = sem_tensor_one_hot.permute(2,0,1)

                if "ins_id_seg" in k:
                    # ins_id_seg -> ins_id_seg_to_edge
                    n, im, i = k
                    ins_id_seg_to_edge = np.expand_dims(self.get_edge(np.asarray(f)), -1)

                    # ins_id_seg_to_edge
                    inputs[(n + "_to_edge", im, 0)] = torch.Tensor(ins_id_seg_to_edge).permute(2,0,1)

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

                    ins_RoI_idx = inputs[("ins_RoI_idx", im, 0)]

                    if ins_RoI_idx is not None:
                        selected_ins_channel = ins_tensor_one_hot[:,:,ins_RoI_idx]

                        cat_ins_tensor_one_hot = []
                        # append bg 
                        cat_ins_tensor_one_hot.append(ins_tensor_one_hot[:,:,0].unsqueeze(2))
                        # append selected channel in order
                        cat_ins_tensor_one_hot.append(selected_ins_channel)

                        remaining_channel = K_num-1-len(ins_RoI_idx)

                        # [h,w,x] all in false
                        if remaining_channel > 0:
                            rem_channel = torch.zeros(self.opt.height, self.opt.width,remaining_channel).type(torch.bool)
                            cat_ins_tensor_one_hot.append(rem_channel)

                        # [h, w, 5]
                        cat_ins_tensor = torch.cat(cat_ins_tensor_one_hot, 2)

                        inputs[(n, im, 0)] = cat_ins_tensor.permute(2,0,1)
                    else:
                        inputs[(n, im, 0)] = ins_tensor_one_hot[:,:,:K_num].permute(2,0,1)

                    # shape: [K_num, 192, 640], 
                    # background mask : [0, :, :]
                    # instance 1 mask : [1, :, :]
                    # instance 2 mask : [2, :, :]
                    # instance 3 mask : [3, :, :]
                    # instance 4 mask : [4, :, :]

    def get_edge(self, ins_id_seg):
        ins_edge_seg = None
        ins_id_seg_edge_gradient = np.gradient(ins_id_seg)
        x = ins_id_seg_edge_gradient[0]
        y = ins_id_seg_edge_gradient[1]

        ins_edge_seg = ((x+y)!=0)*1
        
        return ins_edge_seg

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

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # filenames[index]: 2018-10-11-16-03-19_1192
        # frame_index: 1192
        frame_index = int(self.filenames[index].split("_")[1])
        # frame_name: "2018-10-11-16-03-19"
        frame_name = self.filenames[index].split("_")[0]

        for i in self.frame_idxs: #[0, -1, 1]
            i_frame_name = frame_name + "_" + str(frame_index+i) + self.img_ext
            i_frame_path = os.path.join(self.data_path, self.img_dir, i_frame_name)
            if os.path.isfile(i_frame_path):
                inputs[("color", i, -1)] = self.get_ds_color(i_frame_path, do_flip)
            else:
                print("missing: ", i_frame_path)
                i_frame_name = frame_name + "_" + str(frame_index) + self.img_ext
                i_frame_path = os.path.join(self.data_path, self.img_dir, i_frame_name)
                inputs[("color", i, -1)] = self.get_ds_color(i_frame_path, do_flip)

        if self.opt.SIG or self.opt.instance_pose:
            ins_width = dict()
            ins_height = dict()
            for i in self.frame_idxs: # [0,1,-1]
                i_frame_name = frame_name + "_" + str(frame_index+i) + ".npy"
                i_frame_path = os.path.join(self.data_ins_path, i_frame_name)
                if os.path.isfile(i_frame_path):
                    ins_seg = self.get_sem_ins(i_frame_path, do_flip)
                else:
                    print("missing: ", i_frame_path)
                    i_frame_name = frame_name + "_" + str(frame_index) + ".npy"
                    i_frame_path = os.path.join(self.data_ins_path, i_frame_name)
                    ins_seg = self.get_sem_ins(i_frame_path, do_flip)

                sig_ins_id_seg = Image.fromarray(np.uint8(ins_seg[:,:,1])).convert("L")
                ins_width[i] = sig_ins_id_seg.size[0]
                ins_height[i] = sig_ins_id_seg.size[1]

                sig_ins_id_seg = sig_ins_id_seg.resize((self.opt.width, self.opt.height), Image.NEAREST)
                inputs[("ins_id_seg", i, -1)] = sig_ins_id_seg

                if self.opt.SIG_ignore_fg_loss:
                    ins_tensor = torch.Tensor(np.array(sig_ins_id_seg))
                    ins_id_seg_bk = ins_tensor.type(torch.bool)
                    inputs[("ins_id_seg_bk", i, 0)] = ins_id_seg_bk

            if self.opt.SIG:
                for i in self.frame_idxs: # [0,1,-1]
                    i_frame_name = frame_name + "_" + str(frame_index+i) + ".npy"
                    i_frame_path = os.path.join(self.data_sem_path, i_frame_name)
                    if os.path.isfile(i_frame_path):
                        sem_seg = self.get_sem_ins(i_frame_path, do_flip)
                    else:
                        print("missing: ", i_frame_path)
                        i_frame_name = frame_name + "_" + str(frame_index) + ".npy"
                        i_frame_path = os.path.join(self.data_sem_path, i_frame_name)
                        sem_seg = self.get_sem_ins(i_frame_path, do_flip)
                    
                    sig_sem_seg = Image.fromarray(np.uint8(sem_seg[:,:,0])).convert("L")
                    sig_sem_seg = sig_sem_seg.resize((self.opt.width, self.opt.height), Image.NEAREST)

                    inputs[("sem_seg", i, -1)] = sig_sem_seg

            if self.opt.instance_pose:
                for i in self.frame_idxs: # [0,1,-1]  
                    ratio_w = self.opt.width / ins_width[i]
                    ratio_h = self.opt.height / ins_height[i]

                    i_frame_name = frame_name + "_" + str(frame_index+i) + ".txt"
                    i_frame_path = os.path.join(self.data_bbox_path, i_frame_name)
                    if os.path.isfile(i_frame_path):
                        ins_RoI_bbox, ins_RoI_idx = self.get_ins_bbox(i_frame_path, ratio_w, ratio_h, 
                            self.opt.width, self.opt.height, do_flip)
                    else:
                        print("missing: ", i_frame_path)
                        i_frame_name = frame_name + "_" + str(frame_index) + ".txt"
                        i_frame_path = os.path.join(self.data_bbox_path, i_frame_name)
                        ins_RoI_bbox, ins_RoI_idx = self.get_ins_bbox(i_frame_path, ratio_w, ratio_h, 
                            self.opt.width, self.opt.height, do_flip)
                    
                    inputs[("ins_RoI_bbox", i, 0)] = torch.Tensor(ins_RoI_bbox)
                    inputs[("ins_RoI_idx", i, 0)] = ins_RoI_idx

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

            if self.opt.SIG:
                del inputs[("sem_seg", i, -1)]

            if self.opt.SIG or self.opt.instance_pose:
                del inputs[("ins_id_seg", i, -1)]

            if self.opt.instance_pose:
                del inputs[("ins_RoI_idx", i, 0)]

        if self.load_depth:
            depth_gt = self.get_ds_depth(self.filenames[index], do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def get_ds_color(self, frame_index,  do_flip):
        raise NotImplementedError

    def check_ds_depth(self):
        raise NotImplementedError

    def get_ds_depth(self, frame_index,  do_flip):
        raise NotImplementedError
