# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torchvision 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)

        if not self.opt.fix_depth:
            self.parameters_to_train += list(self.models["encoder"].parameters())
        else:
            for p in self.models["encoder"].parameters():
                p.requires_grad = False

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)

        if not self.opt.fix_depth:
            self.parameters_to_train += list(self.models["depth"].parameters())
        else:
            for p in self.models["depth"].parameters():
                p.requires_grad = False

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)

                if not self.opt.fix_pose:
                    self.parameters_to_train += list(self.models["pose_encoder"].parameters())
                else:
                    pose_encoder_para = self.models["pose_encoder"].parameters()
                    for p in pose_encoder_para:
                        p.requires_grad = False
                    
                    #self.parameters_to_train += list(pose_encoder_para)

                # self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)

            if not self.opt.fix_pose:
                self.parameters_to_train += list(self.models["pose"].parameters())
            else:
                pose_decoder_para = self.models["pose"].parameters()
                for p in pose_decoder_para:
                    p.requires_grad = False
                
                #self.parameters_to_train += list(pose_decoder_para)

            # self.parameters_to_train += list(self.models["pose"].parameters())

        # --------------------------------------------------------------------------------
        if self.opt.instance_pose:
            def weight_init(m):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight)
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
            
            if self.opt.predict_delta:
                self.models["instance_pose"] = networks.InsPoseDecoder(
                    num_RoI_cat_features=1024,
                    num_input_features=1,
                    num_frames_to_predict_for=2,
                    predict_delta=True)
            else:
                self.models["instance_pose"] = networks.InsPoseDecoder(
                    num_RoI_cat_features=1024,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            self.models["instance_pose"].apply(weight_init)
            self.models["instance_pose"].to(self.device)
            instance_pose_para = self.models["instance_pose"].parameters()
            self.parameters_to_train += list(instance_pose_para)

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        # --------------------------------------------------------------------------------
        # 1e-4 -> 1e-5 (15 epoch) -> 1e-6 (30 epoch), step=15
        self.model_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters_to_train), self.opt.learning_rate)
        
        # self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "drivingstereo_eigen": datasets.DSRAWDataset}

        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        '''
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, opt=self.opt, mode="train")
        '''
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, opt=self.opt, mode="train")
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        '''
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, opt=self.opt, mode="val")
        '''
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, opt=self.opt, mode="val")
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                if self.opt.instance_pose:
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data, 
                        losses["ins_loss"].cpu().data, losses["bg_loss"].cpu().data)    
                else:
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                
                
                # print T_dynamic
                for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                    for ins_id in range(4):
                        print(outputs[("T_dynamic", frame_id, ins_id)])

                '''
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                '''
                #self.log("train", inputs, outputs, losses)
                '''
                self.val()
                '''
            #input()
            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            # if self.opt.geometric_loss:
            #     depth_input = [inputs["color_aug", 0, 0], inputs["color_aug", -1, 0], inputs["color_aug", 1, 0]]
            #     features = self.models["encoder"](torch.cat(depth_input, 0))
            # else:

            
            if True:
                features = self.models["encoder"](inputs["color_aug", 0, 0])
            
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        if self.opt.instance_pose:
            self.generate_images_pred(inputs, outputs)
            self.synthesize_layer(inputs, outputs)

            losses = self.compute_losses(inputs, outputs)
            
            weight_fg, weight_bg, ins_losses = self.compute_instance_losses(inputs, outputs)
            
            if ins_losses['ins_loss'].detach().cpu().numpy() == np.nan:
                print('nan')
                input()

            losses['ins_loss'] = ins_losses['ins_loss']

            bg_loss = losses['loss']
            fg_loss = losses['ins_loss']
            losses['bg_loss'] = bg_loss
            
            if self.opt.weight_fg is not None:
                losses['loss'] = (1-self.opt.weight_fg) * bg_loss + self.opt.weight_fg * fg_loss
            
            return outputs, losses
        else:
            self.generate_images_pred(inputs, outputs)
            losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                if True:
                    pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if True:
                        if self.opt.disable_pose_invert:
                            pose_inputs = [pose_feats[0], pose_feats[f_i]]
                        else:
                            if f_i < 0:
                                pose_inputs = [pose_feats[f_i], pose_feats[0]]
                            else:
                                pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    if self.opt.disable_pose_invert:
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)
                    else:                        
                        # Invert the matrix if the frame id is negative
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        bs = self.opt.batch_size
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            # if self.opt.geometric_loss:
            if True:
                outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                # fwd warping: T (tgt->src), warp t+1/t-1 to t
                cam_points = self.backproject_depth[source_scale](
                    outputs[("depth", 0, scale)], inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                # if not self.opt.disable_automasking:
                #     outputs[("color_identity", frame_id, scale)] = \
                #         inputs[("color", frame_id, source_scale)]

                # ----------------------------------------------------------------------
                # ----------------------------------------------------------------------

                # if self.opt.geometric_loss:
                #     outputs[("projected_depth", frame_id, scale)] = F.grid_sample(
                #         outputs[("depth", frame_id, scale)],
                #         pix_coords,
                #         padding_mode="border")
                    
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        if True:
            abs_diff = torch.abs(target - pred)
            l1_loss = abs_diff.mean(1, True)

            if self.opt.no_ssim:
                reprojection_loss = l1_loss
            else:
                ssim_loss = self.ssim(pred, target).mean(1, True)
                reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        bs = self.opt.batch_size
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            # geometry_loss = 0
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]

                # if self.opt.geometric_loss:
                #     computed_depth = outputs[("computed_depth", 0, scale)]
                #     projected_depth = outputs[("projected_depth", frame_id, scale)]

                #     diff_img = (target - pred).abs().clamp(0, 1)
                #     diff_depth = (torch.abs(computed_depth - projected_depth) / (computed_depth + projected_depth)).clamp(0, 1)
                #     geometry_consistency_loss = diff_depth.mean(1, True)
                #     geometry_loss += geometry_consistency_loss.mean()*0.15
                
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            # if self.opt.geometric_loss:
            #     loss += geometry_loss
            #     losses["loss/{}_geometric".format(scale)] = geometry_loss

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            
            # --------------------------------------------------------------------

            # if self.opt.geometric_loss:
            if True:
                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                if self.opt.second_order_disp:
                    smooth_loss = get_sec_smooth_loss(norm_disp, color)
                else:
                    smooth_loss = get_smooth_loss(norm_disp, color)
                loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def get_ins_bbox(self, inputs, frame_id, scale):
        # TODO: [bs, k ,4] -> [[K, 4]*bs], list of [k, 4], length = bs
        ins_RoI_bbox_frame_id = inputs[("ins_RoI_bbox", frame_id, scale)] #[bs, k=4 ,4]
        ins_RoI_bbox_list_frame_id = [x.squeeze(0) for x in list(ins_RoI_bbox_frame_id.split(1, dim=0))]
        return ins_RoI_bbox_list_frame_id

    def compute_IOU(self, mask1, mask2):
        """
        mask1: b, 1, h, w
        """
        inter = mask1 * mask2 # b, 
        outer = 1 - (1-mask1) * (1-mask2) # b, 
        IOU = inter.sum([2, 3]) * 1.0 / (outer.sum([2, 3])+1e-3) # b, 
        return IOU

    def synthesize_layer(self, inputs, outputs):
        # some defition
        scale = 0
        inv_K = inputs[("inv_K", scale)]
        K = inputs[("K", scale)]
        img0_aug = inputs["color_aug", 0, scale]
        img0 = inputs["color", 0, scale]
        
        # compute depth
        disp = outputs[("disp", scale)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        _, depth0 = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth) # min_depth = 0.1, max_depth = 100

        # compute mask of tgt frame, [bs, 1, 192, 640], exclude bg
        tgt_dynamic_layer = torch.sum(inputs[("ins_id_seg", 0, scale)][:, 1:, :, :], 1).unsqueeze(1).float()
        outputs[("cur_mask", 0, scale)] = tgt_dynamic_layer

        # compute dynamic area of tgt frame
        outputs[("f_img_syn", 0, scale)] = inputs["color", 0, scale] * tgt_dynamic_layer
        
        for frame_id in self.opt.frame_ids[1:]: # [-1, 1]
            T_static = outputs[("cam_T_cam", 0, frame_id)] # [bs, 4, 4]
            #img1 = inputs["color_aug", frame_id, scale]
            img1 = inputs["color", frame_id, scale]
            img0_pred = outputs[("color", frame_id, scale)]
            # define the final image and mask
            img0_pred_final = torch.zeros_like(img0_pred, requires_grad=True)   # final image
            mask0_pred_final = torch.zeros_like(outputs[("cur_mask", 0, scale)], requires_grad=True) # bs, 1, 192, 640

            # step1: read ins and mask
            img0_ins_bbox_list = self.get_ins_bbox(inputs, 0, scale) # length = 1, [k=4, 4]
            # FIXME: use img1
            instance_K_num = img0_ins_bbox_list[0].shape[0]

            # step2: compute image feature and crop ROI feature
            img0_feature = self.models["encoder"](img0_aug)[-1] # [bs, 512, 6, 20]
            img0_pred_feature = self.models["encoder"](img0_pred)[-1] # [bs, 512, 6, 20]
            # [bs, 512, 6, 20] -> [k*bs, 512, 6, 20] or [k*bs, 512, 3, 3]
            
            # FIXME: could delete this part
            if self.opt.predict_delta:
                #img0_ins_feature_list = torchvision.ops.roi_align(img0_feature, img0_ins_bbox_list, output_size=(6,20))
                img0_ins_feature_list = torchvision.ops.roi_align(img0_feature, img0_ins_bbox_list, output_size=(self.opt.height//32, self.opt.width//32))
            else:
                img0_ins_feature_list = torchvision.ops.roi_align(img0_feature, img0_ins_bbox_list, output_size=(3,3))
            
            # step3: compute pix_coords of img0_pred
            cam_points = self.backproject_depth[scale](
                depth0, inv_K) # cam_points of frame 0, [12, 4, 122880]
            pix_coords = self.project_3d[scale](
                cam_points, K, T_static)
            
            for ins_id in range(instance_K_num):
                # step4: use T_static to transform mask of each ins
                #img1_ins_mask = img1_ins_mask_list[:, ins_id+1, :, :].unsqueeze(1).float() #[b, 1, h, w]
                img1_ins_mask = inputs[("ins_id_seg", frame_id, scale)][:, ins_id+1, :, :].unsqueeze(1).float() # [b, 1, h, w]
                img0_pred_ins_mask = F.grid_sample(img1_ins_mask, pix_coords) #[b, 1, h, w]
                
                '''
                # TODO: step4.5: compute diff between t_pred and t_gt and then eliminate relative static area
                roi_abs = torch.abs(outputs[("color", frame_id, scale)] * img0_pred_ins_mask - inputs["color", 0, scale] * img0_pred_ins_mask)
                # roi_abs: bs, 3, 192, 640
                roi_sum = torch.sum(roi_abs, dim=[1, 2, 3]) # bs,
                mask_sum = torch.sum(roi_abs, dim=[1, 2, 3]) # bs,
                roi_diff = roi_sum / mask_sum # bs,

                if self.opt.roi_diff_thres is not None:
                    roi_diff = torch.sum(torch.abs(outputs[("color", frame_id, scale)] * img0_pred_ins_mask - inputs["color", 0, scale] * img0_pred_ins_mask))
                    if torch.sum(img0_pred_ins_mask) >= 1:
                        roi_diff = roi_diff / (torch.sum(img0_pred_ins_mask))
                        if roi_diff < self.opt.roi_diff_thres:
                            continue
                '''

                # step5: crop ins feature of img0 and img0_pred
                # [bs, 512, 6, 20] -> [k*bs, 512, 3, 3]
                if self.opt.predict_delta:
                    img0_pred_ins_bbox = self.extract_bbox_from_mask(img0_pred_ins_mask)
                    img0_pred_ins_feature = torchvision.ops.roi_align(img0_pred_feature, img0_pred_ins_bbox, output_size=(self.opt.height//32, self.opt.width//32))
                    
                    if self.opt.use_insid_match:
                        img0_ins_feature = torch.cat([img0_ins_feature_list[i*instance_K_num+ins_id, :, :, :].unsqueeze(0) for i in range(self.opt.batch_size)])
                    else:
                        # use warped bbox
                        img0_ins_feature = torchvision.ops.roi_align(img0_feature, img0_pred_ins_bbox, output_size=(self.opt.height//32, self.opt.width//32))
                else:
                    img0_pred_ins_bbox = self.extract_bbox_from_mask(img0_pred_ins_mask)
                    img0_pred_ins_feature = torchvision.ops.roi_align(img0_pred_feature, img0_pred_ins_bbox, output_size=(3,3)) # [b, 512, 3, 3]
                    
                    if self.opt.use_insid_match:
                        img0_ins_feature = torch.cat([img0_ins_feature_list[i*instance_K_num+ins_id, :, :, :].unsqueeze(0) for i in range(self.opt.batch_size)])
                    else:
                        # use warped bbox
                        img0_ins_feature = torchvision.ops.roi_align(img0_feature, img0_pred_ins_bbox, output_size=(3,3))
                
                # step6: input ins_pose_net and predict ins_pose
                if self.opt.disable_pose_invert:
                    ins_pose_inputs = [img0_ins_feature, img0_pred_ins_feature]
                else:
                    if frame_id < 0:
                        ins_pose_inputs = [img0_pred_ins_feature, img0_ins_feature]
                    else:
                        ins_pose_inputs = [img0_ins_feature, img0_pred_ins_feature]
                
                ins_pose_inputs = torch.cat(ins_pose_inputs, 1)
                if self.opt.predict_delta:
                    ins_axisangle, ins_translation, delta_x_inv, delta_y_inv, delta_z_inv = self.models["instance_pose"](ins_pose_inputs)
                else:
                    ins_axisangle, ins_translation = self.models["instance_pose"](ins_pose_inputs)
                
                if self.opt.disable_pose_invert:
                    ins_cam_T_cam = transformation_from_parameters(ins_axisangle[:, 0], ins_translation[:, 0], invert=False)
                else:
                    ins_cam_T_cam = transformation_from_parameters(ins_axisangle[:, 0], ins_translation[:, 0], invert=(frame_id < 0)) 
                
                # ins_cam_T_cam: b, 4, 4
                #ins_cam_T_cam = list(torch.chunk(ins_cam_T_cam, self.opt.batch_size, dim=0)) # bs x [1, 4, 4]
                #T_dynamic = torch.cat([x for x in ins_cam_T_cam], 0) # [bs, 4, 4]
                T_dynamic = ins_cam_T_cam

                # step7: predict ins
                T_total = torch.matmul(T_dynamic, T_static) # [bs, 4, 4]
                if self.opt.predict_delta:
                    ins_pix_coords = self.project_3d[scale](
                        cam_points, K, T_total, delta_x_inv, delta_y_inv, delta_z_inv, self.opt.min_depth, self.opt.max_depth)
                else:    
                    ins_pix_coords = self.project_3d[scale](cam_points, K, T_total)

                #step8: predict frame 0 from frame 1 based on T_dynamic and T_static
                img0_pred_new = F.grid_sample(img1, ins_pix_coords)
                img0_pred_ins_mask_new = F.grid_sample(img1_ins_mask, ins_pix_coords) # [bs, 1, 192, 640]
                
                #step8.5: filter invalid points
                if self.opt.iou_thres is not None:
                    img0_ins_mask = inputs[("ins_id_seg", 0, scale)][:, ins_id+1, :, :].unsqueeze(1).float()
                    ins_IOU = self.compute_IOU(img0_ins_mask, img1_ins_mask) # [b, 1]
                    IOU_mask = ins_IOU < self.opt.iou_thres # [b, 1]
                    img0_pred_ins_mask_new = img0_pred_ins_mask_new.view(self.opt.batch_size, -1) # [b, 1x192x640]
                    img0_pred_ins_mask_new = img0_pred_ins_mask_new * IOU_mask.float() # [b, 1x192x640]
                    img0_pred_ins_mask_new = img0_pred_ins_mask_new.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

                #step9: predict image
                # img0_pred_final：[bs, 3, 192, 640], img0_pred_ins_mask_new: [bs, 1, 192, 640], ins_pix_coords: [1, 192, 640, 2]
                img0_pred_final = torch.add(img0_pred_final*(1-img0_pred_ins_mask_new), img0_pred_new*img0_pred_ins_mask_new)
                mask0_pred_final = torch.add(mask0_pred_final*(1-img0_pred_ins_mask_new), img0_pred_ins_mask_new)

                # FIXME: save for vis
                outputs[("T_dynamic", frame_id, ins_id)] = T_dynamic
            
            color_ori = outputs[("color", frame_id, scale)]
            color_new = mask0_pred_final * img0_pred_final + (1-mask0_pred_final) * color_ori

            # save for vis
            outputs[("color", frame_id, scale)] = color_new
            # outputs[("color_ori", frame_id, scale)] = color_ori
            # outputs[("color_diff", frame_id, scale)] = color_new - color_ori
            outputs[("color", frame_id, scale)] = color_new
            outputs[("f_img_syn", frame_id, scale)] = img0_pred_final
            # outputs[("warped_mask", frame_id, scale)] = mask0_pred_final
            # # FIXME: just use the max
            # outputs[("mask", frame_id, scale)] = torch.sum(inputs[("ins_id_seg", frame_id, scale)][:, 1:, :, :], 1).unsqueeze(1).float()
            
    '''
    def synthesize_layer_bk(self, inputs, outputs):
        scale = 0
        bs = self.opt.batch_size
        
        # compute cam_points of tgt frame
        disp = outputs[("disp", scale)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth) # min_depth = 0.1, max_depth = 100
        cam_points = self.backproject_depth[scale](depth, inputs[("inv_K", scale)]) # tgt
        
        # compute mask of tgt frame, [bs, 1, 192, 640], exclude bg
        tgt_dynamic_layer = torch.sum(inputs[("ins_id_seg", 0, scale)][:, 1:, :, :], 1).unsqueeze(1).float()
        # compute dynamic area of tgt frame
        outputs[("f_img_syn", 0, scale)] = inputs["color", 0, scale] * tgt_dynamic_layer
        outputs[("cur_mask", 0, scale)] = tgt_dynamic_layer

        if True:
            f_feats_0 = self.models["encoder"](inputs["color_aug", 0, scale])[-1] # [bs, 512, 6, 20]
        
        for frame_id in self.opt.frame_ids[1:]: # [-1, 1]
            instance_K_num = inputs[("ins_id_seg", frame_id, scale)].shape[1] - 1 

            # compute camera pose of static area
            T_static = outputs[("cam_T_cam", 0, frame_id)] # [bs, 4, 4]

            # get image of frame_id 
            total_img_frame_id = inputs[("color", frame_id, scale)]
            # get mask of frame_id, in tgt_dynamic_layer: 0 stands for bg, 1 stands for object instance, [bs, 1, 192, 640]
            total_mask_frame_id = torch.sum(inputs[("ins_id_seg", frame_id, scale)][:, 1:, :, :], 1).unsqueeze(1).float()

            # define the final image and mask
            f_img_syn = torch.zeros_like(total_img_frame_id)   # final image
            f_mask_syn = torch.zeros_like(total_mask_frame_id) # final mask

            ins_RoI_bbox_frame_id = inputs[("ins_RoI_bbox", frame_id, scale)] # [bs, k ,4] -> [[K, 4]*bs]
            ins_RoI_bbox_list_frame_id = [x.squeeze(0) for x in list(ins_RoI_bbox_frame_id.split(1, dim=0))]

            f_feats_frame_id = self.models["encoder"](inputs["color_aug", frame_id, scale])[-1] # [bs, 512, 6, 20]
            
            # [bs, 512, 6, 20] -> [k*bs, 512, 3, 3]
            if self.opt.predict_delta:
                #cur_RoI_feats = torchvision.ops.roi_align(f_feats_frame_id, ins_RoI_bbox_list_frame_id, output_size=(6,20))
                cur_RoI_feats = torchvision.ops.roi_align(f_feats_frame_id, ins_RoI_bbox_list_frame_id, output_size=(self.opt.height//32, self.opt.width//32))
            else:
                cur_RoI_feats = torchvision.ops.roi_align(f_feats_frame_id, ins_RoI_bbox_list_frame_id, output_size=(3,3))

            pix_coords = self.project_3d[scale](cam_points, inputs[("K", scale)], T_static) #[b, h, w, 2]

            for ins_id in range(instance_K_num):
                ins_mask_frame_id = inputs[("ins_id_seg", frame_id, scale)][:, ins_id+1, :, :].unsqueeze(1).float() # [b, 1, h, w]
                ins_warp_mask = F.grid_sample(ins_mask_frame_id, pix_coords) # [b, 1, h, w]
                ins_warp_bbox = self.extract_bbox_from_mask(ins_warp_mask)

                # [b, 512, 3, 3]
                ins_cur_RoI_feats = torch.cat([cur_RoI_feats[i*instance_K_num+ins_id, :, :, :].unsqueeze(0) for i in range(bs)])
                
                if self.opt.predict_delta:
                    #ins_0_RoI_feats = torchvision.ops.roi_align(f_feats_0, ins_warp_bbox, output_size=(6,20))
                    ins_0_RoI_feats = torchvision.ops.roi_align(f_feats_0, ins_warp_bbox, output_size=(self.opt.height//32, self.opt.width//32))
                else:
                    ins_0_RoI_feats = torchvision.ops.roi_align(f_feats_0, ins_warp_bbox, output_size=(3,3))
                
                if self.opt.disable_pose_invert:
                    ins_pose_inputs = [ins_0_RoI_feats, ins_cur_RoI_feats] # 0, 1
                else:
                    if frame_id < 0:
                        ins_pose_inputs = [ins_cur_RoI_feats, ins_0_RoI_feats] # -1, 0
                    else:
                        ins_pose_inputs = [ins_0_RoI_feats, ins_cur_RoI_feats] # 0, 1
                    
                ins_pose_inputs = torch.cat(ins_pose_inputs, 1) # [b, 1024, 3, 3]
                
                if self.opt.predict_delta:
                    axisangle, translation, delta_x_inv, delta_y_inv, delta_z_inv = self.models["instance_pose"](ins_pose_inputs)
                else:
                    axisangle, translation = self.models["instance_pose"](ins_pose_inputs)
                
                # [bs, 4, 4]
                if self.opt.disable_pose_invert:
                    T_dynamic = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=False) 
                else:
                    T_dynamic = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(frame_id < 0)) 
                    
                T_total = torch.matmul(T_dynamic, T_static) # [bs, 4, 4]
                
                if self.opt.predict_delta:
                    T_pix_coords = self.project_3d[scale](
                            cam_points, inputs[("K", scale)], T_total, delta_x_inv, delta_y_inv, delta_z_inv, self.opt.min_depth, self.opt.max_depth)
                else:    
                    T_pix_coords = self.project_3d[scale](cam_points, inputs[("K", scale)], T_total)

                ins_warp_img = F.grid_sample(total_img_frame_id, T_pix_coords)
                T_ins_warp_mask = F.grid_sample(ins_mask_frame_id, T_pix_coords)

                f_img_syn = torch.add(f_img_syn*(1-T_ins_warp_mask), ins_warp_img*T_ins_warp_mask)
                f_mask_syn = torch.add(f_mask_syn*(1-T_ins_warp_mask), T_ins_warp_mask)

            outputs[("f_img_syn", frame_id, scale)] = f_img_syn

            color_ori = outputs[("color", frame_id, scale)]
            color_new = f_mask_syn * f_img_syn + (1-f_mask_syn) * color_ori

            outputs[("color_ori", frame_id, scale)] = color_ori
            outputs[("color_diff", frame_id, scale)] = color_new - color_ori
            outputs[("color", frame_id, scale)] = color_new
            outputs[("warped_mask", frame_id, scale)] = f_mask_syn
    '''
    
    def extract_bbox_from_mask(self, ins_warp_mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        # ins_warp_mask: [bs, 1, 192, 640]
        mask = ins_warp_mask.squeeze(1)
        ins_warp_bbox = []
        for bs_idx in range(mask.shape[0]):
            #idx_mask = mask[bs_idx, :, :].uint8()#.detach().cpu().numpy()
            # Bounding box.
            idx_mask = mask[bs_idx, :, :].type(torch.uint8)
            horizontal_indicies = torch.where(torch.any(idx_mask, axis=0))[0]
            vertical_indicies = torch.where(torch.any(idx_mask, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, y1, x2, y2 = 0, 0, 640, 192
            ins_warp_bbox.append(torch.Tensor([[x1/32, y1/32, x2/32, y2/32]]).to(self.device))
            #ins_warp_bbox.append([[x1, y1, x2, y2]])
            #ins_warp_bbox.append(torch.Tensor([[x1, y1, x2, y2]]).to(self.device))
        
        # list of [1,4]
        return ins_warp_bbox

    '''    
    def extract_bbox_from_mask_qh(self, ins_warp_mask):
        """Compute bounding boxes from masks.
        mask: [height, width]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (x1, y1, x2, y2)].
        """

        # [b, h, w]
        mask = ins_warp_mask.squeeze(1)

        ins_warp_bbox = []
        for bs_idx in range(mask.shape[0]):
            idx_mask = mask[bs_idx, :, :].detach().cpu().numpy() # [h, w]
            # Bounding box.
            horizontal_indicies = np.where(np.any(idx_mask, axis=0))[0]
            vertical_indicies = np.where(np.any(idx_mask, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance
                x1, y1, x2, y2 = 0, 0, 20, 6
            
            ins_warp_bbox.append(torch.Tensor([[x1, y1, x2, y2]]).to(self.device))

        # [[1, 4]*bs]
        return ins_warp_bbox 
    '''

    def compute_instance_losses(self, inputs, outputs):
        """loss of dynamic region"""

        losses = {}
        scale = 0

        total_mask = outputs[("cur_mask", 0, scale)]

        if total_mask.sum() < 1:
            losses["ins_loss"] = torch.zeros(1, requires_grad=True).mean().to(self.device)
            weight_fg = 0
            weight_bg = 1 - weight_fg
        else:
            color = inputs[("color", 0, scale)]
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            
            if self.opt.second_order_disp:
                smooth_loss = get_sec_smooth_loss(norm_disp, color)
            else:
                smooth_loss = get_smooth_loss(norm_disp, color)
            smooth_loss =  self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            losses["ins_loss/{}_smooth_loss".format(scale)] = smooth_loss

            reprojection_losses = []
            tgt_dynamic = outputs[("f_img_syn", 0, scale)]
            for frame_id in self.opt.frame_ids[1:]: # [-1, 1]
                pred_dynamic = outputs[("f_img_syn", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred_dynamic, tgt_dynamic))

            combined = torch.cat(reprojection_losses, 1)
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            #reproj_loss = to_optimise.mean()
            reproj_loss = to_optimise.sum() / total_mask.sum()
            
            losses["ins_loss/{}_reproj".format(scale)] = reproj_loss
            losses["ins_loss_{}".format(scale)] = reproj_loss + smooth_loss

            losses["ins_loss"] = reproj_loss + smooth_loss
            # print("repro\t", reproj_loss.detach().cpu().numpy(), "smooth\t", smooth_loss.detach().cpu().numpy())

            weight_fg = total_mask.sum() / total_mask.nelement()
            weight_bg = 1 - weight_fg
        
        return weight_fg, weight_bg, losses        

    # def log_time(self, batch_idx, duration, loss):
    def log_time(self, batch_idx, duration, loss, ins_loss=None, bg_loss=None):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0

        if ins_loss is not None:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | loss: {:.5f}| ins_loss: {:.5f} | bg_loss: {:.5f} | time elapsed: {} | time left: {}"
            
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, ins_loss, bg_loss,
                                    sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        else:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | loss: {:.5f} | time elapsed: {} | time left: {}"
            
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                    sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses, add_image=False):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        
        if add_image == True:
            for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
                # for s in self.opt.scales:
                for s in [0]:
                    for frame_id in self.opt.frame_ids:
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, self.step)
                        if s == 0 and frame_id != 0:
                            writer.add_image(
                                "color_pred_{}_{}/{}".format(frame_id, s, j),
                                outputs[("color", frame_id, s)][j].data, self.step)

                        if self.opt.instance_pose:
                            if frame_id == 0:
                                writer.add_image(
                                    "outputs_f_img_syn_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("f_img_syn", frame_id, 0)][j].data, self.step)

                            if frame_id != 0:
                                writer.add_image(
                                    "color_pred_ori_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("color_ori", frame_id, s)][j].data, self.step)
                                
                                writer.add_image(
                                    "outputs_f_img_syn_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("f_img_syn", frame_id, 0)][j].data, self.step)
                                
                                writer.add_image(
                                    "color_diff_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("color_diff", frame_id, 0)][j].data, self.step)

                                writer.add_image(
                                    "warped_mask_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("warped_mask", frame_id, 0)][j].data, self.step)

                                writer.add_image(
                                    "mask_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("mask", frame_id, 0)][j].data, self.step)

                                '''
                                outputs[("color_ori", frame_id, scale)] = color_ori
                                outputs[("color_diff", frame_id, scale)] = color_new - color_ori
                                outputs[("color", frame_id, scale)] = color_new
                                outputs[("f_img_syn", frame_id, scale)] = img0_pred_final
                                outputs[("warped_mask", frame_id, scale)] = mask0_pred_final
                                '''
                    
                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("disp", s)][j]), self.step)

                    # if self.opt.predictive_mask:
                    #     for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                    #         writer.add_image(
                    #             "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                    #             outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                    #             self.step)

                    # elif not self.opt.disable_automasking:
                    #     writer.add_image(
                    #         "automask_{}/{}".format(s, j),
                    #         outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
            input()

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
    
    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        models_to_load = self.opt.models_to_load
        if self.opt.instance_pose:
            models_to_load.append("instance_pose")

        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            try:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)
            except Exception as e:
                print(e)
        
        # loading adam state
        if self.opt.fix_pose:
            pass
        else:
            optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
            if os.path.isfile(optimizer_load_path):
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")
        