#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 10:42
# @Author  : wangjie

import math
import numpy as np
import torch
import torch.nn as nn
# from pointnet2_ops import pointnet2_utils
# import sys
from ..models.layers import furthest_point_sample

class PointWOLF_classversion(object):
    def __init__(self, w_num_anchor=4, w_sigma=0.5, w_R_range=10, w_S_range=3, w_T_range=0.25):
        self.num_anchor = w_num_anchor
        self.sigma = w_sigma
        self.R_range = (-abs(w_R_range), abs(w_R_range))
        self.S_range = (1., w_S_range)
        self.T_range = (-abs(w_T_range), abs(w_T_range))

        self.w_R_range = w_R_range
        self.w_S_range = w_S_range
        self.w_T_range = w_T_range


    def __call__(self, xyz):
        """
        Input:
            xyz: [B, N, 3]
        Output:
            xyz: [B, N, 3]
        """
        M = self.num_anchor
        B, N, _ = xyz.shape
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.num_anchor).long()  # [B, M]
        fps_idx = furthest_point_sample(xyz.contiguous(), self.num_anchor).long()  # [B, M]
        xyz_anchor = self.index_points(xyz, fps_idx)                            #   [B, M, 3]

        xyz_repeat = xyz.unsqueeze(dim=1).repeat(1, self.num_anchor, 1, 1)      #   [B, M, N, 3]

        # Move to canonical space
        xyz_normalize = xyz_repeat - xyz_anchor.unsqueeze(dim=-2)               #   [B, M, N, 3]

        # Local transformation at anchor point
        xyz_transformed = self.local_transformaton(xyz_normalize)  # (B,M,N,3)

        # Move to origin space
        xyz_transformed = xyz_transformed + xyz_anchor.reshape(B, M, 1, 3)  # (B,M,N,3)
        xyz_new = self.kernel_regression(xyz, xyz_anchor, xyz_transformed)  #   [B, N, 3]
        xyz_new = self.normalize(xyz_new)


        return xyz, xyz_new



    def index_points(self, points, idx):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points


    def kernel_regression(self, pos, pos_anchor, pos_transformed):
        """
        input :
            pos([B,N,3])
            pos_anchor([B,M,3])
            pos_transformed([B,M,N,3])

        output :
            pos_new([B,N,3]) : Pointcloud after weighted local transformation
        """
        B, M, N, _ = pos_transformed.shape

        # Distance between anchor points & entire points
        sub = pos_anchor.unsqueeze(dim=-2).repeat(1, 1, N, 1) - \
              pos.unsqueeze(dim=1).repeat(1, M, 1, 1)  # (B, M, N, 3)

        project_axis = self.get_random_axis(B, 1).to(pos.device)   #   [B, 1, 3]

        projection = project_axis.unsqueeze(dim=-2) * torch.eye(3).to(pos.device)   #   [B, 1, 3, 3]

        # Project distance
        sub = sub @ projection  # (B, M, N, 3)
        sub = torch.sqrt(((sub) ** 2).sum(dim=-1))  # (B, M, N)

        # Kernel regression
        weight = torch.exp(-0.5 * (sub ** 2) / (self.sigma ** 2)).to(pos.device)  # (B, M, N)

        pos_new = (weight.unsqueeze(dim=-1).repeat(1, 1, 1, 3) * pos_transformed).sum(dim=1)  # (B, N, 3)
        pos_new = (pos_new / weight.sum(dim=1, keepdims=True).transpose(1,2).contiguous())  # normalize by weight   [B, N, 3]

        return pos_new


    def local_transformaton(self, pos_normalize):
        """
        input :
            pos_normalize([B,M,N,3])

        output :
            pos_normalize([B,M,N,3]) : Pointclouds after local transformation centered at M anchor points.
        """
        B, M, N, _ = pos_normalize.shape
        a = torch.Tensor(B, M, 3).uniform_(0, 1)
        transformation_dropout = torch.bernoulli(a).to(pos_normalize.device)     #   [B, M, 3]
        transformation_axis = self.get_random_axis(B, M).to(pos_normalize.device)  # [B, M, 3]

        degree = torch.tensor(math.pi).to(pos_normalize.device) * torch.FloatTensor(B, M, 3).uniform_(*self.R_range).to(pos_normalize.device) / 180.0 \
                 * transformation_dropout[:, :, 0:1]    #   [B, M, 3], sampling from (-R_range, R_range)

        scale = torch.FloatTensor(B, M, 3).uniform_(*self.S_range).to(pos_normalize.device) * transformation_dropout[:, :, 1:2]  # [B, M, 3], sampling from (1, S_range)
        scale = scale * transformation_axis
        scale = scale + 1 * (scale == 0)  # Scaling factor must be larger than 1

        trl = torch.FloatTensor(B, M, 3).uniform_(*self.T_range).to(pos_normalize.device) * transformation_dropout[:, :, 2:3]    # [B, M, 3], sampling from (1, S_range)
        trl *= transformation_axis

        # Scaling Matrix
        S = scale.unsqueeze(dim=-2) * torch.eye(3).to(pos_normalize.device)  # scailing factor to diagonal matrix (M,3) -> (M,3,3)

        # Rotation Matrix
        sin = torch.sin(degree).unsqueeze(dim=-1)
        cos = torch.cos(degree).unsqueeze(dim=-1)
        sx, sy, sz = sin[:, :, 0], sin[:, :, 1], sin[:, :, 2]
        cx, cy, cz = cos[:, :, 0], cos[:, :, 1], cos[:, :, 2]

        R = torch.cat([cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
                      sz * cy, sz * sy * sx + cz * cy, sz * sy * cx - cz * sx,
                      -sy, cy * sx, cy * cx], dim=-1).reshape(B, M, 3, 3)
        pos_normalize = pos_normalize @ R @ S + trl.reshape(B, M, 1, 3)


        return pos_normalize


    def get_random_axis(self, batch, n_axis):
        """
        input :
            batch(int)
            n_axis(int)

        output :
            axis([batch, n_axis,3]) : projection axis
        """
        axis = torch.randint(1, 8, (batch, n_axis))  # 1(001):z, 2(010):y, 3(011):yz, 4(100):x, 5(101):xz, 6(110):xy, 7(111):xyz
        m = 3
        axis = (((axis[:, :, None] & (1 << torch.arange(m)))) > 0).int()
        return axis

    def normalize(self, pos):
        """
        input :
            pos([B, N, 3])

        output :
            pos([B, N, 3]) : normalized Pointcloud
        """
        B, N, C = pos.shape
        pos = pos - pos.mean(axis=-2, keepdims=True)    #   [B, N, 3]
        scale = (1 / torch.sqrt((pos ** 2).sum(dim=-1)).max(dim=-1)[0]) * 0.999999   #   [B, 1]
        pos = scale.reshape(B, 1, 1).repeat(1, N, C) * pos
        # scale = (1 / torch.sqrt((pos ** 2).sum(dim=-1)).max()) * 0.999999   #   [B, 1]
        # pos = scale * pos
        return pos