#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/17 10:21
# @Author  : wangjie

#   Fast PointNet for Point Cloud
from typing import List, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import einsum
# from einops import rearrange, repeat

# from pointnet2_ops import pointnet2_utils

from ..build import MODELS
from ..layers.group import KNN
from ..layers import create_convblock1d, furthest_point_sample, three_interpolation
from ..layers.group import grouping_operation, QueryAndGroup

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)



def index_points(points, idx):
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

class FeaturePropogation(nn.Module):
    """The Feature Propogation module in PointNet++
    """

    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(
                nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=act_args
                                                ))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, px1, px2=None):
        # pxb1 is with the same size of upsampled points
        if px2 is None:
            _, x = px1  # (B, N, 3), (B, C, N)
            x_global = self.pool(x)
            x = torch.cat(
                (x, self.linear2(x_global).unsqueeze(-1).expand(-1, -1, x.shape[-1])), dim=1)
            x = self.linear1(x)
        else:
            p1, x1 = px1
            p2, x2 = px2
            x = self.convs(
                torch.cat((x1, three_interpolation(p1, p2, x2)), dim=1))
        return x




class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)

class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class ConvBNReLURes1D_woRelu(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D_woRelu, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        # print('wo residual')
        return self.net2(self.net1(x)) + x
        # return self.act(self.net2(self.net1(x)) + x)

class Extraction_Feat_Relu(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True, with_relu=None):
        """
        """
        super(Extraction_Feat_Relu, self).__init__()
        # in_channels = 3+2*channels if use_xyz else 2*channels
        in_channels = 3+channels if use_xyz else channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        self.with_relu = with_relu
        operation = []
        for _ in range(blocks):
            if self.with_relu:
                operation.append(
                    ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                    bias=bias, activation=activation)
                )
            else:
                operation.append(
                    ConvBNReLURes1D_woRelu(out_channels, groups=groups, res_expansion=res_expansion,
                                    bias=bias, activation=activation)
                )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        '''
        input:
            x:[b, d, n]
        output:
            x:[b, d_out, n]
        '''
        #  x:[b, d, n]
        b, d, n = x.size()  # torch.Size([32, d, n])
        x = self.transfer(x)
        x = self.operation(x)  # [b, out_channels, n]
        return x

#   PointsetGrouper_paradigm + norm
class PointsetGrouper(nn.Module):
    def __init__(self, channel, reduce, kneighbors, radi, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param radi: radius of ball_query
        :param kwargs: others
        norm_way: batchnorm/pointmlpnorm/pointsetnorm
        """
        super(PointsetGrouper, self).__init__()
        self.reduce = reduce
        self.kneighbors = kneighbors
        self.radi = radi
        self.use_xyz = use_xyz

        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            # add_channel=3 if self.use_xyz else 0
            add_channel=0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))
        self.ballquery = QueryAndGroup(self.radi, self.kneighbors)

    def forward(self, xyz, points):
        '''
        input:
            xyz:[b, p, 3]
            points:[b, p, d]
        output:
            new_xyz:[b, g, 3]
            new_point:[b, d, g]

        '''
        #   xyz:[b, p, 3]  points:[b, p, d]
        B, N, C = points.shape
        xyz = xyz.contiguous()  # xyz [batch, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx = furthest_point_sample(xyz, xyz.shape[1]//self.reduce).long()  # [B, npoint]
        # new_xyz = grouping_operation(xyz.transpose(1,2).contiguous(), fps_idx)      #   [B, 3, npoint]
        # new_points = grouping_operation(points.transpose(1,2).contiguous(), fps_idx)    #   [B, d, npoint]
        new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3))         #   [B, npoint, 3]
        new_points = torch.gather(points, 1, fps_idx.unsqueeze(-1).expand(-1, -1, points.shape[-1]))   #   [B, npoint, d]
        # new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        # new_points = index_points(points, fps_idx)  # [B, npoint, d]


        #   knn -> ballquery
        grouped_xyz, grouped_points = self.ballquery(query_xyz=new_xyz, support_xyz=xyz, features=points.transpose(1,2).contiguous())
        #   grouped_xyz: [B, 3, npoint, k],  grouped_points: [B, d, npoint, k]
        grouped_xyz = grouped_xyz.permute(0, 2, 3, 1).contiguous()          #   [B, 3, npoint, k] -> [B, npoint, k, 3]
        grouped_points = grouped_points.permute(0, 2, 3, 1).contiguous()    #   [B, d, npoint, k] -> [B, npoint, k, d]



        # _, idx = self.knn(xyz, new_xyz)
        # grouped_xyz = grouping_operation(xyz.transpose(1,2).contiguous(), idx).permute(0, 2, 3, 1).contiguous()      #   [B, 3, npoint, k] -> [B, npoint, k, 3]
        # grouped_points = grouping_operation(points.transpose(1,2).contiguous(), idx).permute(0, 2, 3, 1).contiguous()     #   [B, d, npoint, k] -> [B, npoint, k, d]
        # grouped_xyz = index_points(xyz, idx.long())  # [B, npoint, k, 3]
        # grouped_points = index_points(points, idx.long())  # [B, npoint, k, d]

        # if self.use_xyz:
        #     grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                # mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            # std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            # grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = (grouped_points-mean)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.max(grouped_points, dim=2)[0].permute(0, 2, 1)    #   [b, d, npoint]
        if self.use_xyz:
            new_points = torch.cat([new_points, new_xyz.permute(0, 2, 1)], dim=1)   #   [b, d+3, npoint]
        # new_points = torch.mean(grouped_points, dim=2).permute(0, 2, 1)    #   [b, d, npoint]
        return new_xyz, new_points

@MODELS.register_module()
class PointGLEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 with_relu=None,
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], radii= [0.1, 0.2, 0.4, 0.8],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        super().__init__()
        self.stages = len(pre_blocks)
        self.in_channels = in_channels
        self.embedding = ConvBNReLU1D(self.in_channels, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."

        self.use_xyz = use_xyz
        self.extract_feat_list = nn.ModuleList()
        self.pointset_grouper_list = nn.ModuleList()
        self.with_relu = with_relu

        channels = [embed_dim]
        last_channel = embed_dim
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            channels.append(out_channel)
            pre_block_num = pre_blocks[i]
            kneighbor = k_neighbors[i]
            radi = radii[i]
            reduce = reducers[i]

            #   append pre_block_list
            extract_feat = Extraction_Feat_Relu(last_channel, out_channel, pre_block_num, groups=groups,
                                                res_expansion=res_expansion,
                                                bias=bias, activation=activation, use_xyz=self.use_xyz,
                                                with_relu=self.with_relu)
            self.extract_feat_list.append(extract_feat)

            #   append pointset_grouper_list
            local_grouper = PointsetGrouper(out_channel, reduce, kneighbor, radi, self.use_xyz, normalize)
            self.pointset_grouper_list.append(local_grouper)

            last_channel = out_channel

        self.out_channels = last_channel
        self.channel_list = channels


    def forward(self, x, f0=None):

        return self.forward_cls_feat(x, f0)

    def forward_cls_feat(self, p, x=None):   #   p:[B, 3, N]
    # def forward_cls_feat(self, x):   #   x:[B, 3, N]
        if isinstance(p, dict):
            p, x = p['pos'], p.get('x', None)
        if x is None:
            x = p.transpose(1, 2).contiguous()
        xyz = p
        batch_size, _, _ = x.size()
        if self.use_xyz:
            x = torch.cat([x, self.embedding(x)], dim=1 ) #   [B, D+3, N]
        else:
            x = self.embedding(x)  # B,D,N
        for i in range(self.stages):
            #   Give feat[b, d, p], return new_feat[b, d_out, p]
            x = self.extract_feat_list[i](x)
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, d, g]
            xyz, x = self.pointset_grouper_list[i](xyz, x.permute(0, 2, 1))
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1) #   [b, d, g] -> [b, d]
        # x = self.classifier(x)
        return x

    def forward_all_features(self, p0, x0=None):
        '''
            p['pos']: [B, N, 3]
            p['x']: [B, 3, 1024]
        '''
        if hasattr(p0, 'keys'):
            p0, x0 = p0['pos'], p0['x']
        if x0 is None:
            x0 = p0.clone().transpose(1, 2).contiguous()


        xyz = p0
        batch_size, _, _ = x0.size()
        xyz_list = [xyz]
        x_list = [x0]
        if self.use_xyz:
            x = torch.cat([x0, self.embedding(x0)], dim=1 ) #   [B, D+3, N]
        else:
            x = self.embedding(x0)  # B,D,N
        xyz_list.append(xyz)
        x_list.append(x)
        for i in range(self.stages):
            #   Give feat[b, d, p], return new_feat[b, d_out, p]
            x = self.extract_feat_list[i](x)
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, d, g]
            xyz, x = self.pointset_grouper_list[i](xyz, x.permute(0, 2, 1))
            xyz = xyz.contiguous()
            x = x.contiguous()
            xyz_list.append(xyz)
            x_list.append(x)
        return xyz_list, x_list

    def forward(self, p0, x0=None):
        self.forward_all_features(p0, x0)



@MODELS.register_module()
class PointGLDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        # the output channel after interpolation
        fp_channels = encoder_channel_list[:-1]

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f):
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = (
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]
            # f[i - 1] = self.decoder[i][1:](
            #     [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]
        return f[-len(self.decoder) - 1]



class FPNET(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 with_relu=None,
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        super(FPNET, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."

        self.use_xyz = use_xyz
        self.extract_feat_list = nn.ModuleList()
        self.pointset_grouper_list = nn.ModuleList()
        self.with_relu = with_relu


        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce

            #   append pre_block_list
            extract_feat = Extraction_Feat_Relu(last_channel, out_channel, pre_block_num, groups=groups,
                                                res_expansion=res_expansion,
                                                bias=bias, activation=activation, use_xyz=self.use_xyz,
                                                with_relu=self.with_relu)
            self.extract_feat_list.append(extract_feat)

            #   append pointset_grouper_list
            local_grouper = PointsetGrouper(out_channel, anchor_points, kneighbor, self.use_xyz, normalize)
            self.pointset_grouper_list.append(local_grouper)

            last_channel = out_channel

        self.act = get_activation(activation)
        if self.use_xyz:
            self.classifier = nn.Sequential(
                nn.Linear(last_channel+3, 512),
                nn.BatchNorm1d(512),
                self.act,
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                self.act,
                nn.Dropout(0.5),
                nn.Linear(256, self.class_num)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(last_channel, 512),
                nn.BatchNorm1d(512),
                self.act,
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                self.act,
                nn.Dropout(0.5),
                nn.Linear(256, self.class_num)
            )

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        if self.use_xyz:
            x = torch.cat([x, self.embedding(x)], dim=1 ) #   [B, D+3, N]
        else:
            x = self.embedding(x)  # B,D,N

        for i in range(self.stages):
            #   Give feat[b, d, p], return new_feat[b, d_out, p]
            x = self.extract_feat_list[i](x)
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, d, g]
            xyz, x = self.pointset_grouper_list[i](xyz, x.permute(0, 2, 1))

        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1) #   [b, d, g] -> [b, d]
        x = self.classifier(x)
        return x

#   use_xyz=False, normalization=anchor  83.3   1x1080ti 49s/epoch
#   more data(2048points): 84.386/84.316
def fpnet_baseline(num_classes=40, **kwargs) -> FPNET:
    return FPNET(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   with_relu=False,
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2],
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)

#   use_xyz=False, normalization=anchor 81.437    1x1080ti 27s/epoch FLOPs = 0.05195904G Params = 0.489359M
def fpnet_baseline_Lite(num_classes=40, **kwargs) -> FPNET:
    return FPNET(points=1024, class_num=num_classes, embed_dim=32, groups=1, res_expansion=0.25,
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   with_relu=False,
                   dim_expansion = [2, 2, 2, 1], pre_blocks = [1, 1, 2, 1],
                   k_neighbors = [24, 24, 24, 24], reducers = [2, 2, 2, 2], ** kwargs)




if __name__ == '__main__':
    data = torch.rand(1, 3, 1024).cuda()
    print("===> testing pointMLP ...")
    model = fpnet_baseline().cuda()
    model = model.eval()
    # print(model)
    out = model(data)
    print(out.shape)

    from thop import profile
    flops, params = profile(model, (data,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')