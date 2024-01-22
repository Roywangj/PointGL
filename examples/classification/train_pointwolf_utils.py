#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/17 10:54
# @Author  : wangjie

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 11:00
# @Author  : wangjie

import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps
from openpoints.online_aug.pointwolf import PointWOLF_classversion
from openpoints.online_aug import rsmix_provider


def train_one_epoch_pointwolf(model, train_loader, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points

    pointwolf = PointWOLF_classversion(**cfg.pointwolf)
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        points = data['x']
        _, points[:, :, :3] = pointwolf(points[:, :, :3])
        target = data['y']
        """ bebug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0])
        """
        num_curr_pts = points.shape[1]
        if num_curr_pts > npoints:  # point resampling strategy
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
            if  points.size(1) < point_all:
                point_all = points.size(1)
            fps_idx = furthest_point_sample(
                points[:, :, :3].contiguous(), point_all)
            fps_idx = fps_idx[:, np.random.choice(
                point_all, npoints, False)]
            points = torch.gather(
                points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))

        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits, loss = model.get_logits_loss(data, target) if not hasattr(model, 'module') else model.module.get_logits_loss(data, target)
        loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    macc, overallacc, accs = cm.all_acc()
    return loss_meter.avg, macc, overallacc, accs, cm


def train_one_epoch_rsmix(model, train_loader, optimizer, scheduler, epoch, cfg):
    criterion = build_criterion_from_cfg(cfg.criterion_args)
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points

    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        points = data['x']
        target = data['y']
        ##############################
        rsmix = False
        points = points.cpu().numpy()
        target = target.cpu()
        target = target.unsqueeze(dim=-1)
        r = np.random.rand(1)
        if cfg.rsmix_params.beta > 0 and r < cfg.rsmix_params.rsmix_prob:
            rsmix = True
            points, lam, target, target_b = rsmix_provider.rsmix(points, target, beta=cfg.rsmix_params.beta, n_sample=cfg.rsmix_params.nsample,
                                                             KNN=cfg.rsmix_params.knn)
        points = torch.FloatTensor(points)
        if rsmix:
            lam = torch.FloatTensor(lam)
            lam, target_b = lam.cuda(non_blocking=True), target_b.cuda(non_blocking=True).squeeze()
        points, target = points.cuda(non_blocking=True), target.cuda(non_blocking=True).squeeze()
        ########
        """ bebug
        from openpoints.dataset import vis_points 
        vis_points(data['pos'].cpu().numpy()[0])
        """
        num_curr_pts = points.shape[1]
        if num_curr_pts > npoints:  # point resampling strategy
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
            if  points.size(1) < point_all:
                point_all = points.size(1)
            fps_idx = furthest_point_sample(
                points[:, :, :3].contiguous(), point_all)
            fps_idx = fps_idx[:, np.random.choice(
                point_all, npoints, False)]
            points = torch.gather(
                points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))
        # for key in data.keys():
        #     data[key] = data[key].cuda(non_blocking=True)
        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits = model(data)

        if rsmix:
            loss = 0
            for i in range(cfg.batch_size):
                loss_tmp = criterion(logits[i].unsqueeze(0), target[i].unsqueeze(0).long()) * (1 - lam[i]) \
                           + criterion(logits[i].unsqueeze(0), target_b[i].unsqueeze(0).long()) * lam[i]
                loss += loss_tmp
            loss = loss / cfg.batch_size
        else:
            loss = criterion(logits, target)
        loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    macc, overallacc, accs = cm.all_acc()
    return loss_meter.avg, macc, overallacc, accs, cm

def train_one_epoch_wolfmix(model, train_loader, optimizer, scheduler, epoch, cfg):
    criterion = build_criterion_from_cfg(cfg.criterion_args)
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points

    pointwolf = PointWOLF_classversion(**cfg.wolfmix.pointwolf)
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        points = data['x']
        _, points[:, :, :3] = pointwolf(points[:, :, :3])
        target = data['y']
        ##############################
        rsmix = False
        points = points.cpu().numpy()
        target = target.cpu()
        target = target.unsqueeze(dim=-1)
        r = np.random.rand(1)
        if cfg.wolfmix.rsmix_params.beta > 0 and r < cfg.wolfmix.rsmix_params.rsmix_prob:
            rsmix = True
            points, lam, target, target_b = rsmix_provider.rsmix(points, target, beta=cfg.wolfmix.rsmix_params.beta, n_sample=cfg.wolfmix.rsmix_params.nsample,
                                                             KNN=cfg.wolfmix.rsmix_params.knn)
        points = torch.FloatTensor(points)
        if rsmix:
            lam = torch.FloatTensor(lam)
            lam, target_b = lam.cuda(non_blocking=True), target_b.cuda(non_blocking=True).squeeze()
        points, target = points.cuda(non_blocking=True), target.cuda(non_blocking=True).squeeze()
        ########
        """ bebug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0])
        """
        num_curr_pts = points.shape[1]
        if num_curr_pts > npoints:  # point resampling strategy
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
            if  points.size(1) < point_all:
                point_all = points.size(1)
            fps_idx = furthest_point_sample(
                points[:, :, :3].contiguous(), point_all)
            fps_idx = fps_idx[:, np.random.choice(
                point_all, npoints, False)]
            points = torch.gather(
                points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))

        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits = model(data)

        if rsmix:
            loss = 0
            for i in range(cfg.batch_size):
                loss_tmp = criterion(logits[i].unsqueeze(0), target[i].unsqueeze(0).long()) * (1 - lam[i]) \
                           + criterion(logits[i].unsqueeze(0), target_b[i].unsqueeze(0).long()) * lam[i]
                loss += loss_tmp
            loss = loss / cfg.batch_size
        else:
            loss = criterion(logits, target)
        loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    macc, overallacc, accs = cm.all_acc()
    return loss_meter.avg, macc, overallacc, accs, cm
