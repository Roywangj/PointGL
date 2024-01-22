#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 12:57
# @Author  : wangjie


import os, sys, h5py, pickle, numpy as np, logging, os.path as osp
import torch
from torch.utils.data import Dataset
# from ..build import DATASETS
# from openpoints.models.layers import fps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# @DATASETS.register_module()
class ScanObjectNNC(Dataset):
    classes = [
        "bag",
        "bin",
        "box",
        "cabinet",
        "chair",
        "desk",
        "display",
        "door",
        "shelf",
        "table",
        "bed",
        "pillow",
        "sink",
        "sofa",
        "toilet",
    ]
    num_classes = 15
    gravity_dim = 1
    """The hardest variant of ScanObjectNN.
    The data we use is: `training_objectdataset_augmentedrot_scale75.h5`[1],
    where there are 2048 points in training and testing.
    The number of training samples is: 11416, and the number of testing samples is 2882.
    Args:
    """
    def __init__(self,
                 data_dir='./data/ScanObjectNN_C/scanobjectnn_c',
                 split=None,
                 num_points=2048,
                 uniform_sample=True,
                 transform=None,
                 **kwargs):
        self.partition = split
        self.transform = transform
        self.num_points = num_points

        h5_name = os.path.join(
            data_dir, f'{split}.h5')

        if not osp.isfile(h5_name):
            raise FileExistsError(
                f'{h5_name} does not exist, please download dataset at first')
        with h5py.File(h5_name, 'r') as f:
            self.points = np.array(f['data']).astype(np.float32)
            self.labels = np.array(f['label']).astype(int)
        logging.info(f'Successfully load ScanObjectNN {split} '
                     f'size: {self.points.shape}, num_classes: {self.labels.max()+1}')

    @property
    def num_classes(self):
        return self.labels.max() + 1

    def __getitem__(self, idx):
        current_points = self.points[idx][:self.num_points]
        label = self.labels[idx]
        if self.partition == 'train':
            np.random.shuffle(current_points)
        data = {'pos': current_points,
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)

        # height appending. @KPConv
        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = data['pos']
        return data

    def __len__(self):
        return self.points.shape[0]

import pprint
def eval_corrupt_wrapper_scanobjectnnc(model, fn_test_corrupt, args_test_corrupt, path, epoch):
    """
    The wrapper helps to repeat the original testing function on all corrupted test sets.
    It also helps to compute metrics.
    :param model: model
    :param fn_test_corrupt: original evaluation function, returns a dict of metrics, e.g., {'acc': 0.93}
    :param args_test_corrupt: a dict of arguments to fn_test_corrupt, e.g., {'test_loader': loader}
    :return:
    """
    file = open(os.path.join(path, 'outcorruption.txt'), "a")
    file.write(f"epoch: {epoch} \n")
    corruptions = [
        'clean',
        'scale',
        'jitter',
        'rotate',
        'dropout_global',
        'dropout_local',
        'add_global',
        'add_local',
    ]
    # DGCNN_OA = {
    #     'clean': 0.926,
    #     'scale': 0.906,
    #     'jitter': 0.684,
    #     'rotate': 0.785,
    #     'dropout_global': 0.752,
    #     'dropout_local': 0.793,
    #     'add_global': 0.705,
    #     'add_local': 0.725
    # }
    OA_clean = None
    # perf_all = {'OA': [], 'CE': [], 'RCE': []}
    perf_all = {'OA': []}
    for corruption_type in corruptions:
        perf_corrupt = {'OA': []}
        for level in range(5):
            if corruption_type == 'clean':
                split = "clean"
            else:
                split = corruption_type + '_' + str(level)
            test_perf = fn_test_corrupt(split=split, model=model, **args_test_corrupt)
            if not isinstance(test_perf, dict):
                test_perf = {'acc': test_perf}
            perf_corrupt['OA'].append(test_perf['acc'])
            test_perf['corruption'] = corruption_type
            if corruption_type != 'clean':
                test_perf['level'] = level
            pprint.pprint(test_perf, width=200)
            file.write(f"{test_perf} \n")
            if corruption_type == 'clean':
                OA_clean = round(test_perf['acc'], 3)
                break
        for k in perf_corrupt:
            perf_corrupt[k] = sum(perf_corrupt[k]) / len(perf_corrupt[k])
            perf_corrupt[k] = round(perf_corrupt[k], 3)

        if corruption_type != 'clean':
        #     perf_corrupt['CE'] = (1 - perf_corrupt['OA']) / (1 - DGCNN_OA[corruption_type])
        #     perf_corrupt['RCE'] = (OA_clean - perf_corrupt['OA']) / (DGCNN_OA['clean'] - DGCNN_OA[corruption_type])
            for k in perf_all:
                perf_corrupt[k] = round(perf_corrupt[k], 3)
                perf_all[k].append(perf_corrupt[k])
        perf_corrupt['corruption'] = corruption_type
        perf_corrupt['level'] = 'Overall'
        pprint.pprint(perf_corrupt, width=200)
        file.write(f"{perf_corrupt} \n")
    for k in perf_all:
        perf_all[k] = sum(perf_all[k]) / len(perf_all[k])
        perf_all[k] = round(perf_all[k], 3)
    # perf_all['mCE'] = perf_all.pop('CE')
    # perf_all['RmCE'] = perf_all.pop('RCE')
    perf_all['mOA'] = perf_all.pop('OA')
    pprint.pprint(perf_all, width=200)
    file.write(f"{perf_all} \n")
    file.close()


if __name__ == '__main__':
    data_clean = ScanObjectNNC(split='clean')
    data_scale0 = ScanObjectNNC(split='scale_0')
    data = data_clean.__getitem__(0)
    print(f"data_clean size: {data_clean.__len__()}")
    print(f"data_scale0 size: {data_scale0.__len__()}")
    print(f"data.shape: {data['x'].shape}")
    print(f"label.shape: {data['y'].shape}")
    print(f"label: {data['y']}")
