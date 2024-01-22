#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/13 16:45
# @Author  : wangjie

import numpy as np
import re

corruptions = [
    'scale',
    'jitter',
    'rotate',
    'dropout_global',
    'dropout_local',
    'add_global',
    'add_local',
]

DGCNN = {
    'clean': 0.8581,
    'scale': 0.57752,
    'jitter': 0.45579,
    'rotate': 0.7331,
    'dropout_global': 0.62151,
    'dropout_local': 0.69688,
    'add_global': 0.53956,
    'add_local': 0.77294
}

PointNet = {'clean': 0.7408, 'scale': 0.40354, 'jitter': 0.51915, 'rotate': 0.58119, 'dropout_global': 0.70347, 'dropout_local': 0.62019, 'add_global': 0.35892, 'add_local': 0.54587}

PointNet_wpointwolf = {'clean': 0.7512, 'scale': 0.44566, 'jitter': 0.50937, 'rotate': 0.61013, 'dropout_global': 0.69264, 'dropout_local': 0.62061, 'add_global': 0.37287, 'add_local': 0.58799}
PointNet_wrsmix = {'clean': 0.7568, 'scale': 0.40749, 'jitter': 0.54476, 'rotate': 0.60465, 'dropout_global': 0.72755, 'dropout_local': 0.65003, 'add_global': 0.39167, 'add_local': 0.59542}
PointNet_wwolfmix = {'clean': 0.7575, 'scale': 0.43553, 'jitter': 0.51319, 'rotate': 0.63664, 'dropout_global': 0.70597, 'dropout_local': 0.64316, 'add_global': 0.40153, 'add_local': 0.62859}
PointNet_wadaptpoint = {'clean': 0.7418, 'scale': 0.41575, 'jitter': 0.52991, 'rotate': 0.595, 'dropout_global': 0.73185, 'dropout_local': 0.65198, 'add_global': 0.41249, 'add_local': 0.60625}

PointNet2 = {'clean': 0.8619, 'scale': 0.62096, 'jitter': 0.39993, 'rotate': 0.70458, 'dropout_global': 0.79174, 'dropout_local': 0.61291, 'add_global': 0.56405, 'add_local': 0.79452}
PointNet2_wpointwolf = {'clean': 0.8657, 'scale': 0.64532, 'jitter': 0.40867, 'rotate': 0.79681, 'dropout_global': 0.73317, 'dropout_local': 0.52526, 'add_global': 0.59611, 'add_local': 0.79362}
PointNet2_wrsmix = {'clean': 0.873, 'scale': 0.62408, 'jitter': 0.45184, 'rotate': 0.69327, 'dropout_global': 0.78952, 'dropout_local': 0.69986, 'add_global': 0.56454, 'add_local': 0.79764}
PointNet2_wwolfmix = {'clean': 0.8747, 'scale': 0.66384, 'jitter': 0.40694, 'rotate': 0.79667, 'dropout_global': 0.75711, 'dropout_local': 0.64316, 'add_global': 0.59424, 'add_local': 0.81888}
PointNet2_wadaptpoint = {'clean': 0.8671, 'scale': 0.63636, 'jitter': 0.38834, 'rotate': 0.75108, 'dropout_global': 0.83276, 'dropout_local': 0.7653, 'add_global': 0.57523, 'add_local': 0.80382}

DGCNN_wpointwolf = {'clean': 0.856, 'scale': 0.62179, 'jitter': 0.4306, 'rotate': 0.72873, 'dropout_global': 0.60611, 'dropout_local': 0.70201, 'add_global': 0.54872, 'add_local': 0.77099}
DGCNN_wrsmix = {'clean': 0.8654, 'scale': 0.56447, 'jitter': 0.47002, 'rotate': 0.73949, 'dropout_global': 0.65496, 'dropout_local': 0.74136, 'add_global': 0.51728, 'add_local': 0.77592}
DGCNN_wwolfmix = {'clean': 0.8723, 'scale': 0.6109, 'jitter': 0.44143, 'rotate': 0.75566, 'dropout_global': 0.64955, 'dropout_local': 0.74157, 'add_global': 0.55697, 'add_local': 0.80541}
DGCNN_wadaptpoint = {'clean': 0.8439, 'scale': 0.61714, 'jitter': 0.41513, 'rotate': 0.72929, 'dropout_global': 0.72651, 'dropout_local': 0.7787, 'add_global': 0.57037, 'add_local': 0.78806}

PointNext = {'clean': 0.8734, 'scale': 0.66072, 'jitter': 0.41298, 'rotate': 0.73442, 'dropout_global': 0.69473, 'dropout_local': 0.71437, 'add_global': 0.56544, 'add_local': 0.80125}
PointNext_wpointwolf = {'clean': 0.8744, 'scale': 0.656, 'jitter': 0.38584, 'rotate': 0.81027, 'dropout_global': 0.65996, 'dropout_local': 0.72026, 'add_global': 0.5626, 'add_local': 0.80999}
PointNext_wrsmix = {'clean': 0.881, 'scale': 0.64566, 'jitter': 0.41589, 'rotate': 0.7356, 'dropout_global': 0.71638, 'dropout_local': 0.77786, 'add_global': 0.55704, 'add_local': 0.8118}
PointNext_wwolfmix = {'clean': 0.8772, 'scale': 0.65399, 'jitter': 0.35052, 'rotate': 0.81319, 'dropout_global': 0.66037, 'dropout_local': 0.76364, 'add_global': 0.58869, 'add_local': 0.81825}
PointNext_wadaptpoint = {'clean': 0.8845, 'scale': 0.65767, 'jitter': 0.43956, 'rotate': 0.79521, 'dropout_global': 0.80784, 'dropout_local': 0.80951, 'add_global': 0.58064, 'add_local': 0.81291}

RPC = {'clean': 0.7474, 'scale': 0.44372, 'jitter': 0.41617, 'rotate': 0.62582, 'dropout_global': 0.44941, 'dropout_local': 0.6043, 'add_global': 0.47425, 'add_local': 0.63963}
RPC_wpointwolf = {'clean': 0.7099, 'scale': 0.43851, 'jitter': 0.38446, 'rotate': 0.57155, 'dropout_global': 0.46204, 'dropout_local': 0.59473, 'add_global': 0.37779, 'add_local': 0.60673}
RPC_wrsmix = {'clean': 0.6568, 'scale': 0.36912, 'jitter': 0.48383, 'rotate': 0.5694, 'dropout_global': 0.51943, 'dropout_local': 0.59618, 'add_global': 0.29278, 'add_local': 0.52595}
RPC_wwolfmix = {'clean': 0.7967, 'scale': 0.54143, 'jitter': 0.36086, 'rotate': 0.63137, 'dropout_global': 0.5372, 'dropout_local': 0.67495, 'add_global': 0.43227, 'add_local': 0.71707}
RPC_wadaptpoint = {'clean': 0.8352, 'scale': 0.55052, 'jitter': 0.45281, 'rotate': 0.70409, 'dropout_global': 0.74691, 'dropout_local': 0.76405, 'add_global': 0.50583, 'add_local': 0.75954}

PointNext_wadaptpoint_anchor2 = {'clean': 0.8865, 'scale': 0.66357, 'jitter': 0.42117, 'rotate': 0.77793, 'dropout_global': 0.79528, 'dropout_local': 0.80104, 'add_global': 0.60271, 'add_local': 0.80958}
PointNext_wadaptpoint_anchor8 = {'clean': 0.8869, 'scale': 0.6433, 'jitter': 0.40583, 'rotate': 0.766, 'dropout_global': 0.82061, 'dropout_local': 0.81895, 'add_global': 0.57176, 'add_local': 0.8179}
PointNext_wadaptpoint_anchor16 = {'clean': 0.8775, 'scale': 0.64254, 'jitter': 0.42866, 'rotate': 0.76037, 'dropout_global': 0.81811, 'dropout_local': 0.82068, 'add_global': 0.539, 'add_local': 0.81166}
PointNext_wadaptpoint_feedback0 = {'clean': 0.8855, 'scale': 0.62562, 'jitter': 0.38883, 'rotate': 0.77016, 'dropout_global': 0.80951, 'dropout_local': 0.82297, 'add_global': 0.56551, 'add_local': 0.80791}
PointNext_wadaptpoint_feedback0dot5 = {'clean': 0.8869, 'scale': 0.64108, 'jitter': 0.42873, 'rotate': 0.78314, 'dropout_global': 0.8152, 'dropout_local': 0.82221, 'add_global': 0.56724, 'add_local': 0.8229}
PointNext_wadaptpoint_feedback2 = {'clean': 0.8824, 'scale': 0.6492, 'jitter': 0.40361, 'rotate': 0.78973, 'dropout_global': 0.81839, 'dropout_local': 0.81388, 'add_global': 0.58758, 'add_local': 0.81166}
PointNext_wadaptpoint_wodiscriminator = {'clean': 0.873, 'scale': 0.6372, 'jitter': 0.40021, 'rotate': 0.80312, 'dropout_global': 0.78966, 'dropout_local': 0.78397, 'add_global': 0.54268, 'add_local': 0.80416}
PointNext_wadaptpoint_womasking = {'clean': 0.8813, 'scale': 0.67654, 'jitter': 0.4431, 'rotate': 0.78668, 'dropout_global': 0.67974, 'dropout_local': 0.71055, 'add_global': 0.61319, 'add_local': 0.81485}
PointNext_wadaptpoint_wodeformation = {'clean': 0.8747, 'scale': 0.61291, 'jitter': 0.45517, 'rotate': 0.71541, 'dropout_global': 0.80049, 'dropout_local': 0.80298, 'add_global': 0.52033, 'add_local': 0.79611}
PointNext_wadaptpoint_deformwoglobal = {'clean': 0.8779, 'scale': 0.63359, 'jitter': 0.38605, 'rotate': 0.80215, 'dropout_global': 0.74559, 'dropout_local': 0.80298, 'add_global': 0.56495, 'add_local': 0.81187}
PointNext_wadaptpoint_deformwoattention = {'clean': 0.8869, 'scale': 0.64289, 'jitter': 0.45371, 'rotate': 0.78078, 'dropout_global': 0.81055, 'dropout_local': 0.81305, 'add_global': 0.56759, 'add_local': 0.81666}
PointNext_wadaptpoint_maskwoglobal = {'clean': 0.8799, 'scale': 0.63227, 'jitter': 0.39681, 'rotate': 0.77314, 'dropout_global': 0.81201, 'dropout_local': 0.81457, 'add_global': 0.55635, 'add_local': 0.80111}
PointNext_wadaptpoint_maskwoattention = {'clean': 0.8893, 'scale': 0.65642, 'jitter': 0.42075, 'rotate': 0.78841, 'dropout_global': 0.81548, 'dropout_local': 0.81735, 'add_global': 0.58758, 'add_local': 0.80992}
PointNext_wadaptpoint_wrsmix = {'clean': 0.8834, 'scale': 0.66093, 'jitter': 0.43928, 'rotate': 0.72019, 'dropout_global': 0.73664, 'dropout_local': 0.7728, 'add_global': 0.59646, 'add_local': 0.81582}
PointNext_wadaptpoint_noattention = {'clean': 0.8737, 'scale': 0.63803, 'jitter': 0.41679, 'rotate': 0.77092, 'dropout_global': 0.8016, 'dropout_local': 0.81263, 'add_global': 0.57106, 'add_local': 0.81013}
PointNext_wadaptpoint_masknofp = {'clean': 0.8536, 'scale': 0.61693, 'jitter': 0.31839, 'rotate': 0.78855, 'dropout_global': 0.71811, 'dropout_local': 0.79396, 'add_global': 0.55024, 'add_local': 0.78709}
PointNext_wadaptpoint_anchorneighbor1 = {'clean': 0.8775, 'scale': 0.64942, 'jitter': 0.4211, 'rotate': 0.77793, 'dropout_global': 0.80361, 'dropout_local': 0.80645, 'add_global': 0.58425, 'add_local': 0.80867}
PointNext_wadaptpoint_anchorneighbor2 = {'clean': 0.8782, 'scale': 0.64004, 'jitter': 0.38806, 'rotate': 0.81305, 'dropout_global': 0.77772, 'dropout_local': 0.80354, 'add_global': 0.58584, 'add_local': 0.81825}
PointNext_wadaptpoint_anchorneighbor4 = {'clean': 0.8848, 'scale': 0.65094, 'jitter': 0.43241, 'rotate': 0.77183, 'dropout_global': 0.80132, 'dropout_local': 0.80895, 'add_global': 0.55281, 'add_local': 0.80854}
PointNext_wadaptpoint_anchorneighbor8 = {'clean': 0.8893, 'scale': 0.65919, 'jitter': 0.44601, 'rotate': 0.78806, 'dropout_global': 0.82026, 'dropout_local': 0.8238, 'add_global': 0.60819, 'add_local': 0.81957}
PointNext_wadaptpoint_anchorneighbor16 = {'clean': 0.8855, 'scale': 0.63227, 'jitter': 0.42963, 'rotate': 0.78563, 'dropout_global': 0.82221, 'dropout_local': 0.81721, 'add_global': 0.54927, 'add_local': 0.80937}


def CalculateCE(model):
    perf_all = {'CE': [], 'RCE': []}
    for corruption_type in corruptions:
        perf_corruption = {'CE': [], 'RCE': []}
        CE = (1 - model[corruption_type]) / (1 - DGCNN[corruption_type])
        RCE = (model['clean'] - model[corruption_type]) / (DGCNN['clean'] - DGCNN[corruption_type])
        perf_corruption['CE'].append(round(CE, 3))
        perf_corruption['RCE'].append(round(RCE, 3))
        perf_corruption['corruption'] = corruption_type
        perf_corruption['level'] = 'Overall'
        print(f'perf_corruption:{perf_corruption}')
        perf_all['CE'].append(CE)
        perf_all['RCE'].append(RCE)
    print(f'perf_all before average:{perf_all}')
    for k in perf_all:
        perf_all[k] = sum(perf_all[k]) / len(perf_all[k])
        perf_all[k] = round(perf_all[k], 3)
    perf_all['mCE'] = perf_all.pop('CE')
    perf_all['RmCE'] = perf_all.pop('RCE')
    # print(f'perf_corruption:{perf_corruption}')
    print(f'perf_all:{perf_all}')
    pass

def transdata2dict(string):
    data_list = re.findall(r"\d+\.?\d*", string)
    for i in range(len(data_list)):
        data_list[i] = round((float(data_list[i]) / 100), 5)
    data_dict={
    'clean': data_list[0],
    'scale': data_list[2],
    'jitter': data_list[3],
    'rotate': data_list[4],
    'dropout_global': data_list[5],
    'dropout_local': data_list[6],
    'add_global': data_list[7],
    'add_local': data_list[8]
    }
    print(data_dict)
    return data_dict



def main():
    CalculateCE(PointNext_wadaptpoint_anchorneighbor16)
    # data_dict = transdata2dict('88.55	69.223	63.227	42.963	78.563	82.221	81.721	54.927	80.937')


if __name__ == '__main__':
    print('==>begining...')
    main()
    print('==>ending...')
