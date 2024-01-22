CUDA_VISIBLE_DEVICES=1 python examples/classification/main.py --cfg cfgs/scanobjectnn/dcgnn_1_0.yaml
CUDA_VISIBLE_DEVICES=1 python examples/classification/main.py --cfg cfgs/scanobjectnn/dcgnn_1_1.yaml

CUDA_VISIBLE_DEVICES=1 python examples/classification/main.py --cfg cfgs/scanobjectnn/fpnet_ballquery_testingtime_16layers_dropchannel2_2.yaml
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/scanobjectnn/dcgnn_emb_1_2_2.yaml batch_size=128 num_points=1024 timing=True flops=True