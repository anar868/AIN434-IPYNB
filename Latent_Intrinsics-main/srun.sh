#!/bin/bash

#export NCCL_DEBUG=info
#export NCCL_P2P_DISABLE=1
#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_DISTRIBUTED_DEBUG=DETAIL



while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done

echo "$port"
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gn  oded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
data_path=/data/path

python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=${port} main_cls.py \
--data_path ${data_path} \
--reg_weight 1e-4 \
--intrinsics_loss_weight 1e-1 \
--epochs 120 \
--batch_size 64 \
--learning_rate 2e-4 \
--weight_decay 1e-2 \
--resume
