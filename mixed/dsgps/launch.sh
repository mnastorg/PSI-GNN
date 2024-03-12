#!/bin/bash

source activate original_env

backup_dir=$(date +'%d_%m_%Y_%T')

python3 main.py \
--path_dataset      ../dataset/dTest/ \
--path_results      results/${backup_dir} \
--comment           "Deep Statistical Solver" \
--seed              1234 \
--max_epochs        400 \
--batch_size        50 \
--num_gpus          -1 \
--num_workers       10 \
--min_loss_save     1.e5 \
--gradient_clip     0.01 \
--lr                0.01 \
--latent_dim        10 \
--k                 30 \
--alpha             1.e-3 \
--gamma             0.9 \

exit 0
