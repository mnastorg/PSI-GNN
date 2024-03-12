#!/bin/bash

source activate original_env

backup_dir=$(date +'%d_%m_%Y_%T')

python3 main.py \
--path_dataset      ../dataset/dConstant/ \
--path_results      results/${backup_dir} \
--comment           "Temp" \
--seed              1234 \
--max_epochs        400 \
--batch_size        50 \
--num_gpus          -1 \
--num_workers       10 \
--min_loss_save     1.e5 \
--gradient_clip     0.1 \
--sup_weight        0.0 \
--lr_deq            0.01 \
--sched_step_deq    0.5 \
--lr_ae             0.05 \
--sched_step_ae     0.5 \
--solver            broyden \
--jac_weight        1.0 \
--latent_dim        10 \
--hidden_dim        10 \
--n_layers          1 \
--fw_tol            1.e-5 \
--fw_thres          400 \
--bw_tol            1.e-8 \
--bw_thres          400 \

exit 0
