#!/bin/bash

#SBATCH --job-name=DSS                         
#SBATCH --constraint=tesla          	                    
#SBATCH --time=96:00:00            	                    
#SBATCH --gres=gpu:2                	                
#SBATCH --output=out%j.out          	                
#SBATCH --error=out%j.out

source activate dss_env

backup_dir=$(date +'%d_%m_%Y_%T')

python3 main.py \
--path_dataset      ../dataset/dConstant/ \
--path_results      results/${backup_dir} \
--comment           "DSGPS - 30 Iterations - Gamma = 1.0 - No LayerNorm" \
--seed              1234 \
--max_epochs        400 \
--batch_size        50 \
--num_gpus          -1 \
--num_workers       10 \
--min_loss_save     1.e5 \
--gradient_clip     0.01 \
--lr                0.001 \
--latent_dim        10 \
--k                 30 \
--alpha             1.e-3 \
--gamma             1.0 \

exit 0
