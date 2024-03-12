#!/bin/bash

#SBATCH --job-name=loopNLayers                          # nom du job
#SBATCH --partition=tau          	                    # GPU type
#SBATCH --time=72:00:00            	                    # temps maximum d'execution demandé
#SBATCH --gres=gpu:1                	                # nombre de GPU a reserver
#SBATCH --output=out%j.out          	                # nom du fichier de sortie
#SBATCH --error=out%j.out           	                # nom du fichier d'erreur (ici commun avec la sortie)

for i in 1 2 3 4
do
    backup_dir=$(date +'%d_%m_%Y_%T')
    srun python3 main_train.py \
    --path_dataset      ../dataset/d2500/ \
    --path_results      results/${backup_dir} \
    --comment           "Number of layers : $i" \
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
    --n_layers          $i \
    --fw_tol            1.e-5 \
    --fw_thres          400 \
    --bw_tol            1.e-8 \
    --bw_thres          400 \

done 

exit 0
