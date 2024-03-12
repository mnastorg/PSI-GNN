#!/bin/bash

source activate dss_env

python3 test.py \
--path_dataset      ../../dataset/dConstant/ \
--path_results      ../results/dss_results/ckpt/best_model.pt \
--batch_size        50

exit 0