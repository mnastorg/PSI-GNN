#!/bin/bash

source activate dss_env

python3 spec_geo.py \
--path_mesh ../../dataset/special_geo/ \
--saved_mesh saved_mesh/ \
--geometry mesh_2d.py \
--name mesh \
--size_boundary 0.08 \
--path_results ../results/large_iteration_no_gamma/ \
--folder_ckpt ckpt/ \

exit 0

# --path_results ../results/original_with_gamma/ \