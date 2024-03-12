#!/bin/bash

source activate original_env

python3 generate_data.py \
--path_mesh mesh \
--path_data data \
--n_mesh 10 \
--n_samples 20 \

exit 0