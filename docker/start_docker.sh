#!/bin/bash

image_dir = $1;
metadata_file = $2;

docker run --gpus all -it --shm-size=8gb -v ${image_dir}:/images \
-v $HOME/.torch/fvcore_cache:/tmp/ -v {metadata_file}:/images/metadata.json \
--name=detectron2 py-bottom-up-attention;