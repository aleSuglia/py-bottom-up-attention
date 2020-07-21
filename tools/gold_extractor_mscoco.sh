#!/bin/bash

# first symlink inside the docker container
python3 symlink_coco_images.py --image_dir /images/ -data_out /images/;

python3 feature_extractor --image_root /images --images_metadata /images/metadata.json --gold_boxes --output_dir /images/gold_fastrcnn;