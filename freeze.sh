#!/bin/bash

python freeze_graph.py \
  --input_graph=/container_dir/TCT_versionII-master/classification/checkpoint_dir/Resnet50.pb \
  --input_checkpoint=/container_dir/TCT_versionII-master/classification/model.ckpt \
  --input_binary=true --output_graph=/container_dir/TCT_versionII-master/classification/checkpoint_dir/frozen_Resnet50.pb \
  --output_node_names=dense_p3/dense_p3_sigmoid