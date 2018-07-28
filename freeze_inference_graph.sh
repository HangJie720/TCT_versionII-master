#!/bin/bash

python freeze_inference_graph.py \
  --graphdef_file=/container_dir/TCT_versionII-master/classification/checkpoint_dir/Resnet50.pb  \
  --checkpoint_dir=/container_dir/TCT_versionII-master/classification/checkpoint_dir \
  --frozen_graph=/container_dir/TCT_versionII-master/classification/checkpoint_dir/frozen_Resnet50.pb \
  --output_node_name="dense_p3/dense_p3_sigmoid","dense_p4/dense_p4_sigmoid","dense_p5/dense_p5_sigmoid","dense_p6/dense_p6_sigmoid","dense_p7/dense_p7_sigmoid"