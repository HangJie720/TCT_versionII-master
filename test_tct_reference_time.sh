#!/bin/bash

python test_tct_reference_time.py --frozen_graph=/container_dir/TCT_versionII-master/classification/checkpoint_dir/classification.pb \
  --filter_model_path=/container_dir/TCT_versionII-master/deliver_code/models_version/svm_model.pkl \
  --feature_extractor_model_path=/container_dir/TCT_versionII-master/deliver_code/models_version/feature_extractor.h5 \
  --focus_extractor_model_path=/container_dir/TCT_versionII-master/deliver_code/models_version/focus_extractor/ \
  --image_file=2017-07-24-15_32_24.tif \
  --native \
  --output_dir=/container_dir/TCT_versionII-master/classification/checkpoint_dir \
  --input_node=Input \
  --output_node="dense_p3/dense_p3_sigmoid","dense_p4/dense_p4_sigmoid","dense_p5/dense_p5_sigmoid","dense_p6/dense_p6_sigmoid","dense_p7/dense_p7_sigmoid" \
  --batch_size=128