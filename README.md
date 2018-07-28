TCT项目版本二<br>

model_files:
 - 模型构建过程
deliver_code:整片预测模块，包含过滤模块一+离线预测模块<br>
  1.predict_script.py:  对切割后的小图进行离线预测<br>
  2.predict_scirpt_version_two.py: 对整张tif大图进行预测，包含切割+预测<br>
  3.unnormal_patch.py: 过滤模块，包含空白区域的过滤以及焦点区域的提取<br>
  4.xml_generator_v2.py: 预测出的区域转换为bbox信息，生成xml文件用于展示，当前xml的相关信息基于ObjectiveViewer<br>

demo2:
- 第二版demo推理过程，主要逻辑为：空白区域过滤->retinanet->multiscale-resnet101

环境要求：openslide, openslide-python==1.1.1, h5py==2.6.0, keras==2.1.2, tensorflow==1.4.1<br>
  1. yum install openslide<br>
  2. pip install openslide-python==1.1.1<br>
  3. pip install h5py==2.6.0<br>
  4. pip install keras==2.1.2<br>
  5. pip install tensorflow-gpu==1.4.1<br>

数据存放地址：<br>
10.0.2.35: /tct/TCT<br>
