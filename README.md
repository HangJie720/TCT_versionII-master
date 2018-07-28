TCT Program<br>

Environment：
Hardwafre：
CPU：Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz
MEM：256GB
GPU：Tesla P100
NETWORK：56Gbps InfiniBand

Software：
OS：CentOS 7.2
Kernel：3.10.0-327.el7.x86_64
Tensorflow：tensorflow-1.8.0-cp27-cp27mu-linux_x86_64.whl
Python：2.7.5                                                                        TensorRT: TensorRT-3.0.4.Ubuntu-14.04.5.x86_64.cuda-9.0.cudnn7.0.tar.gz                                                                     Cuda：v9.0.176
Cudnn：v7.1.4.18
Nvidia Driver：v384.81
Ceph：v12.2.1

required libraries:
openslide, openslide-python==1.1.1, h5py==2.6.0, keras==2.1.2, tensorflow==1.4.1<br>
  1. yum install openslide<br>
  2. pip install openslide-python==1.1.1<br>
  3. pip install h5py==2.6.0<br>
  4. pip install keras==2.1.2<br>
  5. pip install tensorflow-gpu==1.4.1<br>
