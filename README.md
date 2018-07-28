TCT Program<br>

### Environment：
1. Hardware：
2. CPU：Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz
3. MEM：256GB
4. GPU：Tesla P100
5. NETWORK：56Gbps InfiniBand

### Software：
1. OS：CentOS 7.2
2. Kernel：3.10.0-327.el7.x86_64
3. Tensorflow：tensorflow-1.8.0-cp27-cp27mu-linux_x86_64.whl
4. Python：2.7.5                                                                       
5. TensorRT: TensorRT-3.0.4.Ubuntu-5. 14.04.5.x86_64.cuda-9.0.cudnn7.0.tar.gz                                                 6. Cuda：v9.0.176
7. Cudnn：v7.1.4.18
8. Nvidia Driver：v384.81
9. Ceph：v12.2.1

### Additional libraries:
  1. yum install openslide<br>
  2. pip install openslide-python==1.1.1<br>
  3. pip install h5py==2.6.0<br>
  4. pip install keras==2.1.2<br>
  
## Freezing the exported Graph
If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:

```shell
$ bash freeze_inference_graph.sh
```

### Optimize the Resnet-50 with TensorRT

[TensorRT](https://developer.nvidia.com/tensorrt) is NVIDIA's inference
optimizer for deep learning. Briefly, TensorRT rewrites parts of the
execution graph to allow for faster prediction times.

```shell
$ bash tensorrt_optimize_graph.sh
```
