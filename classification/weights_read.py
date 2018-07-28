import h5py

weights_path = 'F:/keras_pretrain_weights/ResNet-50-model.keras.h5'

f = h5py.File(weights_path)

for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
    print("  {}".format(layer))
    print("    Attributes:")
    for key, value in g.attrs.items(): # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
        print("      {}: {}".format(key, value))

    print("    Dataset:")
    for name, d in g.items(): # 读取各层储存具体信息的Dataset类

        for aa, bb in d.items():
            print("      {}: {}".format(name, d.value.shape)) # 输出储存在Dataset中的层名称和权重，也可以打印dataset的attrs，但是keras中是空的
            print("      {}: {}".format(name. d.value))