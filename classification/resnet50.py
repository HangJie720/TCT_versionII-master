#coding:utf-8
import tensorflow as tf
import cv2
import numpy as np
import pickle

class ResNet50_fpn:
    def __init__(self,
                 sess,
                 inputs,
                 blocks,
                 weights,
                 *args,
                 **kwargs):
        '''
        
        :param inputs: 输入数据shape, ex(None,224, 224, 3)
        :param blocks: 
        :param include_top: 
        :param classes: 
        :param args: 
        :param kwargs: 
        '''
        self.sess=sess
        self.inputs = inputs
        self.blocks = blocks
        self.weights = weights
        self.var_list = []
        self.numerical_names = None
        self.input_tensor = self.Input()
        self.outputs = self.resnet50_build()#获取多个feature map输出
        #self.outputs是list类型，维度是4   表示c2, c3, c4, c5
        self.feature_maps = self.fpn()#获取多尺度feature map
        self.multi_prediction = self.multi_classification()#获取多维度输出

    def Input(self):
        '''
        根据inputs构成input_tensor
        :return: 
        '''
        input_tensor = tf.placeholder(dtype=tf.float32,shape=self.inputs,name='Input')
        return input_tensor

    def multi_classification(self):
        [p3, p4, p5, p6, p7] = self.feature_maps
        p3_output = self.cls_block(p3, 'p3')
        p4_output = self.cls_block(p4, 'p4')
        p5_output = self.cls_block(p5, 'p5')
        p6_output = self.cls_block(p6, 'p6')
        p7_output = self.cls_block(p7, 'p7')
        print(p3_output)
        return [p3_output, p4_output, p5_output, p6_output, p7_output]

    def cls_block(self, x, feature_index):
        '''
        构建最终分类层
        :param x: 
        :param feature_index: 
        :return: 
        '''
        x = self.GlobalAveragePooling2D(x, name='global_avg_pool_' + feature_index)
        prediction = self.Dense(x, name='dense_'+feature_index)
        return prediction

    def Dense(self,x, use_bias = True, name='Dense'):
        with tf.name_scope(name):
            weights_list = self.get_conv_dense_filter(name, use_bias)#获取dense层参数
            weights = weights_list[0]#w
            y = tf.matmul(x, weights)
            if use_bias:
                bias = weights_list[1]
                y = tf.nn.bias_add(y, bias, name=name+'_add_bias')
            y = tf.nn.sigmoid(y, name=name+'_sigmoid')
            self.var_list=self.var_list+[weights, bias]
            return y

    # def mean(self, x, axis=None, keepdims=False):
    #     if x.dtype.base_type == tf.bool:
    #         x = tf.cast(x, tf.float32)
    #     return tf.reduce_mean(x, axis, keepdims)

    def GlobalAveragePooling2D(self, x, name='global_avg_pool'):
        '''
        全局平局池化
        :param x: feature map
        :param name: 
        :return: 
        '''
        with tf.name_scope(name):
            return tf.reduce_mean(x, axis=[1,2], keep_dims=False)

    def fpn(self):
        #基于输出的多个feature map构建不同的p层
        [_, c3, c4, c5] = self.outputs
        p5 = self.Conv2D(c5, strides=(1,1), padding="SAME", name="P5")
        p5_upsampled = self.UpsampleLike(source=p5, targets=c4, name='P5_upsampled')

        #add p5 elementwise to c4
        p4 = self.Conv2D(c4, strides=(1,1), padding='SAME', name='C4_reduced')
        p4 = self.Add(inputs=[p5_upsampled, p4], name='P4_merged')
        p4 = self.Conv2D(p4, strides=(1,1), padding='SAME', name='P4')
        p4_upsampled = self.UpsampleLike(source=p4, targets=c3, name='p4_upsampled')

        #add p4 elementwise to c3
        p3 = self.Conv2D(c3, strides=(1,1), padding='SAME', name='C3_reduced')
        p3 = self.Add(inputs=[p4_upsampled, p3], name='P3_merged')
        p3 = self.Conv2D(p3, strides=(1,1), padding='SAME', name='P3')

        #p6 is obtained via a 3*3 stride-2 conv on c5
        p6 = self.Conv2D(c5, strides=(2,2), padding='SAME', name='P6')

        #p7 is computer by applying RELU followed by a 3*3 stride-2 conv on p6
        p7 = self.Activation(p6, activation_method='relu', name='C6_relu')
        p7 = self.Conv2D(p7, strides=(2,2), padding='SAME', name='P7')

        return [p3, p4, p5, p6, p7]#返回feature map

    def UpsampleLike(self, source, targets, name):
        with tf.name_scope(name):
            target_shape = targets.get_shape().as_list()
            return tf.image.resize_images(source, (target_shape[1], target_shape[2]))

    def resnet50_build(self):

        outputs = []

        if self.numerical_names is None:
            self.numerical_names = [True] * len(self.blocks)

        x = self.ZeroPadding2D(self.input_tensor, padding=3, name='padding_conv1')
        x = self.Conv2D(x, strides=(2,2), padding='VALID', use_bias=False, name='conv1')
        x = self.BatchNormalization(x, name='bn_conv1')
        x = self.Activation(x, name='conv1_relu')
        x = self.MaxPooling2D(x, (3,3), strides=(2,2), padding='SAME', name='pool1')

        for stage_id, iterations in enumerate(self.blocks):#遍历blocks
            for block_id in range(iterations):
                numerical_name = (block_id > 0 and self.numerical_names[stage_id])
                x = self.bottleneck_2d(x, stage=stage_id, block=block_id, numerical_name=numerical_name)

            outputs.append(x)#block输出的feature map用于最后集成
        return outputs

    def bottleneck_2d(self, x, stage=0, block=0, numerical_name=False, stride=None):
        if stride is None:
            if block !=0 or stage ==0:
                stride = 1
            else:
                stride = 2

        # if block > 0 and numerical_name:
        #     block_char = "b{}".format(block)
        # else:
        #     block_char = chr(ord('a') + block)
        block_char = chr(ord('a') + block)

        stage_char = str(stage+2)

        def f(x):
            y = self.Conv2D(x, strides=(stride,stride), padding="VALID",use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char))
            y = self.BatchNormalization(y, epsilon=1e-5, name="bn{}{}_branch2a".format(stage_char, block_char))
            y = self.Activation(y, activation_method='relu', name="res{}{}_branch2a_relu".format(stage_char, block_char))

            y = self.ZeroPadding2D(y, padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))
            y = self.Conv2D(y, padding="VALID", use_bias=False, name="res{}{}_branch2b".format(stage_char, block_char))
            y = self.BatchNormalization(y, epsilon=1e-5, name="bn{}{}_branch2b".format(stage_char, block_char))
            y = self.Activation(y, activation_method='relu', name="res{}{}_branch2b_relu".format(stage_char, block_char))

            y = self.Conv2D(y, padding="VALID", use_bias=False, name="res{}{}_branch2c".format(stage_char, block_char))
            y = self.BatchNormalization(y, epsilon=1e-5, name="bn{}{}_branch2c".format(stage_char, block_char))

            if block==0:
                shortcut = self.Conv2D(x, strides=(stride, stride), padding="VALID", use_bias=False, name="res{}{}_branch1".format(stage_char, block_char))
                shortcut = self.BatchNormalization(shortcut, epsilon=1e-5, name="bn{}{}_branch1".format(stage_char, block_char))
            else:
                shortcut = x
            y = self.Add([y, shortcut], name="res{}{}".format(stage_char, block_char))
            y = self.Activation(y, activation_method='relu', name="res{}{}_relu".format(stage_char, block_char))

            return y

        return f(x)

    def Add(self, inputs, name='add'):
        return tf.add_n(inputs, name=name)

    def ZeroPadding2D(self, x, padding=3, name = 'padding_conv1'):
        '''
        :param x: x是四维张量
        :param padding: 
        :param name: 
        :return: 
        '''
        paddings = [[0,0], [padding, padding], [padding, padding], [0,0]]
        return tf.pad(x, paddings, mode='CONSTANT', name=name)

    def Conv2D(self, x, strides=(1,1), padding='SAME', use_bias = True, name = 'conv2d'):
        with tf.name_scope(name):
            weights_list = self.get_conv_dense_filter(name, use_bias)
            weights = weights_list[0]
            if use_bias:
                bias = weights_list[1]
                conv = tf.nn.conv2d(input=x, filter=weights, strides=[1, strides[0], strides[1], 1], padding=padding, name = name+'_conv')
                conv = tf.nn.bias_add(conv, bias, name=name)
                self.var_list=self.var_list+[weights, bias]
            else:
                conv = tf.nn.conv2d(input=x, filter=weights, strides=[1, strides[0], strides[1], 1], padding=padding, name = name)
                self.var_list=self.var_list+[weights]
            return conv

    def BatchNormalization(self, x, epsilon =0.001, name='bn'):
        with tf.name_scope(name):
            scale, offset, mean, variance = self.get_bn_param(name)
            self.var_list=self.var_list+[scale, offset, mean, variance]
            return tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon=epsilon, name=name)

    def Activation(self, x, activation_method='relu', name='relu'):
        if activation_method == 'relu':#relu激活
            return tf.nn.relu(x, name=name)

    def MaxPooling2D(self, x, kernel_size, strides=(1,1), padding='SAME', name='pool'):
        return tf.nn.max_pool(value=x, ksize=[1, kernel_size[0], kernel_size[1], 1], strides=[1, strides[0], strides[1], 1], padding=padding, name=name)

    def get_conv_dense_filter(self, name, use_bias):
        filters = self.weights[name]
        weights_list = []
        weights = filters[0]
        weights = tf.Variable(tf.constant(value=weights, dtype=tf.float32), trainable=True, dtype=tf.float32, name="weights")
        weights_list.append(weights)
        if use_bias:
            bias = filters[1]
            bias = tf.Variable(tf.constant(value=bias, dtype=tf.float32), trainable=True, dtype=tf.float32, name="bias")
            weights_list.append(bias)
        return weights_list

    def get_bn_param(self, name):
        weights = self.weights[name]
        gama = tf.Variable(tf.constant(value=weights[0], dtype=tf.float32), trainable=True, dtype=tf.float32, name="gama")
        beta = tf.Variable(tf.constant(value=weights[1], dtype=tf.float32), trainable=True, dtype=tf.float32, name="beta")
        moving_mean = tf.Variable(tf.constant(value=weights[2], dtype=tf.float32), trainable=True, dtype=tf.float32, name="moving_mean")
        moving_variance = tf.Variable(tf.constant(value=weights[3], dtype=tf.float32), trainable=True, dtype=tf.float32, name="moving_variance")
        return gama, beta, moving_mean, moving_variance

    #预测接口
    def predict(self, batch_data):
        feed_dict = {self.input_tensor: batch_data}
        output = self.sess.run(fetches = self.multi_prediction, feed_dict=feed_dict)
        print(self.multi_prediction)
        output = np.array(output).reshape(5,)
        label = output.copy()
        label[label>0.5]=1
        label[label<1]=0

        num_one = (label == 1).sum()
        num_zero = (label == 0).sum()

        prob_one = output[label == 1]
        prob_zero = output[label == 0]

        if num_one > num_zero:  # 获取投票多的分类器均值
            prob = np.mean(prob_one)
        else:
            prob = np.mean(prob_zero)

        return prob

    #对获取的结果进行集成

def data_process(x):
    x /= 127.5
    x -= 1.
    return x

def size_process(img, max_size=224):
    w, h, c = img.shape
    size = max(w, h)
    f = max_size / float(size)
    img = cv2.resize(img, dsize=None, fx=f, fy=f)  # 同比例缩放
    data = np.zeros([max_size, max_size, 3])
    w_new, h_new, c = img.shape
    s_x = int((max_size - w_new) / 2)
    s_y = int((max_size - h_new) / 2)
    data[s_x:s_x + w_new, s_y:s_y + h_new, :] = img
    return data

if __name__== "__main__":



    labels = {'NILM': 0, 'HSIL': 1}
    img = cv2.imread('./2017-07-24-16_51_57_0.png')
    img = size_process(img, 224)
    data = img[:, :, ::-1]
    data = np.asarray(data, dtype=np.float32)
    data = data_process(data)
    data = np.expand_dims(data, axis=0)

    fr = open('./checkpoint_dir/weights.pkl', 'rb')
    weights = pickle.load(fr)
    with tf.Session() as sess:
        model = ResNet50_fpn(sess=sess, inputs=(None, 224, 224, 3), blocks=[3, 4, 6, 3], weights=weights)
        init = tf.global_variables_initializer()
        sess.run(init)
        print(model.var_list)
        saver = tf.train.Saver(var_list=model.var_list)
        save_path = saver.save(sess, "./checkpoint_dir/MyModel")
        output = model.predict(data)
        print(output)