import tensorflow as tf
import numpy as np
import cv2

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

#一个数据作为batch输入
def input_process(img):
    img = size_process(img, 224)
    data = img[:, :, ::-1]
    data = np.asarray(data, dtype=np.float32)
    data = data_process(data)
    data = np.expand_dims(data, axis=0)
    return data

def Main(checkpoint_dir, graph_name, img_data):
    img_data = input_process(img_data)

    #new_saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
    new_saver = tf.train.import_meta_graph(checkpoint_dir + "/" + graph_name)

    with tf.Session() as sess:
        graph = tf.get_default_graph()

        # tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
        # for tensor_name in tensor_name_list:
        #     print(tensor_name, '\n')

        new_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        input_tensor = graph.get_tensor_by_name("Input:0")

        # img = cv2.imread('F:/TCT/classification_data/testData/HSIL/2017-07-24-16_51_57_0.png')

        feed_dict = {input_tensor: img_data}
        outputs = [graph.get_tensor_by_name("dense_p3/dense_p3_sigmoid:0"),
                   graph.get_tensor_by_name("dense_p4/dense_p4_sigmoid:0"),
                   graph.get_tensor_by_name("dense_p5/dense_p5_sigmoid:0"),
                   graph.get_tensor_by_name("dense_p6/dense_p6_sigmoid:0"),
                   graph.get_tensor_by_name("dense_p7/dense_p7_sigmoid:0")]
        outs = sess.run(outputs, feed_dict)

        output = np.array(outs).reshape(5,)
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

if __name__=="__main__":
    checkpoint = "./checkpoint_dir"
    graph_name = "MyModel.meta"

    test_img_path = "F:/TCT/classification_data/testData/HSIL/2017-07-24-16_51_57_0.png"
    img_data = cv2.imread(test_img_path)
    result = Main(checkpoint, graph_name, img_data)
    print(result)