# -*- coding: utf-8 -*-
from keras.models import load_model
import keras.backend as K
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
import tensorflow as tf
import os

# Configure Parameters

tf.app.flags.DEFINE_string('graphdef_file','',
                           """Directory exporting grapdef""")
tf.app.flags.DEFINE_string('saved_ckpt','',
                           """Directory saving checkpoint""")
tf.app.flags.DEFINE_string('checkpoint_dir', '',
                           """Directory containing trained checkpoints""")
tf.app.flags.DEFINE_string('keras_model_dir', '',
                           """Directory containing trained keras model h5py.file""")
tf.app.flags.DEFINE_string('output_node_name', '',
                           """The name of the output nodes""")
tf.app.flags.DEFINE_string('frozen_graph', '',
                           """Directory saving frozen graph """)

FLAGS = tf.app.flags.FLAGS

def freeze_keras_model(keras_model_path):
    # Load keras model
    model = load_model(keras_model_path)
    # Observe the input_node_name and output_node_name which are used for creating inference graph with tensorrt
    print(model.inputs)
    print(model.outputs)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    with K.get_session() as sess:
        # saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
        checkpoint_path = saver.save(sess, FLAGS.saved_ckpt,global_step=0,latest_filename='checkpoint_state')
        graph_io.write_graph(sess.graph,'.',FLAGS.graphdef_file)
        print(checkpoint_path)
        freeze_graph.freeze_graph(FLAGS.graphdef_file, '',
                                  False, checkpoint_path, FLAGS.output_node_name,
                                  "save/restore_all", "save/Const:0",
                                  FLAGS.frozen_graph, False, "")

def freeze_tf_model(tf_model_path, graph_name):
    # Create saver to restore from checkpoint
    new_saver = tf.train.import_meta_graph(tf_model_path + "/" + graph_name)
    with tf.Session() as sess:
        graph = tf.get_default_graph()

        tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
        for tensor_name in tensor_name_list:
            print(tensor_name, '\n')

        new_saver.restore(sess, tf.train.latest_checkpoint(tf_model_path))

        graph_io.write_graph(graph, '.', FLAGS.graphdef_file)
        freeze_graph.freeze_graph(FLAGS.graphdef_file, '',
                                  False, tf_model_path+"/"+"MyModel", FLAGS.output_node_name,
                                  "save/restore_all", "save/Const:0",
                                  FLAGS.frozen_graph, False, "")

def main(_):
    if not FLAGS.graphdef_file:
        raise ValueError('You must supply the path to save to with --graphdef_file')
    tf.logging.set_verbosity(tf.logging.INFO)
    # freeze_keras_model(FLAGS.keras_model_dir)
    graph_name = "MyModel.meta"
    freeze_tf_model(FLAGS.checkpoint_dir, graph_name)

if __name__ == '__main__':
    tf.app.run()
