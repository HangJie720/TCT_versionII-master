# -*- coding: utf-8 -*-
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Methods for running the Official Models with TensorRT.

Please note that all of these methods are in development, and subject to
rapid change.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import imghdr
import json
import os
import sys
import time
import functools
import openslide
import numpy as np
import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import reader
import tensorflow.contrib.tensorrt as trt

import preprocess_tct_image
from sklearn.externals import joblib
from keras.models import load_model
from unnormal_patch import invalid_based_filter_one, focus_based_autoencoder
from preprocess_tct_image import data_process
import cv2

from PIL import Image
_GPU_MEM_FRACTION = 0.25
_WARMUP_NUM_LOOPS = 5
_LOG_FILE = "log.txt"
_LABELS_FILE = "labellist.json"
_GRAPH_FILE = "frozen_graph.pb"

def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        retargs = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        return retargs
    return newfunc

def image_to_np_CHW(image):
    return np.asarray(
        image.resize(
            (224, 224),
            Image.ANTIALIAS
        )).transpose([2, 0, 1]).astype(np.float32)

def data_process_1(x):
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

@timeit
def preprocess(tiff_path,
               filter_model_path,
               feature_extractor_model_path,
               focus_extractor_model_path,
               step=300, size=300, tar_size = 128):

    # blank area filter
    filter_model = joblib.load(filter_model_path)

    # focus area extract
    feature_extractor = load_model(feature_extractor_model_path)  # focus area feature extractor

    focus_extractor_names = os.listdir(focus_extractor_model_path)
    focus_extractor_models = []
    for focus_extractor_name in focus_extractor_names:
        focus_extractor_models.append(joblib.load(focus_extractor_model_path + focus_extractor_name))  # 焦点区域提取模型组合'''

    slide = openslide.OpenSlide(tiff_path)
    height, width = slide.dimensions
    print("height: {},width: {}".format(height, width))
    images = []
    for i in range(0, width, step):
        for j in range(0, height, step):

            sub_img = slide.read_region((i, j), 0, (size, size))
            sub_img_f = sub_img.convert("L")# graysacle image are used for filtering and extracting focus area
            sub_img_f = sub_img_f.resize(size=(tar_size, tar_size))# process target size
            sub_img_p = sub_img.convert("RGB")# RGB image are used for predicting

            # convert to numpy matrix
            sub_img_p = np.array(sub_img_p)
            sub_img_f = np.array(sub_img_f)

            # data process at effective area
            if not invalid_based_filter_one(filter_model, sub_img_f):
                if focus_based_autoencoder(feature_extractor, focus_extractor_models, sub_img_f):

                    sub_img_p = data_process(sub_img_p)
                    images.append(sub_img_p)
    np.save("images.npy",images)
    images = images[:1]
    images_arrays = np.resize(images,[len(images),244,224,3])

    return images_arrays


def batch_from_image(image_file,
                     filter_model_path,
                     feature_extractor_model_path,
                     focus_extractor_model_path,
                     batch_size):
  """Produce a batch of data from the passed image file.

  Args:
    file_name: string, path to file containing a JPEG image
    filter_model_path, string, path to file containing filter model
    feature_extractor_model_path, string, path to file containing feature extractor model: autoencoder
    focus_extractor_model_path, string, path to file containing multiple focus extractor model
    batch_size: int, the size of the desired batch of data

  Returns:
    Float array representing copies of the image with shape
      [batch_size, output_height, output_width, num_channels]
  """
  # image_array = preprocess(image_name, filter_model_path, feature_extractor_model_path, focus_extractor_model_path)
  # image = cv2.imread(image_name)
  # data = input_process(image)
  # print(data.shape)
  tstart = time.time()
  data = preprocess(image_file,filter_model_path,feature_extractor_model_path,focus_extractor_model_path)
  tsend = time.time()
  print("elapsed time:", tsend-tstart)

  tiled_array = np.tile(data, [batch_size, 1, 1, 1])
  print(tiled_array.shape)
  return data


def get_iterator(data):
  """Wrap numpy data in a dataset."""
  dataset = tf.data.Dataset.from_tensors(data).repeat()
  return dataset.make_one_shot_iterator()


def write_graph_to_file(graph_name, graph_def, output_dir):
  """Write Frozen Graph file to disk."""
  output_path = os.path.join(output_dir, graph_name)
  with tf.gfile.GFile(output_path, "wb") as f:
    f.write(graph_def.SerializeToString())


def get_frozen_graph(graph_file):
  """Read Frozen Graph file from disk."""
  with tf.gfile.FastGFile(graph_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def get_tftrt_name(graph_name, precision_string):
  return "tftrt_{}_{}".format(precision_string.lower(), graph_name)


def get_trt_graph(graph_name, graph_def, precision_mode, output_dir,
                  output_node, batch_size=128, workspace_size=2<<10):
  """Create and save inference graph using the TensorRT library.

  Args:
    graph_name: string, name of the graph to be used for saving.
    graph_def: GraphDef, the Frozen Graph to be converted.
    precision_mode: string, the precision that TensorRT should convert into.
      Options- FP32, FP16, INT8.
    output_dir: string, the path to where files should be written.
    output_node: string, the names of the output node that will
      be returned during inference.
    batch_size: int, the number of examples that will be predicted at a time.
    workspace_size: int, size in megabytes that can be used during conversion.

  Returns:
    GraphDef for the TensorRT inference graph.
  """
  trt_graph = trt.create_inference_graph(
      graph_def, output_node, max_batch_size=batch_size,
      max_workspace_size_bytes=workspace_size<<20,
      precision_mode=precision_mode)

  write_graph_to_file(graph_name, trt_graph, output_dir)

  return trt_graph


def get_trt_graph_from_calib(graph_name, calib_graph_def, output_dir):
  """Convert a TensorRT graph used for calibration to an inference graph."""
  trt_graph = trt.calib_graph_to_infer_graph(calib_graph_def)
  write_graph_to_file(graph_name, trt_graph, output_dir)
  return trt_graph


################################################################################
# Run the graph in various precision modes.
################################################################################
def get_gpu_config():
  """Share GPU memory between image preprocessing and inference."""
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=_GPU_MEM_FRACTION)
  return tf.ConfigProto(gpu_options=gpu_options)


def time_graph(graph_def, data, input_node, output_node, num_loops=100):
  """Run and time the inference graph.

  This function sets up the input and outputs for inference, warms up by
  running inference for _WARMUP_NUM_LOOPS, then times inference for num_loops
  loops.

  Args:
    graph_def: GraphDef, the graph to be timed.
    data: ndarray of shape [batch_size, height, width, depth], data to be
      predicted.
    input_node: string, the label of the input node where data will enter the
      graph.
    output_node: string, the names of the output node that will
      be returned during inference.
    num_loops: int, number of batches that should run through for timing.

  Returns:
    A tuple consisting of a list of num_loops inference times, and the
    predictions that were output for the batch.
  """
  tf.logging.info("Starting execution")
  tf.reset_default_graph()
  g = tf.Graph()

  # rand_data = np.random.random_sample([1,224,224,3]).astype(np.float32)
  #
  # with g.as_default():
  #     output_graph_def = tf.GraphDef()
  #     with open('./classification/checkpoint_dir/tftrt_fp32_classification.pb', "rb") as f:
  #         output_graph_def.ParseFromString(f.read())
  #         _ = tf.import_graph_def(output_graph_def,name="")
  #     with tf.Session(graph=g, config=get_gpu_config()) as sess:
  #         tf.global_variables_initializer().run()
  #         input_tensor = sess.graph.get_tensor_by_name("Input:0")
  #         print(input_tensor)
  #         # output_tensor = sess.graph.get_tensor_by_name("class/Sigmoid:0")
  #         output_tensor = [sess.graph.get_tensor_by_name("dense_p3/dense_p3_sigmoid:0"),
  #                          sess.graph.get_tensor_by_name("dense_p4/dense_p4_sigmoid:0"),
  #                          sess.graph.get_tensor_by_name("dense_p5/dense_p5_sigmoid:0"),
  #                          sess.graph.get_tensor_by_name("dense_p6/dense_p6_sigmoid:0"),
  #                          sess.graph.get_tensor_by_name("dense_p7/dense_p7_sigmoid:0")]
  #         print(output_tensor)
  #
  #         outs = sess.run(output_tensor,feed_dict={input_tensor:rand_data})
  #         output = np.array(outs).reshape(5, )
  # return output

  with g.as_default():
    iterator = get_iterator(data)
    return_tensors = tf.import_graph_def(
        graph_def=graph_def,
        input_map={input_node: iterator.get_next()},
        return_elements=[output_node]
    )
    # Unwrap the returned output node. For now, we assume we only
    # want the tensor with index `:0`, which is the 0th element of the
    # `.outputs` list.
    output = return_tensors[0].outputs[0]

  timings = []
  with tf.Session(graph=g, config=get_gpu_config()) as sess:
    tf.logging.info("Starting Warmup cycle")

    for _ in range(_WARMUP_NUM_LOOPS):
      sess.run([output])

    tf.logging.info("Starting timing.")

    for _ in range(num_loops):
      tstart = time.time()
      val = sess.run([output])
      timings.append(time.time() - tstart)

    tf.logging.info("Timing loop done!")

  return timings, val[0]


def log_stats(graph_name, log_buffer, timings, batch_size):
  """Write stats to the passed log_buffer.

  Args:
    graph_name: string, name of the graph to be used for reporting.
    log_buffer: filehandle, log file opened for appending.
    timings: list of floats, times produced for multiple runs that will be
      used for statistic calculation
    batch_size: int, number of examples per batch
  """
  times = np.array(timings)
  steps = len(times)
  speeds = batch_size / times
  time_mean = np.mean(times)
  time_med = np.median(times)
  time_99th = np.percentile(times, 99)
  time_99th_uncertainty = np.abs(np.percentile(times[0::2], 99) -
                                 np.percentile(times[1::2], 99))
  speed_mean = np.mean(speeds)
  speed_med = np.median(speeds)
  speed_uncertainty = np.std(speeds, ddof=1) / np.sqrt(float(steps))
  speed_jitter = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))

  msg = ("\n==========================\n"
         "network: %s,\t batchsize %d, steps %d\n"
         "  fps \tmedian: %.1f, \tmean: %.1f, \tuncertainty: %.1f, \tjitter: %.1f\n"  # pylint: disable=line-too-long
         "  latency \tmedian: %.5f, \tmean: %.5f, \t99th_p: %.5f, \t99th_uncertainty: %.5f\n"  # pylint: disable=line-too-long
        ) % (graph_name, batch_size, steps,
             speed_med, speed_mean, speed_uncertainty, speed_jitter,
             time_med, time_mean, time_99th, time_99th_uncertainty)

  log_buffer.write(msg)


def time_and_log_graph(graph_name, graph_def, data, log_buffer, flags):
  timings, result = time_graph(graph_def, data, flags.input_node, flags.output_node, flags.num_loops)
  log_stats(graph_name, log_buffer, timings, flags.batch_size)

  return result


def run_trt_graph_for_mode(
    graph_name, graph_def, mode, data, log_buffer, flags):
  """Convert, time, and log the graph at `mode` precision using TensorRT."""
  g_name = get_tftrt_name(graph_name, mode)
  graph = get_trt_graph(
      g_name, graph_def, mode, flags.output_dir, flags.output_node.split(','),
      flags.batch_size, flags.workspace_size)
  result = time_and_log_graph(g_name, graph, data, log_buffer, flags)
  return result


################################################################################
# Parse predictions
################################################################################
def get_labels():
  """Get the set of possible labels for classification."""
  with open(_LABELS_FILE, "r") as labels_file:
    labels = json.load(labels_file)

  return labels


def top_predictions(result, threshold):
  """Get the top n predictions given the array of softmax results."""
  # We only care about the first example.
  probabilities = result[0]
  # Get the ids of most probable labels. Reverse order to get greatest first.
  if probabilities > threshold:
      label = 1
  else:
      label = 0
  return label


def get_labels_for_ids(labels, ids):
  """Get the human-readable labels for given ids.

  Args:
    labels: dict, string-ID to label mapping from ImageNet.
    ids: list of ints, IDs to return labels for.
    ids_are_one_indexed: whether to increment passed IDs by 1 to account for
      the background category. See ArgParser `--ids_are_one_indexed`
      for details.

  Returns:
    list of category labels
  """
  return [labels[str(x)] for x in ids]


def print_predictions(results, preds_to_print=1):
  """Given an array of mode, graph_name, predicted_ID, print labels."""
  labels = get_labels()

  print("Predictions:")
  for mode, result in results:
    pred_ids = top_predictions(result, preds_to_print)
    pred_labels = get_labels_for_ids(labels, pred_ids)
    print("Precision: ", mode, pred_ids, pred_labels)


################################################################################
# Run this script
################################################################################
def main(argv):
  parser = TensorRTParser()
  flags = parser.parse_args(args=argv[1:])

  # Load the data.
  if flags.image_file:
      data = batch_from_image(flags.image_file, flags.filter_model_path,
                              flags.feature_extractor_model_path,
                              flags.focus_extractor_model_path,
                              flags.batch_size)

  # Load the graph def
  if flags.frozen_graph:
    frozen_graph_def = get_frozen_graph(flags.frozen_graph)
  else:
    raise ValueError(
        "Either a Frozen Graph file or a SavedModel must be provided.")

  # Get a name for saving TensorRT versions of the graph.
  graph_name = os.path.basename(flags.frozen_graph or _GRAPH_FILE)

  # Write to a single file for all tests, continuing from previous logs.
  log_buffer = open(os.path.join(flags.output_dir, _LOG_FILE), "a")

  # Run inference in all desired modes.
  results = []
  if flags.native:
    mode = "native"
    print("Running {} graph".format(mode))
    g_name = "{}_{}".format(mode, graph_name)
    result = time_and_log_graph(
        g_name, frozen_graph_def, data, log_buffer, flags)

    results.append((mode, result))
    print(results)

  if flags.fp32:
    mode = "FP32"
    print("Running {} graph".format(mode))
    g_name = "{}_{}".format(mode, graph_name)
    fp32_graph = get_frozen_graph(flags.optimized_graph)
    result = time_and_log_graph(
        g_name, fp32_graph, data, log_buffer, flags)
    # result = run_trt_graph_for_mode(
    #     graph_name, frozen_graph_def, mode, data, log_buffer, flags)
    results.append((mode, result))
    print(results)

  if flags.fp16:
    mode = "FP16"
    print("Running {} graph".format(mode))
    result = run_trt_graph_for_mode(
        graph_name, frozen_graph_def, mode, data, log_buffer, flags)
    results.append((mode, result))

  if flags.int8:
    mode = "INT8"
    print("Running {} graph".format(mode))
    save_name = get_tftrt_name(graph_name, "INT8_calib")
    calib_graph = get_trt_graph(
        save_name, frozen_graph_def, mode, flags.output_dir, flags.output_node.split(','),
        flags.batch_size, flags.workspace_size)
    time_graph(calib_graph, data, flags.input_node, flags.output_node,
               num_loops=1)

    g_name = get_tftrt_name(graph_name, mode)
    int8_graph = get_trt_graph_from_calib(g_name, calib_graph, flags.output_dir)
    result = time_and_log_graph(g_name, int8_graph, data, log_buffer, flags)
    results.append((mode, result))
    print(results)
  # Print prediction results to the command line.
  # print_predictions(
  #     results, flags.ids_are_one_indexed, flags.predictions_to_print)


class TensorRTParser(argparse.ArgumentParser):
  """Parser to contain flags for running the TensorRT timers."""

  def __init__(self):
    super(TensorRTParser, self).__init__()

    self.add_argument(
        "--frozen_graph", "-fg", default=None,
        help="[default: %(default)s] The location of a Frozen Graph "
        "protobuf file that will be used for inference. Note that either "
        "savedmodel_dir or frozen_graph should be passed in, and "
        "frozen_graph will take precedence.",
        metavar="<FG>",
    )
    self.add_argument(
        "--optimized_graph", "-og", default=None,
        help="[default: %(default)s] The location of a optimized Graph "
             "protobuf file that will be used for inference. Note that either "
             "savedmodel_dir or frozen_graph should be passed in, and "
             "frozen_graph will take precedence.",
        metavar="<OG>",
    )
    self.add_argument(
        "--filter_model_path", "-fmp", default=None,
        help="[default: %(default)s] The location of a Frozen Graph "
             "protobuf file that will be used for inference. Note that either "
             "savedmodel_dir or frozen_graph should be passed in, and "
             "frozen_graph will take precedence.",
        metavar="<FMP>",
    )
    self.add_argument(
        "--feature_extractor_model_path", "-femp", default=None,
        help="[default: %(default)s] Autoencoder module "
             ".h5py file that will be used for extracting feature of the focus area. ",
        metavar="<FEMP>",
    )
    self.add_argument(
        "--focus_extractor_model_path", "-foemp", default=None,
        help="[default: %(default)s] Extracting the focus area based on the focus area feature ",
        metavar="<FOEMP>",
    )

    self.add_argument(
        "--output_dir", "-od", default="/tmp",
        help="[default: %(default)s] The location where output files will "
        "be saved.",
        metavar="<OD>",
    )

    self.add_argument(
        "--output_node", "-on", default="softmax_tensor",
        help="[default: %(default)s] The names of the graph output node "
        "that should be used when retrieving results. Assumed to be a softmax.",
        metavar="<ON>",
    )

    self.add_argument(
        "--input_node", "-in", default="input_tensor",
        help="[default: %(default)s] The name of the graph input node where "
        "the float image array should be fed for prediction.",
        metavar="<ON>",
    )

    self.add_argument(
        "--batch_size", "-bs", type=int, default=128,
        help="[default: %(default)s] Batch size for inference. If an "
        "image file is passed, it will be copied batch_size times to "
        "imitate a batch.",
        metavar="<BS>"
    )

    self.add_argument(
        "--image_file", "-if", default=None,
        help="[default: %(default)s] The location of a JPEG image that will "
        "be passed in for inference. This will be copied batch_size times to "
        "imitate a batch. If not passed, random data will be used.",
        metavar="<IF>",
    )

    self.add_argument(
        "--native", action="store_true",
        help="[default: %(default)s] If set, benchmark the model "
        "with it's native precision and without TensorRT."
    )

    self.add_argument(
        "--fp32", action="store_true",
        help="[default: %(default)s] If set, benchmark the model with TensorRT "
        "using fp32 precision."
    )

    self.add_argument(
        "--fp16", action="store_true",
        help="[default: %(default)s] If set, benchmark the model with TensorRT "
        "using fp16 precision."
    )

    self.add_argument(
        "--int8", action="store_true",
        help="[default: %(default)s] If set, benchmark the model with TensorRT "
        "using int8 precision."
    )

    self.add_argument(
        "--num_loops", "-nl", type=int, default=3,
        help="[default: %(default)s] Number of inferences to time per "
        "benchmarked model.",
        metavar="<NL>"
    )

    self.add_argument(
        "--workspace_size", "-ws", type=int, default=5<<10,
        help="[default: %(default)s] Workspace size in megabytes.",
        metavar="<WS>"
    )


    self.add_argument(
        "--predictions_to_print", "-pp", type=int, default=5,
        help="[default: %(default)s] Number of predicted labels to predict.",
        metavar="<PP>"
    )


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
