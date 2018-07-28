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

import math
import argparse
import cv2
import os
import sys
import time
import functools
import openslide
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
from keras.models import load_model
from unnormal_patch import invalid_based_filter_one, focus_based_autoencoder
from preprocess_tct_image import data_process

_GPU_MEM_FRACTION = 0.5
_WARMUP_NUM_LOOPS = 5
_LOG_FILE = "log.txt"


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
        focus_extractor_models.append(joblib.load(focus_extractor_model_path + focus_extractor_name))  # focus area extractors'''

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

                    # bilateral filter
                    sub_img_p = data_process(sub_img_p)
                    images.append(sub_img_p)
    np.save("images.npy",images)


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


def get_gpu_config():
  """Share GPU memory between image preprocessing and inference."""
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=_GPU_MEM_FRACTION)
  return tf.ConfigProto(gpu_options=gpu_options)


def equal(x, y):
    count=0.0
    for i in range(len(x)):
        if x[i] == y[i]:
            count += 1
    return count
def equal_1(x,y):
    count=0.0
    for i in range(len(x)):
        for id in x[i]:
            if id == y[i]:
                count += 1
                break
    return count


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

@timeit
def execute_graph(mode, graph_def, image, input_node, output_node, flags):

  # Run the inference graph.
  tf.logging.info("Starting execution")
  tf.reset_default_graph()
  g = tf.Graph()
  # output_node = ','.join(output_node)
  print(output_node)
  with g.as_default():
      inp,out0,out1,out2,out3,out4 = tf.import_graph_def(
          graph_def=graph_def,
          return_elements=[input_node,output_node[0],output_node[1],output_node[2],output_node[3],output_node[4]]
      )
      inp = inp.outputs[0]
      out0 = out0.outputs[0]
      out1 = out1.outputs[0]
      out2 = out2.outputs[0]
      out3 = out3.outputs[0]
      out4 = out4.outputs[0]

  image_list = []
  label_list = []

  with tf.Session(graph=g, config=get_gpu_config()) as sess:

      nb_batches = int(math.ceil(float(len(image)) / flags.batch_size))
      assert nb_batches * flags.batch_size >= len(image)
      for batch in range(nb_batches):
          start = batch * flags.batch_size
          end = min(len(image), start + flags.batch_size)
          cur_batch_size = end - start

          image_list[:cur_batch_size] = image[start:end]
          label_list[:cur_batch_size] = np.ones((cur_batch_size,), dtype=int)
          feed_dict = {inp:image_list}

          outs = sess.run([out0,out1,out2,out3,out4], feed_dict=feed_dict)

          output = np.array(outs).reshape(640, )

      print(output)

  return output


def main(argv):
  parser = TensorRTParser()
  flags = parser.parse_args(args=argv[1:])

  # Load the data.
  # preprocess(flags.image_file,
  #            flags.filter_model_path,
  #            flags.feature_extractor_model_path,
  #            flags.focus_extractor_model_path,
  #            step=300, size=300, tar_size = 128)
  image_arrays = np.load("images.npy")
  image = np.resize(image_arrays, [len(image_arrays),224, 224, 3])
  print(image.shape)

  # Load the graph def
  if flags.frozen_graph:
    frozen_graph_def = get_frozen_graph(flags.frozen_graph)
  else:
    raise ValueError(
        "A Frozen Graph file must be provided.")


  # Run inference in all desired modes.
  if flags.native:
    mode = "native"
    print("Running {} graph".format(mode))
    execute_graph(mode,
                  frozen_graph_def,
                  image,
                  flags.input_node,
                  flags.output_node.split(','),
                  flags)



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
