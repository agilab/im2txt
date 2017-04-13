# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")

tf.flags.DEFINE_string("output_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")

def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)
    saver = tf.train.Saver()
    saver.save(sess, FLAGS.output_path + "/im2txt-model")
    tf.train.write_graph(sess.graph_def, FLAGS.output_path, "graph.pbtxt")

if __name__ == "__main__":
  tf.app.run()
