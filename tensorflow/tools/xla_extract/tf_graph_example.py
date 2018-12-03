#!/usr/bin/env python3
"""
This provides a simple Conv-Pool-Conv-FC tensorflow model
for testing xla extraction
"""

import tensorflow as tf
import logging
import tensorflow.contrib.layers as tf_layers
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.compiler.xla import compile

tf.enable_resource_variables()

def model_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=None):
    ''' This function is the input to Estimator constructor.
    More generally it is a python function that returns a computational graph
    given some set of inputs
    '''
    num_classes = 10

    data_format = "channels_first"

    conv1 = tf.layers.conv2d(
        inputs=features,
        filters=4,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        data_format=data_format)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=4,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        data_format=data_format)

    flat = tf.layers.flatten(conv2)
    logits = tf.layers.dense(
        flat, units=num_classes, name='final_node', use_bias=False)

    labels = tf.one_hot(labels, depth=num_classes)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits))

    learning_rate = 0.1
    train_op = \
        tf.train.GradientDescentOptimizer(
            learning_rate, name="train_step").minimize(
            loss, global_step=tf.train.get_global_step())

    with ops.control_dependencies([train_op]):
        return array_ops.identity(loss)

xshape, yshape = [16, 3, 32, 32], [16, 1]

# Constant input version
#x = tf.constant(20, tf.float32, shape=xshape)
#y = tf.constant(1, tf.int32, shape=yshape)

# Placeholder input version
#x = tf.placeholder(tf.float32, shape=xshape)
#y = tf.placeholder(tf.int32, shape=yshape)

# Dataset input version
def input_fn():
    ds = tf.data.Dataset.from_tensor_slices(
        (np.random.rand(*([1] + xshape[1:])).astype(np.float32),
         np.random.rand(*([1] + yshape[1:])).astype(np.int32)))
    ds = ds.repeat(48)
    ds = ds.shuffle(16)
    ds = ds.batch(8, drop_remainder=True)
    return ds

xydataset = input_fn().make_initializable_iterator()
x, y = xydataset.get_next()


def generic_compile(model_fn, inputs):
    placeholder_inputs = [
        tf.placeholder(i.dtype, shape=i.shape, name=i.op.name) for i in inputs]
    return compile(model_fn, inputs=placeholder_inputs)


def strip_graph(output, inputs):
    from tensorflow.tools.graph_transforms import TransformGraph
    transformed_graph_def = TransformGraph(
        output.graph.as_graph_def(add_shapes=True),
        inputs=[i.op.name for i in inputs],
        outputs=[output.op.name],
        transforms=["strip_unused_nodes"]
    )
    return transformed_graph_def

(loss,) = generic_compile(model_fn, inputs=[x, y])
tf_graph_def = strip_graph(output=loss, inputs=[x, y])

with open("tf_graph.pbtxt", 'w') as f:
    f.write(str(tf_graph_def))
    print("Target node: {}".format(loss.op.name))
