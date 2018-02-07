# XLA HLO Session extraction from Graph

This tool allows


## Command line interface
After saving out the graph_def as a proto file, run the following command:

```
    tf_graph_to_xla_hlo \
        --in_graph=graph.pbtxt \
        --target_node="GradientDescent" \
        --out_prefix="xla_output_file" \
        --output_as_txt
```

## Python interface
From within python, one can generate the xla SessionModule proto without saving out the graph by using the `XlaExtract` python interface function

```python

import tensorflow as tf
from tensorflow.contrib.xla_extractor import XlaExtract

batch_size = 4
img = tf.placeholder(tf.float32, shape=(batch_size, 784))
lbl = tf.placeholder(tf.int32, shape=(batch_size, 10))

with tf.device("/job:localhost/replica:0/task:0/device:XLA_CPU:0"):
    with tf.variable_scope('', use_resource=True):
        logits = tf.layers.dense(inputs=img, units=10, use_bias=False)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=lbl))

        train_op = tf.train.GradientDescentOptimizer(
            learning_rate=0.1).minimize(
                loss, global_step=tf.train.get_global_step())

session_proto = XlaExtract(train_op)
```
