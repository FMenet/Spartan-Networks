from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datagen import *
import tensorflow as tf
import sys
import random
import os
import time
from tensorflow.python.saved_model import builder as saved_model_builder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

__author__ = "François Menet <francois.menet@polymtl.ca>"

tf.logging.set_verbosity(tf.logging.INFO)

default_model_dir = "/home/dunamai/code/Projets/HybridCore/Hybrid_core_1/Experiments/Basic_DNN"

tf.app.flags.DEFINE_integer('training_iterations', 10000,
                            'number of training iterations.')
tf.app.flags.DEFINE_string('model_version', "1.0", 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', default_model_dir, 'Working directory.')
tf.app.flags.DEFINE_integer('dataset_magnitude', 5,
                            "The number of datapoints in the NDScheckerboard will be 10^("
                            "dataset_magnitude)")
tf.app.flags.DEFINE_boolean('show_results', False, 'Show the results as a pyplot. Only available for 2D inputs')
FLAGS = tf.app.flags.FLAGS


def dataset_gen(powof10=5, size=3):
    print("Launching the main...")
    checkerboard = NDCBDataGenerator()
    dataset = []
    targets = []
    print("Generating dataset")
    start = time.time()
    for i in range(10 ** powof10):
        rng1 = random.random() * size
        rng2 = random.random() * size
        target3 = checkerboard.get_target([rng1, rng2])
        dataset.append([rng1, rng2])
        targets.append(target3)
    targets=np.array(targets,dtype=np.int32)
    end = time.time()
    print("Dataset of %s samples generated in %s" % (str(len(dataset)), str(end - start)))
    
    train_set = np.array(dataset[:int(round(0.75 * 10 ** powof10))])
    train_targets = np.array(targets[:int(round(0.75 * 10 ** powof10))])
    
    test_set = np.array(dataset[int(round(0.75 * 10 ** powof10)):])
    test_targets = np.array(targets[int(round(0.75 * 10 ** powof10)):])
    
    valid_set = "TODO"
    valid_targets = "TODO"
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = list(map(lambda x: x[0], test_set))
    # y = list(map(lambda x: x[1], test_set))
    # z = list(test_targets)
    #
    # ax.scatter(x, y, z)
    # fig.show()
    
    # not required for now
    
    # train_set = np.array(dataset[:0.7 * 10 ** powof10])
    # valid_set = np.array(dataset[0.7 * 10 ** powof10 + 1:0.85 * 10 ** powof10])
    # test_set = np.array(dataset[0.85 * 10 ** powof10 + 1:])
    #
    # train_targets = np.array(targets[:0.7 * 10 ** powof10])
    # valid_targets = np.array(targets[0.7 * 10 ** powof10 + 1:0.85 * 10 ** powof10])
    # test_targets = np.array(targets[0.85 * 10 ** powof10 + 1:])
    
    return train_set, valid_set, test_set, train_targets, valid_targets, test_targets


# In the following we use Tensorflow coding style even if it is overkill here.

def basic_dnn(features, labels, mode):
    """
    The basic deep neural network
    :param features:
    :param labels:
    :param mode:
    :return:
    """
    input_layer = tf.reshape(features["x"], [-1, 2], name="inputs")
    hidden_layer = tf.layers.dense(inputs=input_layer, units=30, activation=tf.nn.relu)
    dropout_hidden = tf.layers.dropout(inputs=hidden_layer, rate=0.250,
                                             training=(mode == tf.estimator.ModeKeys.TRAIN))
    another_layer = tf.layers.dense(inputs=dropout_hidden, units=100, activation=tf.nn.relu)
    dropout_intermediate = tf.layers.dropout(inputs=another_layer, rate=0.250,
                                             training=(mode == tf.estimator.ModeKeys.TRAIN))
    layer_3=tf.layers.dense(inputs=dropout_intermediate, units=20, activation=tf.nn.relu)
    
    outputter = tf.layers.dense(inputs=layer_3, units=2)
    
    predictions = {
        "classes": tf.argmax(input=outputter, axis=1),
        "probabilities": tf.nn.softmax(logits=outputter, name="sortie")
        
        # Simplest prediction, one neuron
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    #loss = tf.losses.absolute_difference(labels=labels, predictions=tf.reshape(outputter, [-1, 2]))
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=outputter)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.015)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


def main(unused_argv):
    train_set, valid_set, test_set, train_targets, valid_targets, test_targets = dataset_gen()
    model_dir = os.path.join(FLAGS.work_dir, FLAGS.model_version)
    my_classifier = tf.estimator.Estimator(
        model_fn=basic_dnn, model_dir=model_dir
    )
    
    
    print(unused_argv)
    
    tensors_to_log = {"probabilities": "sortie"}
    
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=500)
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_set},
                                                        y=train_targets,
                                                        batch_size=1,
                                                        num_epochs=300,
                                                        shuffle=False)
    
    my_classifier.train(input_fn=train_input_fn,
                        steps=FLAGS.training_iterations,
                        hooks=[logging_hook])
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_set},
        y=test_targets,
        num_epochs=1,
        shuffle=False
    
    )
    
    eval_results = my_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    # exporting the model
    
    
    print("Estimator directory is : %s !" % model_dir)


def print_usage():
    print("""Usage :
    train [--dataset_magnitude powerOf10] [--work_dir path] [--model_version v] :
        trains the model and saves it
    datagen [--work_dir path] [--model_version v] [--show-result] : generates the data for 3D representation
    stdlisten : listen to stdin and give a result""")
    return


def datagen_2D():
    train_set, valid_set, test_set, train_targets, valid_targets, test_targets = dataset_gen(size=4, powof10=4)
    model_dir = os.path.join(FLAGS.work_dir, FLAGS.model_version)
    saved_classifier = tf.estimator.Estimator(
        model_fn=basic_dnn, model_dir=model_dir
    )
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_set},
        y=test_targets,
        num_epochs=1,
        shuffle=False
    
    )
    
    eval_results = saved_classifier.predict(eval_input_fn)
    if not FLAGS.show_results:
        return eval_results
    else:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = list(map(lambda x: x[0], test_set))
        y = list(map(lambda x: x[1], test_set))
        z = list()
        z2 = list(test_targets)
        
        for i in eval_results:
            z.append(i['probabilities'][0])
        
        ax.scatter(x, y, z)
        ax.scatter(x, y, z2)
        fig.show()


def stdlisten():
    print("Not listening yet.")
    return


if __name__ == "__main__":
    print("Launching the benchmark with the main function code...")
    if len(sys.argv) < 2:
        print_usage()
    elif sys.argv[1] == "train":
        tf.app.run()
    elif sys.argv[1] == "datagen":
        eval_results = datagen_2D()
    elif sys.argv[1] == "stdlisten":
        stdlisten()
    else:
        print_usage()
