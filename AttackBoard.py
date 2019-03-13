from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datagen import *
# generally a good idea as np rocks <3
import numpy as np
import tensorflow as tf
# Fake heaviside AF imports
from tensorflow.python.framework import ops
# End Fake heaviside
import sys
import random
import os
import time
from cleverhans.model import Model, CallableModelWrapper
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport, set_log_level
from tensorflow.python.saved_model import builder as saved_model_builder
import matplotlib.pyplot as plt
from cleverhans_tutorials import tutorial_models
from mpl_toolkits.mplot3d import Axes3D

__author__ = "François Menet <francois.menet@polymtl.ca>"

tf.logging.set_verbosity(tf.logging.DEBUG)
default_model_dir = "/home/dunamai/code/Projets/HybridCore/Hybrid_core_1/Experiments/Basic_DNN"

tf.app.flags.DEFINE_integer('training_iterations', 10000,
                            'number of training iterations.')
tf.app.flags.DEFINE_string('model_version', "1.0", 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', default_model_dir, 'Working directory.')
tf.app.flags.DEFINE_integer('dataset_magnitude', 5,
                            "The number of datapoints in the NDScheckerboard will be 10^("
                            "dataset_magnitude)")
tf.app.flags.DEFINE_boolean('show_results', False, 'Show the results as a pyplot. Only available for 2D inputs')
tf.app.flags.DEFINE_boolean('checkerboard', False,
                            'Show the original checkerboard on the pyplot. Only available for 2D inputs')
tf.app.flags.DEFINE_float('epsilon', 0.2, 'the level of perturbation for the FGSM attack')
FLAGS = tf.app.flags.FLAGS

# Global Variables used read-only, to get an easy grasp of the code.

graph = "The computation graph"
input_symbol = "symbolic input that will be poisonned at test time"
mode_setter="a repalcement global variable to feed the mode through cleverhans"

# A default classifier, our "basic dnn"
model_dir = os.path.join(FLAGS.work_dir, FLAGS.model_version)


def dataset_gen(powof10=6, size=3):
    powof10 = FLAGS.dataset_magnitude
    print("Launching the main...")
    checkerboard = NDCBDataGenerator()
    dataset = []
    targets = []
    print("Generating dataset")
    start = time.time()
    for i in range(10 ** powof10):
        rng1 = random.random() * size
        rng2 = random.random() * size
        target3 = checkerboard.get_target_onehot([rng1, rng2])
        dataset.append([rng1, rng2])
        targets.append(target3)
    targets = np.array(targets, dtype=np.int32)
    end = time.time()
    print("Dataset of %s samples generated in %s" % (str(len(dataset)), str(end - start)))
    
    train_set = np.array(dataset[:int(round(0.75 * 10 ** powof10))], dtype=np.float32)
    train_targets = np.array(targets[:int(round(0.75 * 10 ** powof10))])
    
    test_set = np.array(dataset[int(round(0.75 * 10 ** powof10)):], dtype=np.float32)
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


# https://stackoverflow.com/questions/39921607/how-to-make-a-custom-activation-function-with-only-python-in-tensorflow

def heaviside_activation(x):
    if x > 0:
        return 1.0
    else:
        return 0.0


heaviside = np.vectorize(heaviside_activation)

# tensorflow uses float32, numpy float64.

heaviside_32 = lambda x: heaviside(x).astype(np.float32)


@tf.RegisterGradient("FakeHS")
def heaviside_fake_grad(op, grad):
    x = op.inputs[0]
    
    return tf.to_float(1 / (1 + (x * x))) * grad


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def tf_heaviside2(x, name=None):
    with ops.name_scope(name, "heaviside", [x]) as name:
        y = py_func(heaviside_32,
                    [x],
                    (tf.float32),
                    name=name,
                    grad=heaviside_fake_grad)  # <-- here's the call to the gradient
        return y


def tf_heaviside(x, name=None):
    return 0 * x + tf.sign(x)


def BReLU(x, name=None):
    return tf.nn.relu(x) + tf.sign(x) + 1


# def HS(x, name=None):
#     return tf_heaviside2(x, name)


HS = tf.sign

# In the following we use Tensorflow coding style even if it is overkill here.

def basic_dnn(features, labels, mode, config):
    """
    The basic deep neural network
    :param features:
    :param labels:
    :param mode:
    :return:
    """
    input_layer = tf.reshape(features["x"], [-1, 2], name="inputs")
    preproc_layer = input_layer - 1.5
    hidden_layer = tf.layers.dense(inputs=preproc_layer, units=30, activation=tf.nn.relu, name="hiddenLayernum1")
    dropout_hidden = tf.layers.dropout(inputs=hidden_layer, rate=0.25,
                                       training=(mode == tf.estimator.ModeKeys.TRAIN))
    another_layer = tf.layers.dense(inputs=dropout_hidden, units=150, activation=tf.nn.relu,name="StatisticReLU")
    # comment logiclayertoGo and uncomment relu to go from statistical to hybrid.
    #logic_layer = tf.layers.dense(inputs=dropout_hidden, units=150, activation=tf.nn.relu, name="compareReLU")

    # The following is a workaround on the fact we cannot write this as the Activation function FakeGrad
    # is not known by TF
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign": "FakeHS"}):
        logic_layer = tf.layers.dense(inputs=dropout_hidden, units=150, activation=HS, name="logic_layer_toGO")

    logicGD = tf.gradients(logic_layer, dropout_hidden)
    LOGLDG = tf.identity(logicGD, name="Output_Logic_Gradient")

    dropout_intermediate = tf.layers.dropout(inputs=another_layer, rate=0.25,
                                             training=(mode == tf.estimator.ModeKeys.TRAIN))

    concat_layer = tf.keras.layers.concatenate([logic_layer, dropout_intermediate])

    layer_3 = tf.layers.dense(inputs=concat_layer, units=20, activation=tf.nn.relu,name="coucheH")

    outputter = tf.layers.dense(inputs=layer_3, units=2, activation=tf.nn.sigmoid, name="logits")

    out_show = tf.identity(input=outputter, name="outputter")

    probs = tf.nn.softmax(logits=outputter, name="sortie", dim=1)

    # tf.RegisterGradient(rnd_name)(grad)

    predictions = {
        "classes": tf.one_hot(tf.argmax(input=probs, axis=1), 2),
        "probabilities": probs

        # Logits
    }

    total_predictions = {
        'input_layer': input_layer,
        'preproc_layer': preproc_layer,
        'hidden_layer': hidden_layer,
        'dropout_hidden': dropout_hidden,
        'another_layer': another_layer,
        'logic_layer': logic_layer,
        # 'logicGD': logicGD,
        # 'LOGLDG': LOGLDG,
        'dropout_intermediate': dropout_intermediate,
        'concat_layer': concat_layer,
        'layer_3': layer_3,
        'outputter': outputter,
        'logits': out_show,
        'probs': probs,
    }

    # if you want to LOG EVERYTHING
    # for wx in locals().keys():
    #     tab[wx]=wx

    global graph
    graph = tf.get_default_graph()
    global input_symbol
    input_symbol = input_layer

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=total_predictions)

    # loss = tf.losses.absolute_difference(labels=labels, predictions=tf.reshape(outputter, [-1, 2]))
    # loss = tf.losses.absolute_difference(labels=labels, predictions=outputter)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=probs)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.5, global_step,
                                                   1000, 0.98, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)
        # learning_step = (
        #     tf.train.GradientDescentOptimizer(learning_rate)
        #         .minimize(loss=loss, global_step=global_step)
        # )

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


def attack_dnn(layer):
    """
    The basic deep neural network
    :param features:
    :param labels:
    :param mode:
    :return:
    """
    global mode_setter
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[1, 2])
    input_layer = tf.reshape(x, [1, 2], name="inputs")
    preproc_layer = input_layer - 1.5
    hidden_layer = tf.layers.dense(inputs=preproc_layer, units=1, activation=tf.nn.relu, name="hiddenLayer")
    dropout_hidden = tf.layers.dropout(inputs=hidden_layer, rate=0.25,
                                       training=(mode_setter == tf.estimator.ModeKeys.TRAIN))
    another_layer = tf.layers.dense(inputs=dropout_hidden, units=150, activation=tf.nn.relu)
    # comment logiclayertoGo and uncomment relu to go from statistical to hybrid.
    logic_layer = tf.layers.dense(inputs=dropout_hidden, units=150, activation=tf.nn.relu)
    
    # The following is a workaround on the fact we cannot write this as the Activation function FakeGrad
    # is not known by TF
    # g = tf.get_default_graph()
    # with g.gradient_override_map({"Sign": "FakeHS"}):
    #     logic_layer = tf.layers.dense(inputs=dropout_hidden, units=150, activation=HS, name="logic_layer_toGO")
    
    logicGD = tf.gradients(logic_layer, dropout_hidden)
    LOGLDG = tf.identity(logicGD, name="Output_Logic_Gradient")
    
    dropout_intermediate = tf.layers.dropout(inputs=another_layer, rate=0.25,
                                             training=(mode_setter == tf.estimator.ModeKeys.TRAIN))
    
    concat_layer = tf.keras.layers.concatenate([logic_layer, dropout_intermediate])
    
    layer_3 = tf.layers.dense(inputs=concat_layer, units=20, activation=tf.nn.relu)
    
    outputter = tf.layers.dense(inputs=layer_3, units=2, activation=tf.nn.sigmoid, name="logits")
    
    out_show = tf.identity(input=outputter, name="outputter")
    
    probs = tf.nn.softmax(logits=outputter, name="sortie", dim=1)
    
    return locals().get(layer, "")
    
    # tf.RegisterGradient(rnd_name)(grad)


def main(unused_argv):
    train_set, valid_set, test_set, train_targets, valid_targets, test_targets = dataset_gen()
    print(train_set[0:10])
    print(train_targets[0:10])

    model_dir = os.path.join(FLAGS.work_dir, FLAGS.model_version)
    my_classifier = tf.estimator.Estimator(
        model_fn=basic_dnn, model_dir=model_dir
    )

    print(unused_argv)

    tensors_to_log = {"probabilities": "sortie",
                      "entrypoints": "inputs",
                      "outputter": "outputter",
                      "Logic_GDS": "Output_Logic_Gradient"}

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
        shuffle=False,

    )

    eval_results = my_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    # exporting the model


    print("Estimator directory is : %s !" % model_dir)
    return my_classifier


default_classifier = tf.estimator.Estimator(
    model_fn=basic_dnn, model_dir=model_dir
)


def attack(eps=FLAGS.epsilon):
    X_train, valid_set, X_test, Y_train, valid_targets, Y_test = dataset_gen()
    report = AccuracyReport()
    config_args = {}
    sess = tf.Session(config=tf.ConfigProto(**config_args))
    # print(train_set[0:10])
    # print(train_targets[0:10])
    #
    # model_dir = os.path.join(FLAGS.work_dir, FLAGS.model_version)
    # my_classifier = tf.estimator.Estimator(
    #     model_fn=basic_dnn, model_dir=model_dir
    # )
    #
    # tensors_to_log = {"probabilities": "sortie",
    #                   "entrypoints": "inputs",
    #                   "outputter": "outputter",
    #                   "Logic_GDS": "Output_Logic_Gradient"}
    #
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
    #                                           every_n_iter=500)
    #
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_set},
    #                                                     y=train_targets,
    #                                                     batch_size=1,
    #                                                     num_epochs=300,
    #                                                     shuffle=False)
    #
    # my_classifier.train(input_fn=train_input_fn,
    #                     steps=FLAGS.training_iterations,
    #                     hooks=[logging_hook])
    #
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": test_set},
    #     y=test_targets,
    #     num_epochs=1,
    #     shuffle=False,
    #
    # )
    #
    # eval_results = my_classifier.evaluate(input_fn=eval_input_fn)
    # print(eval_results)
    #
    # # exporting the model
    # # def predNN(x):
    # #     pred_input_fn = tf.estimator.inputs.numpy_input_fn(x=x)
    # #     prd = my_classifier.predict(pred_input_fn)
    #
    # print("Estimator directory is : %s !" % model_dir)
    
    # Now we plan the attack ! Create the attacked white-box model using CleverHans and the class below extanding it
    attacked_model = CallableModelWrapper(attack_dnn, 'logits')
    x = tf.placeholder(tf.float32, shape=(1, 2))
    y = tf.placeholder(tf.float32, shape=(1, 2))
    fgsm_params = {'eps': eps,
                   'clip_min': 0.,
                   'clip_max': 3.}
    train_params = {
        'nb_epochs': 10,
        'batch_size': 1,
        'learning_rate': 0.02
    }
    
    preds = attacked_model.get_probs(x)
    fgsm = FastGradientMethod(attacked_model)
    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv = attacked_model.get_probs(adv_x)
    eval_params = {'batch_size': 1}
    
    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test
        # examples
        
        acc = model_eval(
            sess, x, y, preds, X_test, Y_test, args=eval_params)
        report.clean_train_clean_eval = acc
        print('Test accuracy on legitimate examples (training) : %0.4f' % acc)
    global mode_setter
    mode_setter = tf.estimator.ModeKeys.TRAIN
    model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                args=train_params)
    mode_setter = tf.estimator.ModeKeys.EVAL
    acc = model_eval(
        sess, x, y, preds, X_test, Y_test, args=eval_params)
    print('Test accuracy on legitimate examples (test) : %0.4f' % acc)
    print("Precision on Adversarial Examples below.")
    eval_par = {'batch_size': 1}
    acc = model_eval(sess=tf.get_default_session(), x=x, y=y, predictions=preds_adv,
                     X_test=X_test, Y_test=Y_test, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)


# def predNN(x, classifier=default_classifier):
#     """
#     Give the prediction of a neural network layer by layer, for interface with cleverhans
#     :param x: input as list-of-list eg. [[0.1,2.4]], there can be more than one eg [[1.1,2.2],[0.2,0.3]]
#     :return: a dict list of ('layer_name':'layer_value')
#     """
#     if not type(x) == np.ndarray:
#         x = {"x": np.array(x, dtype=np.float32)}
#     else:
#         x = {"x": x}
#
#     pred_input_fn = tf.estimator.inputs.numpy_input_fn(x=x, batch_size=1, shuffle=False)
#     prd = classifier.predict(pred_input_fn)
#     return [i for i in prd]


class AttackedEstimator(Model):
    def __init__(self, model):
        self.layer_names = predNN([[0.01, 0.01]], model).keys()
        self.model = model
        self.input_shape = (1, 2)
    
    def fprop(self, x):
        toreturn = predNN(x, self.model)
        return toreturn
    
    def get_layer_names(self):
        return self.layer_names


def get_classif():
    model_dir = os.path.join(FLAGS.work_dir, FLAGS.model_version)
    my_classifier = tf.estimator.Estimator(
        model_fn=basic_dnn, model_dir=model_dir
    )
    return my_classifier


def absolute_diff(a, b):
    return list(map((lambda x, y: x - y), predNN(a)[0]["logic_layer"], predNN(b)[0]["logic_layer"]))


def print_usage():
    print("""Usage :
    train [--dataset_magnitude powerOf10] [--work_dir path] [--model_version v] :
        trains the model and saves it
    datagen [--work_dir path] [--model_version v] [--show_results] : generates the data for 3D representation
    stdlisten : listen to stdin and give a result""")
    return


def show_plot(test_set, test_targets, eval_results, checkerboard=FLAGS.checkerboard):
    print('showing results ...')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = list(map(lambda x: x[0], test_set))
    y = list(map(lambda x: x[1], test_set))
    z = list()
    z2 = list()
    if type(test_targets[0]) == 'int':
        if checkerboard:
            z2 = list(test_targets)
        for i in eval_results:
            z.append(i['probabilities'][0])
    else:
        
        for i in eval_results:
            z.append(np.argmax(i))
        if checkerboard:
            for i in test_targets:
                z2.append(np.argmax(i))
    ax.scatter(x, y, z)
    if checkerboard:
        ax.scatter(x, y, z2)
    fig.show()


def datagen_2D():
    train_set, valid_set, test_set, train_targets, valid_targets, test_targets = dataset_gen(size=3, powof10=4)
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
        show_plot(test_set, test_targets, eval_results)
        input()
        return eval_results
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # x = list(map(lambda x: x[0], test_set))
        # y = list(map(lambda x: x[1], test_set))
        # z = list()
        # z2 = list(test_targets)
        #
        # for i in eval_results:
        #     z.append(i['probabilities'][0])
        #
        # ax.scatter(x, y, z)
        # ax.scatter(x, y, z2)
        # fig.show()


def stdlisten():
    print("Not listening yet.")
    return


if __name__ == "__main__":
    print("Launching the benchmark with the main function code...")
    if len(sys.argv) < 2:
        print_usage()
    elif sys.argv[1] == "train":
        clsf = tf.app.run()
    elif sys.argv[1] == "attack":
        tf.app.run(main=attack)
    elif sys.argv[1] == "datagen":
        eval_results = datagen_2D()
    elif sys.argv[1] == "stdlisten":
        stdlisten()
    else:
        print_usage()
