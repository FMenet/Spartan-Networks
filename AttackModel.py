"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with Keras.
It is very similar to mnist_tutorial_tf.py, which does the same
thing but without a dependence on keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import keras
from keras import backend
import tensorflow as tf
from tensorflow.python.platform import flags

from keras.layers.merge import _Merge

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
import cleverhans.attacks as atk
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper
import warnings
from distutils.version import LooseVersion
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import *
from TrainableScaler import TrainableScaler, TrainableOffset, Concentrator
from datagen import *
import sys
import random
import os
import time

glb = "global variable for hacking stuff"
mdl = "model globvar"

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


@tf.RegisterGradient("FakeHS")
def heaviside_fake_grad(op, grad):
    #1/sqrt(2pi)=0.39894228
    x = op.inputs[0]
    
    # return tf.to_float(1/(1+x*x))*grad
    # return 0.5*(tf.sign(x+0.1)+1)
    # return grad
    return tf.to_float(0.39894228*tf.exp(-(x*x)/2)) * grad


@tf.RegisterGradient("FakePRN")
def fake_parano_grad(op, grad):
    x = op.inputs[0]
    return tf.sign(0.5 - x)


def Heaviside(x):
    return 0.5 * (tf.sign(x) + 1)

def soft_heaviside(x):
    return 0.5 * (K.softsign(4000*x)+1)

def Paranograd(x):
    return Heaviside(tf.cos(x))


def dataset_gen(powof10=5, size=3):
    # powof10 = FLAGS.dataset_magnitude
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
    # g = tf.get_default_graph()
    # with g.gradient_override_map({"Sign": "FakeHS"}):
    # ax.scatter(x, y, z)
    # fig.show()
    
    # not required for now
    
    train_set = np.array(dataset[:int(0.7 * 10 ** powof10)])
    valid_set = np.array(dataset[int(0.7 * 10 ** powof10 + 1):int(0.85 * 10 ** powof10)])
    test_set = np.array(dataset[int(0.85 * 10 ** powof10 + 1):])
    
    train_targets = np.array(targets[:int(0.7 * 10 ** powof10)])
    valid_targets = np.array(targets[int(0.7 * 10 ** powof10 + 1):int(0.85 * 10 ** powof10)])
    test_targets = np.array(targets[int(0.85 * 10 ** powof10 + 1):])
    
    return train_set, valid_set, test_set, train_targets, valid_targets, test_targets


class XOR(_Merge):
    def _merge_function(self, inputs):
        x=inputs[0]
        y=inputs[1]
        output = tf.add(x,y)-tf.multiply(x,y)
        return output
    


def binary_filter_tf(x):
    """
    An efficient implementation of reduce_precision_tf(x, 2).
    """
    x_bin = tf.nn.relu(tf.sign(x - 0.5))
    return x_bin


class IsNegProb(keras.constraints.Constraint):
    """Constrains the weights to be between -1 and 0.
    """
    
    def __call__(self, w):
        w /= tf.maximum(K.abs(w), K.ones_like(w))
        w *= K.cast(K.less_equal(w, 0.), K.floatx())
        w += K.epsilon()*K.cast(K.equal(w,0.),K.floatx())
        return w

def variance_penalty(wm):
    # return K.var(wm)
    return K.sum(K.abs(wm))


def antientropy(wm):
    return 0.00001 * (1.0 / 0.6931471) * K.sum(tf.multiply(wm,tf.log(K.abs(wm))))
    # definition of entropy, applied to negative inputs
    # we also add a 1000th to do an order-average and avoid non-trainy behaviour


def cnn_model(sess=tf.get_default_session(), debug=False, w=0, rel=2):
    """
    Defines a CNN model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_filters: number of convolutional filters per layer
    :param nb_classes: the number of output classes
    :return:
    """
    return "a"
    # Define the layers successively (convolution layers are version dependent)
    # if keras.backend.image_dim_ordering() == 'th':
    #     input_shape = (channels, img_rows, img_cols)
    # else:
    #     input_shape = (img_rows, img_cols, channels)
    #
    # layers = [Reshape((784,),input_shape=(28,28,1)),
    #           Dense(10),
    #           Activation('relu'),
    #           Dense(50),
    #           Activation('relu'),
    #           Dense(nb_classes)]
    inpt = Input(batch_shape=(None, 28, 28, 1))
    k = Reshape((784,))(inpt)
    # k=Activation(activation=binary_filter_tf)(k) #if you want to use feature squeezing
    
    # if w == 1:
    #     print("No perception ensemble, using only one neuron")
    #     x2 = Activation(micromodel)(k)
    #     x2 = Dropout(0.05)(k)
    # else:
    #
    #     for i in range(w):
    #         percepfilter = Lambda(lambda x: micromodel(x))(k)
    #         percepfilter = Dropout(0.05)(percepfilter)
    #         perceptionlayer.append(percepfilter)
    #     x2 = Concatenate()(perceptionlayer)
    perceptionlayer = []
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign": "FakeHS"}):
        # for i in range(2):
        #     x = TrainableScaler(activation=Paranograd)(k)
        #     x = Dropout(0.15)(x)
        #     x = TrainableScaler(use_bias=False)(x)
        #     x = Dropout(0.15)(x)
        #     perceptionlayer.append(x)
        #
        # for relusnumber in range(0):
        #     relux = TrainableScaler(activation="relu")(k)
        #     relux = Dropout(0.20)(relux)
        #     relux = TrainableScaler(use_bias=False)(relux)
        #     relux = Dropout(0.20)(relux)
        #     perceptionlayer.append(relux)
        #
        # print(perceptionlayer)
        #
        # x2 = Add()(perceptionlayer)
        # x3 = Multiply()(perceptionlayer)
        # x3 = Lambda(lambda x: -2 * x, output_shape=(784,))(x3)
        # print(x2)
        # print(x3)
        #  x2 = Add()([x2, x3])
        # x2 = TrainableOffset()(x2)
        x = TrainableOffset(bias_constraint=IsNegProb(),
                            bias_initializer=keras.initializers.Constant(value=-0.5),
                            bias_regularizer=antientropy,
                            activation=Heaviside)(k)
        x2 = Reshape((28, 28, 1))(x)
        x2 = Conv2D(64, (6, 6))(x2)
        print(x2)
        x2 = Reshape((-1,))(x2)
        x2 = TrainableOffset(
            bias_initializer=keras.initializers.Constant(value=0.0),
            activation='relu')(x2)
        # x2 = TrainableScaler(use_bias=False, kernel_regularizer='l1', activation=Heaviside)(x2)
        # x2 = Activation('relu')(x2)
        x2 = Reshape((23, 23, 64))(x2)
        x2 = Conv2D(64, (4, 4))(x2)
        x2 = Flatten()(x2)
        x2 = Dense(40, activation="relu")(x2)
        # x2 = Dense(400, activation=Paranograd)(x2)
        # x2 = TrainableScaler(use_bias=False, kernel_regularizer='l1', activation=Heaviside)(x2)
        x2 = Dense(10)(x2)
    predictions2 = Activation(activation="softmax")(x2)
    
    model = keras.Model(inputs=inpt, outputs=predictions2)
    
    if debug:
        return keras.Model(inputs=inpt, outputs=debugO)
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                                  train_end=60000,
                                                  test_start=0,
                                                  test_end=10000)
    K.set_session(sess)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(np.array(X_train), np.array(Y_train), 128, 3, verbose=1)
    print(model.evaluate(np.array(X_test), np.array(Y_test)))
    print(model.get_weights())
    return model



# def spartan_network_archive(sess=tf.get_default_session(), debug=False, w=2):
#     inpt = Input(batch_shape=(None, 28, 28, 1))
#     x2 = Reshape((784,))(inpt)
#     g = tf.get_default_graph()
#     with g.gradient_override_map({"Sign": "FakeHS"}):
#
#         # x = TrainableOffset(bias_constraint=IsNegProb(),
#         #                     bias_initializer=keras.initializers.Constant(value=-0.5),
#         #                     bias_regularizer=antientropy,
#         #                     activation=Heaviside)(x2)
#         tf.image.extract_glimpse()
#         x2 = Reshape((28, 28, 1))(x2)
#
#         x2 = Conv2D(64, (6, 6), kernel_constraint=keras.constraints.max_norm(2.))(x2)
#
#         x2 = Flatten()(x2)
#         layer = []
#         for i in range(w):
#             layer.append(TrainableOffset(bias_constraint=IsNegProb(),
#                                          bias_initializer=keras.initializers.random_normal(-0.1),
#                                          bias_regularizer=antientropy,
#                                          activation=Heaviside)(x2))
#
#         x2 = Add()(layer)
#         # x3 = Multiply()(layer)
#         # x3 = Lambda(lambda x:2*x)(x3)
#         # x2 = Subtract()([x2,x3])
#         # x2 = TrainableScaler(use_bias=False, kernel_regularizer='l1', activation=Heaviside)(x2)
#         # x2 = Activation('relu')(x2)
#         x2 = Reshape((23, 23, 64))(x2)
#         x2 = Conv2D(64, (4, 4))(x2)
#         x2 = Flatten()(x2)
#         x2 = Dropout(0.25)(x2)
#         layer=[]
#         # for i in range(w):
#         #     layer.append(TrainableOffset(bias_constraint=IsNegProb(),
#         #                                  bias_initializer=keras.initializers.random_normal(-0.1),
#         #                                  bias_regularizer=antientropy,
#         #                                  activation=Heaviside)(x2))
#
#         # x2 = XOR()(layer)
#         x2 = Dense(40, activation=Heaviside)(x2)
#         # x2 = Dense(400, activation=Paranograd)(x2)
#         # x2 = TrainableScaler(use_bias=False, kernel_regularizer='l1', activation=Heaviside)(x2)
#         x2 = Dense(10)(x2)
#     predictions2 = Activation(activation="softmax")(x2)
#
#     model = keras.Model(inputs=inpt, outputs=predictions2)
#     model.summary()
#
#     X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
#                                                   train_end=60000,
#                                                   test_start=0,
#                                                   test_end=10000)
#     K.set_session(sess)
#     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#     model.fit(np.array(X_train), np.array(Y_train), 128, 3, verbose=1)
#     print(model.evaluate(np.array(X_test), np.array(Y_test)))
#     print(model.get_weights())
#     return model

def unprotected_network(sess=tf.get_default_session(), debug=False, train=False, save=False):
    inpt = Input(batch_shape=(None, 28, 28, 1))
    x2 = Conv2D(32, (3, 3), activation='relu')(inpt)
    x2 = Conv2D(32, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Conv2D(32, (3, 3), activation='relu')(x2)
    x2 = Conv2D(32, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    
    # x2 = Flatten()(x2)
    # layer = []
    # for i in range(w):
    #     layer.append(TrainableOffset(bias_constraint=IsNegProb(),
    #                                  bias_initializer=keras.initializers.random_normal(-0.1),
    #                                  bias_regularizer=antientropy,
    #                                  activation=Heaviside)(x2))
    
    # x2 = Add()(layer)
    # x3 = Multiply()(layer)
    # x3 = Lambda(lambda x:2*x)(x3)
    # x2 = Subtract()([x2,x3])
    # x2 = TrainableScaler(use_bias=False, kernel_regularizer='l1', activation=Heaviside)(x2)
    # x2 = Reshape((23, 23, 64))(x2)
    # x2 = Conv2D(64, (3, 3))(x2)
    # x2 = MaxPool2D()(x2)
    # x2 = Conv2D(128, (3, 3))(x2)
    # x2 = MaxPool2D()(x2)
    
    x2 = Flatten()(x2)
    x2 = Dropout(0.2)(x2)
    layer = []
    # for i in range(w):
    #     layer.append(TrainableOffset(bias_constraint=IsNegProb(),
    #                                  bias_initializer=keras.initializers.random_normal(-0.1),
    #                                  bias_regularizer=antientropy,
    #                                  activation=Heaviside)(x2))
    
    # x2 = XOR()(layer)
    x2 = Dense(50, activation="relu")(x2)
    
    x2 = Dense(50, activation='relu')(x2)
    # x2 = TrainableScaler(use_bias=False, kernel_regularizer='l1', activation=Heaviside)(x2)
    x2 = Dense(10)(x2)


    predictions2 = Activation(activation="softmax")(x2)
    
    model = keras.Model(inputs=inpt, outputs=predictions2)

    if debug:
        model.summary()
    if not train:
        return model
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                                  train_end=60000,
                                                  test_start=0,
                                                  test_end=10000)
    K.set_session(sess)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(np.array(X_train), np.array(Y_train), 128, 6, verbose=1)
    print(model.evaluate(np.array(X_test), np.array(Y_test)))
    # print(model.get_weights())
    if not save:
        return model
    save_path = os.path.join("/tmp", "mnist.ckpt")
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print("Completed model training and saved at: " +
                 str(save_path))
    return model


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def spartan_network(sess=tf.get_default_session(), debug=False, w=4, train=False, create_surrogate=False):
    # we want to integrate an attention mechanism with a feature squeezer
    inpt = Input(batch_shape=(None, 28, 28, 1))
    g = tf.get_default_graph()
    speclayer=[]
    with g.gradient_override_map({"Sign": "FakeHS"}):
        x2 = Conv2D(4, (1,1), activation=Paranograd, bias_initializer=keras.initializers.random_uniform(minval=0.05, maxval=0.5), kernel_initializer=keras.initializers.random_uniform(minval=1, maxval=1.5))(inpt)
        # x2 = SpatialDropout2D(0.1)(x2)
        x2 = Conv2D(32, (3, 3), activation='relu')(x2)
        # x2 = SpatialDropout2D(0.1)(x2)
        x2 = Conv2D(32, (3, 3), activation=Paranograd)(x2)
        x2 = MaxPooling2D(pool_size=(2, 2))(x2)
        x2 = Conv2D(32, (3, 3), activation='relu')(x2)
        # x2 = SpatialDropout2D(0.1)(x2)
        x2 = Conv2D(32, (3, 3), activation='relu')(x2)
        x2 = MaxPooling2D(pool_size=(2, 2))(x2)
        
        # x2 = Flatten()(x2)
        # layer = []
        # for i in range(w):
        #     layer.append(TrainableOffset(bias_constraint=IsNegProb(),
        #                                  bias_initializer=keras.initializers.random_normal(-0.1),
        #                                  bias_regularizer=antientropy,
        #                                  activation=Heaviside)(x2))

        # x2 = Add()(layer)
        # x3 = Multiply()(layer)
        # x3 = Lambda(lambda x:2*x)(x3)
        # x2 = Subtract()([x2,x3])
        # x2 = TrainableScaler(use_bias=False, kernel_regularizer='l1', activation=Heaviside)(x2)
        # x2 = Reshape((23, 23, 64))(x2)
        # x2 = Conv2D(64, (3, 3))(x2)
        # x2 = MaxPool2D()(x2)
        # x2 = Conv2D(128, (3, 3))(x2)
        # x2 = MaxPool2D()(x2)
        
        x2 = Flatten()(x2)
        # x2 = Dropout(0.1)(x2)
        layer=[]
        # for i in range(w):
        #     layer.append(TrainableOffset(bias_constraint=IsNegProb(),
        #                                  bias_initializer=keras.initializers.random_normal(-0.1),
        #                                  bias_regularizer=antientropy,
        #                                  activation=Heaviside)(x2))

        # x2 = XOR()(layer)
        x2 = Dense(100, activation="relu")(x2)
        
        # x2 = Dense(50, activation='relu')(x2)
        # x2 = TrainableScaler(use_bias=False, kernel_regularizer='l1', activation=Heaviside)(x2)
        x2 = Dense(10)(x2)
    predictions2 = Activation(activation="softmax")(x2)
    
    model = keras.Model(inputs=inpt, outputs=predictions2)
    model.summary()
    if not train:
        return model


    hist=LossHistory()
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                                  train_end=60000,
                                                  test_start=0,
                                                  test_end=10000)
    K.set_session(sess)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(np.array(X_train), np.array(Y_train), 128, 8, verbose=1, callbacks=[hist])
    print(model.evaluate(np.array(X_test), np.array(Y_test)))
    print(model.get_weights())
    crntm=time.localtime()
    stnct = str(str(crntm.tm_mday) + "_" + str(crntm.tm_mon) + "_" + str(crntm.tm_year) + "_" + str(crntm.tm_hour) + "_" + str(crntm.tm_min))
    with open('Experiments/'+stnct,"ab") as f:
        import pickle
        pickle.dump([model.get_weights(), hist.losses, model.evaluate(np.array(X_test), np.array(Y_test))], f)
        
    if create_surrogate:
        inpt = Input(batch_shape=(None, 28, 28, 1))
        x2 = Conv2D(4, (1, 1), activation=soft_heaviside,
                    kernel_initializer=keras.initializers.random_uniform(minval=0, maxval=30))(inpt)
        x2 = SpatialDropout2D(0.1)(x2)
        x2 = Conv2D(32, (3, 3), activation='relu')(x2)
        x2 = SpatialDropout2D(0.1)(x2)
        x2 = Conv2D(32, (3, 3), activation=soft_heaviside)(x2)
        x2 = MaxPooling2D(pool_size=(2, 2))(x2)
        x2 = Conv2D(32, (3, 3), activation='relu')(x2)
        x2 = SpatialDropout2D(0.1)(x2)
        x2 = Conv2D(32, (3, 3), activation=soft_heaviside)(x2)
        x2 = MaxPooling2D(pool_size=(2, 2))(x2)
        x2 = Flatten()(x2)
        x2 = Dropout(0.2)(x2)
        x2 = Dense(50, activation="relu")(x2)
        x2 = Dense(50, activation='relu')(x2)
        # x2 = TrainableScaler(use_bias=False, kernel_regularizer='l1', activation=Heaviside)(x2)
        x2 = Dense(10)(x2)
        predictions_surr = Activation(activation="softmax")(x2)
        
        model_surrogate=keras.Model(inputs=inpt, outputs=predictions_surr)
        model_surrogate.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model_surrogate.set_weights(model.get_weights())
        print(model.get_weights()[2],model_surrogate.get_weights()[2])
        return [model_surrogate, model]
        
    return model


FLAGS = flags.FLAGS


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=100, batch_size=128,
                   learning_rate=0.001, train_dir="/tmp",
                   filename="mnist.ckpt", load_model=False,
                   testing=True, w=2, rel=0):
    """
    MNIST CleverHans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param train_dir: Directory storing the saved model
    :param filename: Filename to save model under
    :param load_model: True for load, False for not load
    :param testing: if true, test error is calculated
    :param w : number of perceptive neurons
    :return: an AccuracyReport object
    """
    
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()
    
    # Set TF random seed to improve reproducibility
    #tf.set_random_seed(1234)
    
    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")
    
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")
    
    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)
    keras.layers.core.K.set_learning_phase(1)
    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)
    
    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
    
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    # Define TF model graph
    models = spartan_network(sess=sess,w=w,train=True,create_surrogate=True)
    model_to_attack = models[0]
    spartan_model = models[1]
    ckpt = tf.train.get_checkpoint_state(train_dir)
    print(ckpt)
    trainvalue=True if ckpt is None else False
    #model_to_attack = unprotected_network(sess=sess, train=trainvalue, save=True)
    preds = model_to_attack(x)
    print("Defined TensorFlow model graph.")
    
    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
        report.clean_train_clean_eval = acc
        assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)
    
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': train_dir,
        'filename': filename
    }
    rng = None
    ckpt = tf.train.get_checkpoint_state(train_dir)
    ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path
    
    if load_model and ckpt_path:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
        print("Model loaded from: {}".format(ckpt_path))
        evaluate()
    else:
        print("Model was not loaded, training from scratch.")
        keras.layers.core.K.set_learning_phase(1)
        # model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
          #          args=train_params, save=True, rng=rng)
    keras.layers.core.K.set_learning_phase(0)
    
    # Calculate training error
    if testing:
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
        print("With no Dropout : %s" % acc)
        report.train_clean_train_clean_eval = acc
    
    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    wrap = KerasModelWrapper(model_to_attack)
    global mdl
    mdl = model_to_attack
    fgsm = FastGradientMethod(wrap, sess=sess)
    for epstep in range(40):
        fgsm_params = {'eps': 0.01+0.02*epstep,
                   'clip_min': 0.,
                   'clip_max': 1.}
    # cw_params = {'confidence': 0.5,
    #              'batch_size': 4,
    #              'learning_rate': 2e-2,
    #              'max_iterations': 400,
    #              'clip_min': 0.,
    #              'clip_max': 1.}
        adv_x = fgsm.generate(x, **fgsm_params)
    # Consider the attack to be constant
        adv_x = tf.stop_gradient(adv_x)
    # cwattack = atk.CarliniWagnerL2(model_to_attack,sess=sess)
    # adv_x = cwattack.generate(x, **cw_params)
    # adv_x = tf.stop_gradient(adv_x)
    # adv_x_np = cwattack.generate_np(X_test[500:704], **cw_params)
    # from matplotlib import pyplot as plt
    # plt.rc('figure', figsize=(12.0, 12.0))
    # for j in range(40):
    #
    #     plt.imshow(adv_x_np[j].reshape((28, 28)),
    #                cmap="gray")
    #     plt.pause(0.15)
    # return
    
        preds_adv = spartan_model(adv_x)
    
    
    # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on epsilon-%0.4f-adversarial examples: %0.4f\n' % (fgsm_params["eps"],acc))
    return
    report.clean_train_adv_eval = acc
    
    # Calculating train error
    if testing:
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_train,
                         Y_train, args=eval_par)
        report.train_clean_train_adv_eval = acc
    
    print("Repeating the process, using adversarial training")
    # Redefine TF model graph
    model_2 = cnn_model(w=w, rel=rel)
    preds_2 = model_2(x)
    wrap_2 = KerasModelWrapper(model_2)
    fgsm2 = FastGradientMethod(wrap_2, sess=sess)
    preds_2_adv = model_2(fgsm2.generate(x, **fgsm_params))
    
    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        report.adv_train_clean_eval = accuracy
        
        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        report.adv_train_adv_eval = accuracy
    
    # Perform and evaluate adversarial training
    model_train(sess, x, y, preds_2, X_train, Y_train,
                predictions_adv=preds_2_adv, evaluate=evaluate_2,
                args=train_params, save=False, rng=rng)
    
    # Calculate training errors
    if testing:
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_train, Y_train,
                              args=eval_params)
        report.train_adv_train_clean_eval = accuracy
        accuracy = model_eval(sess, x, y, preds_2_adv, X_train,
                              Y_train, args=eval_params)
        report.train_adv_train_adv_eval = accuracy
    
    


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs,
                   batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   train_dir=FLAGS.train_dir,
                   filename=FLAGS.filename,
                   load_model=FLAGS.load_model,
                   w=FLAGS.w, rel=FLAGS.rel)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('w', 2, 'Number of perceptive neurons per pixel')
    flags.DEFINE_integer('rel', 0, 'Number of hybridation relus per pixel')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_string('train_dir', '/tmp', 'Directory where to save model.')
    flags.DEFINE_string('filename', 'mnist.ckpt', 'Checkpoint filename.')
    flags.DEFINE_boolean('load_model', False, 'Load saved model or train.')
    tf.app.run()
