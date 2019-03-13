from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf

import warnings
from distutils.version import LooseVersion
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import *
from TrainableScaler import TrainableScaler
glb = "global variable for hacking stuff"
mdl = "model globvar"
from keras import backend as K

def cnn_model_with_trainable_scalers_relu(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=1, nb_classes=10, w=10):
    """
    Defines a CNN model using Keras model. It features the ActivableScaling layer
    This model features an exceptional resistance for a uknown reason
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
    inpt = Input(shape=(None, 28, 28, 1))
    k = Lambda(lambda x: x)(inpt)
    perceptionlayer = []
    
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
    
    x2 = TrainableScaler(activation="relu")(k)
    x2 = Dropout(0.15)(x2)
    x2 = Reshape((28, 28, 1))(x2)
    x2 = Conv2D(64, (6, 6))(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(64, (4, 4))(x2)
    x2 = Flatten()(x2)
    x2 = Dense(30, activation='relu')(x2)
    x2 = Dropout(0.25)(x2)
    x2 = Dense(10)(x2)
    predictions2 = Activation(activation="softmax")(x2)
    
    model = keras.Model(inputs=inpt, outputs=predictions2)
    
    return model


def cnn_model_heaviside_spartan_network(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=1, nb_classes=10, w=10):
    """
    This model achieves, with w=4, a precision of 93% but and adversarial precision of 95 % !
    with w=2 precision 95% and 97% on adversarial.
    After 160 epochs, such a net with w=2 was capable of going up to 99.53% and (adv)97.67%

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
    inpt = Input(shape=(None, 28, 28, 1))
    k = Reshape((784,))(inpt)
    
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
        for i in range(w):
            x = TrainableScaler(activation=Heaviside)(k)
            x = Dropout(0.01)(x)
            perceptionlayer.append(x)
        
        x2 = Concatenate()(perceptionlayer)
        x2 = Reshape((784, w))(x2)
        x2 = Dense(1, activation=Heaviside)(x2)
    print(x2)
    x2 = Reshape((28, 28, 1))(x2)
    x2 = Conv2D(64, (6, 6))(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(64, (4, 4))(x2)
    x2 = Flatten()(x2)
    x2 = Dense(30, activation='relu')(x2)
    x2 = Dropout(0.25)(x2)
    x2 = Dense(10)(x2)
    predictions2 = Activation(activation="softmax")(x2)
    
    model = keras.Model(inputs=inpt, outputs=predictions2)
    
    return model


def cnn_model_relu_spartan_network(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=1, nb_classes=10, w=10):
    """
    w=4 and 16 epochs, 99.35% precision but 19.45% on adversarial.
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
    inpt = Input(shape=(None, 28, 28, 1))
    k = Reshape((784,))(inpt)
    
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
        for i in range(w):
            x = TrainableScaler(activation="relu")(k)
            x = Dropout(0.01)(x)
            perceptionlayer.append(x)
        
        x2 = Concatenate()(perceptionlayer)
        x2 = Reshape((784, w))(x2)
        x2 = Dense(1, activation="relu")(x2)
    print(x2)
    x2 = Reshape((28, 28, 1))(x2)
    x2 = Conv2D(64, (6, 6))(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(64, (4, 4))(x2)
    x2 = Flatten()(x2)
    x2 = Dense(30, activation='relu')(x2)
    x2 = Dropout(0.25)(x2)
    x2 = Dense(10)(x2)
    predictions2 = Activation(activation="softmax")(x2)
    
    model = keras.Model(inputs=inpt, outputs=predictions2)
    
    return model


def cnn_hybrid_spartan(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=1, nb_classes=10, w=2, rel=0):
    """
    Relus ! 80 epochs, relu=2, 99.69% precision, 98.01% adv.
    ERROR
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
    inpt = Input(shape=(None, 28, 28, 1))
    k = Reshape((784,))(inpt)
    
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
    # g = tf.get_default_graph()
    # with g.gradient_override_map({"Sign": "FakeHS"}):
    #     for i in range(w):
    #
    #         x = TrainableScaler(activation=Heaviside)(k)
    #         x = Dropout(0.01)(x)
    #         perceptionlayer.append(x)
    
    for relusnumber in range(rel):
        relux = TrainableScaler(activation="relu")(k)
        relux = Dropout(0.01)(relux)
        perceptionlayer.append(relux)
    
    print(perceptionlayer)
    
    x2 = Concatenate()(perceptionlayer)
    x2 = Reshape((784, w + rel))(x2)
    x2 = Dense(1, activation=Heaviside)(x2)
    print(x2)
    x2 = Reshape((28, 28, 1))(x2)
    x2 = Conv2D(64, (6, 6))(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(64, (4, 4))(x2)
    x2 = Flatten()(x2)
    x2 = Dense(30, activation='relu')(x2)
    x2 = Dropout(0.25)(x2)
    x2 = Dense(10)(x2)
    predictions2 = Activation(activation="softmax")(x2)
    
    model = keras.Model(inputs=inpt, outputs=predictions2)
    
    return model


def last_weak_spartan_vs_bb(debug=False, w=0, rel=2):
    """
    vs blackbox : 11%...
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
    inpt = Input(shape=(None, 28, 28, 1))
    k = Reshape((784,))(inpt)
    
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
    # with g.gradient_override_map({"Sign": "FakeHS"}):
    for i in range(w):
        x = TrainableScaler(activation=Heaviside)(k)
        x = Dropout(0.01)(x)
        x = TrainableScaler(use_bias=False)(x)
        x = Dropout(0.01)(x)
        perceptionlayer.append(x)
    
    for relusnumber in range(rel):
        relux = TrainableScaler(activation="relu")(k)
        relux = Dropout(0.01)(relux)
        relux = TrainableScaler(use_bias=False)(relux)
        relux = Dropout(0.01)(relux)
        perceptionlayer.append(relux)
    
    print(perceptionlayer)
    
    x2 = Add()(perceptionlayer)
    x2 = TrainableOffset()(x2)
    print(x2)
    x2 = Reshape((28, 28, 1))(x2)
    debugO = x2
    x2 = Conv2D(64, (6, 6))(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(64, (4, 4))(x2)
    x2 = Flatten()(x2)
    x2 = Dense(30, activation='relu')(x2)
    x2 = Dropout(0.25)(x2)
    x2 = Dense(10)(x2)
    predictions2 = Activation(activation="softmax")(x2)
    
    model = keras.Model(inputs=inpt, outputs=predictions2)
    
    if debug:
        return keras.Model(inputs=inpt, outputs=debugO)
    
    return model

def spartan_doubletraining():
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
        pass
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
        # x2 = Add()([x2, x3])
        # x2 = TrainableOffset()(x2)
        x = TrainableOffset(bias_initializer=keras.initializers.Constant(value=-0.5))(k)
        x2 = Activation(activation=Heaviside)(x)
        x2 = Reshape((28, 28, 1))(x2)
        debugO = x2
        x2 = Conv2D(64, (6, 6))(x2)
        x2 = Activation('relu')(x2)
        x2 = Conv2D(64, (4, 4))(x2)
        x2 = Flatten()(x2)
        x2 = Dense(40, activation='relu')(x2)
    # x2 = Dense(400, activation=Paranograd)(x2)
    x2 = Dense(10)(x2)
    predictions2 = Activation(activation="softmax")(x2)
    
    model = keras.Model(inputs=inpt, outputs=predictions2)
    
    if debug:
        return keras.Model(inputs=inpt, outputs=debugO)
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                                  train_end=60000,
                                                  test_start=0,
                                                  test_end=10000)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit(np.array(X_train), np.array(Y_train), 128, 2, verbose=1)
    # print(model.get_weights())
    return model


def antientropy(wm):
    return tf.multiply(wm, tf.log(K.abs(wm)))


def spartan_network_bb_v1(debug=False, w=0, rel=2):
    """
    98.21 to 97.02 in blackbox situation.
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
        pass
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
        # x2 = Add()([x2, x3])
        # x2 = TrainableOffset()(x2)
        x = TrainableOffset(bias_initializer=keras.initializers.Constant(value=-0.5), bias_regularizer=antientropy)(k)
        x2 = Activation(activation=Heaviside)(x)
        x2 = Reshape((28, 28, 1))(x2)
        debugO = x2
        x2 = Conv2D(64, (6, 6))(x2)
        x2 = Activation('relu')(x2)
        x2 = Conv2D(64, (4, 4))(x2)
        x2 = Flatten()(x2)
        x2 = Dense(40, activation='relu')(x2)
    # x2 = Dense(400, activation=Paranograd)(x2)
    x2 = Dense(10)(x2)
    predictions2 = Activation(activation="softmax")(x2)
    
    model = keras.Model(inputs=inpt, outputs=predictions2)
    
    if debug:
        return keras.Model(inputs=inpt, outputs=debugO)
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                                  train_end=60000,
                                                  test_start=0,
                                                  test_end=10000)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit(np.array(X_train), np.array(Y_train), 128, 4, verbose=1)
    model.evaluate(np.array(X_test), np.array(Y_test))
    # print(model.get_weights())
    return model