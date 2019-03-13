from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# NOW 450% cooler, it is a tester for functions !
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
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport, set_log_level
import keras

from keras.layers import *
from keras.models import Model, Sequential
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K

save = True
load = len(sys.argv) > 1


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
    #
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

def transpose(x):
    return tf.cond(x>tf.zeros_like(x), true_fn=lambda :x, false_fn=lambda :tf.)
    
    
inputs = Input(shape=(2,2,1))


y=Activation('relu')(inputs)
# target=np.array([[[0.1,0.2],[0.3,0.4]],[[0.5,0.6],[0.7,0.8]]])


a = Lambda(lambda x: transpose(x), output_shape=(2,2,1))(y)


# y =
# returns [[[0.1 0.2]],[[0.5 0.6]]]
model = Model(inputs=inputs, outputs=a)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy')
train_set, valid_set, test_set, train_targets, valid_targets, test_targets = dataset_gen(powof10=2)
testar = train_set[0:10]
tagstar = train_targets[0:10]
target=[testar[0],testar[1]]
target=testar[0:2]
target=np.array([[[[3.],[0.]],[[10.],[2.]]],[[[3.],[5.]],[[10.],[2.]]]])
print(target)
print("res")
for i in model.predict(target,batch_size=1):
    print(i)
# print(model.predict(target,batch_size=1))
print("res")
#
#
# @tf.RegisterGradient("FakeHS")
# def heaviside_fake_grad(op, grad):
#     x = op.inputs[0]
#
#     #return (0*x+1)*grad
#
#     return tf.to_float(1 / (1 + (x * x))) * grad
#
#
# def Heaviside(x):
#     return 0.1* (tf.sign(x) + 1)
#
#
# sess = tf.Session()
# keras.backend.set_session(sess)
#
#
# x_train, valid_set, x_test, y_train, valid_targets, y_test = dataset_gen()
# batch_size = [1000,500,100]#[45,32,1,16]#[64,32,16]#  # [1000/(i*i) for i in range(5,10)][45,32]stat[4000,200,100,32,20]hyb
# nano_epochs = 2  # 15 stat 6 hyb
#
# inputs = Input(shape=(2,))
#
# # x = Dense(30, activation='relu')(inputs)
# # x = Dropout(0.25)(x)
# # y = Dense(150, activation='relu')(inputs)
# # y = Dropout(0.25)(y)
# g = tf.get_default_graph()
# with g.gradient_override_map({"Sign": "FakeHS"}):
#     z = Dense(150, activation=Heaviside)(inputs)
#     z = Dropout(0.25)(z)
#     y = Dense(150, activation=Heaviside)(inputs)
#     y = Dropout(0.25)(y)
#
# # z = Dense(150, activation="relu")(inputs)
# # z = Dropout(0.25)(z)
# x = Concatenate()([y, z])
# x = Dense(20, activation='relu')(x)
# x = Dense(30, activation='relu')(x)
# x = Dropout(0.25)(x)
# x = Dense(2, activation='sigmoid')(x)
# predictions = Activation(activation="softmax")(x)
#
# model = Model(inputs=inputs, outputs=predictions)
#
#
# def dnn_model(logits=False, input_ph=None, nb_classes=2):
#     """
#     Defines a CNN model using Keras sequential model
#     :param logits: If set to False, returns a Keras model, otherwise will also
#                     return logits tensor
#     :param input_ph: The TensorFlow tensor for the input
#                     (needed if returning logits)
#                     ("ph" stands for placeholder but it need not actually be a
#                     placeholder)
#     :param img_rows: number of row in the image
#     :param img_cols: number of columns in the image
#     :param channels: number of color channels (e.g., 1 for MNIST)
#     :param nb_filters: number of convolutional filters per layer
#     :param nb_classes: the number of output classes
#     :return:
#     """
#     model = Sequential()
#
#     # Define the layers successively (convolution layers are version dependent)
#     input_shape = (1,2)
#
#     layers = [Dense(20, activation='relu',input_shape=(2,)),
#               Dense(20, activation='relu'),
#               Dense(20, activation='relu'),
#               Dense(nb_classes)]
#
#     for layer in layers:
#         model.add(layer)
#
#     if logits:
#         logits_tensor = model(input_ph)
#     model.add(Activation('softmax'))
#
#     if logits:
#         return model, logits_tensor
#     else:
#         return model
#
# # now we do the same model, with relus only
# #
# #
# #
# # vvvvvvvvvvvvvvvvvvRELUSvvvvvvvvvvvvvvvvvvvv
# model=dnn_model()
#
# if not load:
#     # This creates a model that includes
#     # the Input layer and three Dense layers
#     optt = keras.optimizers.SGD(lr=10.02, decay=0.99)
#     model.compile(optimizer='rmsprop',
#                   loss="categorical_crossentropy",
#                   metrics=['accuracy'])
#
#     print('''
#     -------- START OF HYBRID -----------
#     ''')
#
#     for i in batch_size:
#         print(int(i))
#         model.fit(x_train, y_train,
#                   batch_size=int(i),
#                   epochs=nano_epochs, shuffle=True,
#                   verbose=1,
#                   validation_data=(valid_set, valid_targets))
#
#     score = model.evaluate(x_test, y_test, verbose=0)
#
#     print('Test loss:', score[0])
#     print('Test accuracy:', score[1])
#     print('''
#     -------- END OF HYBRID -----------
#     ''')
#     print("RELU VERSION BELOW !!!")
#
#     for i in batch_size:
#         print(int(i))
#
#     if save:
#         model.save_weights("HybKerasWeights")
#
#         keras.models.save_model(model, "HybKerasModel")
#     print('Test loss:', score[0])
#     print('Test accuracy:', score[1])
#
# else:
#     pass
#     model.load_weights("HybKerasWeights")
#
#
# #Prepare the Cleverhans FGSM attack
# print("Attacking the Heaviside-ReLU Hybrid version...")
# xh = tf.placeholder(tf.float32, shape=(None,2,))
# yh = tf.placeholder(tf.float32, shape=(None,2,))
# predsHYB = model(xh)
# K.set_learning_phase(False)
# # model_train(sess, x, y, preds, x_train, y_train, evaluate=evaluate,
# #                     args=train_params, save=True, rng=rng)
# predsHYB = model(xh)
# eval_params = {'batch_size': 100}
# acc = model_eval(sess, xh, yh, predsHYB, x_test, y_test, args=eval_params)
# print('Test accuracy on legitimate examples: %0.4f' % acc)
# wrapHYB = KerasModelWrapper(model)
# fgsmHYB = FastGradientMethod(wrapHYB)
# fgsm_params = {'eps': 0.0,
#                'clip_min': -1.,
#                'clip_max': 1.}
# adv_xHYB = fgsmHYB.generate(xh, **fgsm_params)
# adv_xHYB = tf.stop_gradient(adv_xHYB)
# preds_advHYB = model(adv_xHYB)
# eval_par = {'batch_size': 100}
# acc = model_eval(sess, xh, yh, preds_advHYB, x_test, y_test, args=eval_par)
# print('Test accuracy on adversarial examples: %0.4f\n' % acc)
# acc = model_eval(sess, xh, yh, predsHYB, x_test, y_test, args=eval_par)
# print('Test accuracy on adversarial examples: %0.4f\n' % acc)
#
#
# #Do it again on the ReLU
