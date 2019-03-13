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
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport, set_log_level
import keras

from keras.layers import Input, Dense, Dropout, Concatenate, Activation
from keras.models import Model
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K

save = True
load = len(sys.argv)>1

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


@tf.RegisterGradient("FakeHS")
def heaviside_fake_grad(op, grad):
    x = op.inputs[0]
    
    #return (0*x+1)*grad
    
    return tf.to_float(1 / (1 + (x * x))) * grad


def Heaviside(x):
    return 0.1* (tf.sign(x) + 1)


sess = tf.Session()
keras.backend.set_session(sess)


x_train, valid_set, x_test, y_train, valid_targets, y_test = dataset_gen()
batch_size = [1000,500,100]#[45,32,1,16]#[64,32,16]#  # [1000/(i*i) for i in range(5,10)][45,32]stat[4000,200,100,32,20]hyb
nano_epochs = 2  # 15 stat 6 hyb

inputs = Input(shape=(2,))

# x = Dense(30, activation='relu')(inputs)
# x = Dropout(0.25)(x)
# y = Dense(150, activation='relu')(inputs)
# y = Dropout(0.25)(y)
g = tf.get_default_graph()
with g.gradient_override_map({"Sign": "FakeHS"}):
    z = Dense(150, activation=Heaviside)(inputs)
    z = Dropout(0.25)(z)
    y = Dense(150, activation=Heaviside)(inputs)
    y = Dropout(0.25)(y)

# z = Dense(150, activation="relu")(inputs)
# z = Dropout(0.25)(z)
x = Concatenate()([y, z])
x = Dense(20, activation='relu')(x)
x = Dense(30, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(2, activation='sigmoid')(x)
predictions = Activation(activation="softmax")(x)

model = Model(inputs=inputs, outputs=predictions)

# now we do the same model, with relus only
#
#
#
# vvvvvvvvvvvvvvvvvvRELUSvvvvvvvvvvvvvvvvvvvv

inputs2 = Input(shape=(2,))

# x = Dense(30, activation='relu')(inputs)
# x = Dropout(0.25)(x)
y2 = Dense(150, activation='relu')(inputs2)
y2 = Dropout(0.25)(y2)
# g = tf.get_default_graph()
# with g.gradient_override_map({"Sign": "FakeHS"}):
#     z = Dense(150, activation=Heaviside)(inputs)

z2 = Dense(150, activation="relu")(inputs2)
z2 = Dropout(0.25)(z2)
x2 = Concatenate()([y2, z2])
x2 = Dense(20, activation='relu')(x2)
x2 = Dense(30, activation='relu')(x2)
x2 = Dropout(0.25)(x2)
x2 = Dense(2, activation='sigmoid')(x2)
predictions2 = Activation(activation="softmax")(x2)

model2 = Model(inputs=inputs2, outputs=predictions2)
if not load:
    # This creates a model that includes
    # the Input layer and three Dense layers
    optt = keras.optimizers.SGD(lr=10.02, decay=0.99)
    model.compile(optimizer='rmsprop',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    model2.compile(optimizer='rmsprop',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    
    print('''
    -------- START OF HYBRID -----------
    ''')
    
    for i in batch_size:
        print(int(i))
        model.fit(x_train, y_train,
                  batch_size=int(i),
                  epochs=nano_epochs, shuffle=True,
                  verbose=1,
                  validation_data=(valid_set, valid_targets))
        
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('''
    -------- END OF HYBRID -----------
    ''')
    print("RELU VERSION BELOW !!!")
    
    for i in batch_size:
        print(int(i))
        model2.fit(x_train, y_train,
                  batch_size=int(i),
                  epochs=nano_epochs, shuffle=True,
                  verbose=1,
                  validation_data=(valid_set, valid_targets))
    
    score = model2.evaluate(x_test, y_test, verbose=0)
    if save:
        model.save_weights("HybKerasWeights")
        model2.save_weights("ReLUKerasWeights")
        keras.models.save_model(model, "HybKerasModel")
        keras.models.save_model(model2, "ReLUKerasModel")
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

else:
    model.load_weights("HybKerasWeights")
    model2.load_weights("ReLUKerasWeights")

#Prepare the Cleverhans FGSM attack
print("Attacking the Heaviside-ReLU Hybrid version...")
xh = tf.placeholder(tf.float32, shape=(None,2,))
yh = tf.placeholder(tf.float32, shape=(None,2,))

K.set_learning_phase(False)

predsHYB = model(xh)
eval_params = {'batch_size': 100}
acc = model_eval(sess, xh, yh, predsHYB, x_test, y_test, args=eval_params)
print('Test accuracy on legitimate examples: %0.4f' % acc)
wrapHYB = KerasModelWrapper(model)
fgsmHYB = FastGradientMethod(wrapHYB, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_xHYB = fgsmHYB.generate(xh, **fgsm_params)
adv_xHYB = tf.stop_gradient(adv_xHYB)
preds_advHYB = model(adv_xHYB)
eval_par = {'batch_size': 100}
acc = model_eval(sess, xh, yh, preds_advHYB, x_test, y_test, args=eval_par)
print('Test accuracy on adversarial examples: %0.4f\n' % acc)

#Do it again on the ReLU

print("Attacking the ReLU version...")
xs = tf.placeholder(tf.float32, shape=(2,))
ys = tf.placeholder(tf.float32, shape=(2,))

predsRLU = model2(xh)
eval_params = {'batch_size': 100}
acc = model_eval(sess, xh, yh, predsRLU, x_test, y_test, args=eval_params)
print('Test accuracy on legitimate examples: %0.4f' % acc)
wrapRLU = KerasModelWrapper(model2)
fgsmRLU = FastGradientMethod(wrapRLU, sess=sess)
adv_xRLU = fgsmRLU.generate(xh, **fgsm_params)
adv_xRLU = tf.stop_gradient(adv_xRLU)
preds_advRLU = model2(adv_xRLU)
eval_par = {'batch_size': 100}
acc = model_eval(sess, xh, yh, preds_advRLU, x_test, y_test, args=eval_par)
print('Test accuracy on adversarial examples: %0.4f\n' % acc)


