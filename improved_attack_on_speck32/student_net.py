import speck as sp

import numpy as np

from pickle import dump

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model, load_model, model_from_json
from keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2

bs = 5000
wdir = './freshly_trained_nets/'


def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return(res)


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)


# initial setting: num_filters=32, d1 = 64, d2 = 64
# v1 : num_filters=16, d1 = 16, d2 = 16
# v2 : num_filters=12, d1 = 12, d2 = 12
# make residual tower of convolutional blocks
def make_resnet(num_blocks=2, num_filters=16, num_outputs=1, d1=16, d2=16, word_size=16, ks=3, depth=5, reg_param=0.0001, final_activation='sigmoid'):
    #Input and preprocessing layers
    inp = Input(shape=(num_blocks * word_size * 2,))
    rs = Reshape((2 * num_blocks, word_size))(inp)
    # rs = Reshape((2, num_blocks * word_size))(inp)
    perm = Permute((2,1))(rs)
    #add a single residual layer that will expand the data to num_filters channels
    #this is a bit-sliced layer
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    #add residual blocks
    shortcut = conv0
    for i in range(depth):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
    #add prediction head
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)

    model.summary()
    return(model)


def extract_sensitive_bits(raw_x, bits=[14,13,12,11,10,9,8,7]):
    # get new-x according to sensitive bits
    id0 = [15 - v for v in bits]
    # print('id0 is ', id0)
    id1 = [v + 16 * i for i in range(4) for v in id0]
    # print('id1 is ', id1)
    new_x = raw_x[:, id1]

    return new_x


def train_speck_distinguisher(num_epochs, num_rounds=5, depth=1, diff=(0x808000, 0x808004), bits=[11,10], folder='./'):
    # generate training data
    raw_x, raw_y = sp.make_train_data(10**7, num_rounds, diff=diff)
    extracted_x = extract_sensitive_bits(raw_x, bits=bits)
    # generate eval data
    raw_x_eval, raw_y_eval = sp.make_train_data(10**6, num_rounds, diff=diff)
    extracted_x_eval = extract_sensitive_bits(raw_x_eval, bits=bits)

    #create student network
    net = make_resnet(num_blocks=2, word_size=len(bits), depth=depth, reg_param=10**-5)
    # net = make_resnet(num_blocks=1, word_size=len(bits), depth=depth, reg_param=10 ** -5)
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    #set up model checkpoint
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5')
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.001, 0.0001))
    #train and evaluate
    h = net.fit(extracted_x, raw_y, epochs=num_epochs, batch_size=bs, shuffle=True,
                validation_data=(extracted_x_eval, raw_y_eval), callbacks=[lr, check])
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_acc'])
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_loss'])
    dump(h.history, open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p','wb'))
    print("Best validation acc: ", np.max(h.history['val_acc']))

    # save model
    net.save(folder + 'student_' + str(num_rounds) + '_distinguisher.h5')

    return(net, h)
