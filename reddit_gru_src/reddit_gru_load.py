# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 14:07:55 2017

@author: Sunrise
"""
#import numpy as np
import reddit_gru_utils as redgu

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping


net_name = 'reddit_text_bot'
vocab_size = 10000
loading_epoch = 15
filepath = "../nets/%s_e%02d.hdf5" % (net_name, loading_epoch-1)
batch_size = 192
epochs = 16
lr = 0.00005

x_train, y_train, i_to_w, w_to_i, max_sentence_length = redgu.load_data(vocab_size=vocab_size)
del i_to_w

w_dim = 300
n_words = len(w_to_i)

x_train, y_train = redgu.structure_data_embw(x_train, y_train, n_words, w_dim, w_to_i, max_sentence_length, loaded_train=True)
del w_to_i

# load the model
print("Loading model {}...".format(net_name))
model = load_model(filepath)
print("Model loaded.")
print(model.summary())

# train the model
n_batches, last_batch_size = redgu.get_training_batches(x_train.shape[0], batch_size)

model.optimizer.lr.set_value(lr)
# train the model
filepath = "../nets/{}_e{{epoch:02d}}.hdf5".format(net_name)
checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True)
early_stop = EarlyStopping(monitor='loss', patience=1)
model.fit_generator(redgu.data_to_generator(x_train, y_train, n_batches, batch_size, last_batch_size, vocab_size), n_batches, epochs, initial_epoch=loading_epoch, callbacks=[checkpoint, early_stop])