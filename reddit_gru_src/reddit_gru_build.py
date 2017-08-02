# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:43:09 2017

@author: Sunrise
"""
import reddit_gru_utils as redgu

from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping

net_name = 'reddit_text_bot'
vocab_size = 10000
batch_size = 96
rnn_dim_1 = 256
rnn_dim_2 = 128
epochs = 1
lr = 0.001


x_train, y_train, i_to_w, w_to_i, max_sentence_length = redgu.load_data(vocab_size=vocab_size)
del i_to_w

w_dim = 300
n_words = len(w_to_i)

x_train, y_train, embedding_weights = redgu.structure_data_embw(x_train, y_train, n_words, w_dim, w_to_i, max_sentence_length)
del w_to_i

# build the model
print("Building model for {}...".format(net_name))
model = Sequential()
print("Adding embedding layer...")
model.add(Embedding(n_words+1, w_dim, mask_zero=True, input_length=max_sentence_length, weights=[embedding_weights]))
print("Embedding layer added.")
del embedding_weights
print("Adding GRU layer...")
model.add(GRU(rnn_dim_1, return_sequences=True, dropout=0.5))
print("GRU layer added.")
print("Adding GRU layer...")
model.add(GRU(rnn_dim_2, return_sequences=True, dropout=0.25))
print("GRU layer added.")
print("Adding softmax layer...")
model.add(TimeDistributed(Dense(n_words+1, activation='softmax'), input_shape=(max_sentence_length, w_dim)))
print("Softmax layer added.")


# compile the model
optimizer = RMSprop(lr=lr)
print("Compiling model {}...".format(net_name))
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              sample_weight_mode="temporal")

print(model.summary())


n_batches, last_batch_size = redgu.get_training_batches(x_train.shape[0], batch_size)

# train the model
filepath = "../nets/{}_e{{epoch:02d}}.hdf5".format(net_name)
checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True)
early_stop = EarlyStopping(monitor='loss', patience=1)
model.fit_generator(redgu.data_to_generator(x_train, y_train, n_batches, batch_size, last_batch_size, vocab_size), n_batches, epochs, callbacks=[checkpoint, early_stop])
