# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 19:49:16 2017

@author: Sunrise
"""

import reddit_gru_utils as redgu

from keras.models import load_model

net_name = 'reddit_text_bot'
n_sentences_desired = 3
vocab_size = 10000
loading_epoch = 15
filepath = "../nets/%s_e%02d.hdf5" % (net_name, loading_epoch-1)
n_variance = 3

x_train, y_train, i_to_w, w_to_i, max_sentence_length = redgu.load_data(vocab_size=vocab_size)

# load the model
print("Loading model {}...".format(net_name))
model = load_model(filepath)
print("Model loaded.")
#print(model.summary())

for i in range(n_sentences_desired):
    redgu.generate_sentence(model, w_to_i, i_to_w, max_sentence_length=max_sentence_length, n_vary=n_variance, vocab_size=vocab_size)