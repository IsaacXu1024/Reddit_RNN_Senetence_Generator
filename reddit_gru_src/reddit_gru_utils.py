# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:31:40 2017

@author: Sunrise
"""

import numpy as np
import nltk
import nltk.data
import itertools
import operator
import gensim
#import os

from keras.preprocessing import sequence
#from keras.preprocessing.text import text_to_word_sequence
    
sentence_start_token = "sentencestarttoken"
sentence_end_token = "sentenceendtoken"
unknown_token = "unknowntoken"

default_vocab_size = 20000

"""
Data structuring functions
"""

def load_data(filename = "../data/comments.txt", vocab_size=default_vocab_size, min_sentence_words=5, max_sentence_words=67):
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = []
    
    print("Reading file...")
    with open(filename, 'r', encoding='utf-8') as f:
        comment_list = f.readlines()
        for comment in comment_list:
            tokenized_comment = tokenizer.tokenize(comment.lower())
            for sentence in tokenized_comment:
                sentences.append("%s %s %s" % (sentence_start_token, sentence, sentence_end_token))
                
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    filtered_tokenized_sentences = [sentence for sentence in tokenized_sentences if len(sentence) >= min_sentence_words and len(sentence) <= max_sentence_words]
    print("Filtered sentences for sentences with more than {} words and less than {}.".format(min_sentence_words-2, max_sentence_words-2))
    
    word_freq = nltk.FreqDist(itertools.chain(*filtered_tokenized_sentences))
    #print("Found %d unique word tokens." % len(word_freq.items()))
    
    # vocab is made by taking the word token x[0] and the frequency x[1] then sorting it by frequency and taking the most frequent vocab_size-1 elements
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocab_size-1]
    #print("Vocabulary size is %d." % vocab_size)
    #print("The least frequent word in vocabulary is '%s', which appeared %d times." % (vocab[-1][0], vocab[-1][1]))
    
    # sorted_vocab takes every element in vocab (every element is a pair of word and frequency), then sorts it by the idx = 1-th item, ie) frequency
    # note that sorted_vocab is not sorted in reverse where as vocab was, ie) start and end tokens are at the bottom of the list now
    sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
    # creates a list of words in the order determined by sorted_vocab and adds the mask tag as element 0 and unknown_token as element 1
    index_to_word = [unknown_token] + [x[0] for x in sorted_vocab]
    # creates a dictionary such that the word in index to word is mapped to its index, note that indices start at 1 to 10000 to account for mask value of 0.
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word, 1)])
    
    # replace words in filtered_tokenized_sentences with unknown_token
    for i, sentence in enumerate(filtered_tokenized_sentences):
        # note that the checks are done in word_to_index, the dictionary, a hashed object, to speed up the program
        filtered_tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sentence]
    
    # x_train is just the translation for each word in filtered_tokenized_sentences to its index in word_to_index and without the last element (end token)
    x_train = np.asarray([np.asarray([word_to_index[w] for w in sentence[:-1]]) for sentence in filtered_tokenized_sentences])
    # y_train is the same as x_train except without the first start token instead of the last one
    y_train = np.asarray([np.asarray([word_to_index[w] for w in sentence[1:]]) for sentence in filtered_tokenized_sentences])
    
    max_sentence_length = 0
    for training_sentence in x_train:
        if max_sentence_length < len(training_sentence):
            max_sentence_length = len(training_sentence)
    
    return x_train, y_train, index_to_word, word_to_index, max_sentence_length

def structure_data_embw(x_train, y_train, n_words, w_dim, w_to_i, max_sentence_length, loaded_train=False):
    if loaded_train == False:
        print("Loading GoogleNews-vectors...")
        word_model = gensim.models.KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)
        
        print("Relating indices to preloaded vectors...")
        embedding_weights = np.zeros((n_words+1, w_dim))
        
        for w, i in w_to_i.items():
            try:
                embedding_weights[i,:] = word_model[w]
            except:
                embedding_weights[i,:] = np.random.uniform(-np.sqrt(1./w_dim), np.sqrt(1./w_dim), w_dim)
                #embedding_weights[i,:] = np.zeros(w_dim)
        del word_model
        print("Indices mapped to preloaded vectors.")
    
    # pad the data with index 0 to ensure all sentences are same length
    print("Padding X, Y training data...")        
    x_train = sequence.pad_sequences(x_train, maxlen=max_sentence_length)
    y_train = sequence.pad_sequences(y_train, maxlen=max_sentence_length)
    print("Padding finished.")
    
    print("Structuring Y training data...")
    #x_train = x_train.reshape(x_train.shape[0], max_sentence_length, 1)
    y_train = y_train.reshape(y_train.shape[0], max_sentence_length, 1)
    print("Training data structured.")
    
    if loaded_train == False:
        return x_train, y_train, embedding_weights
    
    return x_train, y_train

"""
Training functions
"""

def get_training_batches(data_size, batch_size):
    nbatches = int(data_size/batch_size)
    last_batch_size = data_size-nbatches*batch_size
    return nbatches, last_batch_size  

def convert_to_categorical(y, vocab_size=default_vocab_size):
    word_cat_list = np.identity(vocab_size+1)        
    c_y = np.asarray([np.asarray([word_cat_list[w[0]] for w in s]) for s in y])
    del word_cat_list
    return c_y

def data_to_generator(x_train, y_train, n_batches, batch_size, last_batch_size, vocab_size):
    while True:
        for i in range(n_batches):
            init_i = i*batch_size
            
            if i == n_batches-1 and last_batch_size != 0:
                fin_i = init_i + last_batch_size
            elif i == n_batches-1 and last_batch_size == 0:
                pass
            else:
                fin_i = init_i + batch_size            
            x_batch = x_train[init_i:fin_i]
            y_batch = convert_to_categorical(y_train[init_i:fin_i], vocab_size=vocab_size)
            yield x_batch, y_batch
        
"""
Sentence generation functions
"""

def generate_sentence(model, w_to_i, i_to_w, n_vary=3, max_sentence_length=66, vocab_size=default_vocab_size):
    sentence = np.zeros(max_sentence_length)
    print("Generating text...")
    # vocab_size-1 in the net and the w_to_i dict indicate sentence start token
    start_idx = vocab_size-1
    
    sentence[-1] = start_idx
    # put the max_sentence_len size array inside another empty one to process through neural net
    sentence = np.asarray([sentence])
    sentence_len = 1
    
    pw_idx = predict_next_word(model, sentence, sentence_len, n_vary=n_vary, vocab_size=vocab_size)
    
    # index vocab_size is the sentence end token
    while sentence[0][-1] != vocab_size and sentence_len <= max_sentence_length:
        sentence = np.asarray([np.roll(sentence[0], -1)])
        sentence[0][-1] = pw_idx
        sentence_len += 1        
        pw_idx = predict_next_word(model, sentence, sentence_len, n_vary=n_vary, vocab_size=vocab_size)
        
    if sentence_len > max_sentence_length:
        print("Max sentence length reached in sentence generation, check training.")
    
    print_sentence(i_to_w, sentence)

def predict_next_word(model, sentence, sentence_len, n_vary=3, vocab_size=default_vocab_size):
    prediction = model.predict(sentence)
    
    # Predicted word is last element of the only element in the predictions array
    pw = prediction[0][-1]

    # Note that the predicted indices includes 0th position which is the masked value
    pwm_idx = np.argmax(pw)
    
    if pwm_idx == 0:
        print("Possible error detected: predicted word index is 0, check training.")
    
    # Many ways to start a sentence, fewer ways to continue it
    if pwm_idx != vocab_size:
        if sentence_len == 1:
            pws_idx = list(np.argpartition(pw, -n_vary*3)[-n_vary*5:])
        else:
            pws_idx = list(np.argpartition(pw, -n_vary)[-n_vary:])
        if 1 in pws_idx and n_vary >= 2:
            pws_idx.remove(1)
        pw_idx = np.random.choice(pws_idx)
        return pw_idx
    
    return pwm_idx

def print_sentence(i_to_w, sentence):
    # 1 has to be subtracted from each w_idx to account for masking index 0
    word_sentence = [i_to_w[int(w_idx-1)] for w_idx in sentence[0] if w_idx != 0]
    word_sentence.pop(0)
    word_sentence.pop(-1)
    unfiltered_sentence = list(' '.join(word_sentence))
    # Cleaning sentence
    unfiltered_sentence[0] = unfiltered_sentence[0].upper()
    replace_tags(unfiltered_sentence, list(" i "), list(' I '))
    replace_tags(unfiltered_sentence, list(' ,'), list(','))
    replace_tags(unfiltered_sentence, list(' .'), list('.'))
    replace_tags(unfiltered_sentence, list(' !'), list('!'))
    replace_tags(unfiltered_sentence, list(' ?'), list('?'))
    replace_tags(unfiltered_sentence, list(" n't"), list("n't"))
    replace_tags(unfiltered_sentence, list(" '"), list("'"))
    replace_tags(unfiltered_sentence, list("``"), list('"'))
    replace_tags(unfiltered_sentence, list("''"), list('"'))
    replace_tags(unfiltered_sentence, list(' " '), list(' "'))
    clean_sentence = ''.join(unfiltered_sentence)
    
    print(clean_sentence)
    
"""
Sentence cleaner functions (taken from my crawler projects)
"""

def find_tags(p_text, tags):
    length = len(tags)
    for i, v in enumerate(p_text):
        if v == tags[0] and p_text[i:i+length] == tags:
            yield i, i+length
            
def replace_tags(p_text, tag, replacement_tag=[]):
    tags = find_tags(p_text, tag)
    for start, end in tags:
        p_text[start:end] = replacement_tag

"""
Legacy functions
"""

"""

def train_model(model, net_name, x_train, y_train, true_epochs, n_batches, large_batch_size, last_large_batch_size, last_trained_epoch=1, last_trained_large_batch=0, mini_batch_size=32, vocab_size=default_vocab_size):
    print("Training model {}...".format(net_name))

    for epoch in range(last_trained_epoch-1,true_epochs):
        print("Training epoch: {}/{}...".format(epoch+1, true_epochs))
        for i in range(last_trained_large_batch, n_batches):
            print("Training epoch: {}, large_batch: {}/{}...".format(epoch+1, i+1, n_batches))
            init_i = i*large_batch_size
            if i == n_batches-1 and last_large_batch_size != 0:
                fin_i = init_i + last_large_batch_size
            elif i == n_batches-1 and last_large_batch_size == 0:
                pass
            else:
                fin_i = init_i + large_batch_size
                
            x_batch = x_train[init_i:fin_i]
            y_batch = convert_to_categorical(y_train[init_i:fin_i], vocab_size=vocab_size)
            
            model.fit(x_batch, y_batch, batch_size=mini_batch_size, epochs=1)
            
            if i+1 == n_batches:
                last_trained_large_batch = 0
                filepath="../nets/{}_epoch{}_large-batch{}.h5".format(net_name, epoch+2, last_trained_large_batch)
            else:
                filepath="../nets/{}_epoch{}_large-batch{}.h5".format(net_name, epoch+1, i+1)
            model.save(filepath)
            
            try:
                os.remove("../nets/{}_epoch{}_large-batch{}.h5".format(net_name, epoch+1, i))
            except FileNotFoundError:
                print("No older net version detected. No files removed.")
                pass
            print("{} saved.".format(net_name))
"""    