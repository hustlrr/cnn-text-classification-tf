# coding=utf-8

'''
@author:lruoran

created at 17-3-3 下午3:11
'''

import gensim
import numpy as np
from tensorflow.contrib import learn

import data_helpers

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels("./data/rt-polaritydata/rt-polarity.pos",
                                              "./data/rt-polaritydata/rt-polarity.neg")

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
vocab_dict = vocab_processor.vocabulary_._mapping
sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
vocabulary = [word for word, id in sorted_vocab]

# load word2vec model
print("Loading word2vec model...")
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(r"./data/GoogleNews-vectors-negative300.bin",
                                                                 binary=True, encoding="utf-8")
word2vec_embedded = []
for word in vocabulary:
    if word in word2vec_model:
        word2vec_embedded.append(word2vec_model[word])
word2vec_embedded = np.array(word2vec_embedded, dtype=np.float32)
del word2vec_model

var_of_each_dim = np.var(word2vec_embedded, axis=0) * 3
var_of_each_dim = np.sqrt(var_of_each_dim)
np.savetxt("data/rt-polaritydata/uniform_random_a.npy", var_of_each_dim)