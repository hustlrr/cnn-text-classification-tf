# coding=utf-8

'''
@author:lruoran

created at 17-3-3 下午4:20
'''

import matplotlib.pylab as plt


def plot_loss_acc(tr_plot, dev_plot):
    tr_step, tr_loss, _ = zip(*tr_plot)
    d_step, d_loss, _ = zip(*dev_plot)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(tr_step, tr_loss, "b")
    plt.plot(d_step, d_loss, "r")
    plt.xlabel("step", multialignment='right')
    plt.ylabel("loss")

    plt.xlim(0, max(tr_step))
    plt.ylim(min(tr_loss + d_loss), max(tr_loss + d_loss))

    tr_step, _, tr_acc = zip(*tr_plot)
    d_step, _, d_acc = zip(*dev_plot)
    plt.figure(1)
    plt.subplot(212)
    plt.plot(tr_step, tr_acc, "b")
    plt.plot(d_step, d_acc, "r")
    plt.xlabel("step")
    plt.ylabel("acc")
    plt.xlim(0, max(tr_step))
    plt.ylim(min(tr_acc + d_acc), max(tr_acc + d_acc))
    plt.show()
    plt.savefig("data/rt-polaritydata/acc.png")


import data_helpers
import numpy as np
from tensorflow.contrib import learn
import gensim

if __name__ == '__main__':
    x_text, y = data_helpers.load_data_and_labels("./data/rt-polaritydata/rt-polarity.pos",
                                                  "./data/rt-polaritydata/rt-polarity.neg")
    np.random.seed(10)

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
    uniform_border = np.loadtxt("data/rt-polaritydata/uniform_random_a.npy")
    word2vec_embedded = []
    for word in vocabulary:
        if word in word2vec_model:
            word2vec_embedded.append(word2vec_model[word])
        else:
            word2vec_embedded.append(
                np.random.uniform(low=-uniform_border, high=uniform_border, size=300))
    word2vec_embedded = np.array(word2vec_embedded, dtype=np.float32)

