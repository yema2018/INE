#!usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import collections
import math
import random
import numpy as np
from six.moves import xrange
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.preprocessing import *
import gensim
import Transformer as tr


class newsfeature2vec:
    def __init__(self, walks, out_dir, map_dir, batch_size=2, embedding_size=100, skip_window=5,
                 num_skips=2, neg_samples=5, iter=3000000):
        self.iter = iter
        self.data_index = 0

        self.words = walks
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.neg_samples = neg_samples
        self.encode_content()
        self.build_dataset()
        # self.build_dataset_emb()
        self.build_model()

        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True)) as session:
            session.run(tf.global_variables_initializer())
            print("Initialized")
            average_loss = 0
            for step in xrange(self.iter):
                batch, labels = self.generate_batch()
                feed_dict = {self.train_inputs: batch, self.train_labels: labels, self.train: True}

                _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0 and step > 0:

                    average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step {} / {}".format(step, iter), ": ", average_loss)
                    average_loss = 0
            print('\n=======Training over========\n')
            out = np.zeros(shape=(len(self.dictionary), embedding_size))
            df_index = []
            i = 0
            for word, index in self.dictionary.items():
                df_index.append(word)
                out[i] = session.run(self.encode, feed_dict={self.train_inputs: np.array([index], dtype=np.int32),
                                                             self.train: False})
                i += 1
            out = pd.DataFrame(out, index=df_index)
            map = pd.read_csv(map_dir, index_col=[1], names=['id'])
            out = out.join(map).dropna()
            out = out.set_index(out['id'].astype(int)).drop(['id'], axis=1)
            out.to_csv(out_dir)


    def encode_content(self):
        # emb = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
        self.token = text.Tokenizer()
        self.token.fit_on_texts(self.words)
        self.word_size = len(self.token.word_index) + 1

        # self.emb_m = np.zeros(shape=(len(word_dict) + 1, 300), dtype=np.float32)
        # for word, id in word_dict.items():
        #     try:
        #         embedding = emb[word]
        #         self.emb_m[id] = embedding
        #     except:
        #         continue

    def build_dataset(self):
        self.count = []
        self.count.extend([list(item) for item in collections.Counter(self.words).most_common()])
        self.vocabulary_size = len(self.count)
        print(self.count[:20])
        self.encoded_mat = np.array(sequence.pad_sequences(self.token.texts_to_sequences([i[0] for i in self.count]),
                                                           maxlen=410), dtype=np.int32)
        print(self.encoded_mat[0])

        self.dictionary = dict()
        for word, _ in self.count:
            self.dictionary[word] = len(self.dictionary)
        self.data = list()
        for word in self.words:
            if word in self.dictionary:
                index = self.dictionary[word]
            else:
                index = 0
            self.data.append(index)
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

    def generate_batch(self):
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window
        batch = np.ndarray(shape=[self.batch_size], dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        for i in range(self.batch_size // self.num_skips):
            target = self.skip_window  # target label at the center of the buffer
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels

    def build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data.
            self.train_inputs = tf.placeholder(tf.int32, shape=[None])
            self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])
            self.train = tf.placeholder(tf.bool)

            # Look up embeddings for inputs.
            embeddings = tf.Variable(initial_value=self.encoded_mat, trainable=False)
            embed1 = tf.nn.embedding_lookup(embeddings, self.train_inputs)

            embedding_self = tf.Variable(initial_value=tf.truncated_normal([self.vocabulary_size, 512]),
                                         trainable=True)
            embed_self = tf.nn.embedding_lookup(embedding_self, self.train_inputs)
            # embed2 = tf.nn.embedding_lookup(tf.Variable(initial_value=self.emb_m, trainable=False), embed1)

            content_encode = tr.Transformer(num_layers=2, d_model=512, num_heads=8, dff=2048,
                                    input_vocab_size=self.word_size)(embed1, self.train)

            concat = content_encode + embed_self

            self.encode = BatchNormalization(epsilon=1e-6)(Dense(self.embedding_size)(Dropout(0.1)(concat, self.train)))

            nce_weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                    stddev=1.0 / math.sqrt(self.embedding_size)))
            nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]), dtype=tf.float32)

            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                 biases=nce_biases,
                                                 inputs=self.encode,
                                                 labels=self.train_labels,
                                                 num_sampled=self.neg_samples,
                                                 num_classes=self.vocabulary_size))

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            lr = tf.train.exponential_decay(0.001, global_step=self.global_step, decay_steps=int(0.05*self.iter),
                                            decay_rate=0.95)
            self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self.loss, global_step=self.global_step)




