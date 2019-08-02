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
from Classification import SVMclassifier

class newsfeature2vec:
    def __init__(self, walks, out_dir, map_dir, embedding_size=200, skip_window=5,
                 neg_samples=5, epoch=2, batch_size=64):

        self.data_index = 0
        self.walks = walks
        self.batch_size = batch_size
        self.words = []
        for walk in walks:
            self.words.extend(walk)
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.neg_samples = neg_samples
        self.encode_content()
        self.build_dataset()
        # self.build_dataset_emb()
        self.total = np.math.ceil((len(walks[0]) - 2 * skip_window) * 2 * skip_window / batch_size) * epoch * len(walks)
        self.build_model()

        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True)) as session:
            session.run(tf.global_variables_initializer())
            print("Initialized")
            average_loss = 0
            batch = self.generate_batch(epoch)
            f1_best = 0
            for step, bat in enumerate(batch):
                # print(bat)
                batch, labels = bat
                feed_dict = {self.train_inputs: batch, self.train_labels: labels, self.train: True}

                _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val

                if (step % 16000 == 0 and step > 0) or step == self.total-1:

                    average_loss /= 16000
                    print("Average loss at step {} / {}".format(step, self.total), ": ", average_loss)
                    out = np.zeros(shape=(len(self.dictionary), embedding_size))
                    df_index = []
                    i = 0
                    for word, index in self.dictionary.items():
                        df_index.append(word)
                        out[i] = session.run(self.encode,
                                             feed_dict={self.train_inputs: np.array([index], dtype=np.int32),
                                                        self.train: False})
                        i += 1
                    out = pd.DataFrame(out, index=df_index)
                    map = pd.read_csv(map_dir, index_col=[1], names=['id'])
                    out = out.join(map).dropna()
                    out = out.set_index(out['id'].astype(int)).drop(['id'], axis=1)
                    # out.to_csv(out_dir)
                    f1, _ = SVMclassifier(out, embedding_size, group='dblp/group.txt', test_ratio=0.9)
                    if f1_best < f1:
                        f1_best = f1
                        out.to_csv(out_dir)
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('highest f1 {:g}'.format(f1_best))
                    average_loss = 0
            print('\n=======Training over========\n')
            # out = np.zeros(shape=(len(self.dictionary), embedding_size))
            # df_index = []
            # i = 0
            # for word, index in self.dictionary.items():
            #     df_index.append(word)
            #     out[i] = session.run(self.encode, feed_dict={self.train_inputs: np.array([index], dtype=np.int32),
            #                                                  self.train: False})
            #     i += 1
            # out = pd.DataFrame(out, index=df_index)
            # map = pd.read_csv(map_dir, index_col=[1], names=['id'])
            # out = out.join(map).dropna()
            # out = out.set_index(out['id'].astype(int)).drop(['id'], axis=1)
            # out.to_csv(out_dir)


    def encode_content(self):
        #emb = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
        self.token = text.Tokenizer()
        self.token.fit_on_texts(self.words)
        self.word_size = len(self.token.word_index) + 1

        #word_dict = self.token.word_index
        """     
        self.emb_m = np.zeros(shape=(len(word_dict) + 1, 300), dtype=np.float32)
        for word, id in word_dict.items():
            try:
                embedding = emb[word]
                self.emb_m[id] = embedding
            except:
                continue
        """
    def build_dataset(self):
        self.count = []
        self.count.extend([list(item) for item in collections.Counter(self.words).most_common()])
        self.vocabulary_size = len(self.count)
        print(self.count[:20])
        self.encoded_mat = np.array(sequence.pad_sequences(self.token.texts_to_sequences([i[0] for i in self.count]),
                                                           maxlen=None, padding='post'), dtype=np.int32)
        print(self.encoded_mat[0])

        self.dictionary = dict()
        for word, _ in self.count:
            self.dictionary[word] = len(self.dictionary)
        self.data = list()
        for walk in self.walks:
            data = []
            for word in walk:
                if word in self.dictionary:
                    index = self.dictionary[word]
                else:
                    index = 0
                data.append(index)
            self.data.append(data)

        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

    def generate_batch(self, epoch):
        num_skips = 2 * self.skip_window
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]

        for _ in range(epoch):
            for da in self.data:
                data_index = 0
                sam_num = len(da) - span + 1
                batch_num = np.math.ceil(sam_num * num_skips / self.batch_size)

                batch = np.ndarray(shape=[sam_num * num_skips], dtype=np.int32)
                labels = np.ndarray(shape=(sam_num * num_skips, 1), dtype=np.int32)
                buffer = collections.deque(maxlen=span)
                for _ in range(span):
                    buffer.append(da[data_index])
                    data_index = data_index + 1
                for i in range(sam_num):
                    target = self.skip_window  # target label at the center of the buffer
                    targets_to_avoid = [self.skip_window]
                    for j in range(num_skips):
                        while target in targets_to_avoid:
                            target = random.randint(0, span - 1)
                        targets_to_avoid.append(target)
                        batch[i * num_skips + j] = buffer[self.skip_window]
                        labels[i * num_skips + j, 0] = buffer[target]
                    buffer.append(da[data_index])
                    data_index = (data_index + 1) % len(da)
                for b in range(batch_num):
                    start_index = b * self.batch_size
                    end_index = min((b + 1) * self.batch_size, sam_num * num_skips)
                    batch1 = batch[start_index: end_index]
                    labels1 = labels[start_index: end_index]
                    yield batch1, labels1

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

            # embedding_self = tf.Variable(initial_value=tf.truncated_normal([self.vocabulary_size, 100]),
            #                          trainable=True)
            # embed_self = tf.nn.embedding_lookup(embedding_self, self.train_inputs)

            mask1 = tr.create_padding_mask(embed1)

            mask2 = tr.create_output_mask(embed1)

            content_encode = tr.Transformer(num_layers=1, d_model=self.embedding_size, num_heads=4,
                                            dff=512, input_vocab_size=self.word_size)(embed1, self.train, mask1)

            att_to_out = tr.Attention_Layer(content_encode, mask2)

            self.encode = att_to_out

            nce_weights1 = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                    stddev=1.0 / math.sqrt(self.embedding_size)))
            nce_biases1 = tf.Variable(tf.zeros([self.vocabulary_size]), dtype=tf.float32)

            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights1,
                                                 biases=nce_biases1,
                                                 inputs=self.encode,
                                                 labels=self.train_labels,
                                                 num_sampled=self.neg_samples,
                                                 num_classes=self.vocabulary_size))

            # nce_weights2 = tf.Variable(
            #    tf.truncated_normal([self.vocabulary_size, self.embedding_size],
            #                        stddev=1.0 / math.sqrt(self.embedding_size)))
            # nce_biases2 = tf.Variable(tf.zeros([self.vocabulary_size]), dtype=tf.float32)
            #
            # loss2 = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights2,
            #                                          biases=nce_biases2,
            #                                          inputs=content_encode,
            #                                          labels=self.train_labels,
            #                                          num_sampled=self.neg_samples,
            #                                          num_classes=self.vocabulary_size))



            #self.loss = loss1 + loss2

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            lr = tf.train.exponential_decay(0.001, global_step=self.global_step, decay_steps=int(0.05 * self.total),
                                            decay_rate=0.95)
            self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss, global_step=self.global_step)




