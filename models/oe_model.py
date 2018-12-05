"""Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tf_utils
from models import unit_cube
import tensorflow as tf

my_seed = 20180112
tf.set_random_seed(my_seed)

class tf_model(object):
    def __init__(self, data, placeholder, FLAGS):
        self.optimizer = FLAGS.optimizer
        self.opti_epsilon = FLAGS.epsilon
        self.lr = FLAGS.learning_rate
        self.vocab_size = data.vocab_size
        self.measure = FLAGS.measure
        self.embed_dim = FLAGS.embed_dim
        self.batch_size = FLAGS.batch_size
        self.rel_size = FLAGS.rel_size
        self.tuple_model = FLAGS.tuple_model
        self.init_embedding = FLAGS.init_embedding
        self.rang=tf.range(0,FLAGS.batch_size,1)
        # LSTM Params
        self.term = FLAGS.term
        self.hidden_dim = FLAGS.hidden_dim
        self.peephole = FLAGS.peephole
        self.freeze_grad = FLAGS.freeze_grad

        self.t1x = placeholder['t1_idx_placeholder']
        self.t1mask = placeholder['t1_msk_placeholder']
        self.t1length = placeholder['t1_length_placeholder']
        self.t2x = placeholder['t2_idx_placeholder']
        self.t2mask = placeholder['t2_msk_placeholder']
        self.t2length = placeholder['t2_length_placeholder']
        self.rel = placeholder['rel_placeholder']
        self.relmsk = placeholder['rel_msk_placeholder']
        self.label = placeholder['label_placeholder']

        """Initiate embeddings"""
        self.embed = tf.Variable(tf.random_uniform([self.vocab_size, FLAGS.embed_dim], 0.0, 0.1), trainable = True, name = 'word_embed')
        self.Rel = tf.Variable(tf.random_uniform([FLAGS.rel_size, FLAGS.embed_dim],  0.0, 0.1), trainable = True, name = 'rel_embed')


        self.t1_embed = tf.squeeze(tf.nn.embedding_lookup(self.embed, self.t1x), [1])
        self.t2_embed = tf.squeeze(tf.nn.embedding_lookup(self.embed, self.t2x), [1])
        self.t1_embed = tf.abs(self.t1_embed)
        self.t2_embed = tf.abs(self.t2_embed)

        if FLAGS.neg == 'uniform':
            neg_num = 1
            self.nt1x = tf.random_uniform([self.batch_size*neg_num, 1], 0, self.vocab_size, dtype = tf.int32)
            self.nt2x = tf.random_uniform([self.batch_size*neg_num, 1], 0, self.vocab_size, dtype = tf.int32)
            self.generated_nt1_embed = tf.squeeze(tf.nn.embedding_lookup(self.embed, self.nt1x), [1])
            self.generated_nt2_embed = tf.squeeze(tf.nn.embedding_lookup(self.embed, self.nt2x), [1])
            self.nt1_embed = tf.concat([tf.tile(self.t1_embed, [neg_num, 1]), self.generated_nt1_embed], axis = 0)
            self.nt2_embed = tf.concat([self.generated_nt2_embed, tf.tile(self.t2_embed, [neg_num, 1])], axis = 0)
            self.label = tf.concat([self.label, tf.zeros([self.batch_size*neg_num*2])], 0)
            self.t1_uniform_embed = tf.concat([self.t1_embed, self.nt1_embed], axis=0)
            self.t2_uniform_embed = tf.concat([self.t2_embed, self.nt2_embed], axis=0)
            errors = self.error_func(self.t1_uniform_embed, self.t2_uniform_embed)
        else:
            errors = self.error_func(self.t1_embed, self.t2_embed)
        """model cond prob loss"""

        self.eval_prob = self.error_func(self.t1_embed, self.t2_embed)
        self.pos = tf.multiply(errors, self.label)
        self.neg = tf.multiply(tf.maximum(tf.zeros([1], tf.float32), 1.0-errors), (1-self.label))
        self.all = self.pos + self.neg
        self.cond_loss = tf.reduce_mean(self.all, 0)

        """model marg prob loss"""
        self.marg_loss = tf.constant(0.0)
        self.regularization = tf.constant(0.0)
        self.debug = tf.constant(0.0)
        self.temperature = tf.constant(0.0)

        """model final loss"""
        self.loss = self.cond_loss + self.marg_loss

    def error_func(self, specific, general):
        error = tf.reduce_sum(tf.pow(tf.maximum(tf.zeros_like(general, dtype = tf.float32), general-specific + 0.0),2), 1) #self.batch_size
        return error





    def training(self, loss, epsilon, learning_rate):
        tf.summary.scalar(loss.op.name, loss)
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate, epsilon)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('expected adam or sgd, got', self.optimizer)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op


    def rel_embedding(self, Rel, rel, relmsk):
        embed_rel = tf.nn.embedding_lookup(Rel, rel)
        embed_rel = embed_rel * relmsk
        return embed_rel
