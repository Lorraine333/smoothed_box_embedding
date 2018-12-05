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
        self.temperature = tf.Variable(FLAGS.temperature, trainable=False)
        self.decay_rate = FLAGS.decay_rate

        self.t1x = placeholder['t1_idx_placeholder']
        self.t1mask = placeholder['t1_msk_placeholder']
        self.t1length = placeholder['t1_length_placeholder']
        self.t2x = placeholder['t2_idx_placeholder']
        self.t2mask = placeholder['t2_msk_placeholder']
        self.t2length = placeholder['t2_length_placeholder']
        self.rel = placeholder['rel_placeholder']
        self.relmsk = placeholder['rel_msk_placeholder']
        self.label = placeholder['label_placeholder']

        """Initiate box embeddings"""
        self.min_embed, self.delta_embed = self.init_word_embedding(data)
        self.projector = unit_cube.MinMaxHyperCubeProjectorDeltaParam(self.min_embed, self.delta_embed, 0.0, 1e-10)
        self.project_op = self.projector.project_op
        """get unit box representation for both term, no matter they are phrases or words"""
        if self.term:
            # if the terms are phrases, need to use either word average or lstm to compose the word embedding
            # Then transform them into unit cube.
            raw_t1_min_embed, raw_t1_delta_embed, raw_t2_min_embed, raw_t2_delta_embed = self.get_term_word_embedding(self.t1x, self.t1mask,
                                                                                                                        self.t1length, self.t2x,
                                                                                                                        self.t2mask, self.t2length,
                                                                                                                        False)
            self.t1_min_embed, self.t1_max_embed, self.t2_min_embed, self.t2_max_embed = self.transform_cube(raw_t1_min_embed, raw_t1_delta_embed,
                                                                                         raw_t2_min_embed, raw_t2_delta_embed)

        else:
            self.t1_min_embed, self.t1_max_embed, self.t2_min_embed, self.t2_max_embed = self.get_word_embedding(self.t1x, self.t2x)

        """get negative example unit box representation, if it's randomly generated during training."""
        if FLAGS.neg == 'uniform':
            neg_num = 5
            self.nt1x = tf.random_uniform([self.batch_size*neg_num, 1], 0, self.vocab_size, dtype = tf.int32)
            self.nt2x = tf.random_uniform([self.batch_size*neg_num, 1], 0, self.vocab_size, dtype = tf.int32)
            self.nt1_min_embed, self.nt1_max_embed, self.nt2_min_embed, self.nt2_max_embed = self.get_word_embedding(self.nt1x, self.nt2x)
            # combine the original word embedding with the new embeddings.
            self.nt1_min_embed = tf.concat([tf.tile(self.t1_min_embed, [neg_num, 1]),
                                           self.nt1_min_embed], axis=0)
            self.nt1_max_embed = tf.concat([tf.tile(self.t1_max_embed, [neg_num, 1]),
                                            self.nt1_max_embed], axis=0)
            self.nt2_min_embed = tf.concat([self.nt2_min_embed,
                                            tf.tile(self.t2_min_embed, [neg_num, 1])], axis=0)
            self.nt2_max_embed = tf.concat([self.nt2_max_embed,
                                            tf.tile(self.t2_max_embed, [neg_num, 1])], axis=0)
            self.label = tf.concat([self.label, tf.zeros([self.batch_size*neg_num*2])], 0)
            self.t1_uniform_min_embed = tf.concat([self.t1_min_embed, self.nt1_min_embed], axis=0)
            self.t1_uniform_max_embed = tf.concat([self.t1_max_embed, self.nt1_max_embed], axis=0)
            self.t2_uniform_min_embed = tf.concat([self.t2_min_embed, self.nt2_min_embed], axis=0)
            self.t2_uniform_max_embed = tf.concat([self.t2_max_embed, self.nt2_max_embed], axis=0)
            """calculate box stats, join, meet and overlap condition"""
            self.join_min, self.join_max, self.meet_min, self.meet_max, self.disjoint = unit_cube.calc_join_and_meet(
                self.t1_uniform_min_embed, self.t1_uniform_max_embed, self.t2_uniform_min_embed, self.t2_uniform_max_embed)
            self.nested = unit_cube.calc_nested(self.t1_uniform_min_embed, self.t1_uniform_max_embed,
                                                self.t2_uniform_min_embed, self.t2_uniform_max_embed, self.embed_dim)
            """calculate -log(p(term2 | term1)) if overlap, surrogate function if not overlap"""
            # two surrogate function choice. lambda_batch_log_upper_bound or lambda_batch_disjoint_box
            if FLAGS.surrogate_bound:
                surrogate_func = unit_cube.lambda_batch_log_upper_bound
            else:
                surrogate_func = unit_cube.lambda_batch_disjoint_box
            """tf.where"""
            pos_tensor1 = surrogate_func(self.join_min, self.join_max, self.meet_min, self.meet_max,
                                        self.t1_uniform_min_embed, self.t1_uniform_max_embed,
                                        self.t2_uniform_min_embed, self.t2_uniform_max_embed)
            pos_tensor2 = unit_cube.lambda_batch_log_prob(self.t1_uniform_min_embed, self.t1_uniform_max_embed,
                                                          self.t2_uniform_min_embed, self.t2_uniform_max_embed)
            pos_tensor1 = tf.multiply(pos_tensor1, tf.cast(self.disjoint, tf.float32))
            pos_tensor2 = tf.multiply(pos_tensor2, tf.cast(tf.logical_not(self.disjoint), tf.float32))
            # pos_tensor1 = tf.Print(pos_tensor1, [pos_tensor1, pos_tensor2], 'pos_tensor1')
            train_pos_prob = pos_tensor1 + pos_tensor2
            """slicing where"""
            # train_pos_prob = tf_utils.slicing_where(condition=self.disjoint,
            #                                         full_input=tf.tuple([self.join_min, self.join_max, self.meet_min, self.meet_max,
            #                                                      self.t1_uniform_min_embed, self.t1_uniform_max_embed,
            #                                                      self.t2_uniform_min_embed, self.t2_uniform_max_embed]),
            #                                         true_branch=lambda x: surrogate_func(*x),
            #                                         false_branch=lambda x: unit_cube.lambda_batch_log_prob_emgerncy(*x))
            """tf.print"""
            # train_pos_prob = tf.Print(train_pos_prob, [tf.reduce_sum(tf.cast(tf.logical_and(
            #     self.disjoint, tf.logical_not(tf.cast(self.label, tf.bool))), tf.float32)), self.disjoint],
            #                           'neg disjoint value', summarize=3)
            # train_pos_prob = tf.Print(train_pos_prob, [tf.reduce_sum(tf.cast(tf.logical_and(
            #     self.disjoint, tf.cast(self.label, tf.bool)), tf.float32))],
            #                           'pos disjoint value', summarize=3)
            """calculate -log(1-p(term2 | term1)) if overlap, 0 if not overlap"""
            neg_tensor1 = unit_cube.lambda_zero(self.join_min, self.join_max, self.meet_min, self.meet_max,
                                                self.t1_uniform_min_embed, self.t1_uniform_max_embed,
                                                self.t2_uniform_min_embed, self.t2_uniform_max_embed)
            neg_tensor2 = unit_cube.lambda_batch_log_1minus_prob(self.join_min, self.join_max, self.meet_min, self.meet_max,
                                                self.t1_uniform_min_embed, self.t1_uniform_max_embed,
                                                self.t2_uniform_min_embed, self.t2_uniform_max_embed)
            neg_tensor1 = tf.multiply(neg_tensor1, tf.cast(self.disjoint, tf.float32))
            neg_tensor2 = tf.multiply(neg_tensor2, tf.cast(tf.logical_not(self.disjoint), tf.float32))
            train_neg_prob = neg_tensor1 + neg_tensor2

        else:
            """calculate box stats, join, meet and overlap condition"""
            self.join_min, self.join_max, self.meet_min, self.meet_max, self.disjoint = unit_cube.calc_join_and_meet(
                self.t1_min_embed, self.t1_max_embed, self.t2_min_embed, self.t2_max_embed)
            self.nested = unit_cube.calc_nested(self.t1_min_embed, self.t1_max_embed,
                                                self.t2_min_embed, self.t2_max_embed, self.embed_dim)
            """calculate -log(p(term2 | term1)) if overlap, surrogate function if not overlap"""
            # two surrogate function choice. lambda_batch_log_upper_bound or lambda_batch_disjoint_box
            if FLAGS.surrogate_bound:
                surrogate_func = unit_cube.lambda_batch_log_upper_bound
            else:
                surrogate_func = unit_cube.lambda_batch_disjoint_box
            """tf.where"""
            pos_tensor1 = 500*surrogate_func(self.join_min, self.join_max, self.meet_min, self.meet_max,
                                        self.t1_min_embed, self.t1_max_embed,
                                        self.t2_min_embed, self.t2_max_embed)
            pos_tensor2 = unit_cube.lambda_batch_log_prob(self.t1_min_embed, self.t1_max_embed,
                                                          self.t2_min_embed, self.t2_max_embed)
            pos_tensor1 = tf.multiply(pos_tensor1, tf.cast(self.disjoint, tf.float32))
            pos_tensor2 = tf.multiply(pos_tensor2, tf.cast(tf.logical_not(self.disjoint), tf.float32))
            # pos_tensor1 = tf.Print(pos_tensor1, [pos_tensor1, pos_tensor2], 'pos_tensor1')
            train_pos_prob = pos_tensor1 + pos_tensor2
            """slicing where"""
            # train_pos_prob = tf_utils.slicing_where(condition=self.disjoint,
            #                                         full_input=tf.tuple([self.join_min, self.join_max, self.meet_min, self.meet_max,
            #                                                      self.t1_min_embed, self.t1_max_embed,
            #                                                      self.t2_min_embed, self.t2_max_embed]),
            #                                         true_branch=lambda x: surrogate_func(*x),
            #                                         false_branch=lambda x: unit_cube.lambda_batch_log_prob(*x))
            """tf.print"""
            # train_pos_prob = tf.Print(train_pos_prob, [tf.reduce_sum(tf.cast(tf.logical_and(
            #     self.disjoint, tf.logical_not(tf.cast(self.label, tf.bool))), tf.float32)), self.disjoint],
            #                           'neg disjoint value', summarize=3)
            # train_pos_prob = tf.Print(train_pos_prob, [tf.reduce_sum(tf.cast(tf.logical_and(
            #     self.disjoint, tf.cast(self.label, tf.bool)), tf.float32))],
            #                           'pos disjoint value', summarize=3)
            """calculate -log(1-p(term2 | term1)) if overlap, 0 if not overlap"""
            neg_tensor1 = unit_cube.lambda_zero(self.join_min, self.join_max, self.meet_min, self.meet_max,
                                                self.t1_min_embed, self.t1_max_embed,
                                                self.t2_min_embed, self.t2_max_embed)
            neg_tensor2 = unit_cube.lambda_batch_log_1minus_prob(self.join_min, self.join_max, self.meet_min, self.meet_max,
                                                self.t1_min_embed, self.t1_max_embed,
                                                self.t2_min_embed, self.t2_max_embed)
            neg_tensor1 = tf.multiply(neg_tensor1, tf.cast(self.disjoint, tf.float32))
            neg_tensor2 = tf.multiply(neg_tensor2, tf.cast(tf.logical_not(self.disjoint), tf.float32))
            train_neg_prob = neg_tensor1 + neg_tensor2

        self.temperature_update = tf.assign_sub(self.temperature, FLAGS.decay_rate)
        # train_neg_prob = tf_utils.slicing_where(condition=self.disjoint,
        #                                         full_input=([self.join_min, self.join_max, self.meet_min, self.meet_max,
        #                                                      self.t1_min_embed, self.t1_max_embed,
        #                                                      self.t2_min_embed, self.t2_max_embed]),
        #                                         true_branch=lambda x: unit_cube.lambda_zero(*x),
        #                                         false_branch=lambda x: unit_cube.lambda_batch_log_1minus_prob(*x))
        """calculate negative log prob when evaluating pairs. The lower, the better"""
        # when return hierarchical error, we return the negative log probability, the lower, the probability higher
        # if two things are disjoint, we return -tf.log(1e-8).
        _, _, _, _, self.eval_disjoint = unit_cube.calc_join_and_meet(
                self.t1_min_embed, self.t1_max_embed, self.t2_min_embed, self.t2_max_embed)
        eval_tensor1 = unit_cube.lambda_hierarchical_error_upper(self.t1_min_embed, self.t1_max_embed,
                                                                 self.t2_min_embed, self.t2_max_embed)
        eval_tensor2 = unit_cube.lambda_batch_log_prob(self.t1_min_embed, self.t1_max_embed,
                                                       self.t2_min_embed, self.t2_max_embed)
        self.eval_prob = tf.where(self.eval_disjoint, eval_tensor1, eval_tensor2)
        # self.eval_prob = tf_utils.slicing_where(condition = self.disjoint,
        #                                         full_input = [self.join_min, self.join_max, self.meet_min, self.meet_max,
        #                                                       self.t1_min_embed, self.t1_max_embed,
        #                                                       self.t2_min_embed, self.t2_max_embed],
        #                                         true_branch = lambda x: unit_cube.lambda_hierarchical_error_upper(*x),
        #                                         false_branch = lambda x: unit_cube.lambda_batch_log_prob(*x))
        """model marg prob loss"""
        if FLAGS.w2 > 0.0:
            self.marg_prob = tf.constant(data.margina_prob)
            kl_difference = unit_cube.calc_marginal_prob(self.marg_prob, self.min_embed, self.delta_embed)
            kl_difference = tf.reshape(kl_difference, [-1]) / self.vocab_size
            self.marg_loss = FLAGS.w2 * (tf.reduce_sum(kl_difference))
        else:
            self.marg_loss = tf.constant(0.0)

        """model cond prob loss"""
        self.pos = FLAGS.w1 * tf.multiply(train_pos_prob, self.label)
        self.neg = FLAGS.w1 * tf.multiply(train_neg_prob, (1 - self.label))
        if FLAGS.debug:
            self.pos_disjoint = tf.logical_and(tf.cast(self.label, tf.bool), self.disjoint)
            self.pos_overlap = tf.logical_and(tf.cast(self.label, tf.bool), tf.logical_not(self.disjoint))
            self.neg_disjoint = tf.logical_and(tf.logical_not(tf.cast(self.label, tf.bool)), self.disjoint)
            self.neg_overlap = tf.logical_and(tf.logical_not(tf.cast(self.label, tf.bool)), tf.logical_not(self.disjoint))
            self.pos_disjoint.set_shape([None])
            self.neg_disjoint.set_shape([None])
            self.pos_overlap.set_shape([None])
            self.neg_overlap.set_shape([None])
            self.pos = tf.Print(self.pos, [tf.reduce_mean(tf.boolean_mask(self.pos, self.pos_disjoint)), tf.reduce_sum(tf.cast(self.pos_disjoint, tf.int32))], 'pos disjoint loss')
            self.pos = tf.Print(self.pos, [tf.reduce_mean(tf.boolean_mask(self.pos, self.pos_overlap)), tf.reduce_sum(tf.cast(self.pos_overlap, tf.int32))], 'pos overlap loss')
            self.neg = tf.Print(self.neg, [tf.reduce_mean(tf.boolean_mask(self.neg, self.neg_disjoint)), tf.reduce_sum(tf.cast(self.neg_disjoint, tf.int32))], 'neg disjoint loss')
            self.neg = tf.Print(self.neg, [tf.reduce_mean(tf.boolean_mask(self.neg, self.neg_overlap)), tf.reduce_sum(tf.cast(self.neg_overlap, tf.int32))], 'neg overlap loss')
            self.pos = tf.Print(self.pos, [tf.reduce_sum(self.pos)], 'pos loss')
            self.neg = tf.Print(self.neg, [tf.reduce_sum(self.neg)], 'neg loss')
            self.pos = tf.Print(self.pos, [tf.reduce_mean(tf.exp(-tf.boolean_mask(self.pos, self.pos_overlap)))], 'pos conditional prob')
            self.neg = tf.Print(self.neg, [tf.reduce_mean(tf.exp(-tf.boolean_mask(train_pos_prob, self.neg_overlap)))], 'neg conditional prob')
            self.pos = tf.Print(self.pos, [tf.reduce_mean(self.min_embed), tf.reduce_mean(self.delta_embed)], 'embedding mean')
            # self.neg = tf.Print(self.neg, [tf.reduce_mean(tf.exp(tf.boolean_mask(unit_cube.batch_log_prob(self.meet_min, self.meet_max), self.neg_overlap)))], 'neg joint prob')
            # self.neg = tf.Print(self.neg, [tf.reduce_mean(tf.exp(tf.boolean_mask(unit_cube.batch_log_prob(self.t1_min_embed, self.t1_max_embed), self.neg_overlap)))], 'neg marg prob')
            self.pos_nested = tf.logical_and(tf.cast(self.label, tf.bool), self.nested)
            self.neg_nested = tf.logical_and(tf.logical_not(tf.cast(self.label, tf.bool)), self.nested)
            self.pos_nested.set_shape([None])
            self.neg_nested.set_shape([None])
            self.pos = tf.Print(self.pos, [tf.reduce_mean(tf.boolean_mask(self.pos, self.pos_nested)), tf.reduce_sum(tf.cast(self.pos_nested, tf.int32))], 'pos nested loss')
            self.neg = tf.Print(self.neg, [tf.reduce_mean(tf.boolean_mask(self.neg, self.neg_nested)), tf.reduce_sum(tf.cast(self.neg_nested, tf.int32))], 'neg nested loss')





        self.cond_loss = tf.reduce_sum(self.pos) / (self.batch_size / 2) + \
                         tf.reduce_sum(self.neg) / (self.batch_size / 2)
        # self.cond_loss = tf.Print(self.cond_loss, [tf.reduce_sum(self.pos), tf.reduce_sum(self.neg)], 'pos and neg loss')
        # self.cond_loss = tf.Print(self.cond_loss, [tf.gradients(self.cond_loss, [self.min_embed, self.delta_embed])[0], self.min_embed, self.delta_embed], 'gradient')


        """model regurlization: make box to be poe-ish"""
        self.regularization = FLAGS.r1 * tf.reduce_sum(tf.abs(1 - self.min_embed - self.delta_embed)) / self.vocab_size

        """model final loss"""
        # self.cond_loss = tf.Print(self.cond_loss, [self.pos, self.neg])
        self.debug = tf.constant(0.0)
        self.loss = self.cond_loss + self.marg_loss + self.regularization
        # self.loss = self.cond_loss + self.marg_loss

        if not self.freeze_grad:
            grads=tf.gradients(self.loss, tf.trainable_variables())
            grad_norm=0.0
            for g in grads:
                new_values=tf.clip_by_value(g.values,-0.5, 0.5)
                grad_norm += tf.reduce_sum(new_values * new_values)
            grad_norm=tf.sqrt(grad_norm)
            self.grad_norm = grad_norm


    @property
    def init_embedding_scale(self):
        """For different measures, min and delta have different init value. """
        if self.measure == 'exp' and not self.term:
            min_lower_scale, min_higher_scale = 0.0, 0.001
            delta_lower_scale, delta_higher_scale = 10.0, 10.5
        elif self.measure == 'uniform' and not self.term:
            # min_lower_scale, min_higher_scale = 1e-4, 0.99
            # delta_lower_scale, delta_higher_scale = 0.4, 0.6
            # min_lower_scale, min_higher_scale = 1e-4, 0.001
            # delta_lower_scale, delta_higher_scale = 0.9, 0.99
            # min_lower_scale, min_higher_scale = 1e-4, 0.7
            # delta_lower_scale, delta_higher_scale = 0.5, 0.6
            min_lower_scale, min_higher_scale = 1e-4, 0.9
            delta_lower_scale, delta_higher_scale = 0.1, 0.9
        elif self.term and self.measure == 'uniform':
            min_lower_scale, min_higher_scale = 1.0, 1.1
            delta_lower_scale, delta_higher_scale = 5.0, 5.1
        else:
            raise ValueError("Expected either exp or uniform but received", self.measure)
        return min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale

    def init_word_embedding(self, data):
        min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale = self.init_embedding_scale
        if self.init_embedding == 'random':
            # random init word embedding
            min_embed = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embed_dim], min_lower_scale, min_higher_scale, seed=my_seed),
                trainable=True, name='word_embed')
            delta_embed = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embed_dim], delta_lower_scale, delta_higher_scale,
                                  seed=my_seed), trainable=True, name='delta_embed')

        elif self.init_embedding == 'pre_train':
            # init min/delta word embedding with pre trained prob order embedding
            min_embed = tf.Variable(data.min_embed, trainable=True, name='word_embed')
            delta_embed = tf.Variable(data.delta_embed, trainable=True, name='delta_embed')

        else:
            raise ValueError("Expected either random or pre_train but received", self.init_embedding)

        return min_embed, delta_embed

    def get_term_word_embedding(self, t1x, t1mask, t1length, t2x, t2mask, t2length, reuse):

        """

        Args:
            t1x, t1mask, t1length: entity one stats.
            t2x, t2mask, t2length: entity two stats.
            reuse: whether to reuse lstm parameters. Differs in training or eval

        Returns: word embedding for entity one phrase and entity two phrase.

        """
        if self.tuple_model == 'ave':
            t1_min_embed = tf_utils.tuple_embedding(t1x, t1mask, t1length, self.min_embed)
            t2_min_embed = tf_utils.tuple_embedding(t2x, t2mask, t2length, self.min_embed)
            t1_delta_embed = tf_utils.tuple_embedding(t1x, t1mask, t1length, self.delta_embed)
            t2_delta_embed = tf_utils.tuple_embedding(t2x, t2mask, t2length, self.delta_embed)

        elif self.tuple_model == 'lstm':
            term_rnn = tf.contrib.rnn.LSTMCell(self.hidden_dim, use_peepholes=self.peephole,num_proj=self.embed_dim, state_is_tuple=True)
            if reuse:
                with tf.variable_scope('term_embed', reuse=True):
                    t1_min_embed = tf_utils.tuple_lstm_embedding(t1x, t1mask, t1length, self.min_embed, term_rnn, False)
            else:
                with tf.variable_scope('term_embed'):
                    t1_min_embed = tf_utils.tuple_lstm_embedding(t1x, t1mask, t1length, self.min_embed, term_rnn, False)
            with tf.variable_scope('term_embed', reuse=True):
                t1_delta_embed = tf_utils.tuple_lstm_embedding(t1x, t1mask, t1length, self.delta_embed, term_rnn, True)
            with tf.variable_scope('term_embed', reuse=True):
                t2_min_embed = tf_utils.tuple_lstm_embedding(t2x, t2mask, t2length, self.min_embed, term_rnn, True)
            with tf.variable_scope('term_embed', reuse=True):
                t2_delta_embed = tf_utils.tuple_lstm_embedding(t2x, t2mask, t2length, self.delta_embed, term_rnn, True)
        else:
            raise ValueError("Expected either ave or lstm but received", self.tuple_model)

        return t1_min_embed, t1_delta_embed, t2_min_embed, t2_delta_embed

    def transform_cube(self, t1_min_embed, t1_delta_embed, t2_min_embed, t2_delta_embed):
        if self.cube == 'sigmoid':
            t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed = tf_utils.make_sigmoid_cube(t1_min_embed,
                                                                                                t1_delta_embed,
                                                                                                t2_min_embed,
                                                                                                t2_delta_embed)
        elif self.cube == 'softmax':
            t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed = tf_utils.make_softmax_cube(t1_min_embed,
                                                                                                t1_delta_embed,
                                                                                                t2_min_embed,
                                                                                                t2_delta_embed)
        else:
            raise ValueError("Expected either sigmoid or softmax but received", self.cube)
        return t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed

    def get_word_embedding(self, t1_idx, t2_idx):
        """read word embedding from embedding table, get unit cube embeddings"""

        t1_min_embed = tf.squeeze(tf.nn.embedding_lookup(self.min_embed, t1_idx), [1])
        t1_delta_embed = tf.squeeze(tf.nn.embedding_lookup(self.delta_embed, t1_idx), [1])
        t2_min_embed = tf.squeeze(tf.nn.embedding_lookup(self.min_embed, t2_idx), [1])
        t2_delta_embed = tf.squeeze(tf.nn.embedding_lookup(self.delta_embed, t2_idx), [1])

        t1_max_embed = t1_min_embed + t1_delta_embed
        t2_max_embed = t2_min_embed + t2_delta_embed
        return t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed


    def generate_neg(self, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
        # randomly generate negative examples by swaping to the next examples
        nt1_min_embed = tf.nn.embedding_lookup(t1_min_embed,(self.rang+1)%self.batch_size)
        nt2_min_embed = tf.nn.embedding_lookup(t2_min_embed,(self.rang+2)%self.batch_size)
        nt1_max_embed = tf.nn.embedding_lookup(t1_max_embed,(self.rang+1)%self.batch_size)
        nt2_max_embed = tf.nn.embedding_lookup(t2_max_embed,(self.rang+2)%self.batch_size)

        return nt1_min_embed, nt1_max_embed, nt2_min_embed, nt2_max_embed

    def get_grad(self, optimizer):
        self.pos_grad = tf.gradients(self.pos, [self.min_embed, self.delta_embed])
        self.neg_grad = tf.gradients(self.neg, [self.min_embed])
        self.neg_delta_grad = tf.gradients(self.neg, [self.delta_embed])
        self.kl_grad = tf.gradients(self.marg_loss, [self.min_embed, self.delta_embed])
        # self.kl_grad = tf.Print(self.kl_grad, [self.pos_grad, self.neg_grad, self.neg_delta_grad], 'gradient')
        # self.pos_grad = optimizer.compute_gradients(loss = self.pos, var_list = [self.min_embed, self.delta_embed])
        # self.neg_grad = optimizer.compute_gradients(loss = self.neg, var_list = [self.min_embed])
        # self.neg_delta_grad = optimizer.compute_gradients(loss = self.neg, var_list = [self.delta_embed])
        # self.kl_grad = optimizer.compute_gradients(loss = self.marg_loss, var_list = [self.min_embed, self.delta_embed])


    def training(self, loss, epsilon, learning_rate):
        tf.summary.scalar(loss.op.name, loss)
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon = epsilon, use_locking=True)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('expected adam or sgd, got', self.optimizer)
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # whether to freeze the negative gradient
        if self.freeze_grad:
            self.get_grad(optimizer)
            # optimizer.compute_gradients()
            # train_op = optimizer.apply_gradients(self.pos_grad)
            train_op1 = optimizer.apply_gradients(zip(self.pos_grad, [self.min_embed, self.delta_embed]))
            with tf.control_dependencies([train_op1]):
                train_op_neg = optimizer.apply_gradients(zip(self.neg_grad, [self.min_embed]))
                # kl_train_op = optimizer.apply_gradients(zip(self.kl_grad, [self.min_embed, self.delta_embed]))
                # train_op3 = tf.group(train_op1, train_op_neg, kl_train_op)
                train_op3 = tf.group(train_op1, train_op_neg)
                # train_op2 = tf.group(train_op1, train_op_neg)
            # with tf.control_dependencies([train_op2]):
            #     kl_train_op = optimizer.apply_gradients(zip(self.kl_grad, [self.min_embed, self.delta_embed]))
            #     train_op3=tf.group(train_op2, kl_train_op)
            grad_norm=0.0
            for g in self.pos_grad:
                new_values=tf.clip_by_value(g,-10, 10)
                grad_norm += tf.reduce_sum(new_values * new_values)
            for g in self.neg_grad:
                new_values=tf.clip_by_value(g,-10, 10)
                grad_norm += tf.reduce_sum(new_values * new_values)
            grad_norm=tf.sqrt(grad_norm)
            self.grad_norm = grad_norm
        else:
            # train_op3 = tf.Print(self.kl_grad, [self.pos_grad, self.neg_grad, self.neg_delta_grad], 'gradient')
            train_op3 = optimizer.minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_op3]):
            if self.measure == 'exp':
                clipped_we,clipped_delta=unit_cube.exp_clip_embedding(self.min_embed, self.delta_embed)
            elif self.measure == 'uniform':
                clipped_we,clipped_delta=unit_cube.uniform_clip_embedding(self.min_embed, self.delta_embed)
                # clipped_we = tf.Print(clipped_we, [tf.equal(clipped_we, self.min_embed)], 'clipped')
                # clipped_we = tf.Print(clipped_we, [clipped_delta, clipped_we, self.min_embed, self.delta_embed], 'after clip')
                # clipped_we, clipped_delta = unit_cube.projection(self.min_embed, self.delta_embed)
            else:
                raise ValueError('Expected exp or uniform, but got', self.measure)
            project=tf.group(tf.assign(self.min_embed,clipped_we),tf.assign(self.delta_embed,clipped_delta))
            train_op=tf.group(train_op3,project)
        return train_op


    # def rel_embedding(self, Rel, rel, relmsk):
    #     embed_rel = tf.nn.embedding_lookup(Rel, rel)
    #     embed_rel = embed_rel * relmsk
    #     return embed_rel
