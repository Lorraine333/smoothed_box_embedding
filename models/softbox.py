""""Licensed to the Apache Software Foundation (ASF) under one
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
        self.temperature = tf.Variable(FLAGS.temperature, trainable=False)
        self.decay_rate = FLAGS.decay_rate
        self.log_space = FLAGS.log_space
        # LSTM Params
        self.term = FLAGS.term
        self.hidden_dim = FLAGS.hidden_dim
        self.peephole = FLAGS.peephole
        self.freeze_grad = FLAGS.freeze_grad
        self.regularization_method = FLAGS.regularization_method
        self.marginal_method = FLAGS.marginal_method

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

        self.t1_min_embed, self.t1_max_embed, self.t2_min_embed, self.t2_max_embed = self.get_word_embedding(self.t1x, self.t2x)
        """get negative example unit box representation, if it's randomly generated during training."""
        if FLAGS.neg == 'uniform':
            neg_num = 1
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
            conditional_logits, self.meet_min, self.meet_max, self.disjoint, self.nested, self.overlap_volume, self.rhs_volume = self.get_conditional_probability(
                self.t1_uniform_min_embed, self.t1_uniform_max_embed, self.t2_uniform_min_embed, self.t2_uniform_max_embed
            )
        else:
            conditional_logits, self.meet_min, self.meet_max, self.disjoint, self.nested, self.overlap_volume, self.rhs_volume = self.get_conditional_probability(
                self.t1_min_embed, self.t1_max_embed, self.t2_min_embed, self.t2_max_embed
            )

        evaluation_logits, _, _, _, _, _, _ = self.get_conditional_probability(self.t1_min_embed, self.t1_max_embed, self.t2_min_embed, self.t2_max_embed)
        self.eval_prob = -evaluation_logits

        """get conditional probability loss"""
        self.cond_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.label, logits=conditional_logits))
        self.cond_loss = FLAGS.w1 * self.cond_loss

        """model marg prob loss"""
        if FLAGS.w2 > 0.0:
            if self.log_space:
                self.max_embed = self.min_embed + tf.exp(self.delta_embed)
            else:
                self.max_embed = self.min_embed + self.delta_embed
            if self.marginal_method == 'universe':
                self.universe_min = tf.reduce_min(self.min_embed, axis=0, keep_dims=True)
                self.universe_max = tf.reduce_max(self.max_embed, axis=0, keep_dims=True)
                self.universe_volume = tf.reduce_prod(tf.nn.softplus((self.universe_max - self.universe_min)
                                                                     /self.temperature)*self.temperature, axis=-1)
                self.box_volume = tf.reduce_prod(tf.nn.softplus((self.max_embed - self.min_embed)
                                                                /self.temperature)*self.temperature, axis=-1)
                self.predicted_marginal_logits = tf.log(self.box_volume) - tf.log(self.universe_volume)
            elif self.marginal_method == 'softplus':
                self.box_volume = tf.reduce_prod(unit_cube.normalized_softplus(self.delta_embed, self.temperature), axis=-1)
                self.predicted_marginal_logits = tf.log(self.box_volume)
            elif self.marginal_method == 'sigmoid':
                self.box_volume = tf.reduce_prod(unit_cube.sigmoid_normalized_softplus(self.delta_embed, self.temperature), axis=-1)
                self.predicted_marginal_logits = tf.log(self.box_volume)
            else:
                raise ValueError("Expected either softplus or universe but received", self.marginal_method)
            self.marginal_probability = tf.constant(data.margina_prob)
            self.marginal_probability = tf.reshape(self.marginal_probability, [self.vocab_size])
            self.marg_loss = FLAGS.w2 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.marginal_probability, logits=self.predicted_marginal_logits))
        else:
            self.marg_loss = tf.constant(0.0)
        self.debug = tf.constant(0.0)
        self.temperature_update = tf.assign_sub(self.temperature, FLAGS.decay_rate)

        if FLAGS.debug:
            # """model cond prob loss"""
            self.pos_disjoint = tf.logical_and(tf.cast(self.label, tf.bool), self.disjoint)
            self.pos_overlap = tf.logical_and(tf.cast(self.label, tf.bool), tf.logical_not(self.disjoint))
            self.neg_disjoint = tf.logical_and(tf.logical_not(tf.cast(self.label, tf.bool)), self.disjoint)
            self.neg_overlap = tf.logical_and(tf.logical_not(tf.cast(self.label, tf.bool)), tf.logical_not(self.disjoint))
            self.pos_nested = tf.logical_and(tf.cast(self.label, tf.bool), self.nested)
            self.neg_nested = tf.logical_and(tf.logical_not(tf.cast(self.label, tf.bool)), self.nested)
            self.pos_disjoint.set_shape([None])
            self.neg_disjoint.set_shape([None])
            self.pos_overlap.set_shape([None])
            self.neg_overlap.set_shape([None])
            self.pos_nested.set_shape([None])
            self.neg_nested.set_shape([None])
            if self.marginal_method == 'universe':
                lhs_volume = tf.reduce_prod(tf.nn.softplus((self.t2_max_embed - self.t2_min_embed)
                                                           /self.temperature)*self.temperature, axis=-1)
                logx = tf.log(rhs_volume)-tf.log(self.universe_volume)
                logy = tf.log(lhs_volume)-tf.log(self.universe_volume)
                logxy = tf.log(overlap_volume)-tf.log(self.universe_volume)
            elif self.marginal_method == 'softplus':
                logx = tf.log(tf.reduce_prod(unit_cube.normalized_softplus((self.t1_max_embed - self.t1_min_embed), self.temperature), axis=-1))
                logy = tf.log(tf.reduce_prod(unit_cube.normalized_softplus((self.t2_max_embed - self.t2_min_embed), self.temperature), axis=-1))
                logxy = tf.log(tf.reduce_prod(unit_cube.normalized_softplus((self.meet_max - self.meet_min), self.temperature), axis=-1))
            elif self.marginal_method == 'sigmoid':
                logx = tf.log(tf.reduce_prod(unit_cube.sigmoid_normalized_softplus((self.t1_max_embed - self.t1_min_embed), self.temperature), axis=-1))
                logy = tf.log(tf.reduce_prod(unit_cube.sigmoid_normalized_softplus((self.t2_max_embed - self.t2_min_embed), self.temperature), axis=-1))
                logxy = tf.log(tf.reduce_prod(unit_cube.sigmoid_normalized_softplus((self.meet_max - self.meet_min), self.temperature), axis=-1))
            else:
                raise ValueError("Expected either softplus or universe but received", self.marginal_method)
            lognume1 = logxy
            lognume2 = logx + logy
            logdomi = 0.5 * (logx + logy + tf_utils.log1mexp(-logx) + tf_utils.log1mexp(-logy))
            correlation = tf.exp(lognume1 - logdomi) - tf.exp(lognume2 - logdomi)
            self.marg_loss = tf.Print(self.marg_loss,
                                      [tf.exp(self.predicted_marginal_logits), self.marginal_probability, self.box_volume],
                                      'marginal prediction and label')
            self.cond_loss = tf.Print(self.cond_loss,
                                      [tf.exp(conditional_logits), self.label],
                                      'conditional prediction and label')
            self.cond_loss = tf.Print(self.cond_loss, [tf.reduce_sum(tf.cast(self.pos_nested, tf.int32)),
                                                       tf.boolean_mask(tf.exp(conditional_logits), self.pos_nested)],
                                      'pos nested number')
            self.cond_loss = tf.Print(self.cond_loss, [tf.reduce_sum(tf.cast(self.neg_nested, tf.int32)),
                                                       tf.boolean_mask(tf.exp(conditional_logits), self.neg_nested)],
                                      'neg nested number')
            self.cond_loss = tf.Print(self.cond_loss, [tf.reduce_mean(tf.boolean_mask(tf.exp(conditional_logits), self.pos_disjoint)),
                                                       tf.reduce_sum(tf.cast(self.pos_disjoint, tf.int32)),
                                                       tf.count_nonzero(tf.less_equal(
                                                           tf.boolean_mask(correlation, self.pos_disjoint), 0)),
                                                       tf.reduce_mean(tf.boolean_mask(tf.exp(logxy), self.pos_disjoint)),
                                                       tf.reduce_mean(tf.boolean_mask(tf.exp(logx), self.pos_disjoint)),
                                                       tf.boolean_mask(self.t2_max_embed, self.pos_disjoint),
                                                       tf.boolean_mask(self.t2_min_embed, self.pos_disjoint)], 'pos disjoint loss')

            self.cond_loss = tf.Print(self.cond_loss, [tf.reduce_mean(tf.boolean_mask(tf.exp(conditional_logits), self.pos_overlap)),
                                                       tf.reduce_sum(tf.cast(self.pos_overlap, tf.int32)),
                                                       tf.count_nonzero(tf.less_equal(
                                                           tf.boolean_mask(correlation, self.pos_overlap), 0)),
                                                       tf.reduce_mean(tf.boolean_mask(tf.exp(logxy), self.pos_overlap)),
                                                       tf.reduce_mean(tf.boolean_mask(tf.exp(logx), self.pos_overlap))], 'pos overlap loss')

            self.cond_loss = tf.Print(self.cond_loss, [tf.reduce_mean(tf.boolean_mask(tf.exp(conditional_logits), self.neg_disjoint)),
                                                       tf.reduce_sum(tf.cast(self.neg_disjoint, tf.int32)),
                                                       tf.count_nonzero(tf.less_equal(
                                                           tf.boolean_mask(correlation, self.neg_disjoint), 0)),
                                                       tf.reduce_mean(tf.boolean_mask(tf.exp(logxy), self.neg_disjoint)),
                                                       tf.reduce_mean(tf.boolean_mask(tf.exp(logx), self.neg_disjoint))], 'neg disjoint loss')

            self.cond_loss = tf.Print(self.cond_loss, [tf.reduce_mean(tf.boolean_mask(tf.exp(conditional_logits), self.neg_overlap)),
                                                       tf.reduce_sum(tf.cast(self.neg_overlap, tf.int32)),
                                                       tf.count_nonzero(tf.less_equal(
                                                           tf.boolean_mask(correlation, self.neg_overlap), 0)),
                                                       tf.boolean_mask(self.t1x, self.neg_overlap),
                                                       tf.boolean_mask(self.t2x, self.neg_overlap),
                                                       tf.reduce_mean(tf.boolean_mask(tf.exp(logxy), self.neg_overlap)),
                                                       tf.reduce_mean(tf.boolean_mask(tf.exp(logx), self.neg_overlap))], 'neg overlap loss')

        """model regurlization"""
        if self.regularization_method == 'universe_edge' and FLAGS.r1>0.0:
            self.regularization = FLAGS.r1 * tf.reduce_mean(
                tf.nn.softplus(self.universe_max - self.universe_min)
            )
        elif self.regularization_method == 'delta' and FLAGS.r1>0.0:
            if self.log_space:
                self.regularization = FLAGS.r1 * tf.reduce_mean(
                    tf.square(tf.exp(self.delta_embed)))
            else:
                self.regularization = FLAGS.r1 * tf.reduce_mean(
                    tf.square(self.delta_embed))
        else:
            self.regularization = tf.constant(0.0)


        """model final loss"""

        self.loss = self.cond_loss + self.marg_loss + self.regularization
        """loss gradient"""
        grads=tf.gradients(self.loss, tf.trainable_variables())
        grad_norm=0.0
        for g in grads:
            grad_norm += tf.reduce_sum(g.values * g.values)
        grad_norm=tf.sqrt(grad_norm)
        self.grad_norm = grad_norm
        # self.loss = tf.Print(self.loss, [grad_norm], 'gradient')



    @property
    def init_embedding_scale(self):
        """For different measures, min and delta have different init value. """
        if self.measure == 'exp' and not self.term:
            min_lower_scale, min_higher_scale = 0.00001, 0.001
            delta_lower_scale, delta_higher_scale = 10.0, 10.5
        elif self.measure == 'uniform' and not self.term:
            min_lower_scale, min_higher_scale = 1e-4, 1e-2
            delta_lower_scale, delta_higher_scale = 0.9, 0.999
            if self.log_space:
                min_lower_scale, min_higher_scale = 1e-4, 0.9
                delta_lower_scale, delta_higher_scale = -1.0, -0.1
                # min_lower_scale, min_higher_scale = 1e-4, 0.5
                # delta_lower_scale, delta_higher_scale = -0.1, -0.001
            if self.marginal_method == 'sigmoid':
                min_lower_scale, min_higher_scale = -4.5, -4.0
                delta_lower_scale, delta_higher_scale = 4.0, 4.5
            # min_lower_scale, min_higher_scale = 1e-4, 0.9
            # delta_lower_scale, delta_higher_scale = 1e-3, 0.999
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



    def get_word_embedding(self, t1_idx, t2_idx):
        """read word embedding from embedding table, get unit cube embeddings"""

        t1_min_embed = tf.squeeze(tf.nn.embedding_lookup(self.min_embed, t1_idx), [1])
        t1_delta_embed = tf.squeeze(tf.nn.embedding_lookup(self.delta_embed, t1_idx), [1])
        t2_min_embed = tf.squeeze(tf.nn.embedding_lookup(self.min_embed, t2_idx), [1])
        t2_delta_embed = tf.squeeze(tf.nn.embedding_lookup(self.delta_embed, t2_idx), [1])

        t1_max_embed = t1_min_embed + t1_delta_embed
        t2_max_embed = t2_min_embed + t2_delta_embed
        if self.log_space:
            t1_max_embed = t1_min_embed + tf.exp(t1_delta_embed)
            t2_max_embed = t2_min_embed + tf.exp(t2_delta_embed)
        if self.marginal_method == 'sigmoid':
            t1_max_embed = t1_min_embed + tf.nn.sigmoid(t1_delta_embed)
            t2_max_embed = t2_min_embed + tf.nn.sigmoid(t2_delta_embed)
        return t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed


    def get_conditional_probability(self, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
        _, _, meet_min, meet_max, disjoint = unit_cube.calc_join_and_meet(
        t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed)
        nested = unit_cube.calc_nested(t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, self.embed_dim)
        """get conditional probabilities"""
        overlap_volume = tf.reduce_prod(tf.nn.softplus((meet_max - meet_min)
                                                       /self.temperature)*self.temperature, axis=-1)
        rhs_volume = tf.reduce_prod(tf.nn.softplus((t1_max_embed - t1_min_embed)
                                                   /self.temperature)*self.temperature, axis=-1)
        conditional_logits = tf.log(overlap_volume+1e-10) - tf.log(rhs_volume+1e-10)
        return conditional_logits, meet_min, meet_max, disjoint, nested, overlap_volume, rhs_volume

    def generate_neg(self, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
        # randomly generate negative examples by swaping to the next examples
        nt1_min_embed = tf.nn.embedding_lookup(t1_min_embed,(self.rang+1)%self.batch_size)
        nt2_min_embed = tf.nn.embedding_lookup(t2_min_embed,(self.rang+2)%self.batch_size)
        nt1_max_embed = tf.nn.embedding_lookup(t1_max_embed,(self.rang+1)%self.batch_size)
        nt2_max_embed = tf.nn.embedding_lookup(t2_max_embed,(self.rang+2)%self.batch_size)

        return nt1_min_embed, nt1_max_embed, nt2_min_embed, nt2_max_embed

    def training(self, loss, epsilon, learning_rate):
        tf.summary.scalar(loss.op.name, loss)
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon = epsilon, use_locking=True)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('expected adam or sgd, got', self.optimizer)
        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_op = optimizer.minimize(loss)

        return train_op


    def rel_embedding(self, Rel, rel, relmsk):
        embed_rel = tf.nn.embedding_lookup(Rel, rel)
        embed_rel = embed_rel * relmsk
        return embed_rel
