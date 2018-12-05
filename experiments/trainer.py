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

from __future__ import print_function
import os
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from utils import data_loader
from utils import feeder
from evaluation import evaluater
from models.model import tf_model
import models.transE as transe_model
import models.multi_relation_model as multi_model
import models.softbox as soft_model
import models.oe_model as oe_model
import tensorflow as tf
from datetime import datetime
import time
import pickle
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

def save_model(name, sess, model):
    f = open(name,'wb')
    print("Model saved in file: %s" % name)
    save_model = {}
    save_model['min_embeddings'] = sess.run(model.min_embed)
    save_model['delta_embeddings'] = sess.run(model.delta_embed)
    pickle.dump(save_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def run_training():
    exp_name = 'time' + str(datetime.now()) + 'train_file' + str(FLAGS.train_file) + 'freeze_grad' + str(
        FLAGS.freeze_grad) + 'neg' + str(FLAGS.neg) + 'model' + str(FLAGS.model) + '_measure' + str(FLAGS.measure) + \
               '_w1' + str(FLAGS.w1) + '_w2' + str(FLAGS.w2) + '_learning_rate' + str(
        FLAGS.learning_rate) + '_batchsize' + str(FLAGS.batch_size) + '_dim' + str(FLAGS.embed_dim) + \
               '_cube_eps' + str(FLAGS.cube_eps) + '_steps' + str(FLAGS.max_steps) + '_softfreeze' + str(
        FLAGS.softfreeze) + '_r1' + str(FLAGS.r1) + '_paireval' + str(FLAGS.pair_eval)
    print('experiment file name', exp_name)
    error_file_name = FLAGS.error_file + exp_name + '.txt'
    save_model_name = FLAGS.params_file + exp_name + '.pkl'
    log_folder = FLAGS.log_file + exp_name + '/'

    # define evalution number list
    train_acc_list, dev2_acc_list = [], []
    curr_best = 0.0

    # read data set is a one time thing, so even it takes a little bit longer, it's fine.
    data_sets = data_loader.read_data_sets(FLAGS)
    if FLAGS.overfit:
        train_data = data_sets.dev
        train_test_data = data_sets.dev
    else:
        train_data = data_sets.train
        train_test_data = data_sets.train_test

    with tf.Graph().as_default():
        print('Build Model...')
        placeholder = feeder.define_placeholder()
        if FLAGS.model == 'transe':
             model = transe_model.tf_model(data_sets, placeholder, FLAGS)
        elif FLAGS.model == 'oe':
            print('OE')
            model = oe_model.tf_model(data_sets, placeholder, FLAGS)
        elif FLAGS.model == 'cube' and FLAGS.rel_size > 1:
            model = multi_model.tf_model(data_sets, placeholder, FLAGS)
        elif FLAGS.model == 'cube' and FLAGS.rel_size == 1:
            print('hard box')
            model = tf_model(data_sets, placeholder, FLAGS)
        elif FLAGS.model == 'softbox':
            print('softbox')
            model = soft_model.tf_model(data_sets, placeholder, FLAGS)
        else:
            raise ValueError('no valid model combination, transe or cube')
        eval_neg_prob = model.eval_prob
        print('Build Training Function...')
        train_op = model.training(model.loss, FLAGS.epsilon, FLAGS.learning_rate)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        # gradient plot
        # grad_norm_list = []
        # plt.ion()

        i = 0
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_folder, graph=sess.graph)
        if FLAGS.marginal_method == 'softplus' or FLAGS.model == 'box':
                sess.run([model.project_op])
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            train_feed_dict = feeder.fill_feed_dict(train_data, placeholder, data_sets.rel2idx, FLAGS.batch_size)
            if FLAGS.marginal_method == 'softplus' or FLAGS.model == 'box':
                sess.run([model.project_op], feed_dict=train_feed_dict)
            _ , cond_loss, marg_loss, reg_loss, loss_value, temperature, summary = sess.run([train_op, model.cond_loss, model.marg_loss,
                                                                                model.regularization, model.loss, model.temperature, summary_op],
                                                                               feed_dict=train_feed_dict)
            # grad_norm_list.append(grad_norm)
            # moving_average = np.convolve(grad_norm_list, np.ones((50,))/50, mode='valid')
            # if step %5:
            #     plt.figure(1)
            #     plt.plot(moving_average)
            #     plt.draw()
            #     plt.pause(0.0001)
            #     plt.clf()
            # min_embed, delta_embed = sess.run([model.min_embed, model.delta_embed], feed_dict=train_feed_dict)
            debug, loss_value, summary = sess.run([model.debug, model.loss, summary_op], feed_dict=train_feed_dict)
            summary_writer.add_summary(summary, step)
            duration = time.time() - start_time
            if (step%(FLAGS.print_every) == 0) and step > 1:
                print('='*100)
                print('step', step)
                print('temperature',temperature)
                if temperature >0.0001:
                    sess.run(model.temperature_update)
                print('Epoch %d: kb_loss = %.5f (%.3f sec)' % (train_data._epochs_completed, loss_value, duration))
                print('Conditional loss: %.5f, Marginal loss: %.5f , Regularization loss: %.5f' % (cond_loss, marg_loss, reg_loss))
                print('Training Stats:', end = '')
                train_acc = evaluater.accuracy_eval(sess, eval_neg_prob, placeholder, train_test_data, data_sets.rel2idx, FLAGS, error_file_name)
                train_acc_list.append(train_acc)

                dev2_acc = evaluater.do_eval(sess, eval_neg_prob, placeholder, data_sets.dev, data_sets.devtest, curr_best, FLAGS, error_file_name, data_sets.rel2idx, data_sets.word2idx)
                dev2_acc_list.append(dev2_acc)
                print("Accuracy for Devtest: %.5f" % dev2_acc)

                print(i)
                if dev2_acc >= curr_best:
                    i = 0
                    curr_best = dev2_acc
                    if FLAGS.save:
                        save_model(save_model_name, sess, model)
                # elif dev2_acc < curr_best and i<50:
                #     i+=1
                # elif i>=50:
                    # sys.exit()
                print('current best accurancy: %.5f' %curr_best)
                print('Average of dev2 score %.5f' % (np.mean(sorted(dev2_acc_list, reverse = True)[:10])))

        print('Average of Top 10 Training Score', np.mean(sorted(train_acc_list, reverse = True)[:10]))
        opt_idx = np.argmax(np.asarray(dev2_acc_list))
        print('Epoch', opt_idx)
        # print('Best Dev2 Score: %.5f' %dev2_acc_list[opt_idx])
        print('Average of dev2 score %.5f' % (np.mean(sorted(dev2_acc_list, reverse = True)[:10])))
        # plt.ioff()
        # plt.figure(3)
        # plt.plot(moving_average)
        # plt.draw()
        # plt.show()
        # with open('./init_analysis/'+exp_name+'.txt', 'w') as outputfile:
        #     for i in grad_norm_list:
        #         outputfile.write(str(i) + '\n')




def main(argv):
    run_training()


if __name__ == '__main__':
    """basic parameters"""
    flags.DEFINE_boolean('save', False, 'Save the model')
    flags.DEFINE_integer('random_seed', 20180112, 'random seed for model')
    flags.DEFINE_string('params_file', './params/', 'file to save parameters')
    flags.DEFINE_string('error_file', './error_analysis/', 'dictionary to save error analysis result')
    flags.DEFINE_string('ouput_file', './result/', 'print the result to this file')
    flags.DEFINE_string('log_file', './log/', 'tensorboard log files')

    """dataset parameters"""
    # # flags.DEFINE_string('train_dir', '/iesl/canvas/xiangl/canvas/data/wordnet/has_isa', 'Directory to put the data.')
    # # flags.DEFINE_string('train_dir', '/iesl/canvas/xiangl/canvas/data/wordnet', 'Directory to put the data.')
    # # flags.DEFINE_string('train_dir', '/Users/lorraine/UMass/Projects/data/wordnet', 'Directory to put the data.')
    # # flags.DEFINE_string('train_dir', '/Users/lorraine/UMass/2018Research/Data/Octavian_WordNet', 'Directory to put the data.')
    flags.DEFINE_string('train_dir', '/Users/lorraine/UMass/2018Research/Data/Mammal_WordNet', 'Directory to put the data.')
    # # flags.DEFINE_string('train_dir', '/Users/lorraine/UMass/2018Research/Data/WordNet', 'Directory to put the data.')
    # # flags.DEFINE_string('train_dir', '/Users/lorraine/UMass/2017Research/probabilistic-hypercube-lattices/data/wordnet', 'Directory to put the data.')
    # # flags.DEFINE_string('train_dir', '/home/xiangl/workspace/data/country/s1', 'Directory to put the data.')
    # # flags.DEFINE_string('train_dir', '/mnt/nfs/work1/mccallum/xiangl/data/wordnet', 'Directory to put the data.')
    # # flags.DEFINE_string('train_file', 'wordnet_train_aug_with_cpr0.6.txt', 'which training file to use')
    # flags.DEFINE_string('train_file', 'wordnet_train_preneg.txt', 'which training file to use')
    flags.DEFINE_string('train_file', 'wordnet_train_0percent.txt', 'which training file to use')
    # # flags.DEFINE_string('train_file', 'wordnet_train.txt', 'which training file to use')
    flags.DEFINE_string('train_test_file', 'wordnet_train_test.txt', 'which dev file to use')
    # flags.DEFINE_string('dev_file', 'wordnet_valid.txt', 'which dev file to use')
    # flags.DEFINE_string('test_file', 'wordnet_test.txt', 'which test file to use')
    # flags.DEFINE_string('dev_file', 'wordnet_balanced_valid.txt', 'which dev file to use')
    # flags.DEFINE_string('test_file', 'wordnet_balanced_test.txt', 'which test file to use')
    flags.DEFINE_string('dev_file', 'wordnet_valid.txt', 'which dev file to use')
    flags.DEFINE_string('test_file', 'wordnet_test.txt', 'which test file to use')
    flags.DEFINE_string('marg_prob_file', 'count.txt', 'which marginal probability file to use')
    # # flags.DEFINE_string('train_file', 'tiny_test.txt', 'which training file to use')
    # # flags.DEFINE_string('train_test_file', 'tiny_test.txt', 'which dev file to use')
    # # flags.DEFINE_string('dev_file', 'tiny_test.txt', 'which dev file to use')
    # # flags.DEFINE_string('test_file', 'tiny_test.txt', 'which test file to use')
    # # flags.DEFINE_string('marg_prob_file', 'tiny_test_count.txt', 'which marginal probability file to use')
    # # flags.DEFINE_string('train_file', 'train.txt', 'which training file to use')
    # # flags.DEFINE_string('train_test_file', 'train.txt', 'which dev file to use')
    # # flags.DEFINE_string('dev_file', 'valid.txt', 'which dev file to use')
    # # flags.DEFINE_string('test_file', 'test.txt', 'which test file to use')
    # new wordnet data 2018.11.11
    # flags.DEFINE_string('train_dir', '/Users/lorraine/UMass/2018Research/Data/hypernym_merynym_wordnet', 'Directory to put the data.')
    # flags.DEFINE_string('train_file', 'hypernym_train.txt', 'which training file to use')
    # flags.DEFINE_string('train_test_file', 'hypernym_train.txt', 'which dev file to use')
    # flags.DEFINE_string('dev_file', 'hypernym_test.txt', 'which dev file to use')
    # flags.DEFINE_string('test_file', 'hypernym_test.txt', 'which test file to use')
    # flags.DEFINE_string('marg_prob_file', 'count.txt', 'which marginal probability file to use')


    flags.DEFINE_string('neg', 'uniform', 'uniformly generate negative examples or use pre generated negative examplse')
    flags.DEFINE_integer('rel_size', 1,
                         'relation_size. one means only test for isa relations, else will test for multiple relations')
    flags.DEFINE_boolean('term', False,
                         'whether to use word or term for each input, if using term, need to specify tuple_model')

    """init parameters"""
    flags.DEFINE_string('init_embedding', 'random',
                        'whether to use pre trained min word embedding to init. pre_train or random')
    flags.DEFINE_string('init_embedding_file',
                        '../params/train_datawordnettrain_nameisacubesoftmaxtermFalsefreeze_gradFalsetuple_modelavenegpre_negmodelpoe_measureuniform_w11.0_w210.0_learning_rate0.001_batchsize459776_dim50_steps100000.pkl',
                        'if choose pre_train at init_embedding, specify which embedding you want to use')

    """tensorflow model parameters"""
    flags.DEFINE_string('model', 'cube', 'which model to use, poe cube, or transe')
    flags.DEFINE_string('measure', 'uniform',
                        'exp or uniform represent for different measure. Attention: for different measure, embedding initialization is different')
    flags.DEFINE_boolean('surrogate_bound', True, 'whether to use upper bound for disjoint functions.')
    flags.DEFINE_string('cube', 'softbox', 'use sigmoid or softmax to construct cube when apply term embedding')
    # flags.DEFINE_float('lambda_value', 1e-100, 'smoothe distribution parameter')
    flags.DEFINE_float('lambda_value', 1e-6, 'smoothe distribution parameter')
    flags.DEFINE_float('cube_eps', 1e-6, 'minimize size of each cube')
    flags.DEFINE_string('tuple_model', 'ave', 'how to compose term vector when need to use terms, ave or lstm')
    # if using term as input, and using lstm as tuple model
    flags.DEFINE_integer('hidden_dim', 100, 'lstm hidden layer dimension')
    flags.DEFINE_boolean('peephole', True, 'whether to use peephole in lstm layer')
    flags.DEFINE_float('temperature', 1.0, 'temperature for softplus functions')
    flags.DEFINE_float('decay_rate', 0.0, 'decay rate for temperature')
    flags.DEFINE_boolean('log_space', True, 'represent delta in log space for softbox model.')

    """optimization parameters"""
    flags.DEFINE_string('optimizer', 'adam', 'which optimizer to use: adam or sgd')
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    # flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

    flags.DEFINE_float('epsilon', 1e-8, 'Optimizer epsilon')
    flags.DEFINE_float('softfreeze', '0.0', 'whether to use soft gradient on neg delta embedding')
    flags.DEFINE_boolean('freeze_grad', True, 'whether freeze delta embedding when calculate for negative examples')

    """loss parameters"""
    flags.DEFINE_float('w1', 1.0, 'weight on conditional prob loss')
    flags.DEFINE_float('w2', 0.0, 'weight on marginal prob loss')
    flags.DEFINE_float('r1', 0.0, 'regularization parameter to reduce poe to be box-ish') # 0.1 for universe
    flags.DEFINE_string('regularization_method', 'delta', 'method to regularizing the embedding, either delta'
                                                                  'or universe_edge')
    flags.DEFINE_string('marginal_method', 'universe', 'softplus, universe or sigmoid')

    """training parameters"""
    # flags.DEFINE_integer('max_steps',50000, 'Number of steps to run trainer.')
    # flags.DEFINE_integer('batch_size', 74, 'Batch size. Must divide evenly into the dataset sizes.')
    # flags.DEFINE_integer('print_every', 10, 'Every 20 step, print out the evaluation results')
    # flags.DEFINE_integer('embed_dim', 5, 'word embedding dimension')
    flags.DEFINE_integer('max_steps', 50000, 'Number of steps to run trainer.')
    flags.DEFINE_integer('batch_size', 1176, 'Batch size. Must divide evenly into the dataset sizes.')
    flags.DEFINE_integer('print_every', 10, 'Every 20 step, print out the evaluation results')
    flags.DEFINE_integer('embed_dim', 5, 'word embedding dimension')
    flags.DEFINE_boolean('overfit', False, 'Over fit the dev data to check model')
    # flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer.')
    # flags.DEFINE_integer('batch_size', 74, 'Batch size. Must divide evenly into the dataset sizes.')
    # flags.DEFINE_integer('print_every', 10, 'Every 20 step, print out the evaluation results')
    # flags.DEFINE_integer('embed_dim', 2, 'word embedding dimension')
    # flags.DEFINE_boolean('overfit', False, 'Over fit the dev data to check model')


    """evalution and error analysis parameters"""
    flags.DEFINE_boolean('pair_eval', False, 'whether to use pair eval')
    flags.DEFINE_boolean('rel_acc', False, 'check the different relation accurancy for test dataset')
    flags.DEFINE_boolean('error_analysis', False, 'do error analysis for evaluation data')
    flags.DEFINE_string('eval', 'acc', 'evaluate on MAP, acc or taxo')
    flags.DEFINE_boolean('debug', False , 'whether in debug mode')

    tf.app.run()