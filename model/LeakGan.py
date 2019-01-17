import tensorflow as tf
import numpy as np
import os
import pickle
from time import time
from utils.utils import text_precess, code_to_text, text_to_code
from utils.utils import get_tokens
from utils.DataLoader import DataLoader, DisDataloader, TestDataloader
from model.Discriminator import Discriminator
from model.Generator import Generator
from model.Reward import Reward
from model.Gan import Gan
from utils.metrics.Nll import Nll
from utils.metrics.Bleu import Bleu


def pre_train_epoch_gen(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        inputs, target = data_loader.next_batch()
        _, g_loss, _, _ = trainable_model.pretrain_step(sess, inputs, target, .8)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def generate_samples_gen(sess, trainable_model, batch_size, generated_num, data_loader,
                         output_file=None, get_code=True, train=0):
    # Generate Samples
    generated_samples = []
    data_loader.reset_pointer()
    for i in range(int(generated_num / batch_size)):
        inputs, targets = data_loader.next_batch()
        one_batch = trainable_model.generate(sess, 1.0, inputs, train)
        for poem, input_x in zip(one_batch, inputs):
            poem[:len(input_x)] = input_x[:]
        generated_samples.extend(one_batch)

    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)

    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) + '\n'
        codes += buffer
    return codes


def trunc_data(origin, target, input_length):
    token = get_tokens(origin)
    with open(target, 'w', encoding='utf-8') as file:
        for t in token:
            file.write(' '.join(t[input_length:]) + '\n')


class LeakGan(Gan):
    def __init__(self, wi_dict_path, iw_dict_path, train_data, val_data=None):
        super().__init__()

        self.vocab_size = 20
        self.emb_dim = 64
        self.hidden_dim = 64

        self.input_length = 8
        self.sequence_length = 32
        self.filter_size = [2, 3]
        self.num_filters = [100, 200]
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 64
        self.generate_num = 256
        self.start_token = 0
        self.dis_embedding_dim = 64
        self.goal_size = 16

        self.save_path = 'save/model/LeakGan/LeakGan'
        self.model_path = 'save/model/LeakGan'
        self.best_path_pre = 'save/model/best-pre-gen/best-pre-gen'
        self.best_path = 'save/model/best-leak-gan/best-leak-gan'
        self.best_model_path = 'save/model/best-leak-gan'

        self.truth_file = 'save/truth.txt'
        self.generator_file = 'save/generator.txt'
        self.test_file = 'save/test_file.txt'

        self.trunc_train_file = 'save/trunc_train.txt'
        self.trunc_val_file = 'save/trunc_val.txt'
        trunc_data(train_data, self.trunc_train_file, self.input_length)
        trunc_data(val_data, self.trunc_val_file, self.input_length)

        if not os.path.isfile(wi_dict_path) or not os.path.isfile(iw_dict_path):
            print('Building word/index dictionaries...')
            self.sequence_length, self.vocab_size, word_index_dict, index_word_dict = text_precess(train_data, val_data)
            print('Vocab Size: %d' % self.vocab_size)
            print('Saving dictionaries to ' + wi_dict_path + ' ' + iw_dict_path + '...')
            with open(wi_dict_path, 'wb') as f:
                pickle.dump(word_index_dict, f)
            with open(iw_dict_path, 'wb') as f:
                pickle.dump(index_word_dict, f)
        else:
            print('Loading word/index dectionaries...')
            with open(wi_dict_path, 'rb') as f:
                word_index_dict = pickle.load(f)
            with open(iw_dict_path, 'rb') as f:
                index_word_dict = pickle.load(f)
            self.vocab_size = len(word_index_dict) + 1
            print('Vocab Size: %d' % self.vocab_size)

        self.wi_dict = word_index_dict
        self.iw_dict = index_word_dict
        self.train_data = train_data
        self.val_data = val_data

        goal_out_size = sum(self.num_filters)
        self.discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2,
                                           vocab_size=self.vocab_size,
                                           dis_emb_dim=self.dis_embedding_dim, filter_sizes=self.filter_size,
                                           num_filters=self.num_filters,
                                           batch_size=self.batch_size, hidden_dim=self.hidden_dim,
                                           start_token=self.start_token,
                                           goal_out_size=goal_out_size, step_size=4,
                                           l2_reg_lambda=self.l2_reg_lambda)

        self.generator = Generator(num_classes=2, num_vocabulary=self.vocab_size, batch_size=self.batch_size,
                                   emb_dim=self.emb_dim, dis_emb_dim=self.dis_embedding_dim, goal_size=self.goal_size,
                                   hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                   input_length=self.input_length,
                                   filter_sizes=self.filter_size, start_token=self.start_token,
                                   num_filters=self.num_filters, goal_out_size=goal_out_size,
                                   D_model=self.discriminator,
                                   step_size=4)

        self.saver = tf.train.Saver()
        self.best_pre_saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

        self.val_bleu1 = Bleu(real_text=self.trunc_val_file, gram=1)
        self.val_bleu2 = Bleu(real_text=self.trunc_val_file, gram=2)

    def train_discriminator(self):
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num,
                             self.gen_data_loader, self.generator_file)
        self.dis_data_loader.load_train_data(self.truth_file, self.generator_file)
        for _ in range(3):
            self.dis_data_loader.next_batch()
            x_batch, y_batch = self.dis_data_loader.next_batch()
            feed = {
                self.discriminator.D_input_x: x_batch,
                self.discriminator.D_input_y: y_batch,
            }
            _, _ = self.sess.run([self.discriminator.D_loss, self.discriminator.D_train_op], feed)
            self.generator.update_feature_function(self.discriminator)

    def eval(self):
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num,
                             self.gen_data_loader, self.generator_file)
        if self.log is not None:
            if self.epoch == 0 or self.epoch == 1:
                for metric in self.metrics:
                    self.log.write(metric.get_name() + ',')
                self.log.write('\n')
            scores = super().eval()
            for score in scores:
                self.log.write(str(score) + ',')
            self.log.write('\n')
            return scores
        return super().eval()

    def init_metric(self):
        # docsim = DocEmbSim(oracle_file=self.truth_file, generator_file=self.generator_file,
        #                    num_vocabulary=self.vocab_size)
        # self.add_metric(docsim)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

        bleu1 = Bleu(test_text=self.test_file, real_text=self.trunc_train_file, gram=1)
        bleu1.set_name('BLEU-1')
        self.add_metric(bleu1)

        bleu2 = Bleu(test_text=self.test_file, real_text=self.trunc_train_file, gram=2)
        bleu2.set_name('BLEU-2')
        self.add_metric(bleu2)

    def train(self, restore=False, model_path=None):
        self.gen_data_loader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length,
                                          input_length=self.input_length)
        self.dis_data_loader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        tokens = get_tokens(self.train_data)
        with open(self.truth_file, 'w', encoding='utf-8') as outfile:
            outfile.write(text_to_code(tokens, self.wi_dict, self.sequence_length))

        wi_dict, iw_dict = self.wi_dict, self.iw_dict

        self.init_metric()

        def get_real_test_file(dict=iw_dict):
            codes = get_tokens(self.generator_file)
            with open(self.test_file, 'w', encoding='utf-8') as outfile:
                outfile.write(code_to_text(codes=codes[self.input_length:], dict=dict))

        if restore:
            self.pre_epoch_num = 0
            if model_path is not None:
                self.model_path = model_path
            savefile = tf.train.latest_checkpoint(self.model_path)
            self.saver.restore(self.sess, savefile)
        else:
            self.sess.run(tf.global_variables_initializer())
            self.pre_epoch_num = 80

        # self.adversarial_epoch_num = 100
        self.log = open('log/experiment-log.txt', 'w', encoding='utf-8')
        self.gen_data_loader.create_batches(self.truth_file)
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num,
                             self.gen_data_loader, self.generator_file)

        self.gen_data_loader.reset_pointer()
        for a in range(1):
            inputs, target = self.gen_data_loader.next_batch()
            g = self.sess.run(self.generator.gen_x,
                              feed_dict={self.generator.drop_out: 1, self.generator.train: 1,
                                         self.generator.inputs: inputs})

        print('start pre-train generator:')
        best = 0
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.epoch += 1
            if epoch % 5 == 0:
                generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num,
                                     self.gen_data_loader, self.generator_file)
                get_real_test_file()
                scores = self.eval()
                self.saver.save(self.sess, self.save_path, global_step=epoch)
                if scores[3] > best:
                    print('--- Saving best-pre-gen...')
                    best = scores[3]
                    self.best_pre_saver.save(self.sess, self.best_path_pre, global_step=epoch)

        print('start pre-train discriminator:')
        # self.epoch = 0
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()
        self.saver.save(self.sess, self.save_path, global_step=self.pre_epoch_num*2)

        self.epoch = 0
        best = 0
        self.reward = Reward(model=self.generator, dis=self.discriminator, sess=self.sess, rollout_num=4)
        for epoch in range(self.adversarial_epoch_num // 10):
            for epoch_ in range(10):
                print('epoch:' + str(epoch) + '--' + str(epoch_))
                start = time()
                for index in range(1):
                    inputs, target = self.gen_data_loader.next_batch()
                    samples = self.generator.generate(self.sess, 1, inputs=inputs)
                    rewards = self.reward.get_reward(samples, inputs)
                    feed = {
                        self.generator.x: samples,
                        self.generator.reward: rewards,
                        self.generator.drop_out: 1,
                        self.generator.inputs: inputs
                    }
                    _, _, g_loss, w_loss = self.sess.run(
                        [self.generator.manager_updates, self.generator.worker_updates, self.generator.goal_loss,
                         self.generator.worker_loss, ], feed_dict=feed)
                    print('epoch', str(epoch), 'g_loss', g_loss, 'w_loss', w_loss)
                end = time()
                self.epoch += 1
                print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
                if self.epoch % 5 == 0 or self.epoch == self.adversarial_epoch_num - 1:
                    generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num,
                                         self.gen_data_loader, self.generator_file)
                    get_real_test_file()
                    scores = self.eval()

                    print('--- Generating poem on val data... ')
                    target_file = 'save/gen_val/val_%d_%f.txt' % (self.epoch, scores[1])
                    self.infer(test_data=self.val_data, target_path=target_file,
                               model_path=self.model_path, restore=False, trunc=True)
                    self.val_bleu1.test_data = target_file
                    self.val_bleu2.test_data = target_file
                    bleu1, bleu2 = self.val_bleu1.get_score(), self.val_bleu2.get_score()
                    print('--- BLEU on val data: \t bleu1: %f \t bleu2: %f' % (bleu1, bleu2))
                    if bleu2 > best:
                        best = bleu2
                        print('--- Saving best-leak-gan...')
                        self.best_saver.save(self.sess, self.best_path, global_step=epoch*10+epoch_)


                for _ in range(15):
                    self.train_discriminator()

            self.saver.save(self.sess, self.save_path, global_step=1+epoch+self.pre_epoch_num*2)
            for epoch_ in range(5):
                start = time()
                loss = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader)
                end = time()
                print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
                if epoch % 5 == 0:
                    generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num,
                                         self.gen_data_loader, self.generator_file)
                    get_real_test_file()
                    self.eval()

            for epoch_ in range(5):
                print('epoch:' + str(epoch) + '--' + str(epoch_))
                self.train_discriminator()

    def infer(self, test_data, target_path, model_path, restore=True, trunc=False):
        if model_path is None:
            model_path = self.model_path

        if restore:
            savefile = tf.train.latest_checkpoint(model_path)
            self.saver.restore(self.sess, savefile)

        tokens = get_tokens(test_data)
        sentence_num = len(tokens)
        temp_file = 'save/infer_temp.txt'
        with open(temp_file, 'w', encoding='utf-8') as outfile:
            outfile.write(text_to_code(tokens, self.wi_dict, self.input_length))

        test_data_loader = TestDataloader(batch_size=self.batch_size, input_length=self.input_length)
        test_data_loader.create_batches(temp_file)

        generate_samples_gen(self.sess, self.generator, self.batch_size, test_data_loader.num_batch * self.batch_size,
                             test_data_loader, target_path)

        codes = get_tokens(target_path)[:sentence_num]
        with open(target_path, 'w', encoding='utf-8') as outfile:
            if trunc:
                outfile.write(code_to_text(codes=codes[self.input_length:], dict=self.iw_dict))
            else:
                outfile.write(code_to_text(codes=codes, dict=self.iw_dict))

        print('Finished generating %d poems to %s' % (sentence_num, target_path))
