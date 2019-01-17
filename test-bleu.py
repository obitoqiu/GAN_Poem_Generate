import nltk
import tensorflow as tf
from multiprocessing import Pool
import os
from utils.utils import get_tokens
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_boolean('sentence', False, 'Test Sentence-based Average BLEU')
flags.DEFINE_string('test_path', None, 'Test file path or directory')
flags.DEFINE_string('ref_file', 'data/val.txt', 'Reference file path')
flags.DEFINE_integer('gram', 2, 'Maximum n-gram BLEU')
flags.DEFINE_string('output', 'BLEU-result.txt', 'Output file')
flags.DEFINE_boolean('trunc', False, 'Use this flag if the test file contains only last 3 sentences')


def calc_bleu(reference, hypothesis, weights):
    return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights,
                                                   smoothing_function=SmoothingFunction().method1)


def get_bleu_parallel(ngram, reference, test_data):
    weights = tuple((1. / ngram for _ in range(ngram)))
    pool = Pool(os.cpu_count())
    result = list()
    for hypothesis in test_data:
        result.append(pool.apply_async(calc_bleu, args=(reference, hypothesis, weights)))
    score = 0.0
    cnt = 0
    for i in result:
        score += i.get()
        cnt += 1
    pool.close()
    pool.join()
    return score / cnt


def flatten(lst):
    a = np.array(lst)
    return list(a.flatten())


def main(_):
    ref_s = get_tokens(FLAGS.ref_file)
    ref_sentences = list()
    for poem in ref_s:
        ref_sentences.append(poem[8:])

    filelist = list()
    if os.path.isfile(FLAGS.test_path):
        filelist.append(FLAGS.test_path)
    elif os.path.isdir(FLAGS.test_path):
        filelist.extend([os.path.join(FLAGS.test_path, i)
                         for i in os.listdir(FLAGS.test_path) if i.endswith('.txt')])

    metrics = ['Filename']
    for i in range(FLAGS.gram):
        metrics.append('BLEU-%d' % (i+1))

    for item in metrics:
        print(item, end='\t')
    print()

    avg = list(np.zeros(FLAGS.gram))
    result = [metrics]
    for test_file in filelist:
        test_s = get_tokens(test_file)
        test_sentences = list()
        for poem in test_s:
            if FLAGS.trunc:
                test_sentences.append(poem)
            else:
                test_sentences.append(poem[8:])
        num_sentences = len(test_sentences)
        file_result = [test_file]

        scores = list(np.zeros(FLAGS.gram))
        for i in range(FLAGS.gram):
            scores[i] = get_bleu_parallel(i+1, ref_sentences, test_sentences)
            avg[i] += scores[i]
        file_result.extend(scores)

        '''
        for i in range(FLAGS.gram):
            scores[i] /= num_sentences
        file_result.extend(scores)  # S-BLEU

        scores = list(np.zeros(FLAGS.gram))
        ref = flatten(ref_sentences)
        test = flatten(test_sentences)
        for i in range(FLAGS.gram):
            weights = [1. / (i+1)] * (i+1)
            scores[i] += nltk.translate.bleu_score.sentence_bleu([ref], test, weights=weights,
                                                                 smoothing_function=SmoothingFunction().method1)
        file_result.extend(scores)  # S-whole-BLEU

        scores = list(np.zeros(FLAGS.gram))
        for poem in ref_sentences:
            poem = [poem]
        for i in range(FLAGS.gram):
            weights = [1. / (i+1)] * (i+1)
            scores[i] += nltk.translate.bleu_score.corpus_bleu(ref_sentences, test_sentences, weights=weights,
                                                               smoothing_function=SmoothingFunction().method1)
        file_result.extend(scores)
        '''

        result.append(file_result)

        for item in file_result:
            print(item, end='\t')
        print()

    avg_result = ['Average \t\t\t']
    for i in range(FLAGS.gram):
        avg_result.append(avg[i] / len(filelist))
    for item in avg_result:
        print(item, end='\t')
    print()
    result.append(avg_result)

    with open(FLAGS.output, 'w', encoding='utf-8') as fout:
        for line in result:
            for item in line:
                fout.write(str(item) + '\t')
            fout.write('\n')


if __name__ == '__main__':
    tf.app.run()