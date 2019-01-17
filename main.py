import tensorflow as tf
from model.LeakGan import LeakGan

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_data', 'data/train.txt', 'Training data path')
flags.DEFINE_string('val_data', 'data/val.txt', 'Validation data path')
flags.DEFINE_string('wi_dict', 'save/word_index_dict.pk', 'Word to Index dict path')
flags.DEFINE_string('iw_dict', 'save/index_word_dict.pk', 'Index to Word dict path')
flags.DEFINE_boolean('restore', False, 'Restore from model checkpoint')
flags.DEFINE_boolean('infer', False, 'Training or inferring using a trained model')
flags.DEFINE_boolean('best', False, 'Use model with best nll or latest(default)')
flags.DEFINE_string('model_path', 'save/model/LeakGan/', 'Specify model checkpoint path')
flags.DEFINE_string('test_data', 'data/test.txt', 'Test data path to infer')
flags.DEFINE_string('target_path', 'save/generated_poem.txt', 'Generating to')


def main(_):
    gan = LeakGan(FLAGS.wi_dict, FLAGS.iw_dict, FLAGS.train_data, FLAGS.val_data)

    if FLAGS.infer:
        if FLAGS.test_data is None:
            print('No test data specified! Exiting...')
            return

        model_path = FLAGS.model_path
        if FLAGS.best:
            model_path = gan.best_model_path
        gan.infer(FLAGS.test_data, FLAGS.target_path, model_path, FLAGS.train_data)

    else:
        gan.train(restore=FLAGS.restore, model_path=FLAGS.model_path)


if __name__ == '__main__':
    tf.app.run()