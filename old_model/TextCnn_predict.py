# coding = utf-8
import time
import TextCnn_inference
import instrument
import numpy as np
import tensorflow as tf 
from tensorflow.contrib import learn

# Data loading parameters
tf.flags.DEFINE_float('test_sample_percentage', 0.1, 'Percentage of the training data to use for testing')
tf.flags.DEFINE_string('training_data_file', './data_bunch/cnn_non_stop_words_train_bunch.dat', 'Data for training')
tf.flags.DEFINE_string('training_data_label_file', './data_bunch/cnn_train_label_bunch.dat', 'Some infomation for training,such as label, picture list and so on')

# Model Hyperparameters
tf.flags.DEFINE_integer('embedding_size', 64, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string('filter_sizes', '2,3,4', "Comma-separated filter sizes (defulat: '2,3,4')")
tf.flags.DEFINE_integer('num_filters', 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float('dropout_keep_prob', 1.0, "Dropout keep probality (default: 0.5)")
tf.flags.DEFINE_integer('fc1_nodes', 50, "Number of nodes in first full connection layer (default: 50) ")

# Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, "Allow device soft device placement")
tf.flags.DEFINE_boolean('log_device_placement', False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# Load data set
from instrument import read_bunch
train_data = read_bunch(FLAGS.training_data_file)
train_label = read_bunch(FLAGS.training_data_label_file)

# Map vocabulary to index
vocab_path = './vocabulary'
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
train_matrix = np.array(list(vocab_processor.transform(train_data.news_words_jieba)))

eval_interval_secs = 120

# 模型保存路径和文件名
model_save_path = './model/'
model_name = 'model.ckpt'

def evaluate(x_array, y_label, FLAGS):
    with tf.Graph().as_default() as g:
        # Split train/test set
        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(x_array,
                                                            y_label,
                                                            test_size=FLAGS.test_sample_percentage,
                                                            random_state=33)
        del x_array, y_label, train_x,  train_y


        # Placeholders for input, output and dropout
        x  = tf.placeholder(tf.int32, [None, test_x.shape[1]], name='input-x')
        y_ = tf.placeholder(tf.float32, [None, len(test_y[1])], name='input-y')

        # use the inference function to compute logit
        y = TextCnn_inference.inference(input_tensor=x,
                                        train=False,
                                        sequence_length=test_x.shape[1],
                                        embedding_size=FLAGS.embedding_size,
                                        vocab_size=len(vocab_processor.vocabulary_),
                                        filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                                        num_filters=FLAGS.num_filters,
                                        dropout_keep_prob=FLAGS.dropout_keep_prob,
                                        fc1_nodes=FLAGS.fc1_nodes,
                                        num_classes=len(test_y[1]))

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        validate_feed = {x: test_x, y_: test_y}
        
        saver = tf.train.Saver()
        # session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
        #                             log_device_placement=FLAGS.log_device_placement)
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(model_save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(eval_interval_secs)

def main(argv=None):
    evaluate(train_matrix, train_label.news_pic_label_one_hot, FLAGS=FLAGS)

if __name__ == '__main__':
    main()