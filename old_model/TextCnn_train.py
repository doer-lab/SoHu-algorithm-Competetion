# coding = utf-8

# 导入函数库
import datetime
import random
import numpy as np 
from TextCnn_inference import inference
import tensorflow as tf
from tensorflow.contrib import learn

# pycharm not need the “/souhu_algorithm_competition"
# 参数设置
#=====================================================#
# Data loading params
tf.flags.DEFINE_float('test_sample_percentage', 0.005, 'Percentage of the training data to use for testing')
tf.flags.DEFINE_string('training_data_file', './data_bunch/cnn_non_stop_words_train_bunch.dat', 'Data for training')
tf.flags.DEFINE_string('training_data_label_file', './data_bunch/cnn_train_label_bunch.dat', 'Some infomation for training,such as label, picture list and so on')

# Model Hyperparameters
tf.flags.DEFINE_integer('embedding_size', 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string('filter_sizes', '2,3,4', "Comma-separated filter sizes (defulat: '2,3,4')")
tf.flags.DEFINE_integer('num_filters', 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, "Dropout keep probality (default: 0.5)")
tf.flags.DEFINE_integer('fc1_nodes', 50, "Number of nodes in first full connection layer (default: 50) ")
# tf.flags.DEFINE_float('moving_average_decay', 0.99, "移动平均指数")
tf.flags.DEFINE_float('learning_rate_basic', 0.9, "基础学习率")
tf.flags.DEFINE_float('learning_rate_decay', 0.96, "学习率衰减速度")

# Training parameters
tf.flags.DEFINE_integer('batch_size', 500, "Batch size (default: 64)")
tf.flags.DEFINE_integer('num_epochs', 200000, "Number of training epochs (default: 2000)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, "Allow device soft device placement")
tf.flags.DEFINE_boolean('log_device_placement', False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# Load data
from instrument import read_bunch
train_data = read_bunch(FLAGS.training_data_file)
train_label = read_bunch(FLAGS.training_data_label_file)
# validate_data = read_bunch(FLAGS.validate_data_file)

# Build vocabulary
# sequence_length = max(train_data.news_length)                        # 取句子的最大长度
sequence_length = int(np.percentile(train_data.news_length,95))        #95%分位数
vocab_processor = learn.preprocessing.VocabularyProcessor(sequence_length)
train_matrix = np.array(list(vocab_processor.fit_transform(train_data.news_words_jieba)))
# Writer vocabulary to local
vocab_path = './vocabulary'
vocab_processor.save(vocab_path)
# release memeory
del train_data

# save trained model to local with a name
model_save_path = './model/'
model_name = 'TextCnn.ckpt'

# Training
#=========================================================#
def TextCnn_train(x_array,y_label, FLAGS):
    # Split train/test set
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(x_array,
                                                        y_label,
                                                        test_size=FLAGS.test_sample_percentage,
                                                        random_state=33)
    # Placeholders for input, output and dropout
    x  = tf.placeholder(tf.int32, [None, train_x.shape[1]], name='input-x')
    y_ = tf.placeholder(tf.float32, [None, len(train_y[1])], name='input-y')
    DropOut_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
    # use the inference function to compute logit
    y = inference(input_tensor= x,
                  sequence_length=train_x.shape[1],
                  embedding_size=FLAGS.embedding_size,
                  vocab_size=len(vocab_processor.vocabulary_),
                  filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                  num_filters=FLAGS.num_filters,
                  dropout_keep_prob= DropOut_keep_prob,
                  fc1_nodes=FLAGS.fc1_nodes,
                  num_classes=len(train_y[1]))
    global_step = tf.Variable(0, trainable=False)

    # define accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # define cost function, learning rate, moving average operation and train step
    cross_entroy = -(tf.reduce_sum(y_*tf.log(tf.clip_by_value(tf.nn.softmax(y), 1e-30, 1.0)), reduction_indices=[1]))
    loss = tf.reduce_mean(cross_entroy)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate_basic,
                                                global_step, train_x.shape[0]/FLAGS.batch_size,
                                                FLAGS.learning_rate_decay)
    train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()

    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement)
    with tf.Session(config=session_conf) as sess:
        tf.global_variables_initializer().run()
        
        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序完成
        for epoch in range(FLAGS.num_epochs):
            # 随机从训练集中抽取 batch_size 条数据参加模型训练
            shuffle_indices = np.random.permutation(np.arange(train_x.shape[1]))
            batch_indices = random.sample(list(shuffle_indices), FLAGS.batch_size)
            batch_x, batch_y = zip(*np.array(list(zip(train_x, train_y)))[batch_indices])
            print(''.center(130, '*'))
            print(batch_x)
            print(batch_y)

            # Training
            train_feed_dict = {x: batch_x, y_: batch_y, DropOut_keep_prob: FLAGS.dropout_keep_prob}
            _, step, train_loss, train_accuracy, y_value = sess.run([train_step, global_step, loss, accuracy, y], train_feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, train_loss, train_accuracy))
            print(y_value)

            current_step = tf.train.global_step(sess, global_step)
            if current_step % 10 == 0:   # 每 FLAGS.evaluate_every 步进行验证一次
                print("\nEvaluation:")
                # data structure is tuple
                test_feed_dict = {x: tuple(test_x), y_: tuple(test_y), DropOut_keep_prob: 1.0}
                step, valid_accuracy = sess.run([global_step, accuracy], test_feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, acc {:g}".format(time_str, step, valid_accuracy))
                print("")
            if current_step % FLAGS.checkpoint_every == 0:  # 每 FLAGS.checkpoint_every 保存一次模型
                path = saver.save(sess, './model/TextCnn_model.ckpt', global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    TextCnn_train(train_matrix, train_label.news_pic_label_one_hot, FLAGS=FLAGS)

if __name__ == '__main__':
    main()

