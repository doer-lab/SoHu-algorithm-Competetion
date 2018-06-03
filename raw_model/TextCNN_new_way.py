import os
import time
import datetime
import data_helper
import tensorflow as tf
import numpy as np
from TextCNN_model import textcnn

# Parameters /sohu_algorithm_competition
# ==================================================
# Data Parameters
tf.flags.DEFINE_string("pred_valid_file", "./submittion/results_textcnn.txt", "prediction of validation use to submit")
tf.flags.DEFINE_string('validate_data_file', './data_bunch/cnn_non_stop_words_validate_bunch.dat', "get it's label to submmit")
tf.flags.DEFINE_string('word2vec_model_path', './word2vec_new/trained_word2vec.model',"path of word2vec model, use it to representate chinese data")
tf.flags.DEFINE_integer('concatenate_size',100,'need to concatenate samples for a time')
tf.flags.DEFINE_integer("num_classes", 3, "Number of labels for data")
# Model hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 400, "Dimensionality of character embedding (default: 400)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5,6", "Comma-spearated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Checkpoint directory from training run")
# Parse parameters from commands
FLAGS = tf.flags.FLAGS

# validate
# ==================================================
# load data of needing to predict
validate_data = data_helper.read_bunch(FLAGS.validate_data_file)
# extract the data of needing to predict and submmit
id_validate = validate_data.news_id
x_validate = validate_data.news_words_jieba
del validate_data

# fix the length of per sentences
sequence_length = 1657           # depend on training data

# first: convert the string data to word vectors(Get Embedding vector x_validate)
print('convert the string data to word vectors ...')
x_validate = data_helper.sentence2vector(x_validate,length_per_sequence=sequence_length,word2vec_model_path=FLAGS.word2vec_model_path)
NumIters = int((len(x_validate)-1)/FLAGS.concatenate_size) + 1

# Prediction
# ==================================================
print("\nPrediction ...\n")
# loading trained model parameters and use it to predict
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
    )
    session_conf.gpu_options.allow_growth = True

    with tf.Session(config = session_conf) as sess:
        cnn = textcnn(
        sequence_length = sequence_length,
        num_classes = FLAGS.num_classes,
        embedding_size = FLAGS.embedding_dim,
        filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters = FLAGS.num_filters)

        # Initialize all variables
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,checkpoint_file)

        # Collect the predictions here
        all_predictions = []

        test_valid_batchs = data_helper.batch_concat(x_validate, batch_size=FLAGS.concatenate_size, num_iters=NumIters)
        for test_valid_batch in test_valid_batchs:
            print("it.s validate set ...")
            print(test_valid_batch)
            pred_scores = sess.run(cnn.scores, {cnn.input_x: test_valid_batch, cnn.dropout_keep_prob: 1.0})
            # pred_validate = tf.argmax(pred_scores,1)
            print(pred_scores)
            pred_validate = sess.run(cnn.predictions, {cnn.input_x: test_valid_batch, cnn.dropout_keep_prob: 1.0})
            print(pred_validate)
            all_predictions.extend(pred_validate)
        
   

    # with tf.Session(config=session_conf) as sess:
    #     # Load the saved meta graph and restore variables
    #     saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    #     saver.restore(sess, checkpoint_file)

    #     # Get the placeholders from the graph by name
    #     input_x = graph.get_operation_by_name("input_x").outputs[0]
    #     dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

    #     # Tensors we want to evaluate
    #     scores = graph.get_operation_by_name("output/scores").outputs[0]
    #     predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # batches = data_helper.batch_iter(list(x_validate), FLAGS.concatenate_size, 1, shuffle=False)

        # # Collect the predictions here
        # all_predictions = []

        # for x_validate_batch in batches:
        #     batch_predictions = sess.run(predictions, {input_x: x_validate_batch, dropout_keep_prob: 1.0})
        #     all_predictions = np.concatenate([all_predictions, batch_predictions])
        #     print(all_predictions)

        # Collect the predictions here
        # all_predictions = []

        # # for sample in x_validate:
        # #     pred_validate = sess.run(predictions, {input_x: np.array(sample), dropout_keep_prob: 1.0})
        # #     print(pred_validate)
        # #     all_predictions.extend(pred_validate)


        # test_valid_batchs = data_helper.batch_concat(x_validate, batch_size=FLAGS.concatenate_size, num_iters=NumIters)
        # for test_valid_batch in test_valid_batchs:
        #     print("it.s validate set ...")
        #     print(test_valid_batch)
        #     pred_scores = sess.run(scores, {input_x: test_valid_batch, dropout_keep_prob: 1.0})
        #     # pred_validate = tf.argmax(pred_scores,1)
        #     print(pred_scores)
        #     pred_validate = sess.run(predictions, {input_x: test_valid_batch, dropout_keep_prob: 1.0})
        #     print(pred_validate)
        #     all_predictions.extend(pred_validate)

print(all_predictions)
# store prediction to local and use it to submmit
# combine prediction and other information to submmition
results_text = []
for i in range(len(id_validate)):
    results_text.append('NULL')
label_predict = all_predictions
all_results = []
for i in range(len(id_validate)):
    all_results.append(id_validate[i]+'\t'+str(label_predict[i])+'\t'+results_text[i]+'\t'+results_text[i])
data_helper.save_text(FLAGS.pred_valid_file, all_results)


'''
if checkpoint_file is None:
    print("Cannot find a valid checkpoint file!")
    exit(0)
print("Using checkpoint file : {}".format(checkpoint_file))
# validate word2vec model file
trained_word2vec_model_file = os.path.join(FLAGS.checkpoint_dir, "..", "trained_word2vec.model")
if not os.path.exists(trained_word2vec_model_file):
    print("Word2vec model file \'{}\' doesn't exist!".format(trained_word2vec_model_file))
print("Using word2vec model file : {}".format(trained_word2vec_model_file))
'''
