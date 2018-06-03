import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from TextCNN import textcnn

# Parameters
# ==================================================
# Data Parameters
tf.flags.DEFINE_string("pred_valid_file", "./submittion/results_textcnn.txt", "prediction of validation use to submit")
tf.flags.DEFINE_string('validate_data_file', './data_bunch/cnn_non_stop_words_validate_bunch.dat', "get it's label to submmit")
tf.flags.DEFINE_string('word2vec_model_path', './trained_word2vec.model',"path of word2vec model, use it to representate chinese data")
# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", "./checkpoints/textcnn_model.ckpt", "Checkpoint directory from training run")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Parse parameters from commands
FLAGS = tf.flags.FLAGS

# validate
# ==================================================

# load data of needing to predict
validate_data = data_helper.read_bunch(FLAGS.validate_data_file)
# extract the data of needing to predict and submmit
id_validate = validate_data.news_id
x_validate = validate_data.news_words_jieba

# fix the length of per sentences
sequence_length = 1657           # depend on training data

# first: convert the string data to word vectors(Get Embedding vector x_validate)
x_validate = np.array(data_helper.sentence2vector(x_validate,length_per_sequence=sequence_length,word2vec_model_path=FLAGS.word2vec_model_path))
print("x_validate.shape = {}".format(x_validate.shape))

# Prediction
# ==================================================
print("\nPrediction ...\n")
# loading trained model parameters and use it to predict
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    
    with tf.Session(config=session_conf) as sess:
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Collect the predictions here
        all_predictions = []

        pred_validate = sess.run(predictions, {input_x: x_validate, dropout_keep_prob: 1.0})
        print(pred_validate)

# store prediction to local and use it to submmit
# combine prediction and other information to submmition
bayes_text = []
for i in range(len(id_validate)):
    bayes_text.append('NULL')
label_predict = pred_validate
bayes_result = []
for i in range(len(id_validate)):
    bayes_result.append(id_validate[i]+'\t'+label_predict[i]+'\t'+bayes_text[i]+'\t'+bayes_text[i])
data_helper.save_text(FLAGS.pred_valid_file, bayes_result)


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
