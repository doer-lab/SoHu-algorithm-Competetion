# coding = utf-8

import tensorflow as tf

# 配置神经网络参数
# input_tensor
# sequence_length                 # 句子的长度（每条句子的长度一致）
# embedding_size                  # 将每个字或词表示为 1 x embedding_size 的向量
# vocab_size                      # 词典的大小（词典中包含训练数据集分词后的所有字、词和某些标点符号）
# filter_sizes                    # filter 的尺寸列表
# num_filters                     # 每个尺寸 filter 的个数
# dropout_keep_prob               # 训练过程中以概率 dropout_keep_prob 随机选择一部分节点进行训练
# fc1_nodes                       # 第一个全连接层中神经元的个数
# num_classes                     # 第二个全连接层中神经元的个数（默认为标签种类的数目）

def inference(input_tensor, sequence_length, embedding_size, vocab_size,
              filter_sizes, num_filters, dropout_keep_prob, fc1_nodes, num_classes):
    # embedding layer
    with tf.variable_scope('layer1-embeddeing'):
        embededd_weights = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="weights")
        # tf.get_variable('weights', [vocab_size, embedding_size],
        #                                    initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
        embedded_chars = tf.nn.embedding_lookup(embededd_weights, input_tensor)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    
    # create a convolutional layer and max pooling layer for each filter size
    pooled_outputs = []
    for _, filter_size in enumerate(filter_sizes):
        with tf.variable_scope('layer2-conv-maxpool-%s' % filter_size):
            conv_weights = tf.get_variable('weights', [filter_size, embedding_size, 1, num_filters],  #####
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv_biases = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(embedded_chars_expanded, conv_weights,
                                strides=[1, 1, 1, 1], padding='VALID')
            # Apply nonlinearity
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
            # maxpooling over the outputs
            pooled = tf.nn.max_pool(relu, ksize=[1, sequence_length-filter_size+1, 1, 1],
                                    strides=[1, 1, 1, 1], padding='VALID')
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    pool = tf.concat(pooled_outputs, 3)
    fc_pool = tf.reshape(pool, [-1, num_filters_total])

    # Add dropout
    with tf.variable_scope('layer3-dropout'):
        fc_pool = tf.nn.dropout(fc_pool, dropout_keep_prob)

    # full connection layer one
    with tf.variable_scope('layer4-fc1'):
        fc1_weights = tf.get_variable('weights', [num_filters_total, fc1_nodes],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_biases = tf.get_variable('biases', [fc1_nodes], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(fc_pool, fc1_weights) + fc1_biases)
        fc1 = tf.nn.dropout(fc1, dropout_keep_prob)

    # full connection layer two
    with tf.variable_scope('layer5-fc2'):
        fc2_weights = tf.get_variable('weights', [fc1_nodes, num_classes],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable('biases', [num_classes],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        logit = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)

    print(logit)
    
    return logit

