import pickle
import multiprocessing
import numpy as np
from gensim.models import Word2Vec

def read_bunch(bunch_path):
    '''
    1、读取指定的 bunch 对象
    2、for example（读取保存在当前工作目录下的 train_bunch_balance.dat 文件）:
        train_bunch_path = './data_bunch/train_bunch_balance.dat'
        train_bunch = read_bunch(train_bunch_path)
    '''
    with open(bunch_path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

def save_text(save_path, save_data):
    '''
    将数据 save_data 保存为指定路径下的 txt 文件
    1、save_path 为存储 txt 文件的地址（目录）
    2、save_data 为需要存储到本地 txt 文件的数据
    3、 for example（保存为当前工作目录下的 submittion_validate_0.001.txt 文件）:
        save_path = './submittion_validate_0.001.txt'
        save_text(save_path,bayes_result)
    '''
    with open(save_path, 'w') as f:
        for i in range(len(save_data)):
            f.write(str(save_data[i])+'\n')
    return

def padding_sentences(input_sentences, padding_token, padding_sentence_length=None):
    # 固定句子的长度：太长的截取最前面的，不足的在句尾添加 padding_token
    # 返回固定句长样本及每条样本的长度
    sentences = [sentence.split(' ') for sentence in input_sentences]
    sentence_length = padding_sentence_length if padding_sentence_length is not None else max(
            [len(sentence) for sentence in sentences])
    all_vector=[]
    for sentence in sentences:
        if len(sentence) > sentence_length:
            sentence = sentence[:sentence_length]
        else:
            sentence.extend([padding_token] * (sentence_length - len(sentence)))
        all_vector.append(sentence)
    return (all_vector, sentence_length)

def train_word2vec_model(sentences,embedding_size=400,window=5,min_count=5,iter=200,word2vec_model_path=None):
    '''
    1、模型参数说明
        embeding_size: 输出词向量的维数.值太小会导致词映射因为冲突而影响结果,值太大则会耗内存并使算法计算变慢,一般值取为100到200之间
        window: 句子中当前词与目标词之间的最大距离,3表示在目标词前看3-b个词,后面看b个词（b在0-3之间随机）
        min_count: 对词进行过滤,频率小于 min-count 的单词则会被忽视,默认值为5
        hs=1: 表示层级softmax将会被使用,默认hs=0且negative不为0,则负采样将会被选择使用
        workers: 控制训练的并行.此参数只有在安装了Cpython后才有效,否则只能使用单核
        sg=1: 是skip-gram算法,对低频词敏感;默认sg=0为CBOW算法
        iter: 随机梯度下降算法中迭代的最大次数,默认是5;对于大语料,可以增大这个值
    2、模型训练结束后返回的结果
        返回样本的向量空间（3维的）,每个句子被表示为 [sequence_length, embedding_size], [num_examples, sequence_length, embeeding_dize]
    3、句子的长短不影响 word2vec 模型的训练
    4、统一模型的保存地址为 word2vec_model_path
    '''
    sentences = [sentence.split(' ') for sentence in sentences]

    if word2vec_model_path is not None:   # 验证存在的模型和需要的模型是否一致
        print('find trained word2vec model, load it and contine to train ...')
        w2vModel = Word2Vec.load(word2vec_model_path)
        w2vModel.train(sentences)
        w2vModel.save(word2vec_model_path)
        print('save word2vec model to {}'.format(word2vec_model_path))
    else:
        print('not find trained word2vec model, start to train it ...')
        w2vModel = Word2Vec(sentences,size=embedding_size,iter=iter,window=window,min_count=min_count,workers=multiprocessing.cpu_count())
        w2vModel.save(word2vec_model_path)
        print('save word2vec model to {}'.format(word2vec_model_path))

def sentence2vector(sentence_need_to_vector,length_per_sequence,word2vec_model_path=None):
    """
    在训练好的 word2vec 模型的基础上进行句子向量化，训练好的词向量中，每个词被表示为一个400维的矩阵
    sentence_word: 每个句子为一个词列表，词与词之间通过空格相连接
    返回的 all_vectors ，其维度为[num_simples,1657,400]。其中，每个句子的维度为 [sequence_length, embedding_size], 也即 1657,400]
    句子的长短不影响得到最后的词向量 
    """
    print("start sentence padding ...")
    sentences, padding_sentence_length = padding_sentences(sentence_need_to_vector, '<PADDING>', padding_sentence_length=length_per_sequence)
    print("embedding finished!")
    print("number of examples is {}, the length of per sentence after padding is {}".format(len(sentences[0]),padding_sentence_length))
    
    print("word representation ...")
    if word2vec_model_path is None:
        print("cannot find trained word2vec model, please checking again ...")
    else:
        print("find trained word2vec model, loading it ...")
        w2vModel = Word2Vec.load(word2vec_model_path)
    
    print("sentences to vectors is running ...")
    all_vectors = []
    embeddingDim = w2vModel.vector_size
    print('embedding dimension is ', embeddingDim)
    # 嵌入维数
    embeddingUnknown = [0 for i in range(embeddingDim)]
    for sentence in sentences:
        this_vector = []
        for word in sentence:
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)
        all_vectors.append(this_vector)
    print("Fnished!".center(100,'*-*'))

    return all_vectors

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    '''
    Generate a batch iterator for a dataset
    Total num_epchs * num_batches_per_epoch iters
    '''
    data = np.array(data)
    data_size = len(data)                                      # 训练样本的总数
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1   # batch_size 表示每个batch所含样本数，计算出总共有多少个 batch
    for epoch in range(num_epochs):
        print('bath_iter函数中的第{}个epoch'.format(epoch))
        if shuffle:
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch in range(num_batches_per_epoch):              # 对于每个 batch 
            start_idx = batch * batch_size                      # 数据的起始点为 第几个batch * examples for per batch
            end_idx = min((batch + 1) * batch_size, data_size)  # 数据的结束点为 起始点+一个batch的大小 
            yield shuffled_data[start_idx: end_idx]

def embedding_sentences(sentences, embedding_size=400, window=5, min_count=5, iter=200, model_to_load=None, model_to_save=None):
    '''
    embeding_size : 词嵌入维数
    window : 上下文窗口
    min_count : 词频少于min_count会被删除
    iter : 训练次数
    返回样本的向量空间（3维的），每个句子被表示为 [sequence_length, embedding_size], [num_examples, sequence_length, embeeding_dize]
    '''
    if model_to_load is not None:
        print('find trained_word2vec.model and load it...')
        w2vModel = Word2Vec.load(model_to_load)
    else:
        print('not find trained_word2vec.model and start to train it ...')
        w2vModel = Word2Vec(sentences, size=embedding_size, iter=iter, window=window, min_count=min_count, workers=multiprocessing.cpu_count())
        if model_to_save is not None:
            w2vModel.save(model_to_save)
            print('save word2vec model to {}'.format(model_to_save))

    all_vectors = []
    embeddingDim = w2vModel.vector_size
    # 嵌入维数
    embeddingUnknown = [0 for i in range(embeddingDim)]
    for sentence in sentences:
        this_vector = []
        for word in sentence:
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)
        all_vectors.append(this_vector)
    return all_vectors

'''
def sentence2vector(sentence_need_to_vector,length_per_sequence=sequence_length,word2vec_model_path=None):
    """
    在训练好的 word2vec 模型的基础上进行句子向量化，训练好的词向量中，每个词被表示为一个400维的矩阵
    sentence_word: 每个句子为一个词列表，词与词之间通过空格相连接
    返回的 all_vectors ，其维度为[num_simples,1657,400]。其中，每个句子的维度为 [sequence_length, embedding_size], 也即 1657,400] 
    """
    print("start sentence padding ...")
    sentences, padding_sentence_length = data_helper.padding_sentences(sentence_need_to_vector, '<PADDING>', padding_sentence_length=length_per_sequence)
    print("embedding finished!")
    print("number of examples is {}, the length of per sentence after padding is {}".format(len(sentences[0]),padding_sentence_length))
    
    print("word representation ...")
    if word2vec_model_path is None:
        print("cannot find trained word2vec model, please checking again ...")
    else:
        print("find trained word2vec model, loading it ...")
        w2vModel = Word2Vec.load(word2vec_model_path)
    
    print("sentences to vectors is running ...")
    all_vectors = []
    embeddingDim = w2vModel.vector_size
    print('embedding dimension is ', embeddingDim)
    # 嵌入维数
    embeddingUnknown = [0 for i in range(embeddingDim)]
    for sentence in sentences:
        this_vector = []
        for word in sentence:
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)
        all_vectors.append(this_vector)
    print("Fnished!".center(100,'*-*'))

    return all_vectors
'''

'''
def data_preprocess():
    # 用 word2vec 将训练样本转换为样本空间，最后的输出为三维数据， [num_examples, sentence_legth, embedding_size]
    # Data preprocess
    # =======================================================
    # Load data
    print("Loading data...")
    x = train_data.news_words_jieba
    y = train_label.news_pic_label_one_hot

    # Get padding sentences
    sentences, document_length = data_helper.padding_sentences(x, '<PADDING>', padding_sentence_length=sequence_length)
    print(len(sentences[0]))

    # embedding vector
    if not os.path.exists(os.path.join(out_dir,"trained_word2vec.model")):
        x = np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size = FLAGS.embedding_dim, model_to_save = os.path.join(out_dir, 'trained_word2vec.model')))
    else:
        print('word2vec model found...')
        x = np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size = FLAGS.embedding_dim, model_to_save = os.path.join(out_dir, 'trained_word2vec.model'),model_to_load=os.path.join(out_dir, 'trained_word2vec.model')))
    y = np.array(y)
    # np.save(os.path.join(out_dir,"data_x.npy"),x)
    # np.save(os.path.join(out_dir,"data_y.npy"),y)
    del sentences

    # Shuffle data randomly
    # np.random.seed(10)
    # shuffle_indices = np.random.permutation(np.arange(len(y)))
    # x_shuffled = x[shuffle_indices]
    # y_shuffled = y[shuffle_indices]
    # del x,y

    # x_train, x_test, y_train, y_test = train_test_split(x_shuffled, y_shuffled, test_size=0.2, random_state=42)  # split into training and testing set 80/20 ratio
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=FLAGS.test_sample_percentage, random_state=42)  # split into training and testing set 80/20 ratio
    del x, y
    return x_train, x_test, y_train, y_test
'''
