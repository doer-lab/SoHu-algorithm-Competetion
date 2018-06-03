import datetime
import pickle
import multiprocessing
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

def train_word2vec_model(sentences,embedding_size=400,window=5,min_count=5,iter=500,word2vec_model_path=None):
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
    print('training word2vec model ...')
    w2vModel = Word2Vec(sentences,size=embedding_size,iter=iter,window=window,min_count=min_count,workers=multiprocessing.cpu_count())
    w2vModel.save(word2vec_model_path)
    print('save word2vec model to {}'.format(word2vec_model_path))

    # print('find trained word2vec model, load it and continue to train ...')
    # w2vModel = Word2Vec.load(word2vec_model_path)
    # print('loading finished!')
    # w2vModel.train(sentences, total_examples=len(sentences), epochs=400)
    # w2vModel.save(word2vec_model_path)
    # print('save word2vec model to {}'.format(word2vec_model_path))


    # if word2vec_model_path is not None:   # 验证存在的模型和需要的模型是否一致
    #     print('find trained word2vec model, load it and continue to train ...')
    #     w2vModel = Word2Vec.load(word2vec_model_path)
    #     print('loading finished!')
    #     w2vModel.train(sentences, total_examples=len(sentences), epochs=400)
    #     w2vModel.save(word2vec_model_path)
    #     print('save word2vec model to {}'.format(word2vec_model_path))
    # else:
    #     print('not find trained word2vec model, start to train it ...')
    #     w2vModel = Word2Vec(sentences,size=embedding_size,iter=iter,window=window,min_count=min_count,workers=multiprocessing.cpu_count())
    #     w2vModel.save(word2vec_model_path)
    #     print('save word2vec model to {}'.format(word2vec_model_path))

# parameters
train_data_path = 'E:/Competition/souhu_algorithm_competition/data_bunch/cnn_non_stop_words_train_bunch.dat'
unlabel_data_path = 'E:/Competition/souhu_algorithm_competition/data_bunch/cnn_non_stop_words_unlabel_news_bunch.dat'
train_label_path = 'E:/Competition/souhu_algorithm_competition/data_bunch/cnn_train_label_bunch.dat'
validate_data_path = 'E:/Competition/souhu_algorithm_competition/data_bunch/cnn_non_stop_words_validate_bunch.dat'

word2vec_model_path = 'E:/Competition/souhu_algorithm_competition/word2vec/word2vec/trained_word2vec.model'

# train word2vec model on training data and validating data(3小时20分钟)
print('loading data ...')
train_data = read_bunch(train_data_path).news_words_jieba
validate_data = read_bunch(validate_data_path).news_words_jieba
time_str = datetime.datetime.now().isoformat()
print("now is {}".format(time_str))

sentences = train_data + validate_data

print('word2vec model is training ...')
train_word2vec_model(sentences, word2vec_model_path=word2vec_model_path)
print('training finished!')
time_str = datetime.datetime.now().isoformat()
print("now is {}".format(time_str))

# train word2vec model on training data ( 一小时42分钟 )
# print('loading training data ...')
# train_data = read_bunch(train_data_path).news_words_jieba
# time_str = datetime.datetime.now().isoformat()
# print("now is {}".format(time_str))
#
# print('word2vec model is training ...')
# train_word2vec_model(train_data, word2vec_model_path=word2vec_model_path)
# print('training finished!')
#
# time_str = datetime.datetime.now().isoformat()
# print("now is {}".format(time_str))

# # continue to train word2vec model on unlabel data
# print('loading unlabel data ...')
# unlabel_data = read_bunch(unlabel_data_path).news_words_jieba
# time_str = datetime.datetime.now().isoformat()
# print("now is {}".format(time_str))
# print("")

# print('word2vec model is training ...')
# train_word2vec_model(unlabel_data, word2vec_model_path=word2vec_model_path)
# print('training finished!')

# time_str = datetime.datetime.now().isoformat()
# print("now is {}".format(time_str))
