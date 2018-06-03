# coding = utf-8

import pickle
from sklearn.datasets.base import Bunch

# 读取文件
def readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content
# 读取bunch对象
def readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

# 读取训练集与待预测数据集
trainset_path = 'E:/Desktop/ZhuFei/CNN/TensorFlow/云移动/subject_1/train_set.dat'
testset_path = 'E:/Desktop/ZhuFei/CNN/TensorFlow/云移动/subject_1/test_set.dat'
train_bunch = readbunchobj(trainset_path)
test_bunch = readbunchobj(testset_path)

# 计算 TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# 停用词表
stopword_path = 'E:/Desktop/ZhuFei/CNN/TensorFlow/云移动/subject_1/stopwords_1.txt'
with open(stopword_path) as file:
    stopwd = [line.strip('\n') for line in file.readlines()]  # 删除 \n
stopwdlst = stopwd
# 构建训练集的 TF-IDF 词向量空间对象
tfidf_train = Bunch(Id=train_bunch.review_id, Score=train_bunch.review_score, tdm=[], vocabulary={})
train_vectorizer = TfidfVectorizer(stop_words=stopwdlst, sublinear_tf=True, max_df=0.7)
tfidf_train.tdm = train_vectorizer.fit_transform(train_bunch.review_content)
tfidf_train.vocabulary = train_vectorizer.vocabulary_
# 构建测试集的 TF-IDF 词向量空间对象
tfidf_test = Bunch(Id=test_bunch.review_id, tdm=[], vocabulary={})
tfidf_test.vocabulary = tfidf_train.vocabulary
test_vectorizer = TfidfVectorizer(stop_words=stopwdlst, sublinear_tf=True,
                                  max_df=0.7, vocabulary=tfidf_train.vocabulary)
tfidf_test.tdm = test_vectorizer.fit_transform(test_bunch.review_content)
# 训练集的字典
tfidf_voca = tfidf_train.vocabulary
print('The length of vocabulary:', len(tfidf_voca))

#
from sklearn import feature_selection
from xgboost.sklearn import XGBClassifier

xgb_classifier = XGBClassifier(silent=1,                      # 设置为1则没有运行信息输出，设置为0则有运行信息输出
                               learning_rate=0.007,             # 学习率
                               min_child_weight=3,            # 该参数越小，越容易过拟合
                               max_depth=12,                   # 构建的树的深度，越大越容易过拟合
                               gamma=0.1,                       # 越大越保守，一般取值为0.1，0.2
                               subsample=0.7,
                               max_delta_step=0,              # 最大增量步长，我们允许每个树的权重估计
                               colsample_bytree=0.7,          # 生成树时进行的列采样
                               reg_lambda=2,                  # L2 正则化参数，越大越不容易过拟合
                               scale_pos_weight=1,            # 取值大于0，在类别样本不平衡时有助于快速收敛
                               objective='multi:softmax',     # 多分类问题
                               num_class=5,                   # 类别数
                               n_estimators=800,              # 树的个数
                               eval_metric='merror',          # 多分类的损失函数
                               seed=1000)

# split data to train and test dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(tfidf_train.tdm,
                                                    tfidf_train.Score,
                                                    test_size=0.2,
                                                    random_state=33)

# 通过交叉验证的方法，按照固定间隔的百分比筛选特征，并作图展示性能随特征帅选比例的变化
from sklearn.model_selection import cross_val_score
import numpy as np
percentiles = range(15, 20, 2)
results = []
for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    x_train_fs = fs.fit_transform(x_train, y_train)
    scores = cross_val_score(xgb_classifier, x_train_fs, y_train, cv=4)
    print('percent', i, 'The accuracy with test', scores)
    results = np.append(results, scores.mean())
print(results)

# 找到实现最佳性能的特征筛选百分比
opt = np.where(results == results.max())[0]
print('Optimal number of features %d' % percentiles[opt[0]])
import pylab as pl
pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

# # 对待测试集进行预测
# fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=30)
# x_train_fs = fs.fit_transform(tfidf_train.tdm, tfidf_train.Score)
# xgb_classifier.fit(x_train_fs, tfidf_train.Score)
# x_test_fs = fs.transform(tfidf_test.tdm)
# test_predict = xgb_classifier.predict(x_test_fs)
#
# result = []
# for i in range(len(test_bunch.review_id)):
#     result.append([test_bunch.review_id[i], test_predict[i]])
#
# # store result to local, and use it to hand in
# import csv
# store_predict_path = 'E:/Desktop/ZhuFei/CNN/TensorFlow/云移动/subject_1/evaluation_public.csv'
# with open(store_predict_path, 'w', newline="") as file:
#     csv.writer(file).writerows(result)
