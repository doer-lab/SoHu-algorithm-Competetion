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

trainset_path = 'E:/Desktop/ZhuFei/CNN/TensorFlow/云移动/subject_1/train_set.dat'
testset_path = 'E:/Desktop/ZhuFei/CNN/TensorFlow/云移动/subject_1/test_set.dat'

train_bunch = readbunchobj(trainset_path)
test_bunch = readbunchobj(testset_path)

# create TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# 停用词表
stopword_path = 'E:/Desktop/ZhuFei/CNN/TensorFlow/云移动/subject_1/stopwords_1.txt'
with open(stopword_path) as file:
    stopwd = [line.strip('\n') for line in file.readlines()]  # 删除 \n

stopwdlst = stopwd
# 构建训练集的tf-idf词向量空间对象
tfidf_train = Bunch(Id=train_bunch.review_id, Score=train_bunch.review_score, tdm=[], vocabulary={})
train_vectorizer = TfidfVectorizer(stop_words=stopwdlst, sublinear_tf=True, max_df=0.7)
tfidf_train.tdm = train_vectorizer.fit_transform(train_bunch.review_content)
tfidf_train.vocabulary = train_vectorizer.vocabulary_
# 构建测试集的tf-idf词向量空间对象
tfidf_test = Bunch(Id=test_bunch.review_id, tdm=[], vocabulary={})
tfidf_test.vocabulary = tfidf_train.vocabulary
test_vectorizer = TfidfVectorizer(stop_words=stopwdlst, sublinear_tf=True,
                                  max_df=0.5, vocabulary=tfidf_train.vocabulary)
tfidf_test.tdm = test_vectorizer.fit_transform(test_bunch.review_content)

tfidf_voca = tfidf_train.vocabulary

# split data to train and test dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(tfidf_train.tdm,
                                                    tfidf_train.Score,
                                                    test_size=0.25,
                                                    random_state=33)

# 特征筛选
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=25)
# just for test the model
x_train = fs.fit_transform(x_train, y_train)
x_test = fs.transform(x_test)
# print('The accuracy of classifying train_bunch with Naive Bayes(with filtering stopwords):', mnb.score(x_test_fs, y_test))
# print(classification_report(y_test, mnb.predict(x_test_fs)))
#

# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(x_data, y_labels):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.1)
    model.fit(x_data, y_labels)
    return model
bayes_classifier = naive_bayes_classifier(x_train, y_train)
print('The accuracy of classifying training data with Naive Bayes',
      bayes_classifier.score(x_test, y_test))
print(classification_report(y_test, bayes_classifier.predict(x_test)))

# KNN Classifier
def knn_classifier(x_data, y_labels):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(x_data, y_labels)
    return model
knn_classifier = knn_classifier(x_train, y_train)
print('The accuracy of classifying training data with Nearest Neighbors',
      knn_classifier.score(x_test, y_test))
print(classification_report(y_test, knn_classifier.predict(x_test)))

# Logistic Regression Classifier
def logistic_regression_classifier(x_data, y_labels):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(x_data, y_labels)
    return model
linear_classifier = logistic_regression_classifier(x_train, y_train)
print('The accuracy of classifying training data with linear regression',
      linear_classifier.score(x_test, y_test))
print(classification_report(y_test, linear_classifier.predict(x_test)))

# Random Forest Classifier
def random_forest_classifier(x_data, y_labels):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(x_data, y_labels)
    return model
random_forest_classifier = random_forest_classifier(x_train, y_train)
print('The accuracy of classifying training data with Random Forest classifier',
      random_forest_classifier.score(x_test, y_test))
print(classification_report(y_test, random_forest_classifier.predict(x_test)))

# Decision Tree Classifier
def decision_tree_classifier(x_data, y_labels):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(x_data, y_labels)
    return model
decision_tree_classifier = decision_tree_classifier(x_train, y_train)
print('The accuracy of classifying training data with Decision Tree classifier',
      decision_tree_classifier.score(x_test, y_test))
print(classification_report(y_test, decision_tree_classifier.predict(x_test)))


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_calssifier(x_data, y_labels):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(x_data, y_labels)
    return model
gradient_boosting_calssifier = gradient_boosting_calssifier(x_train, y_train)
print('The accuracy of classifying training data with Gradient Decision Tree classifier',
      gradient_boosting_calssifier.score(x_test, y_test))
print(classification_report(y_test, gradient_boosting_calssifier.predict(x_test)))

# SVM(Support Vector Machine) Classifier
def svm_classifier(x_data, y_labels):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(x_data, y_labels)
    return model
svm_classifier = svm_classifier(x_train, y_train)
print('The accuracy of classifying training data with SVM classifier',
      svm_classifier.score(x_test, y_test))
print(classification_report(y_test, svm_classifier.predict(x_test)))

# XGBoost(eXtreme Gradient Boosting) Classifier
def xgboost_classifier(x_data, y_labels):
    from xgboost import XGBClassifier
    model = XGBClassifier(max_depth=8, learning_rate=0.5, min_child_weight=1,
                          scale_pos_weight=1, n_estimators=1000, reg_lambda=4,
                          objective='multi:softmax', num_class=5, eval_metric='merror')
    model.fit(x_data, y_labels)
    return model
xgboost_classifier = xgboost_classifier(x_train, y_train)
print('The accuracy of classifying training data with XGBoost Classifier',
      xgboost_classifier.score(x_test, y_test))
print(classification_report(y_test, xgboost_classifier.predict(x_test)))

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
# cross_validation


# 通过交叉验证的方法，按照固定间隔的百分比帅选特征，并作图展示性能随特征帅选比例的变化
from sklearn.model_selection import cross_val_score
import numpy as np
percentiles = range(1, 100, 1)
results = []
for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    x_train_fs = fs.fit_transform(x_train, y_train)
    scores = cross_val_score(mnb, x_train_fs, y_train, cv=5)
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
#使用最佳筛选后的特征，利用相同配置的模型在测试集上进行性能评估
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=22)
x_train_fs = fs.fit_transform(x_train, y_train)
mnb.fit(x_train_fs, y_train)
x_test_fs = fs.transform(x_test)
print('The accuracy of classifying train_bunch with Naive Bayes(with filtering stopwords):', mnb.score(x_test_fs, y_test))
print(classification_report(y_test, mnb.predict(x_test_fs)))

# # predict on test data
# x_train = fs.fit_transform(tfidf_train.tdm, tfidf_train.Score)
# mnb.fit(x_train, tfidf_train.Score)
# x_test = fs.transform(tfidf_test.tdm)
# test_predict = mnb.predict(x_test)
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
