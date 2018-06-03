# coding = utf-8

# 读入数据
from instrument import read_bunch
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer

# load data from local
train_bunch_path = './data_bunch/train_bunch.dat'
validate_bunch_path = './data_bunch/validate_bunch.dat'
train_bunch = read_bunch(train_bunch_path)
validate_bunch = read_bunch(validate_bunch_path)

# 创建词向量空间
stop_words_list = None
max_df = 0.7

# create TF-IDF words vector space with train data
tfidf_train = Bunch(Id=train_bunch.news_id, Label=train_bunch.news_pic_label, tdm=[], vocabulary={})
train_vectorizer = TfidfVectorizer(stop_words=stop_words_list, sublinear_tf=True, max_df=max_df)
tfidf_train.tdm = train_vectorizer.fit_transform(train_bunch.news_words_jieba)                # jieba 分词结果或
tfidf_train.vocabulary = train_vectorizer.vocabulary_

# create TF-IDF words vector space with validate data
tfidf_validate = Bunch(Id=validate_bunch.news_id, tdm=[], vocabulary={})
tfidf_validate.vocabulary = tfidf_train.vocabulary
validate_vectorizer = TfidfVectorizer(stop_words=stop_words_list, sublinear_tf=True, max_df=max_df,
                                      vocabulary=tfidf_train.vocabulary)
tfidf_validate.tdm = validate_vectorizer.fit_transform(validate_bunch.news_words_jieba)        # jieba 分词结果

# 将数据分为训练集与测试集
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(tfidf_train.tdm,
                                                    tfidf_train.Label,
                                                    test_size=0.3,
                                                    random_state=33)

# 构建模型
from sklearn.metrics import classification_report

# 1、Naive Bayes
# Multinomial Naive Bayes Classifier
def classifier_naive_bayes(x_data, y_labels):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.1453)
    model.fit(x_data, y_labels)
    return model

## 1.1、基于训练集和测试集进行模型的训练
model_naive_bayes = classifier_naive_bayes(x_train, y_train)
print('The accuracy of classifying training data with Naive Bayes is :',
      model_naive_bayes.score(x_test, y_test))
print(classification_report(y_test, model_naive_bayes.predict(x_test)))

## 1.2、预测
model_naive_bayes = classifier_naive_bayes(tfidf_train.tdm, tfidf_train.Label)
predict_naive_bayes = model_naive_bayes.predict(tfidf_validate.tdm)

# store the result of predict to local, and ust it to submittion
bayes_text = []
for i in range(len(validate_bunch.news_id)):
    bayes_text.append('NULL')

label_predict = predict_naive_bayes
bayes_result = []
for i in range(len(validate_bunch.news_id)):
    bayes_result.append(validate_bunch.news_id[i]+'\t'+label_predict[i]+'\t'+bayes_text[i]+'\t'+bayes_text[i])

from instrument import save_text

save_path = './submittion/result_bayes.txt'
save_text(save_path, bayes_result)