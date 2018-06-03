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
max_df = 0.8

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

# 将训练集拆分为训练集和测试集
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

x_train, x_test, y_train, y_test = train_test_split(tfidf_train.tdm,
                                                    tfidf_train.Label,
                                                    test_size=0.3,
                                                    random_state=33)

# 构建GBDT模型
# GBDT(Gradient Boosting Decision Tree) Classifier
def calssifier_gradient_boosting(x_data, y_labels):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(x_data, y_labels)
    return model

model_gbdt = calssifier_gradient_boosting(x_train, y_train)
print('The accuracy of classifying training data with GBDT is :',
      model_gbdt.score(x_test, y_test))
print(classification_report(y_test,model_gbdt.predict(x_test)))

# 用全部训练集训练模型
model_gbdt = calssifier_gradient_boosting(tfidf_train.tdm,tfidf_train.Label)
predict_gbdt = model_gbdt.predict(tfidf_validate.tdm)

# 7. store the result of predict to local, and ust it to submittion
bayes_text = []
for i in range(len(validate_bunch.news_id)):
    bayes_text.append('NULL')

label_predict = predict_gbdt
bayes_result = []
for i in range(len(validate_bunch.news_id)):
    bayes_result.append(validate_bunch.news_id[i]+'\t'+label_predict[i]+'\t'+bayes_text[i]+'\t'+bayes_text[i])

# 保存预测数据至本地用以提交
from instrument import save_text

save_path = './submittion/result_gbdt.txt'
save_text(save_path, bayes_result)