# coding = utf-8

from sklearn.datasets.base import Bunch
from instrument import read_bunch

# 读取训练集与验证集
train_bunch_path = './data_bunch/train_bunch.dat'
validate_bunch_path = './data_bunch/validate_bunch.dat'
train_bunch = read_bunch(train_bunch_path)
validate_bunch = read_bunch(validate_bunch_path)

# 停止词
stop_words_list = None
max_df = 0.7

# 计算 TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# 训练集的 TF-IDF
tfidf_train = Bunch(Id=train_bunch.news_id, Label=train_bunch.news_pic_label, tdm=[], vocabulary={})
train_vectorizer = TfidfVectorizer(stop_words=stop_words_list, sublinear_tf=True, max_df=max_df)
tfidf_train.tdm = train_vectorizer.fit_transform(train_bunch.news_words_ltp)   # jieba 分词结果或
tfidf_train.vocabulary = train_vectorizer.vocabulary_

# 验证集的 TF-IDF
tfidf_validate = Bunch(Id=validate_bunch.news_id, tdm=[], vocabulary={})
tfidf_validate.vocabulary = tfidf_train.vocabulary
validate_vectorizer = TfidfVectorizer(stop_words=stop_words_list, sublinear_tf=True, max_df=max_df,
                                      vocabulary=tfidf_train.vocabulary)
tfidf_validate.tdm = validate_vectorizer.fit_transform(validate_bunch.news_words_ltp)        # jieba 分词结果

# 训练集的字典
tfidf_voca = tfidf_train.vocabulary
print('The length of vocabulary:', len(tfidf_voca))

# 从训练集中提取一部分数据对模型进行性能评估
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(tfidf_train.tdm,
                                                    tfidf_train.Label,
                                                    test_size=0.2,
                                                    random_state=33)

classifier_xgb = XGBClassifier(silent=0,                      # 设置为1则没有运行信息输出，设置为0则有运行信息输出
                               learning_rate=0.01,            # 学习率
                               min_child_weight=1,            # 该参数越小，越容易过拟合
                               max_depth=8,                   # 构建的树的深度，越大越容易过拟合
                               gamma=0,                       # 越大越保守，一般取值为0.1，0.2
                               subsample=0.8,
                               max_delta_step=0,              # 最大增量步长，我们允许每个树的权重估计
                               colsample_bytree=0.8,          # 生成树时进行的列采样
                               reg_lambda=1,                  # L2 正则化参数，越大越不容易过拟合
                               scale_pos_weight=1,            # 取值大于0，在类别样本不平衡时有助于快速收敛
                               objective='multi:softmax',     # 多分类问题
                               num_class=5,                   # 类别数
                               n_estimators=900,              # 树的个数
                               eval_metric='merror',          # 多分类的损失函数
                               seed=1000)
parameters_xgb = [
    {
        'learning_rate': [0.01, 0.1, 1, 3, 10],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.75,  0.85, 0.9],
        'reg_lambda': [0.5, 1, 5, 10, 50],
        'n_estimators': [800, 1000, 1500, 2000],
        'min_child_weight': range(1, 6, 2),
        'gamma': [i/10.0 for i in range(0, 5)],
        'colsample_bytree': [i/100.0 for i in range(75, 90, 5)]
    }
]
gs_xgb = GridSearchCV(classifier_xgb, parameters_xgb, verbose=True, cv=4, n_jobs=-1)

if __name__ == '__main__':
    gs_xgb.fit(x_train, y_train)
    print(gs_xgb.best_params_, gs_xgb.best_score_)
    y_true, y_pred = y_test, gs_xgb.predict(x_test)
    print('Accuracy %.4g' % metrics.accuracy_score(y_true, y_pred))
    print('The accuracy of classifying training data with XGBoost Classifier',
          classifier_xgb.score(x_test, y_test))
    print(classification_report(y_test, y_pred))
