# coding = utf-8

def read_text(file_path):
    '''
    读取指定的 txt 文件
    1、file_path 为所读文件的具体地址
    2、for example（读取当前工作目录下的 News_info_train_filter.txt 文件）:
        news_info_train_path = './News_info_train_filter.txt'
        news_info_train = read_text(news_info_train_path)
    '''
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    return content

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

def save_bunch(save_path, save_bunch):
    '''
    1、save_path 为存储 Bunch 对象的地址（目录）
    2、save_bunch 为要存储的 Bunch 对象
    3、for example （保存为当前工作目录下的 train_bunch_balance.dat 文件）:
        save_path = './train_bunch_balance.dat'
        save_bunch(save_path,TRAIN_BUNCH)
    '''
    import pickle
    with open(save_path, 'wb') as file_obj:
        pickle.dump(save_bunch, file_obj)
    return

def read_bunch(bunch_path):
    '''
    1、读取指定的 bunch 对象
    2、for example（读取保存在当前工作目录下的 train_bunch_balance.dat 文件）:
        train_bunch_path = './data_bunch/train_bunch_balance.dat'
        train_bunch = read_bunch(train_bunch_path)
    '''
    import pickle
    with open(bunch_path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

def read_image(file_path):
    # 读取指定地址中图片
    pass
def save_image(save_path):
    # 将图片保持至指定的文件下
    pass

def text2words(need_to_segment_text, method='jieba', ltp_model_path='',postdict={},stop_words=[]):
    '''
        该函数实现对每条新闻文本进行分词（分词方法包括jieba和ltp）,并将繁体字替换为简体字
    1、need_to_segment_text 为需要进行分词的文本列表
    2、method 为中文分词方法，可选 jieba 或 ltp
    3、ltp_model_path 为选择ltp分词方法时，ltp 分词模型在本地的目录地址
    4、postdist 数据格式：
        postdict = {'解 空间':'解空间','深度 优先':'深度优先'}
    
    '''
    from langconv import Converter
    import re
    text_words = []

    if method == 'jieba':
        import jieba
#         jieba.enable_parallel(4)             # 多线程分词，仅支持 Linux
        for sentence in need_to_segment_text:
            #             print(sentence)                                # 查看新闻原文
            sentence = Converter('zh-hans').convert(sentence)            # 将繁体中文转换为简体中文
            content = re.sub('[^\u4e00-\u9fa5a-zA-Z0-9.]',' ', sentence)  # 将中文、大小写字母和数字外的字符全替换为空格
            content_word = jieba.cut(content)
            seg_sent = ' '.join([word for word in content_word if word not in stop_words])  # 去除停止词
            seg_sent = re.sub('\s+',' ',seg_sent)
            for key in postdict:
                seg_sent = seg_sent.replace(key,postdict[key])    # 在分词后处理某些被分错的词和词语
            text_words.append(seg_sent)
    elif method == 'ltp' and ltp_model_path != '':
        from pyltp import Segmentor
        #         model_path = 'E:/Desktop/ZhuFei/Competition/NLP/ltp_data_v3.4.0/cws.model'   # Ltp 3.4 分词模型库
        segmentor = Segmentor()   # 实例化分词模块
        segmentor.load(ltp_model_path)  # 加载分词库
        for sentence in need_to_segment_text:
            #             print(sentence)                         # 查看新闻原文
            sentence = Converter('zh-hans').convert(sentence)     # 将繁体中文转换为简体中文
            content = re.sub('[^\u4e00-\u9fa5a-zA-Z0-9.:]',' ', sentence)  # 将中文、大小写字母和数字外的字符全替换为空格
            content_word = jieba.cut(content)
            seg_sent = ' '.join([word for word in content_word if word not in stop_words])  # 去除停止词
            seg_sent = re.sub('\s+',' ',seg_sent)
            for key in postdict:
                seg_sent = seg_sent.replace(key,postdict[key])    # 在分词后处理某些被分错的词和词语
            text_words.append(seg_sent) # 去除停止词
    else:
        fill_to_length = 130
        print(''.center(fill_to_length, '#'))
        print(' Method or model path is wrong! Please check it!!!! '.center(fill_to_length, '#'))
        print(''.center(fill_to_length, '#'))

    return text_words

def text2words_v2(need_to_segment_text, method='jieba', ltp_model_path='',postdict={},stop_words=[]):
    '''
        该函数实现对每条新闻文本进行分词（分词方法包括jieba和ltp）,并将繁体字替换为简体字
    1、need_to_segment_text 为需要进行分词的文本列表
    2、method 为中文分词方法，可选 jieba 或 ltp
    3、ltp_model_path 为选择ltp分词方法时，ltp 分词模型在本地的目录地址
    4、postdist 数据格式：
        postdict = {'解 空间':'解空间','深度 优先':'深度优先'}
    
    '''
    from langconv import Converter
    import re
    text_words = []
    sentence_len = []            # 记录句子的长度（按分词后句子所含字或词的个数计算）

    if method == 'jieba':
        import jieba
#         jieba.enable_parallel(4)             # 多线程分词，仅支持 Linux
        for sentence in need_to_segment_text:
#             print(sentence)                       # 查看新闻原文
            sentence = Converter('zh-hans').convert(sentence)             # 将繁体中文转换为简体中文
#             content = re.sub('[^\u4e00-\u9fa5a-zA-Z0-9.]',' ', sentence)     # 将中文、大小写字母和数字外的字符全替换为空格
            content = re.sub("[^\u4e00-\u9fa5a-zA-Z0-9|，。？！、；：“”‘’（）【】{}……《》%,.?!;:'()[]-——/&]",' ', sentence)
            content = re.sub('\s+',' ',content)
            content = re.sub('//\s+(//)?','/',content)
            word_list = [word for word in jieba.cut(content) if word not in stop_words]  # 分词，去停止词
            sentence_len.append(len(word_list))             # 记录该句话的长度
            seg_sent = ' '.join(word_list)  # 去除停止词
            seg_sent = re.sub('\s+',' ',seg_sent)
            for key in postdict:
                seg_sent = seg_sent.replace(key,postdict[key])    # 在分词后处理某些被分错的词和词语
            text_words.append(seg_sent)
    elif method == 'ltp' and ltp_model_path != '':
        from pyltp import Segmentor
        #         model_path = 'E:/Desktop/ZhuFei/Competition/NLP/ltp_data_v3.4.0/cws.model'   # Ltp 3.4 分词模型库
        segmentor = Segmentor()   # 实例化分词模块
        segmentor.load(ltp_model_path)  # 加载分词库
        for sentence in need_to_segment_text:
            #             print(sentence)                         # 查看新闻原文
            sentence = Converter('zh-hans').convert(sentence)     # 将繁体中文转换为简体中文
            content = re.sub("[^\u4e00-\u9fa5a-zA-Z0-9|，。？！、；：“”‘’（）【】{}……《》%,.?!;:'()[]-——/&]",' ', sentence)
            content = re.sub('\s+','',content)
            content = re.sub('//\s+(//)?','/',content)
            word_list = [word for word in segmentor.cut(content) if word not in stop_words]
            sentence_len.append(len(word_list))                       # 记录该句话的长度
            seg_sent = ' '.join(word_list)  # 去除停止词
            seg_sent = re.sub('\s+',' ',seg_sent)
            for key in postdict:
                seg_sent = seg_sent.replace(key,postdict[key])    # 在分词后处理某些被分错的词和词语
            text_words.append(seg_sent) # 去除停止词
    else:
        fill_to_length = 130
        print(''.center(fill_to_length, '#'))
        print(' Method or model path is wrong! Please check it!!!! '.center(fill_to_length, '#'))
        print(''.center(fill_to_length, '#'))

    return [text_words,sentence_len]

def data2bunch(data_file, label=False):
    '''
    将新闻数据转化为bunch对象，包括文本数据与标签数据
    '''
    if label:
        '''
        读取与新闻和图片相对应的标签
        '''
        news_pic_id = []
        news_pic_label = []
        news_pic_pic = []
        news_pic_text = []
        for line in data_file:
            news_pic_split = line.split('\t')           # 以tab键为分隔符对文本进行分割
            news_pic_id.append(news_pic_split[0])       # 获取每条新闻的ID
            news_pic_label.append(news_pic_split[1])    # 获取每条新闻的标注类别
            news_pic_pic.append(news_pic_split[2])      # 获取每条有营销意图新闻中 有营销意图的配图ID列表
            news_pic_text.append(news_pic_split[3])     # 获取每条有营销意图新闻中 有营销意图的文本片段
        # 将分离出来的数据重组为一个Bunch对象
        from sklearn.datasets.base import Bunch
        bunch = Bunch(news_pic_id=[], news_pic_label=[], news_pic_pic=[], news_pic_text=[])
        bunch.news_pic_id = news_pic_id
        bunch.news_pic_label = news_pic_label
        bunch.news_pic_pic = news_pic_pic
        bunch.news_pic_text = news_pic_text
    else:
        ''' 
        1、data_file ：原始新闻数据经过去标签处理后得到的数据，数据的具体样式见News_info_train_example100_filter
        2、对去除网页等标签的新闻数据
            （1）、先按‘新闻ID’、‘新闻文本’和‘新闻配图ID列表’三项进行数据分离，
            （2）、将上述提取出来的三项数据重组为一个Bunch对象
        '''
        # 将整个新闻数据分离为‘新闻ID’、‘新闻文本’及‘新闻配图ID列表’三个部分
        news_id = []
        news_content = []
        news_pic_list = []
        for line in data_file:
            news_split = line.split('\t')        # 以tab键为分隔符对文本进行分割
            news_id.append(news_split[0])        # 获取每条新闻的ID
            news_content.append(news_split[1])   # 获取每条新闻的内容
            news_pic_list.append(news_split[2])       # 获取每条新闻的配图ID
        # 将分离出来的数据重组为一个Bunch对象
        from sklearn.datasets.base import Bunch
        bunch = Bunch(news_id=[], news_content=[], news_pic_list=[])
        bunch.news_id = news_id
        bunch.news_content = news_content
        bunch.news_pic_list = news_pic_list
    return bunch

def search_best_para_bayes(min_df_list, max_df_list, feature_percent_list, alpha_list, method='jieba',
                           stop_words_list=None):
    '''
        该函数实现从众参数组成的解空间中寻找使朴素贝叶斯模型性能最佳的参数组合并进行返回
    '''
    from sklearn.model_selection import GridSearchCV
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import metrics, feature_selection
    from sklearn.metrics import classification_report
    from sklearn.datasets.base import Bunch
    from sklearn.feature_extraction.text import TfidfVectorizer
    # from instrument import read_bunch

    train_bunch_path = './data_bunch/train_bunch_balance.dat'
    train_bunch = read_bunch(train_bunch_path)

    results = [['min_df', 'max_df', 'feature percent', 'alpha', 'accuracy_best', 'accuracy_test']]
    for min_df in min_df_list:  # 控制 min_df ，尽量小
        for max_df in max_df_list:  # 控制 max_df ，尽量大
            tfidf_train = Bunch()
            tfidf_train = Bunch(Id=train_bunch.news_id, Label=train_bunch.news_pic_label, tdm=[], vocabulary={})
            train_vectorizer = TfidfVectorizer(stop_words=stop_words_list, sublinear_tf=True, min_df=min_df,
                                               max_df=max_df)
            if method == 'jieba':
                tfidf_train.tdm = train_vectorizer.fit_transform(train_bunch.news_words_jieba)  # 取 jieba 分词结果进行模型训练
            elif method == 'ltp':
                tfidf_train.tdm = train_vectorizer.fit_transform(train_bunch.news_words_ltp)  # 取 ltp 分词结果进行模型训练
            else:
                print(''.center(130, '*'))
                print("Warning: method not correct, please input 'jieba' or 'ltp'!")
                print(''.center(130, '*'))
            tfidf_train.vocabulary = train_vectorizer.vocabulary_

            # 训练集与测试集划分
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(tfidf_train.tdm, tfidf_train.Label, test_size=0.3,
                                                                random_state=30)
            for feature_percent in feature_percent_list:  # 控制参加模型训练的特征的比例
                feature_select = feature_selection.SelectPercentile(feature_selection.chi2, percentile=feature_percent)
                x_train_fs = feature_select.fit_transform(x_train, y_train)
                x_test_fs = feature_select.transform(x_test)

                model = MultinomialNB(alpha=0.1)
                param = {'alpha': alpha_list}
                gs_model = GridSearchCV(model, param, scoring='accuracy', verbose=0, cv=5, n_jobs=-1)

                gs_model.fit(x_train_fs, y_train)
                y_true, y_pred = y_test, gs_model.predict(x_test_fs)

                print(''.center(130, '*'))
                print('Accuracy is %g' % metrics.accuracy_score(y_true, y_pred))
                print('The accuracy of Naive_Bayes Classifier on dataset wich splis from train data is :',
                      gs_model.score(x_test_fs, y_test))
                print(classification_report(y_test, y_pred))
                print('Current best parameters and result :',
                      [min_df, max_df, feature_percent, gs_model.best_params_['alpha'], gs_model.best_score_,
                       gs_model.score(x_test_fs, y_test)])
                results.append([min_df, max_df, feature_percent, gs_model.best_params_['alpha'], gs_model.best_score_,
                                gs_model.score(x_test_fs, y_test)])
                print(''.center(130, '*'))

    # 查找使贝叶斯模型在验证集上性能最佳的参数组合并进行输出
    accu = []
    for index in range(len(results)):
        accu.append(results[index][5])
    index_best_param = accu.index(max(accu[1:]))
    print('Best parameters: ', results[index_best_param])

    # 将迭代结果写入本地
    import csv
    if method == 'jieba':
        store_pamaters_path = './paramter/pamaters_jieba.csv'
    else:
        store_pamaters_path = './paramter/pamaters_ltp.csv'
    with open(store_pamaters_path, 'w', newline="") as file:
        csv.writer(file).writerows(results)

    return results[index_best_param]

def perdict_bayes(best_min_df, best_max_df, best_feature_percentage,
                  best_alpha, method='jieba', stop_words_list=None):
    '''
        该函数用以对验证集进行预测
    '''
    from sklearn.datasets.base import Bunch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import feature_selection
    from sklearn.naive_bayes import MultinomialNB
    # from instrument import read_bunch

    # load train data and validate data
    train_bunch_path = './data_bunch/train_bunch_balance.dat'
    train_bunch = read_bunch(train_bunch_path)
    validate_bunch_path = './data_bunch/validate_bunch_balance.dat'
    validate_bunch = read_bunch(validate_bunch_path)

    # 构建训练集的 TF-IDF 词向量空间对象（分为 jieba 和 ltp ）
    tfidf_train = Bunch(Id=train_bunch.news_id, Label=train_bunch.news_pic_label, tdm=[], vocabulary={})
    train_vectorizer = TfidfVectorizer(stop_words=stop_words_list, sublinear_tf=True, min_df=best_min_df,
                                       max_df=best_max_df)
    if method == 'jieba':
        tfidf_train.tdm = train_vectorizer.fit_transform(train_bunch.news_words_jieba)  # 取 jieba 分词结果进行模型训练
    elif method == 'ltp':
        tfidf_train.tdm = train_vectorizer.fit_transform(train_bunch.news_words_ltp)  # 取 ltp 分词结果进行模型训练
    else:
        print(''.center(130, '*'))
        print("Warning: method not correct, please input 'jieba' or 'ltp'!")
        print(''.center(130, '*'))
    tfidf_train.vocabulary = train_vectorizer.vocabulary_
    # 构建验证集的 TF-IDF 词向量空间对象
    tfidf_validate = Bunch(Id=validate_bunch.news_id, tdm=[], vocabulary={})
    tfidf_validate.vocabulary = tfidf_train.vocabulary
    validate_vectorizer = TfidfVectorizer(stop_words=stop_words_list, sublinear_tf=True,
                                          min_df=best_min_df, max_df=best_max_df, vocabulary=tfidf_train.vocabulary)
    if method == 'jieba':
        tfidf_validate.tdm = validate_vectorizer.fit_transform(validate_bunch.news_words_jieba)  # jieba 分词结果
    else:
        tfidf_validate.tdm = validate_vectorizer.fit_transform(validate_bunch.news_words_ltp)  # ltp 分词结果

    # 利用训练集所有数据训练模型并对验证集进行预测
    model = MultinomialNB(best_alpha)
    feature_select = feature_selection.SelectPercentile(feature_selection.chi2, percentile=best_feature_percentage)
    x_train = feature_select.fit_transform(tfidf_train.tdm, tfidf_train.Label)
    x_validate = feature_select.transform(tfidf_validate.tdm)
    model.fit(x_train, tfidf_train.Label)
    predict_label = model.predict(x_validate)

    return predict_label

if __name__ == '__main__':
    pass

