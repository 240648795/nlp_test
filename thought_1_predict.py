import jieba
import jieba.posseg as pseg
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
from user_dict import jieba_test

def predict_element(predict_sentenc_raw_,filename_='quality_feedback_fault2020_1000'):
    # 设置停用词，常用词和最大序列
    max_sequence_length = 50
    jieba.load_userdict('user_dict/liugong_element_keyword.txt')
    stopwords = jieba_test.stopwordslist('user_dict/liugong_stopwords.txt')
    filename = filename_

    # 读取保存的标签
    encoder = joblib.load('model/' + filename + '_label.joblib')
    # 读取保存的词袋
    tokenizer = joblib.load('model/' + filename + '_tok.joblib')
    # 读取保存的模型
    model = load_model('model/' + filename + '.h5')

    # 此处开始预测
    predict_sentenc_raw = predict_sentenc_raw_
    predict_words = pseg.cut(str(predict_sentenc_raw))
    predict_sentenc = ''
    for predict_word in predict_words:
        if predict_word.word not in stopwords:
            if predict_word.word != '\t':
                predict_sentenc += predict_word.word
                predict_sentenc += " "

    sentencs_vec_predict = tokenizer.texts_to_sequences([predict_sentenc])
    sentencs_vec_pad_predict = pad_sequences(sentencs_vec_predict, padding='post', maxlen=max_sequence_length)
    ys = model.predict_classes(sentencs_vec_pad_predict)
    y_label = encoder.inverse_transform(ys[0])
    return y_label[0]


if __name__ == '__main__':
    y_label=predict_element(predict_sentenc_raw_='O形密封圈损坏漏油')
    print (y_label)
