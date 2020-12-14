import jieba
import jieba.posseg as pseg
import joblib
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import Adam
from keras.models import Sequential
from keras import layers


def stopwordslist(file_path):
    stopwords = [line.strip() for line in open(file_path, encoding='UTF-8').readlines()]
    return stopwords


if __name__ == '__main__':

    max_sequence_length = 50
    jieba.load_userdict('user_dict/liugong_element_keyword.txt')
    stopwords = stopwordslist('user_dict/liugong_stopwords.txt')

    # 读取数据
    df = pd.read_csv(r'data/quality/quality_feedback_fault_use38_all.csv', sep=',')
    df = df[['fault_element', 'new_element', 'fault_desc', 'defect_desc', 'fault_phen', 'fault_reason']]

    df['static'] = df['fault_element']
    df['sentence'] = df['fault_desc'] + df['fault_phen'] + df['fault_reason']
    df['sentence']=df['sentence'].astype(str)
    df = df[['static', 'sentence']]

    # 读取语句和标签
    sentencs = []
    labels = []
    for index, row in df.iterrows():
        sentencs.append(row['sentence'])
        labels.append(row['static'])

    print (sentencs)

    # 设置标签类别，将str对应成数字
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoder_labels = encoder.transform(labels)

    # 词向量化
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(sentencs)
    word_index = tokenizer.word_index
    sentencs_vec = tokenizer.texts_to_sequences(sentencs)
    sentencs_vec_pad = pad_sequences(sentencs_vec, padding='post', maxlen=max_sequence_length)

    print (sentencs_vec_pad)

    # 划分测试集和训练集
    sentences_train, sentences_test, encoder_labels_train, encoder_labels_test = train_test_split(sentencs_vec_pad,
                                                                                                  encoder_labels,
                                                                                                  test_size=0.2,
                                                                                                  random_state=1000)

    classifier = LogisticRegression()
    classifier.fit(sentences_train, encoder_labels_train)
    score = classifier.score(sentences_test, encoder_labels_test)
    print("Accuracy:", score)

