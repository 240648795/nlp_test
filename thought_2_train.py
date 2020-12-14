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
    # 设置停用词，常用词和最大序列
    max_sequence_length = 50
    jieba.load_userdict('user_dict/liugong_element_keyword.txt')
    stopwords = stopwordslist('user_dict/liugong_stopwords.txt')

    # 读取数据
    df = pd.read_csv(r'data/quality/quality_feedback_fault_use38_all.csv', sep=',')
    df = df[['fault_element', 'new_element', 'fault_desc', 'defect_desc', 'fault_phen', 'fault_reason']]

    df['static'] = df['fault_element']
    df['sentence'] = df['fault_desc'] + df['fault_phen'] + df['fault_reason']
    df = df[['static', 'sentence']]

    # 读取语句和标签，分词
    sentencs = []
    labels = []
    for index, row in df.iterrows():
        sentence = ""
        sentence_word = pseg.cut(str(row['sentence']))
        for word in sentence_word:
            if word.word not in stopwords:
                # 可以设置其词性为名词的，这里暂且注释起来
                if word.word != '\t' and word.flag=='n':
                # if word.word != '\t':
                    sentence += word.word
                    sentence += " "
        sentencs.append(sentence)
        labels.append(row['static'])

    print(sentencs)

    # 设置标签类别，将str对应成数字
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoder_labels = encoder.transform(labels)

    # 词向量化
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(sentencs)
    word_index = tokenizer.word_index
    print(word_index)
    sentencs_vec = tokenizer.texts_to_sequences(sentencs)

    sentencs_vec_pad = pad_sequences(sentencs_vec, padding='post', maxlen=max_sequence_length)

    print (sentencs_vec_pad)

    # 划分测试集和训练集
    sentences_train, sentences_test, encoder_labels_train, encoder_labels_test = train_test_split(sentencs_vec_pad,
                                                                                                  encoder_labels,
                                                                                                  test_size=0.2,
                                                                                                  random_state=1000)

    # 加入词嵌入模式模式的神经网络
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 50
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size,
                               output_dim=embedding_dim,
                               input_length=max_sequence_length))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(sentences_train, encoder_labels_train,
                        epochs=100,
                        verbose=False,
                        validation_data=(sentences_test, encoder_labels_test),
                        batch_size=10)
    loss, accuracy = model.evaluate(sentences_train, encoder_labels_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(sentences_test, encoder_labels_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    model.save('model/thought2.h5')
    joblib.dump(tokenizer, 'model/thought2_tok.joblib')
    joblib.dump(encoder, 'model/thought2_label.joblib')