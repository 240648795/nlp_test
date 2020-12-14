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
from gensim.models import Word2Vec

def stopwordslist(file_path):
    stopwords = [line.strip() for line in open(file_path, encoding='UTF-8').readlines()]
    return stopwords


if __name__ == '__main__':
    # 设置停用词，常用词和最大序列
    max_sequence_length = 50
    jieba.load_userdict('user_dict/liugong_element_keyword.txt')
    stopwords = stopwordslist('user_dict/liugong_stopwords.txt')

    # 读取word2vec
    word2vecModelFile = 'word2vec/quality_feedback_fault2020_all'
    word2vecModel = Word2Vec.load(word2vecModelFile)

    vocab_list = [word for word, Vocab in word2vecModel.wv.vocab.items()]

    word_index = {" ": 0}
    word_vector = {}

    embeddings_matrix = np.zeros((len(vocab_list) + 1, word2vecModel.vector_size))

    #填充大矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1  # 词语：序号
        word_vector[word] = word2vecModel.wv[word]  # 词语：词向量
        embeddings_matrix[i + 1] = word2vecModel.wv[word]  # 词向量矩阵

    #读取要训练的数据
    train_df = pd.read_csv('data/quality/quality_feedback_fault_use38_all.csv', sep=',')
    train_df = train_df[['fault_element', 'new_element', 'fault_desc', 'defect_desc', 'fault_phen', 'fault_reason']]
    train_df['static'] = train_df['fault_element']
    train_df['sentence'] = train_df['fault_desc'] + train_df['fault_phen'] + train_df['fault_reason']
    train_df = train_df[['static', 'sentence']]

    # 读取语句和标签
    sentencs = []
    labels = []
    for index, row in train_df.iterrows():
        sentence = []
        sentence_word = pseg.cut(str(row['sentence']))
        for word in sentence_word:
            try:
                sentence.append(word_index[word.word])  # 把句子中的 词语转化为index
            except:
                sentence.append(0)
        sentencs.append(sentence)
        labels.append(row['static'])

    # 设置标签类别，将str对应成数字
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoder_labels = encoder.transform(labels)
    sentencs_vec_pad = pad_sequences(sentencs, padding='post', maxlen=max_sequence_length)

    sentences_train, sentences_test, encoder_labels_train, encoder_labels_test = train_test_split(sentencs_vec_pad,
                                                                                                  encoder_labels,
                                                                                                  test_size=0.2,
                                                                                                  random_state=1000)

    # 加入词嵌入模式模式的神经网络
    embedding_dim = 100
    model = Sequential()
    model.add(layers.Embedding(len(embeddings_matrix), embedding_dim,
                               weights=[embeddings_matrix],
                               input_length=max_sequence_length,
                               trainable=False))
    model.add(LSTM(64, return_sequences=False))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(sentences_train, encoder_labels_train,
                        epochs=10,
                        verbose=1,
                        validation_data=(sentences_test, encoder_labels_test),
                        batch_size=100)
    loss, accuracy = model.evaluate(sentences_train, encoder_labels_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(sentences_test, encoder_labels_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))



