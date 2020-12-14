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


if __name__ == '__main__':

    # 读取word2vec
    word2vecModelFile = 'word2vec/quality_feedback_fault2020_all'
    word2vecModel = Word2Vec.load(word2vecModelFile)

    for key in word2vecModel.wv.similar_by_word('发动机', topn=5):
        print (key)

    vocab_list = [word for word, Vocab in word2vecModel.wv.vocab.items()]
    word_index = {" ": 0}
    word_vector = {}
    embeddings_matrix = np.zeros((len(vocab_list) + 1, word2vecModel.vector_size))

    # 填充大矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1  # 词语：序号
        word_vector[word] = word2vecModel.wv[word]  # 词语：词向量
        embeddings_matrix[i + 1] = word2vecModel.wv[word]  # 词向量矩阵

    print (embeddings_matrix)