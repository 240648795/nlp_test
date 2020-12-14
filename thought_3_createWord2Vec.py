import jieba
import jieba.posseg as pseg
import pandas as pd
from gensim.models import Word2Vec
import gensim
import numpy as np

def stopwordslist(file_path):
    stopwords = [line.strip() for line in open(file_path, encoding='UTF-8').readlines()]
    return stopwords


if __name__ == '__main__':
    # 设置停用词，常用词和最大序列
    max_sequence_length = 50
    jieba.load_userdict('user_dict/liugong_element_keyword.txt')
    stopwords = stopwordslist('user_dict/liugong_stopwords.txt')

    # 读取语料数据，语料数据比类别数据要多
    word_df = pd.read_csv(r'data/quality/quality_feedback_fault_all.csv', sep=',')
    word_df = word_df[['fault_element', 'new_element', 'fault_desc', 'defect_desc', 'fault_phen', 'fault_reason']]
    word_df['sentence'] = word_df['fault_desc'] + word_df['fault_phen'] + word_df['fault_reason']

    # 读取语句和标签
    word_df_sentencs = []
    for index, row in word_df.iterrows():
        sentence =  []
        sentence_word = pseg.cut(str(row['sentence']))
        for word in sentence_word:
            if word.word not in stopwords:
                # 可以设置其词性为名词的，这里暂且注释起来
                if word.word != '\t':
                    # if word.word != '\t':
                    sentence.append(word.word)
        word_df_sentencs.append(sentence)

    word2VecModel = Word2Vec(word_df_sentencs, size=100, window=5, min_count=2, workers=1)
    vocab_list = [word for word, Vocab in word2VecModel.wv.vocab.items()]

    print (vocab_list)
    #保存word2vec对象
    print (vocab_list)
    word2VecModel.save('word2vec/quality_feedback_fault2020_all')