import jieba
import jieba.posseg as pseg

def stopwordslist(file_path):
    stopwords = [line.strip() for line in open(file_path, encoding='UTF-8').readlines()]
    return stopwords

if __name__ == '__main__':
    t='加力器内部泄露'
    jieba.load_userdict('liugong_element_keyword.txt')
    stopwords = stopwordslist('liugong_stopwords.txt')
    words=pseg.cut(str(t))
    sentence = ""
    for word in words:
        if word.word not in stopwords:
            if word.word != '\t':
                sentence += word.word
                sentence += " "
    print (sentence)