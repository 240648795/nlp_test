import codecs

from bert_train import data_generator, get_train_test_data


raw_data_path=r'data\quality_feedback_fault_use38_all.csv'
dict_path = r'bert\vocab.txt'
def get_token_dict():
    """
    # 将词表中的字编号转换为字典
    :return: 返回自编码字典
    """
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict

if __name__ == '__main__':
    # X_train, X_valid, data_test, nclass, encoder = get_train_test_data(raw_data_path)
    # train_D = data_generator(X_train, shuffle=True)
    #
    # for val in train_D:
    #     print (val)


    print (get_token_dict())
