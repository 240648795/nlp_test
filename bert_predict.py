import joblib
from keras_bert import get_custom_objects
from keras.models import load_model
import numpy as np
from bert_train import acc_top2, data_generator
from keras.utils import to_categorical
import gc
import keras.backend as K

def load_model_encoder_details(model_path, encoder_path, details_path):
    custom_objects = get_custom_objects()
    my_objects = {'acc_top2': acc_top2}
    custom_objects.update(my_objects)
    model = load_model(model_path, custom_objects=custom_objects)
    encoder = joblib.load(encoder_path)
    nclass_dict = joblib.load(details_path)
    return model, encoder, nclass_dict['nclass']

def predict_one(raw_text,model, encoder, nclass):
    text = raw_text
    DATA_text = []
    DATA_text.append((text, to_categorical(0, nclass)))
    DATA_text = np.array(DATA_text)
    text = data_generator(DATA_text, shuffle=False)
    test_model_pred = model.predict_generator(text.__iter__(), steps=len(text), verbose=1)

    predict_num=np.argmax(test_model_pred)
    predict_label=encoder.inverse_transform([predict_num])
    return predict_label[0]


if __name__ == '__main__':
    #必须加载模型带r
    model, encoder, nclass = load_model_encoder_details(r'model\bert_model.h5',
                                                        r'model\bert_model_encoder.joblib',
                                                        r'model\bert_model_details.joblib')

    # 单独评估一个本来分类
    text = '支架过于单薄。导致支撑不平衡。更换新件后试车正常'
    predict_label=predict_one(text,model, encoder, nclass)
    print (predict_label)

    del model  # 删除模型减少缓存
    gc.collect()  # 清理内存
    K.clear_session()  # clear_session就是清除一个session
