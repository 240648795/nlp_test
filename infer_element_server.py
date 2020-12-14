from flask import Flask, render_template, jsonify,request
import bert_predict

model, encoder, nclass = bert_predict.load_model_encoder_details(r'model\bert_model.h5',
                                                        r'model\bert_model_encoder.joblib',
                                                        r'model\bert_model_details.joblib')
app = Flask(__name__)

@app.route('/get_predict_element_html')
def get_predict_element_html():
    return render_template('predict_element.html')

@app.route('/predict_element',methods=['GET','POST'])
def predict_element():
    input_txt = request.args.get('input_txt')
    rs='输入错误'
    if input_txt is None or input_txt=='':
        rs = '输入错误'
    else:
        rs = bert_predict.predict_one(input_txt,model, encoder, nclass)
    return jsonify(rs)

if __name__ == '__main__':
    app.run()