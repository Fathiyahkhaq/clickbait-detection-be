from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model('model_best.tf')

classes = ['News', 'Clickbait']

@app.route("/")
def home():
    return "<h1>Hi, there!</h1>"

@app.route("/predict", methods=['GET','POST'])
def model_prediction():
    if request.method == "POST":
        content = request.json
        try: 
            data = [content['headline']]
            res = model.predict([data])
            res = np.where(res>0.5,1,0)
            response = {"code": 200, "status":"OK", 
                        "result":{"prediction":str(res[0].item()),
                        "description":classes[res[0].item()]}}
            return jsonify(response)
        except Exception as e:
            response = {"code":500, "status":"ERROR", 
                        "result":{"error_msg":str(e)}}
            return jsonify(response)
    return "<p>Silahkan gunakan method POST untuk mengakses hasil prediksi dari model</p>"

