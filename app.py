from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from pickle import load
import requests
import sklearn


app = Flask(__name__)
MODEL_PATH = "covid19predictor.h5"
model = load_model(MODEL_PATH)
model.make_predict_function()
scaler = load(open('scaler.pkl', 'rb'))

def predictor(data, days = 7):
    future_days = days
    y_predict_future = []
    while(future_days):
        data = data.reshape(1, data.shape[0], data.shape[1]).astype(np.float64)
        predict = model.predict(data)
        y_predict_future.append(predict[0])
        data = data[0]
        data = np.concatenate((data, predict))
        data = data[1:]
        data = data.reshape(data.shape[0], 1)
        future_days -= 1
    y_predict_future = np.array(y_predict_future)
    return y_predict_future

@app.route("/")
def home():
    response = requests.get('https://api.covid19india.org/data.json')
    json_data = response.json()
    covid = pd.DataFrame.from_dict(json_data['cases_time_series'])
    data = covid['dailyconfirmed']
    data = data[-50:]
    data = np.array(data)
    data = scaler.transform(data.reshape(50, 1))
    prediction = scaler.inverse_transform(predictor(data))
    prediction = prediction.ravel()
    date = np.array(covid[-1:]['dateymd'], dtype=np.datetime64)
    date = date + np.arange(1, 8)
    values = [i for i in prediction]
    labels = [str(i) for i in date]
    return render_template("index.html" , labels = labels, values= values)


if __name__ == "__main__":
    app.run(debug = True)
