from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application

ridge = pickle.load(open("models/ridge.pkl", "rb"))
elastic = pickle.load(open("models/elastic.pkl", "rb"))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict_datapoint", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        result = elastic.predict([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI]])

        return render_template('home.html', result=result[0])
    
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")