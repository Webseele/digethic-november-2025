from flask import Flask, Response, request
import pandas as pd
import os
import pickle

app = Flask(__name__)


@app.route("/")
def root():
    return {"hello": "world"}


@app.route("/hello_world")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/training_data")
def training_data():
    training_data = pd.read_csv(os.path.join("data", "auto-mpg.csv"), sep=";")
    return Response(training_data.to_json(), mimetype="application/json")




@app.route("/predict")
def predict():
    zylinder = float(request.args.get("zylinder"))
    ps = float(request.args.get("ps"))
    gewicht = float(request.args.get("gewicht"))
    beschleunigung = float(request.args.get("beschleunigung"))
    baujahr = float(request.args.get("baujahr"))
    print ("zylinder", zylinder)
    print ("ps", ps)
    print ("gewicht", gewicht)
    print ("beschleunigung", beschleunigung)
    print ("baujahr", baujahr)

    file_to_open = open(os.path.join("data", "models", "baummethoden_lr.pickle"), "rb")
    trained_model = pickle.load(file_to_open)
    file_to_open.close()
    prediction = trained_model.predict([[zylinder, ps, gewicht, beschleunigung, baujahr]])
    print("prediction", prediction[0])
    return {"result": prediction[0]}
