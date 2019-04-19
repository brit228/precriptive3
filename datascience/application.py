#!flask/bin/python
import json
import dill
import numpy as np
from flask import Flask, request
from flaskrun import flaskrun


application = Flask(__name__, static_folder=None)

with open("/home/ec2-user/python-flask-service/predict.pk", "rb") as fp:
    model = dill.load(fp)

@application.route("/predict", methods=["POST"])
def predict():
    smiles = request.form.get('smiles')
    pr_seq = request.form.get('sequence')
    if smiles != "" and pr_seq != "":
        return "{:.5f}".format(model.predict_proba(np.array([[smiles, pr_seq]]))[0,1])
    return str(-1.0)

if __name__ == '__main__':
    flaskrun(application)
