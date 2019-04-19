#!flask/bin/python
import json
import dill
import numpy as np
import requests
from decouple import config
from flask import Flask, request, Response
from flaskrun import flaskrun


USERNAME = config("USERNAME")
PASSWORD = config("PASSWORD")
BACKEND_URL = config("BACKEND_URL")
AUTH_TOKEN = ""

application = Flask(__name__, static_folder=None)

with open("predict.pk", "rb") as fp:
    model = dill.load(fp)

@application.route("/predict", methods=["POST"])
def predict():
    smiles = request.form.get('smiles')
    pr_seq = request.form.get('sequence')
    effect_id = request.form.get('effect_id')
    d = {
        'effect_id': effect_id,
        'smiles': smiles,
        'sequence': pr_seq,
        'bind_chance': -1.0
    }
    status = 401
    if smiles != "" and pr_seq != "" and effect_id != "" and effect_id:
        try:
            if effect_id:
                effect_id = int(effect_id)
            d = {
                'effect_id': effect_id,
                'smiles': smiles,
                'sequence': pr_seq,
                'bind_chance': -1.0
            }
            d["bind_chance"] = float(model.predict_proba(np.array([[smiles, pr_seq]]))[0,1])
            status = 200
        except:
            pass
    resp = Response(json.dumps(d), status=status, mimetype='application/json')
    resp.headers['Authorization'] = AUTH_TOKEN
    return resp

if __name__ == '__main__':
    r = requests.post(BACKEND_URL, data={"username": USERNAME, "password": PASSWORD})
    AUTH_TOKEN = r.json().get('token')
    flaskrun(application)
