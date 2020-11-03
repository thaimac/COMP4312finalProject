import os
import pymysql
from flask import Flask, request, jsonify, render_template, url_for, flash, redirect
from baselines import load
from db import get_table, add_inference
import numpy as np


app = Flask(__name__)

model_api = load("./data/tuned_RFR.pickle")


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/getapi', methods=['GET'])
def getapi():
    """
    GET request at a sentence level
    http://127.0.0.1:5000/getapi?text=hi%20there,%20how%20are%20you
    """
    text = request.args.get('text', default='the best hotel i stayed so far', type=str)
    app.logger.info("Input: " + text)
    label = model_api.predict([text]).tolist()[0]
    prob = model_api.predict_proba([text]).max()
    result = dict()
    result["input"] = text
    result["Output"] = label
    result["Probability"] = prob
    app.logger.info("model_output: " + str(result))
    result = jsonify(result)
    return result


@app.route('/postapi', methods=['POST'])
def postapi():
    """
    POST request at a sentence level
    """
    json_data = request.json
    text = json_data['rv_text']
    app.logger.info("Input: " + text)
    label = model_api.predict([text]).tolist()[0]
    prob = model_api.predict_proba([text]).max()
    result = dict()
    result["input"] = text
    result["Output"] = label
    result["Probability"] = prob
    app.logger.info("model_output: " + str(result))
    result = jsonify(result)
    return result


@app.route('/inference', methods=('GET', 'POST'))
def inference():
    if request.method == 'POST':

        req = [int(x) for x in request.form.values()]
        final_features = [np.array(req)]

        if not req:
            flash('Missing values')
        else:
            price_prediction = model_api.predict(final_features).tolist()[0]
            result = dict()
            result["input"] = req
            result["output"] = price_prediction
            app.logger.info("model_output: " + str(result))
            return render_template('inference.html', price_prediction=price_prediction)
    return render_template("inference.html")

@app.route('/stored_inferences', methods=['POST', 'GET'])
def stored_inferences():
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({"msg": "Missing JSON in request"}), 400  

        add_inference(request.get_json())
        return 'Song Added'

    return get_table()
    return render_template("stored_inferences.html")

@app.route('/about')
def about():
    return "about page"

if __name__ == '__main__':
    app.run(debug=True)
