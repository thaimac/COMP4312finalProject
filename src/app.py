from flask import Flask, request, jsonify, render_template, url_for, flash, redirect
from baselines import load
import numpy as np
from db import get_data, add_data

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
    form_data = request.form.to_dict()
    req = [int(x) for x in request.form.values()]
    final_features = [np.array(req)]
    price_prediction = model_api.predict(final_features).tolist()[0]
    result = dict()
    result["input"] = form_data
    result["Output"] = price_prediction
    app.logger.info("model_output: " + str(result))
    return result

@app.route('/postapi', methods=['POST'])
def postapi():
    """
    POST request at a sentence level
    """
    form_data = request.form.to_dict()
    app.logger.info("Input: " + form_data)
    req = [int(x) for x in request.form.values()]
    final_features = [np.array(req)]
    price_prediction = model_api.predict([final_features]).tolist()[0]
    result = dict()
    result["input"] = form_data
    result["Output"] = price_prediction
    app.logger.info("model_output: " + str(result))
    return result


@app.route('/inference', methods=('GET', 'POST'))
def inference():
    if request.method == 'POST':

        req = [int(x) for x in request.form.values()]
        final_features = [np.array(req)]
        data = request.form.to_dict()
        print(data)

        if not req:
            flash('Missing values')
        else:
            price_prediction = model_api.predict(final_features).tolist()[0]
            result = dict()
            result["input"] = req
            result["output"] = price_prediction
            data.update({'price_prediction': result['output']})
            add_data(data)
            app.logger.info("model_output: " + str(result))
            return render_template('inference.html', price_prediction=price_prediction)
    return render_template("inference.html")


@app.route('/stored_inferences', methods=['POST', 'GET'])
def stored_inferences():
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({"msg": "Missing JSON in request"}), 400

        add_data(request.get_json())
        return 'Inference Added'

    #return render_template("stored_inferences.html", data=get_data())
    return get_data()


@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=True, host="0.0.0.0", port=8000)
