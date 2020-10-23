from flask import Flask, request, jsonify, render_template, url_for, flash, redirect
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/inference')
def inference():
    return "Inference page"

@app.route('/about')
def about():
    return "about page"

if __name__ == '__main__':
    app.run(debug=True)
