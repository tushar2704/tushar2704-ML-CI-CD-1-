from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app=applcation

## Route for a Home Page

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template("home.html")
    else:
        data =CustomData