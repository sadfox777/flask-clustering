import os
import numpy as np
import flask
import pickle
from flask import Flask, redirect, url_for, request, render_template


# creating instance of the class
app = Flask(__name__, template_folder='templates')

# to tell flask what url should trigger the function index()


@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 2)
    loaded_model = pickle.load(
        open("./model/model.pkl", "rb"))  # load the model
    # predict the values using loded model
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        name = request.form['name']
        gender = request.form['gender']
        age = request.form['age']
        height = request.form['height']
        weight = request.form['weight']

        to_predict_list = list(map(float, [height, weight]))
        result = ValuePredictor(to_predict_list)

        #CLuster Index 
        #0 = Obesitas
        #1 = Tidak Obesitas
        #2 = Normal
        
        if float(result) == 0:
            prediction = 'You are Obesitas'
        elif float(result) == 1:
            prediction = 'You are not Obesitas'
        elif float(result) == 2:
            prediction = 'You are Normal'


        return render_template("result.html", prediction=prediction, name=name)


if __name__ == "__main__":
    app.run(debug=False)  # use debug = False for jupyter notebook
