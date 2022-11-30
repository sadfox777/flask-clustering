import os
import flask
import numpy as np
import pickle
from flask import Flask, redirect, url_for, request, render_template

# Membuat instance dari kelas
app = Flask(__name__, template_folder='templates')

#app.route memetakan URL ke fungsi tertentu yang akan menangani logika untuk URL tersebut
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

# Membuat fungsi untuk memprediksi
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 4)
    loaded_model = pickle.load(
        open("./model/model.pkl", "rb")) # Load "model.pkl"
    # Memprediksi nilai menggunakan model yang dimuat
    result = loaded_model.predict(to_predict)
    return result[0]

# Membuat fungsi untuk mengambil nilai dari form
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        total_cases = request.form['total_cases']
        total_active_cases = request.form['total_active_cases']
        population = request.form['population']
        total_deaths = request.form['total_deaths']
        
        to_predict_list = list(map(float, [total_cases, total_active_cases, population, total_deaths]))
        result = ValuePredictor(to_predict_list)

        # Hasil clustering inputan
        if float(result) == 0:
            clustering = 'Cluster 0: Termasuk dalam populasi jumlahnya palang yang rendah, total kasus paling rendah, total kasus aktif paling rendah, dan total kematian paling rendah.'
        elif float(result) == 1:
            clustering = 'Cluster 1: Termasuk dalam populasi yang tinggi, tingkat kematian yang tinggi, dan tingkat kasus aktif yang tinggi.'
        else:
            clustering = 'Cluster 2: Termasuk dalam populasi yang sedang, tingkat kematian sedang, dan tingkat kasus aktif sedang.'


        return render_template("result.html", clustering=clustering)

if __name__ == "__main__":
    app.run(debug=False)  # gunakan debug = False untuk jupyter notebook
