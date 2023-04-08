from flask import Flask, render_template, request
import pickle
from joblib import load

app = Flask(__name__)

# Load the trained models
odi_model = load('../scorePredictorODI.joblib')
t20_model = load('../scorePredictorT20.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/t20')
def t20():
    return render_template('t20-form.html')


@app.route('/odi')
def odi():
    return render_template('odi-form.html')


@app.route('/predict', methods=['POST'])
def predict():
    venue = request.form['venue']
    bat_team = request.form['bat_team']
    bowl_team = request.form['bowl_team']
    runs = int(request.form['runs'])
    wickets = int(request.form['wickets'])
    overs = float(request.form['overs'])
    runs_last_5 = int(request.form['runs_last_5'])
    wickets_last_5 = int(request.form['wickets_last_5'])
    striker = request.form['striker']
    non_striker = request.form['non_striker']

    # Check which form was submitted
    form_type = request.referrer.split('/')[-1]

    # Make the prediction based on the form type
    if form_type == 't20':
        prediction = t20_model.predict([[venue,bat_team,bowl_team,runs, wickets, overs, runs_last_5, wickets_last_5,striker,non_striker]])
    else:
        prediction = odi_model.predict([[venue,bat_team,bowl_team,runs, wickets, overs, runs_last_5, wickets_last_5,striker,non_striker]])

    output = round(prediction[0])

    return render_template('result.html',prediction=output)


if __name__ == '__main__':
    app.run(debug=True)
