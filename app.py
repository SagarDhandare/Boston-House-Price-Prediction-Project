from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('LinReg.pickle', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        CRIM = float(request.form['CRIM'])
        ZN = float(request.form['ZN'])
        INDUS = float(request.form['INDUS'])
        CHAS = float(request.form['CHAS'])
        NOX = float(request.form['NOX'])
        RM = float(request.form['RM'])
        AGE = float(request.form['AGE'])
        DIS = float(request.form['DIS'])
        PTRATIO = float(request.form['PTRATIO'])
        B = float(request.form['B'])
        LSTAT = float(request.form['LSTAT'])

        prediction = model.predict(standard_to.fit_transform([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, PTRATIO, B, LSTAT]]))
        output=round(prediction[0],2)
        return render_template('index.html', prediction_text = 'The Price is {}'.format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

