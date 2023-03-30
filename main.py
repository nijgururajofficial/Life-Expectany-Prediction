from flask import Flask, render_template, request
import numpy as np
from joblib import load
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

model = load('savedmodels/model.joblib')

@app.route('/result',methods=['POST','GET'])
def predict():
    features = [float(x) for x in request.form.values()]
    final = [np.array(features)]
    result = model.predict(final)
    return render_template('result.html', prediction = result)

if __name__ == "__main__":
    app.run(debug=True, port=8000)