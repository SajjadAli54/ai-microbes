import joblib as job
import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import StandardScaler

# Create flask app
app = Flask(__name__)
# loaded the joblib model
model = job.load("./random_forest.joblib")
sc = StandardScaler()

# this app will take me to the home page

# defining the function for home page


@app.route('/')
def Home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    features = sc.fit_transform(features)
    prediction = model.predict(features)

    return render_template("index.html", prediction_text="Mirco-life is  {}".format(prediction))

# export FLASK_ENV=development
# export FLASK_APP=app
# python -m flask run
# flask run

if __name__ == "__main__":
    app.run()
