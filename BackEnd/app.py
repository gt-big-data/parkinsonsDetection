from flask import Flask
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
import clock_test.py as ct

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/clock-test")
def get_clock_test_results(): 
    x = np.array(request.args['clockData'])
    model = ct.get_clock_model()
if __name__ == '__main__':
    app.run()