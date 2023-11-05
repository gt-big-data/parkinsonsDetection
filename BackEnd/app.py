from flask import Flask, request
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import tensorflow as tf
from clock_test import get_clock_model

app = Flask(__name__)
clockModel = None

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/clock-test")
def get_clock_test_results(): 
    content = request.json
    print(content['clockData'])
    arr = np.array(content['clockData']).reshape((1,8))
    print(arr.shape)
    pred = clockModel.predict(arr)
    print(pred[0])
    return "yes"
    # if 'clockData' in request.args:
    #     print("yes")
    #     return str(request.args['clockData'])
    # x = np.array(request.args['clockData'])
    # model = ct.get_clock_model()
if __name__ == '__main__':
    clockModel = get_clock_model()
    app.run()