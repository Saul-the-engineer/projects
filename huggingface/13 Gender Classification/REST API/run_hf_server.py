# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 21:48:05 2022

@author: saulg
"""

import flask
from transformers import pipeline
import re 

app = flask.Flask(__name__)
model = None

def load_model():
    # Change `transformersbook` to your Hub username
    model_id = "Saulr/distilbert-base-uncased-finetuned-gender-classification"
    global model
    model = pipeline("text-classification", model=model_id)
    
def prepare_datapoint(string):
    string = str(string)
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    string = preprocessing(text_cleaning_re, string)
    return string

def preprocessing(regex, text):
  text = re.sub(regex, ' ', str(text).lower()).strip()
  return text
    
@app.route("/predict", methods = ["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":

        # read the data
        text = flask.request.files["text"].read()

        # preprocess the image and prepare it for classification
        text = prepare_datapoint(text)

        # classify the input image and then initialize the list
        # of predictions to return to the client
        preds = model(text, return_all_scores = True)

        data["predictions"] = []

        # loop over the results and add them to the list of
        # returned predictions
        for instance in preds:
            p = {"Male": float(instance[0]["score"]),
                 "Female": float(instance[1]["score"])
                 }
            data["predictions"].append(p)

        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading gender classificaiton model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(host='localhost', port=5000)
    
