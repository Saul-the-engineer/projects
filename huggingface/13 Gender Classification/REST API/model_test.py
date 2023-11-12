# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 23:44:48 2022

@author: saulg
"""


from transformers import pipeline


def load_model():
    # Change `transformersbook` to your Hub username
    model_id = "Saulr/distilbert-base-uncased-finetuned-gender-classification"
    global model
    model = pipeline("text-classification", model=model_id)
    
text = "rainbows and butterflies"
 
load_model()   
preds = model(text, return_all_scores = True)