# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 08:32:25 2021

@author: Neel
"""

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from flask import Flask, request, render_template


df=pd.read_csv('static/news.csv')

df.shape
df.head()

labels=df.label
labels.head()

x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

pac=PassiveAggressiveClassifier(max_iter=20)
pac.fit(tfidf_train,y_train)

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
score = str(score)

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html',text="Real/Fake")

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    inp_string = [text]
    tfidf_input = tfidf_vectorizer.transform(inp_string)
    processed_text = pac.predict(tfidf_input)[0]
    return render_template('index.html',text="The article is "+processed_text)

if __name__ == "__main__":
    app.run(debug=True)
    