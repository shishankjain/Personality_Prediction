# -*- coding: utf-8 -*-
from flask import Flask,render_template,url_for,request
import pickle
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
import numpy as np
from ipywidgets import widgets, interact
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
app.config ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

db = SQLAlchemy(app)

class database(db.Model):
    id=db.Column('user_id',db.Integer, primary_key=True)
    name = db.Column(db.String(20))
    text=db.Column(db.String(1000))
    personality_score=db.Column(db.Integer(3))

def __init__(self, name, text, personality_score):
    self.name=name
    self.text=text
    self.personality_score= personality_score

# load the model from disk
nlp = spacy.load("en_core_web_sm")


def tokeniser(sentence):
 
    # Remove ||| from kaggle dataset
    sentence = re.sub("[]|||[]", " ", sentence)

    # remove reddit subreddit urls
    sentence = re.sub("/r/[0-9A-Za-z]", "", sentence)

    # remove MBTI types
    MBTI_types = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
              'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ',
              'MBTI']
    MBTI_types = [ti.lower() for ti in MBTI_types] + [ti.lower() + 's' for ti in MBTI_types]

    tokens = nlp(sentence)

    tokens = [ti for ti in tokens if ti.lower_ not in STOP_WORDS]
    tokens = [ti for ti in tokens if not ti.is_space]
    tokens = [ti for ti in tokens if not ti.is_punct]
    tokens = [ti for ti in tokens if not ti.like_num]
    tokens = [ti for ti in tokens if not ti.like_url]
    tokens = [ti for ti in tokens if not ti.like_email]
    tokens = [ti for ti in tokens if ti.lower_ not in MBTI_types]


    # lemmatize
    tokens = [ti.lemma_ for ti in tokens if ti.lemma_ not in STOP_WORDS]
    tokens = [ti for ti in tokens if len(ti) > 1]

    return tokens

dummy_fn = lambda x:x


with open('cv.pickle', 'rb') as f:
    cv = pickle.load(f)
    
with open('idf_transformer.pickle', 'rb') as f:
    idf_transformer = pickle.load(f)
    
# loading the pickle files with the classifiers
with open('LR_clf_IE_kaggle.pickle', 'rb') as f:
    lr_ie = pickle.load(f)
with open('LR_clf_JP_kaggle.pickle', 'rb') as f:
    lr_jp = pickle.load(f)
with open('LR_clf_NS_kaggle.pickle', 'rb') as f:
    lr_ns = pickle.load(f)
with open('LR_clf_TF_kaggle.pickle', 'rb') as f:
    lr_tf = pickle.load(f)

    



@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        if not request.form['name'] or not request.form['city'] or not request.form['text']:
            print('Please enter all the fields', error)
        else:
            message = request.form['message']
        if(len(message)>10):
            def eval_string(my_post):
                c = cv.transform([tokeniser(my_post)])
                x = idf_transformer.transform(c)
    
                ie = lr_ie.predict_proba(x).flatten()
                ns = lr_ns.predict_proba(x).flatten()
                tf = lr_tf.predict_proba(x).flatten()
                jp = lr_jp.predict_proba(x).flatten()
    
                score=((ie[1]+ns[1]+tf[0]+jp[0])/4)*100
                score=int(round(score))
                predicted_score=score
            my_prediction=eval_string(message)
            data=database(request.form['name'],request.form['text'],['score'])
            db.session.add(data)
            db.session.commit()
            print('Record was successfully submitted')
        else:
            my_prediction=3
        
    return render_template('home.html',prediction = my_prediction)


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)

