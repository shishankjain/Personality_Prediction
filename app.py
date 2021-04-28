# -*- coding: utf-8 -*-
from flask import Flask , render_template,url_for,request, flash, redirect
import pickle
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
from pyresparser import ResumeParser
import numpy as np
#from flask_bcrypt import bcrypt
from flask_sqlalchemy import SQLAlchemy
import os
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

app = Flask(__name__)
app.config ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config ['UPLOAD_FOLDER'] = "G:\\NEWW\\Personality_Prediction"
db = SQLAlchemy(app)
#bcrypt= Bcrypt(app)

#SKILLS REQUIRED in SET
text={'Python','Machine Learning', 'Java', 'SQL', 'C'}
#EXPERIENCE RANGE MENTIONED IN LIST
req_experience=[0,2]


#CREATING DATABASE table name=database
class database(db.Model):
    id=db.Column('user_id',db.Integer, primary_key=True)
    #rank=db.Column(db.Integer)
    name = db.Column(db.String(20))
    text=db.Column(db.String(1000))
    personality_score=db.Column(db.Integer)
    skills_score=db.Column(db.Integer)
    experience_score=db.Column(db.Integer)
    total_score=db.Column(db.Integer)

    def __init__(self, name, text, personality_score, skills_score, experience_score, total_score):
        #self.rank=rank
        self.name=name
        self.text=text
        self.personality_score= personality_score
        self.experience_score=experience_score
        self.skills_score=skills_score
        self.total_score=total_score


        
# load the model from disk
nlp = spacy.load("en_core_web_sm")

upload_location="C:\\Users\\jshas\\Desktop\\Personality_Prediction\\Personality_Prediction\\static"

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
        if not request.form['name'] or not request.form['message']:
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
                return score
            personality_score=eval_string(message)
            my_prediction=personality_score
            f=request.files['upload']
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
            resume_data = ResumeParser(f.filename).get_extracted_data()
            print(resume_data)
            skills=(resume_data.get("skills"))
            experience=resume_data.get("total_experience")
            
            #Calculating SKills Score out of 40
            skills_score=((len((text & set(skills))))/len(text))*40
            
            #Calculating Experience Score out of 40
            if(experience>req_experience[1]):
                experience_score=40
            elif (experience<req_experience[0]):
                experience_score=0
            else:
                experience_score=(experience/req_experience[1])*40
            
            candidate_total=experience_score+skills_score+(personality_score*0.2)
            experience_score=experience_score*2.5
            skills_score=skills_score*2.5
            # candidates = database.query.order_by(database.total_score.desc()).all() #fetch them all in one query
            # rank=candidates.id
            data=database(request.form['name'],request.form['message'], personality_score,skills_score, experience_score,candidate_total)
            db.session.add(data)
            db.session.commit()
            print('Record was successfully submitted')
            
        else:
            my_prediction=3
        
    return render_template('home.html',predicton=my_prediction)



    #return render_template('resume.html')

# @app.route('/resumesubmit')
# def resumesubmit():
#     return render_template('resume.html')

# @app.route('/resume', methods=['POST'])
# def resume():
#     if request.method == 'POST':
#         message = request.form['message']
#         print(type(message))
#         resume_data = ResumeParser(message).get_extracted_data()
#         print (resume_data)
#     return render_template('home.html')



@app.route('/admin')
def admin():
    return render_template('admin.html',database=database.query.order_by(database.total_score.desc()).all())

if __name__ == '__main__':
    db.create_all()
    db.session.commit()
    app.run(debug=True)

