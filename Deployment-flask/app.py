import json
import numpy as np
from flask import Flask, request, jsonify, render_template
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk 
import re
import seaborn as sns
from google_trans_new import google_translator 
from pprint import pprint
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.utils import resample
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

def userid(tweet):
    ''' This function calculates the number of userids in the tweets'''
    count = 0
    for i in tweet.split():
        if i[0] == '@':
            count += 1
    return count

def profanity_vector(tweet):
    
    ''' This functions calculates the profanity vector for a given tweet '''
    
    bad_words = pd.read_csv('Hinglish_Profanity_List.csv', engine='python', header=None, encoding='cp1252')
    bad_words.columns = ['Hinglish', 'English', 'Level']
    hinglish = bad_words['Hinglish'].values
    level = bad_words['Level'].values
    PV = [0] * len(level)
    for word in tweet.split():
        if word in hinglish:
            idx = np.where(hinglish == word)
            PV[level[idx][0]] = 1
    return PV

def translation(tweet):
    
    ''' This function translates the hinglish tweet into english '''
    from googletrans import Translator
    translator = Translator()
    trans = translator.translate(tweet)
    trans_tweet = trans.text
    
    return trans_tweet.lower()

def stopword(tweet):
    
    ''' This function removes the stopwords from the given sentence'''
    stop_words = set(STOPWORDS)
    
    sentence = []
    for word in tweet.split():
        if word not in stop_words:
            sentence.append(word)
    return sentence

def Lemmatizer(tweet):
    
    ''' This function uses NLTK lemmatization method and clean the sentence'''
    lemma = []
    sentence = []
    lemmatizer = WordNetLemmatizer()
    
    for word in tweet:
        sentence.append(lemmatizer.lemmatize(word))
    lemma.append(' '.join(sentence))
    return lemma

def SID(tweet):
    
    ''' This function calculates the NLTK sentiments and return the negative, neutral, postive and compound values'''
    negative = []
    neutral = []
    positive = []
    compound = []
    
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(tweet)
    
    negative.append(sentiment_score['neg'])
    neutral.append(sentiment_score['neu'])
    positive.append(sentiment_score['pos'])
    compound.append(sentiment_score['compound'])
    
    return negative, neutral, positive, compound

def feature_process_pred(clean_data, userids, PV, test):
    ''' This function except the clean translated data and return dataset after stacking userids, profanity vector, negative sentiment, neutral sentiment, 
                    positive sentiment, compound sentiment, n-grams and tfidf features'''
    
    vectorizer = CountVectorizer()
    tfidf = TfidfVectorizer()
    scaler = MinMaxScaler()
    negative = []
    neutral = []
    positive = []
    compound = []
    clean_data = clean_data.tolist()
    neg, neu, pos, comp = SID(clean_data)
    negative.append(neg), neutral.append(neu), positive.append(pos), compound.append(comp)

    clean_data_SW = stopword(clean_data)
    clean_data_lemm = Lemmatizer(clean_data_SW)
    if test == False:
        ohe_model = vectorizer.fit(clean_data_lemm)
        tfidf_model = tfidf.fit(clean_data_lemm)
        #scaler_fit = scaler.fit(userids)
        pickle.dump((ohe_model), open('ohe_model.pkl', 'wb'))
        pickle.dump((tfidf_model), open('tfidf_model.pkl', 'wb'))
        #pickle.dump((scaler_fit), open('scaler_fit.pkl', 'wb'))

    else:
        ohe_model = pickle.load(open('ohe_model.pkl', 'rb'))
        tfidf_model = pickle.load(open('tfidf_model.pkl', 'rb'))
        #scaler_fit = pickle.load(open('scaler_fit.pkl', 'rb'))

    n_grams = ohe_model.transform(clean_data_lemm)
    tfidf_ngrams = tfidf_model.transform(clean_data_lemm)
    #userids = scaler_fit.transform(userids)

    negative = np.asarray(negative)
    neutral = np.asarray(neutral)
    positive = np.asarray(positive)
    compound = np.asarray(compound)
    dataset = hstack((userids, PV.reshape(1,-1), negative, neutral, positive, compound, n_grams, tfidf_ngrams))

    return dataset

def cleaning_pred(tweet):
    
    ''' This functions clean the input text'''
    
    clean_data_hinglish = []
    clean_translated_data = []
    prof_vector = []
    
    userids = [userid(tweet)]
    clean_text = []
    tweet = re.sub(r'\\n', ' ', tweet)  # replacing '\\n' with a space
    tweet = re.sub(r',', ' ', tweet)    # replacing ','  with a space
    tweet = re.sub(r'RT|rt', '', tweet)
        
    for word in tweet.split():
        if word[0] == '@':              # removing user_ids 
            clean_word = re.sub(word, 'username', word)
        else:
            clean_word = word.lower()       # lowercase all the words
            clean_word = re.sub(r'^#\w+', ' ', clean_word)
            #clean_word = re.sub(r'^\\[a-z0-9].*\\[a-z0-9{3}+]*[^\\n]$', '', clean_word)   # removing emotions in unicode
            clean_word = re.sub(r'\\', ' ', clean_word)
            clean_word = re.sub(r'^https:[\a-zA-Z0-9]+', '', clean_word)              # replacing url link with 'url'
            #clean_word = re.sub(r'[^a-z].\w+', '', clean_word)           # removing evering thing except a-z
            clean_word = re.sub(r'[!,.:_;$%^\'\#"&]', '', clean_word)
            clean_text.append(clean_word)
                
    clean_text = (' ').join(clean_text)
    
    PV = profanity_vector(clean_text)  # calling profanity_vector function
    translated_tweet = translation(clean_text)  #calling translated_tweet function

        
    clean_data_hinglish = np.asarray(clean_text)
    user_ids = np.asarray(userids).reshape(1,-1)
    prof_vector = np.asarray(PV).reshape(1,-1)
    clean_translated_data = np.asarray(translated_tweet)

    return clean_data_hinglish, user_ids, prof_vector, clean_translated_data

@app.route('/predict',methods=['POST'])
def predict():
    XGB = XGBClassifier()
    CatBoost = CatBoostClassifier()
    data = request.form.values()
    data = str(data)
    clean_data_hinglish, user_ids, prof_vector, clean_translated_data = cleaning_pred(data)
    Test = feature_process_pred(clean_translated_data, user_ids, prof_vector, test=True)
    pred_test = pd.DataFrame()
    base_classifier = ['SVC_Linear', 'SVC_RBF', 'SVC_Poly', 'XGB', 'CatBoost', 'KNN']
    for classifier in base_classifier:
        if classifier == 'XGB':
            clf = XGBClassifier()
            clf.load_model('XGB')
        elif classifier == 'CatBoost':
             clf = CatBoostClassifier()
             clf.load_model('CatBoost')
        else:
            clf = pickle.load(open(classifier+'.pkl', 'rb'))
        if classifier == 'XGB':
            pred = clf.predict_proba(Test)
            pred = pred[0:,0].reshape(1,-1)
        else:
            pred = clf.predict_proba(Test)
        for i in range(len(pred[0])):
           pred_test[f"{classifier}{i}"] = pred[0:,i]
    meta_classifier = pickle.load(open('meta_classifier.pkl', 'rb'))
    meta_data = hstack((Test, pred_test))

    final_pred = meta_classifier.predict(meta_data)
    
    if final_pred == 0:
      output = 'Non Offensive'
      return render_template('index.html', prediction_text='Your Speech is {}'.format(output))
    
    elif final_pred == 1:
      output = 'Hate Speech'
      return render_template('index.html', prediction_text='Your Speech is {}'.format(output))
    
    else:
      output = 'Offensive'
      return render_template('index.html', prediction_text='Your Speech is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)