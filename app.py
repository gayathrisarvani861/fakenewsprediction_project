import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


pickle_in=open("fnpickle.pkl","rb")
model=pickle.load(pickle_in)


dataframe = pd.read_csv('news1.csv')

x = dataframe['ORIGINAL']
y = dataframe['fakeOrNot']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)


def welcome():
    return "Welcome All"

def predict_news(news):
    tfidf_train =tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test = tfidf_vectorizer.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfidf_vectorizer.transform(input_data)
    prediction=model.predict(vectorized_input_data)
    return prediction

    

def main():
    """st.title( ")"""
    html_temp = """
    <div style="background-color:violet;padding:10px">
    <h2 style="color:white;text-align:center;">Fake News Prediction </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    news = st.text_input("ENTER THE NEWS")
    
    result=""
    if st.button("Predict the news"):
        result=predict_news(news)
    st.success('The news you have given is {}'.format(result))
    

     
    
if __name__=='__main__':
    main()