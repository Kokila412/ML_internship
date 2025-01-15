import pandas as pd
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import emoji

model=pkl.load(open('model.pkl','rb'))
cv=pkl.load(open('cv.pkl','rb'))
review=st.text_input("enter movie review")

if st.button('predict'):
    # scaler=TfidfVectorizer(max_features=2500)
    review_scale=cv.transform([review]).toarray()
    prediction=model.predict(review_scale)
    if prediction==1:
        st.write('Positive Review ',emoji.emojize(':smile:'))
    else:
        st.write('Negative Review',emoji.emojize(':disappointed:'))
    