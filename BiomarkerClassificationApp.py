import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

st.write("""
# COVID mortality Prediction App
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('age',18, 88, 54)
    systolicBP = st.sidebar.slider('systolicBP', 50, 230, 130)
    rr = st.sidebar.slider('rr', 0, 75, 23)
    cr = st.sidebar.slider('cr', 0.0, 10.0, 2.0)
    crp = st.sidebar.slider('crp', 0.0, 10.0, 2.0)
    data = {'age': age,
            'systolicBP': systolicBP,
            'rr': rr,
            'cr': cr,
            'crp': crp}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

biomarkers = pd.read_csv('biomarkers.csv')
X = biomarkers[['age','systolicBP','rr','cr','crp']]
Y = biomarkers['death']

clf = LogisticRegression()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

#st.subheader('Class labels and their corresponding index number')
#st.write(biomarkers.target_names)

st.subheader('Prediction')
#st.write(biomarkers.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)