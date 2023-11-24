# App to predict the health of fetus
# Using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

st.title('Fetal Health Classification: A Machine Learning App')

# display image
st.image('fetal_health_image.gif', width = 650)

st.subheader("Use this application to predict the health status of mothers' fetuses")

# Asking users to input their own data
fetalhealth_file = st.file_uploader('Upload patient data for a diagnostic')

# Display an example dataset and prompt the user 
# to submit the data in the required format.
st.write("Please ensure that your data adheres to this specific format:")

# Cache the dataframe so it's only loaded once
@ st.cache_data
def load_data(filename):
  df = pd.read_csv(filename)
  df.drop('fetal_health',axis=1,inplace = True)
  return df.head(3)
data_format = load_data('fetal_health.csv')
st.dataframe(data_format, hide_index = True)

# Check to ensure that proper file is uploaded
if fetalhealth_file is not None :
  fetalhealth_df = pd.read_csv(fetalhealth_file)
else:
    st.stop()

# Reading the pickle files that we created before 
rf_pickle = open('randomforest_fetalhealth.pickle', 'rb') 
clf = pickle.load(rf_pickle)
rf_pickle.close()

fetalhealth_df.dropna(inplace = True)

predictions = clf.predict(fetalhealth_df)
probabilities = np.max(clf.predict_proba(fetalhealth_df), axis = 1)

# adding predictions and probabilities to dataframe
fetalhealth_df['Predicted Fetal Health'] = predictions
fetalhealth_df['Prediction Probability'] = probabilities

# convert numerary coding of fetal health to string descriptors
fetalhealth_df.loc[fetalhealth_df['Predicted Fetal Health'] == 1, 'Predicted Fetal Health'] = 'Normal'
fetalhealth_df.loc[fetalhealth_df['Predicted Fetal Health'] == 2, 'Predicted Fetal Health'] = 'Suspect'
fetalhealth_df.loc[fetalhealth_df['Predicted Fetal Health'] == 3, 'Predicted Fetal Health'] = 'Pathological'

#define function for conditional formatting
def cond_formatting(x):
    if x == 'Normal':
        return 'background-color: green'
    elif x == 'Suspect':
        return 'background-color: yellow'
    elif x == 'Pathological':
       return 'background-color: red'
    
st.title("Here are your results")

#display DataFrame with conditional formatting applied    
st.dataframe(fetalhealth_df.style.applymap(cond_formatting, subset = "Predicted Fetal Health"))

# Showing additional items
st.subheader("Prediction Performance")
tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

with tab1:
  st.image('featureimportance.svg')
with tab2:
  st.image('confusionmatrix.svg')
with tab3:
    df = pd.read_csv('class_report.csv', index_col=0)
    st.dataframe(df)