#%%
import streamlit as st
from pyAudioAnalysis import audioTrainTest as aT

st.header("Machine Learning Demonstrator")

#%%
aT.extract_features_and_train(["./data/training/links","./data/training/rechts"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)

#%%

aT.file_classification("./data/test/test.wav", "svmSMtemp","svm")
# %%
