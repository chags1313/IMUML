import streamlit as st
from pycaret.classification import *
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="🤾‍♀️ IMU ML",
    page_icon="🤾‍♀️",
    layout="wide"
)

st.title('🤾‍♀️ IMU ML')
st.markdown("Predicting activities from movement data and machine learning")

upload, annotation, ml, results = st.tabs(['Upload', 'Annotation', 'ML', 'Results'])

with upload:
  file = st.file_uploader("Upload Your Phyphox Data File")

  if file is not None:
    with upload:
      data = pd.read_csv(file)
      col1, col2 = st.columns(2)
      x = col1.selectbox('X Axis', data.columns)
      y = col2.multiselect('Y Axis', data.columns)
      fig = px.line(data, x = x, y = y)
      st.plotly_chart(fig, use_container_width = True)
    with annotation:
        st.success("Activity 1")
        Activity_1 = st.text_input("Activity 1")
        Activity_1_Start_Time = st.number_input("Activity 1 Start Time")
        Activity_1_Finish_Time = st.number_input("Activity 1 End Time")
        st.success("Activity 2")
        Activity_2 = st.text_input("Activity 2")
        Activity_2_Start_Time = st.number_input("Activity 2 Start Time")
        Activity_2_Finish_Time = st.number_input("Activity 2 End Time")
        st.success("Activity 3")
        Activity_3 = st.text_input("Activity 3")
        Activity_3_Start_Time = st.number_input("Activity 3 Start Time")
        Activity_3_Finish_Time = st.number_input("Activity 3 End Time")
