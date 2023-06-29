import streamlit as st
from pycaret.classification import *
import pandas as pd
import numpy as np
import plotly.express as px

def add_class(df, Activity_1_Start_Time,Activity_1_Finish_Time, Activity_1, Activity_2_Start_Time,Activity_2_Finish_Time, Activity_2, Activity_3_Start_Time,Activity_3_Finish_Time, Activity_3):
    df['class'] = 'NAN'
    df['class'] = np.where(df['Time (s)'].between(Activity_1_Start_Time,Activity_1_Finish_Time), Activity_1, 0)
    df['class'] = np.where(df['Time (s)'].between(Activity_2_Start_Time,Activity_2_Finish_Time), Activity_2, df['class'])
    df['class'] = np.where(df['Time (s)'].between(Activity_3_Start_Time,Activity_3_Finish_Time), Activity_3, df['class'])
    df = df[df['class'] != 'NAN']
    df = df[df['class'] != '0']
    return df

def plot_annotations():
    fig = px.line()
    for class_name in df.class.unique():
        class_data = df[df['class'] == class_name]
        fig.add_line(class_data, x = 'Time (s)', y = 'Absolute acceleration (m/s^2)', name = class_name)
        return fig


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
        Activity_1_Start_Time = st.slider("Activity 1 Start Time", min_value = 0.0, max_value = len(data)/100)
        Activity_1_Finish_Time = st.slider("Activity 1 End Time", min_value = 0.0, max_value = len(data)/100)
        st.success("Activity 2")
        Activity_2 = st.text_input("Activity 2")
        Activity_2_Start_Time = st.slider("Activity 2 Start Time", min_value = 0.0, max_value = len(data)/100)
        Activity_2_Finish_Time = st.slider("Activity 2 End Time", min_value = 0.0, max_value = len(data)/100)
        st.success("Activity 3")
        Activity_3 = st.text_input("Activity 3")
        Activity_3_Start_Time = st.slider("Activity 3 Start Time", min_value = 0.0, max_value = len(data)/100)
        Activity_3_Finish_Time = st.slider("Activity 3 End Time", min_value = 0.0, max_value = len(data)/100)
        proceed = st.button("Annotate Data", use_container_width = True)
        if proceed:
            df = add_class(data, Activity_1_Start_Time,Activity_1_Finish_Time, Activity_1, Activity_2_Start_Time,Activity_2_Finish_Time, Activity_2, Activity_3_Start_Time,Activity_3_Finish_Time, Activity_3)
            st.plotly_chart(plot_annotations())
