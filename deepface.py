import streamlit as st
import time
import pandas as pd
from PIL import Image
from deepface import DeepFace
import numpy as np

st.title("DeepFace Analysis App")

# Upload an image
image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if image is not None:
    # Perform DeepFace analysis on the uploaded image
    img = Image.open(image)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    res = DeepFace.analyze(np.array(img), actions=("gender", "age", "race", "emotion"))

    data = {}

    for gender, probability in res[0]['gender'].items():
        data[f'Gender_{gender}'] = [probability]

    for race, probability in res[0]['race'].items():
        data[f'Race_{race}'] = [probability]

    data['Age'] = [res[0]['age']]
    data['Dominant Gender'] = [res[0]['dominant_gender']]
    data['Dominant Race'] = [res[0]['dominant_race']]
    data['Dominant Emotion'] = [res[0]['dominant_emotion']]

    df = pd.DataFrame(data)

    dominant_gender_value = df.loc[0, 'Dominant Gender']
    dominant_race_value = df.loc[0, 'Dominant Race']
    dominant_emotion_value = df.loc[0, 'Dominant Emotion']

    original_result = res[0]
    dominant_gender_result = original_result['gender'][dominant_gender_value]
    dominant_race_result = original_result['race'][dominant_race_value]
    dominant_emotion_result = original_result['emotion'][dominant_emotion_value]

    dominant_gender_percentage = dominant_gender_result
    dominant_race_percentage = dominant_race_result
    dominant_emotion_percentage = dominant_emotion_result

    report = f"The estimated age is {res[0]['age']} years. The dominant gender is {dominant_gender_value} ({dominant_gender_percentage:.5f}%). The dominant race is {dominant_race_value} ({dominant_race_percentage:.5f}%). The dominant emotion is {dominant_emotion_value} ({dominant_emotion_percentage:.5f}%)."

    st.subheader("Analysis Report:")

    st.text(report)
