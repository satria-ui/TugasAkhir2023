from math import log2, pow
import os
import numpy as np
from scipy.fftpack import fft
import gradio as gr
from datetime import date
import librosa
import pandas as pd
import pickle
import joblib

def get_pitch(freq):
    A4 = 440
    C0 = A4 * pow(2, -4.75)
    name = ["neutral",  "calm",  "happy",  "sad",  "angry",  "fearful",  "disgust",  "surprised"]
    h = round(len(name) * log2(freq / C0))
    n = h % len(name)
    return name[n]

def main_note(input):
    rate, y = input
    print(rate)
    print(y)
    print(len(y))
    if len(y.shape) == 2:
        y = y.T[0]
    N = len(y)
    T = 1.0 / rate
    x = np.linspace(0.0, N * T, N)
    yf = fft(y)
    yf2 = 2.0 / N * np.abs(yf[0 : N // 2])
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    volume_per_pitch = {}
    total_volume = np.sum(yf2)
    for freq, volume in zip(xf, yf2):
        if freq == 0:
            continue
        pitch = get_pitch(freq)
        if pitch not in volume_per_pitch:
            volume_per_pitch[pitch] = 0
        volume_per_pitch[pitch] += 1.0 * volume / total_volume
    volume_per_pitch = {k: float(v) for k, v in volume_per_pitch.items()}
    # dictionary
    return volume_per_pitch

def reverse_label_encoder(data):
    mapping = {'angry': 0, 'fear': 1, 'disgust': 2, 'happy': 3, 'neutral': 4, 'sad': 5}
    reverse_mapping_dict = {v: k for k, v in mapping.items()}
    keys = list(reverse_mapping_dict.values())

    result = {f'{keys[i]}': data[0, i] for i in range(len(keys))}

    return result

def mfcc_formula(audio):
    y,sr = librosa.load(audio, duration=4, offset=0.5)
    n_fft = int(sr * 0.02)
    hop_length = n_fft // 2
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length).T, axis=0)

    extracted_audio = pd.DataFrame(mfcc)
    return extracted_audio.T

def svm_prediction(input):
    loaded_model = pickle.load(open("./ML_Model/svm_model.sav", 'rb'))

    dataset = mfcc_formula(input)
    scaler = joblib.load('./Scaler/Z-ScoreScaler.joblib')
    x_scaled = scaler.transform(dataset)
    x_test = pd.DataFrame(x_scaled)

    prediction = loaded_model.predict_proba(x_test)
    prediction = reverse_label_encoder(prediction)
    return prediction

def main():
    today = date.today()
    tgl = today.strftime("%B %d, %Y")
    desc = (
        f"<p id='Name'>by Adam Satria</p><br><p>{tgl}</p>"
        )

    gr.Interface(
    svm_prediction,
    inputs = [
        gr.Audio(source="upload", type="filepath"),
    ],
    outputs = [
        gr.Label(),
    ],
    examples=[
        [os.path.join(os.path.dirname(__file__),"audio/recording1.wav")],
        [os.path.join(os.path.dirname(__file__),"audio/recording2.wav")],
    ],
    title = "English Audio Emotion Recognition",
    description = desc,
    allow_flagging = "never",
    css = (
        "p{color: gray; text-align: center; font-size: 	1.2500em;}"
        "#Name{color: lightgrey; text-align: center; font-size: 1.875em; margin-bottom: 0px;}"
        "footer{visibility: hidden;}"
    )
    ).launch(share=True)

if __name__ == "__main__":
    main()
